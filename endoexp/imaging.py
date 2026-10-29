"""Image utilities shared by sources, views and engines (Pillow and numpy only)."""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

# Qwen3.5 uses 16-pixel patches merged 2x2: one visual token per 32x32 pixels.
PATCH = 32


def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def fit(img: Image.Image, max_pixels: int, factor: int = PATCH, upscale: bool = False) -> Image.Image:
    """Resize so both sides are multiples of ``factor`` and the area is at most ``max_pixels``.

    Engines then see exactly these pixels: Qwen processors leave an image whose
    sides are already multiples of 32 and within their pixel bounds unchanged.
    """
    width, height = img.size
    scale = math.sqrt(max_pixels / (width * height))
    if not upscale:
        scale = min(scale, 1.0)
    new_w = max(factor, round(width * scale / factor) * factor)
    new_h = max(factor, round(height * scale / factor) * factor)
    while new_w * new_h > max_pixels and (new_w > factor or new_h > factor):
        if new_w >= new_h:
            new_w -= factor
        else:
            new_h -= factor
    if (new_w, new_h) == (width, height):
        return img
    return img.resize((new_w, new_h), Image.BICUBIC)


def visual_tokens(size: tuple[int, int], factor: int = PATCH) -> int:
    return (size[0] // factor) * (size[1] // factor)


def fingerprint(img: Image.Image, side: int = 32, margin: float = 0.15) -> np.ndarray:
    """Unit vector of the high-passed thumbnail of the central crop.

    The dot product of two fingerprints is near 1 for the same frame after
    resizing or re-encoding (how one frame reappears across Kvasir releases)
    and near 0 for different frames. The central crop and the high-pass keep
    the black border and the vignetting, which every frame shares, out of it.
    """
    width, height = img.size
    crop = img.crop((round(margin * width), round(margin * height),
                     round((1 - margin) * width), round((1 - margin) * height)))
    thumb = crop.convert("L").resize((side, side), Image.BOX)
    detail = (np.asarray(thumb, dtype=np.float32)
              - np.asarray(thumb.filter(ImageFilter.BoxBlur(2)), dtype=np.float32)).ravel()
    detail -= detail.mean()
    norm = float(np.linalg.norm(detail))
    return detail / norm if norm > 1e-3 else np.zeros_like(detail)


def _components(binary: np.ndarray) -> list[list[tuple[int, int]]]:
    """4-connected components of a boolean array (pure Python breadth-first search)."""
    height, width = binary.shape
    seen = np.zeros_like(binary, dtype=bool)
    components = []
    for y0, x0 in zip(*np.nonzero(binary)):
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        queue, pixels = deque([(y0, x0)]), []
        while queue:
            y, x = queue.popleft()
            pixels.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        components.append(pixels)
    return components


def _downscaled(img: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    width, height = img.size
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        img = img.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.BILINEAR)
    return img, scale


def mask_regions(mask: Image.Image, threshold: int = 128, max_side: int = 256,
                 min_fraction: float = 0.001) -> list[dict]:
    """Connected foreground regions of a (JPEG) mask, largest first.

    Each region has ``box`` in the mask's pixel coordinates, ``center`` in
    normalized coordinates and ``fraction`` of the image area.
    """
    small, scale = _downscaled(mask.convert("L"), max_side)
    binary = np.asarray(small) >= threshold
    height, width = binary.shape
    regions = []
    for pixels in _components(binary):
        fraction = len(pixels) / (width * height)
        if fraction < min_fraction:
            continue
        ys = np.fromiter((p[0] for p in pixels), dtype=np.int64, count=len(pixels))
        xs = np.fromiter((p[1] for p in pixels), dtype=np.int64, count=len(pixels))
        box = [xs.min() / scale, ys.min() / scale, (xs.max() + 1) / scale, (ys.max() + 1) / scale]
        regions.append({
            "box": [round(min(v, lim), 1) for v, lim in zip(box, (*mask.size, *mask.size))],
            "center": [float((xs.mean() + 0.5) / width), float((ys.mean() + 0.5) / height)],
            "fraction": float(fraction),
        })
    regions.sort(key=lambda r: -r["fraction"])
    return regions


def field_of_view(img: Image.Image, threshold: int = 25, max_side: int = 256) -> Image.Image:
    """Binary mask (mode "L", original size) of the endoscopic field of view.

    The field of view is the largest bright region with its interior holes
    (dark lumen) filled; burned-in text in the black border lies outside it.
    """
    small, _ = _downscaled(img, max_side)
    bright = np.asarray(small, dtype=np.int16).max(axis=2) > threshold
    components = _components(bright)
    if not components:
        return Image.new("L", img.size, 255)
    largest = max(components, key=len)
    fov = np.zeros_like(bright)
    for y, x in largest:
        fov[y, x] = True
    # Fill holes: background is what is reachable from the border without crossing the field of view.
    height, width = fov.shape
    outside = np.zeros_like(fov)
    queue = deque()
    for y in range(height):
        for x in (0, width - 1):
            if not fov[y, x] and not outside[y, x]:
                outside[y, x] = True
                queue.append((y, x))
    for x in range(width):
        for y in (0, height - 1):
            if not fov[y, x] and not outside[y, x]:
                outside[y, x] = True
                queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < height and 0 <= nx < width and not fov[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                queue.append((ny, nx))
    mask = Image.fromarray(np.where(outside, 0, 255).astype(np.uint8), mode="L")
    return mask.resize(img.size, Image.NEAREST)


def green_insets(img: Image.Image, max_side: int = 256, min_fraction: float = 0.002) -> list[list[float]]:
    """Boxes of strongly green regions in the lower-left part of the image.

    Olympus ScopeGuide insets in HyperKvasir are bright green drawings on black
    in the lower-left corner. White-light mucosa is never this green.
    """
    small, scale = _downscaled(img, max_side)
    rgb = np.asarray(small, dtype=np.int16)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    binary = (green > 120) & (green - red > 60) & (green - blue > 60)
    height, width = binary.shape
    boxes = []
    for pixels in _components(binary):
        if len(pixels) / (width * height) < min_fraction:
            continue
        ys = [p[0] for p in pixels]
        xs = [p[1] for p in pixels]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs) + 1, max(ys) + 1
        if x1 <= 0.45 * width and y0 >= 0.45 * height:
            boxes.append([x0 / scale, y0 / scale, x1 / scale, y1 / scale])
    return boxes
