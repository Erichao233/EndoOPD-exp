"""Image views: privileged views for a teacher and shortcut manipulations.

Privileged views add information that is in principle recoverable from the
student's own image (where the finding is, a closer look). Shortcut
manipulations remove or insert burned-in artefacts without touching tissue.
"""

from __future__ import annotations

from typing import Callable

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from endoexp.imaging import field_of_view, green_insets

ROI_COLOR = (255, 255, 0)


def _line_width(img: Image.Image) -> int:
    return max(2, min(img.size) // 120)


def _union(boxes: list[list[float]]) -> list[float]:
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def draw_boxes(img: Image.Image, boxes: list[list[float]], color=ROI_COLOR) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    width = _line_width(img)
    for x0, y0, x1, y1 in boxes:
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=color, width=width)
    return out


def draw_contours(img: Image.Image, masks: list[Image.Image], color=ROI_COLOR) -> Image.Image:
    """Outline mask boundaries; masks are resized to the image if needed."""
    out = img.copy()
    size = 2 * (_line_width(img) // 2) + 1
    for mask in masks:
        binary = mask.convert("L").resize(img.size, Image.NEAREST).point(lambda v: 255 if v >= 128 else 0)
        edge = ImageChops.subtract(binary, binary.filter(ImageFilter.MinFilter(3)))
        out.paste(color, (0, 0, *img.size), edge.filter(ImageFilter.MaxFilter(size)))
    return out


def crop_context(img: Image.Image, boxes: list[list[float]], context: float = 0.5,
                 min_fraction: float = 0.25) -> Image.Image:
    """Crop around the union of boxes, padded by ``context`` of the box size per side.

    The crop is at least ``min_fraction`` of each image side so tiny findings
    keep some surrounding mucosa.
    """
    x0, y0, x1, y1 = _union(boxes)
    width, height = img.size
    half_w = max((x1 - x0) * (0.5 + context), min_fraction * width / 2)
    half_h = max((y1 - y0) * (0.5 + context), min_fraction * height / 2)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left, right = max(0, cx - half_w), min(width, cx + half_w)
    top, bottom = max(0, cy - half_h), min(height, cy + half_h)
    return img.crop((round(left), round(top), round(right), round(bottom)))


def clean_overlays(img: Image.Image) -> Image.Image:
    """Black out everything outside the field of view and any green inset."""
    out = Image.new("RGB", img.size, (0, 0, 0))
    out.paste(img, mask=field_of_view(img))
    draw = ImageDraw.Draw(out)
    pad = max(2, min(img.size) // 100)
    for x0, y0, x1, y1 in green_insets(img):
        draw.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], fill=(0, 0, 0))
    return out


def insert_inset(img: Image.Image) -> Image.Image:
    """Paste a synthetic scope-position inset (green drawing on black) at the lower left."""
    out = img.copy()
    width, height = img.size
    w, h = int(0.24 * width), int(0.30 * height)
    x0, y0 = int(0.01 * width), height - h - int(0.01 * height)
    draw = ImageDraw.Draw(out)
    draw.rectangle([x0, y0, x0 + w, y0 + h], fill=(0, 0, 0))
    line = max(2, min(img.size) // 150)
    points = [(x0 + w * fx, y0 + h * fy) for fx, fy in
              ((0.15, 0.85), (0.30, 0.55), (0.55, 0.40), (0.75, 0.55), (0.70, 0.80), (0.45, 0.70))]
    draw.line(points, fill=(40, 220, 60), width=line * 2, joint="curve")
    draw.rectangle([x0, y0, x0 + w, y0 + h], outline=(40, 220, 60), width=line)
    return out


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has only the fixed bitmap font
        return ImageFont.load_default()


def insert_text(img: Image.Image, text: str) -> Image.Image:
    """Write ``text`` in white on a black label at the upper left."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    font = _font(max(12, img.size[1] // 14))
    margin = max(4, img.size[1] // 60)
    left, top, right, bottom = draw.textbbox((margin, margin), text, font=font)
    draw.rectangle([left - margin // 2, top - margin // 2, right + margin // 2, bottom + margin // 2],
                   fill=(0, 0, 0))
    draw.text((margin, margin), text, fill=(255, 255, 255), font=font)
    return out


# Each view maps (image, regions, masks) to a new image. ``regions`` are boxes
# in the image's pixel coordinates; ``masks`` are mask images for those boxes.
ViewFn = Callable[[Image.Image, list, list], Image.Image]

VIEWS: dict[str, ViewFn] = {
    "plain": lambda img, boxes, masks: img,
    "box": lambda img, boxes, masks: draw_boxes(img, boxes),
    "contour": lambda img, boxes, masks: draw_contours(img, masks) if masks else draw_boxes(img, boxes),
    "crop": lambda img, boxes, masks: crop_context(img, boxes),
    "clean": lambda img, boxes, masks: clean_overlays(img),
    "insert_inset": lambda img, boxes, masks: insert_inset(img),
    "insert_text_polyp": lambda img, boxes, masks: insert_text(img, "POLYP"),
}

# Short instruction added for views whose marks the model has to interpret.
VIEW_HINTS = {
    "box": "The region of interest is marked with a yellow box.",
    "contour": "The region of interest is outlined in yellow.",
    "crop": "This image is a zoomed-in crop centered on the region of interest.",
}

# Views that need a region of interest.
REGION_VIEWS = {"box", "contour", "crop"}
