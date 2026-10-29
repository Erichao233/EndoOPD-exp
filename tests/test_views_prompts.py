import numpy as np
from PIL import Image

from endoexp.fields import FIELDS
from endoexp.imaging import fingerprint, fit, mask_regions, visual_tokens
from endoexp.prompts import letter_distribution, letter_variants, mcq, parse_report, report_template
from endoexp.views import VIEWS, clean_overlays, crop_context, draw_boxes, insert_inset, insert_text

from conftest import frame


def test_fit_respects_patch_grid_and_budget():
    img = Image.new("RGB", (720, 576))
    out = fit(img, 409600)
    assert out.size[0] % 32 == 0 and out.size[1] % 32 == 0
    assert out.size[0] * out.size[1] <= 409600 and visual_tokens(out.size) <= 400
    small = Image.new("RGB", (100, 80))
    assert fit(small, 409600).size == (96, 64)
    assert fit(small, 409600, upscale=True).size[0] > 300


def test_fingerprint_survives_resizing_but_separates_frames():
    a = frame(1, polyp=(60, 60, 20))
    b = a.resize((180, 144))
    c = frame(2, polyp=(120, 90, 15))
    assert float(fingerprint(a) @ fingerprint(b)) > 0.9
    assert abs(float(fingerprint(a) @ fingerprint(c))) < 0.5
    assert not fingerprint(Image.new("RGB", (64, 64))).any()      # flat images never link


def test_mask_regions_finds_components():
    mask = Image.new("L", (200, 100), 0)
    mask.paste(255, (10, 10, 30, 40))
    mask.paste(255, (150, 50, 190, 90))
    regions = mask_regions(mask)
    assert len(regions) == 2
    assert regions[0]["box"] == [150.0, 50.0, 190.0, 90.0]
    assert abs(regions[0]["fraction"] - 0.08) < 0.01


def test_region_views_change_only_what_they_should():
    img = frame(3, polyp=(100, 80, 20))
    boxed = np.asarray(draw_boxes(img, [[80, 60, 120, 100]]), dtype=int)
    plain = np.asarray(img, dtype=int)
    changed = np.argwhere((boxed != plain).any(axis=2))
    assert changed[:, 0].min() >= 60 and changed[:, 0].max() < 100
    crop = crop_context(img, [[80, 60, 120, 100]])
    assert crop.size[0] < img.size[0] and crop.size[1] < img.size[1]
    contour = VIEWS["contour"](img, [[80, 60, 120, 100]], [Image.new("L", (200, 160), 0)])
    assert contour.size == img.size


def test_shortcut_views():
    img = frame(4, green=True, text=True)
    arr = np.asarray(img, dtype=int)
    green = (arr[..., 1] > 120) & (arr[..., 1] - arr[..., 0] > 60)
    assert green.any()
    cleaned = np.asarray(clean_overlays(img), dtype=int)
    assert not ((cleaned[..., 1] > 120) & (cleaned[..., 1] - cleaned[..., 0] > 60)).any()
    assert cleaned[0:16].max() == 0                    # text in the top border is gone
    assert (cleaned[60:100, 60:140] == arr[60:100, 60:140]).all()   # tissue untouched
    inset = np.asarray(insert_inset(frame(5)), dtype=int)
    assert ((inset[..., 1] > 120) & (inset[..., 1] - inset[..., 0] > 60)).any()
    assert np.asarray(insert_text(frame(5), "POLYP")).max() == 255


def test_mcq_and_report_round_trip():
    text = mcq(FIELDS["paris_type"], hint="The region of interest is marked with a yellow box.")
    assert "A. 0-Ip (pedunculated)" in text and "C. 0-IIa (slightly elevated)" in text
    assert text.splitlines()[0].startswith("The region")
    assert mcq(FIELDS["procedure"], blind=True).startswith("No image")
    template = report_template()
    assert "Paris classification: [0-Ip (pedunculated) | 0-Is (sessile) | 0-IIa (slightly elevated)]" in template
    parsed = parse_report("Procedure: colonoscopy\nParis classification: 0-Is\n**Polyp size:** 5 to 10 mm\nAbnormality: tumor")
    assert parsed["procedure"] == "colonoscopy"
    assert parsed["paris_type"] == "0-Is (sessile)"
    assert parsed["abnormality"] is None


class _Tokenizer:
    vocab = {"A": 1, " A": 2, "B": 3, " B": 4, "C": 5, " C": 6, "D": 7, " D": 8, "E": 9, " E": 10,
             "F": 11, " F": 12, "G": 13, " G": 14, "H": 15, " H": 16, "I": 17, " I": 18, "J": 19, " J": 20}

    def encode(self, text, add_special_tokens=False):
        return [self.vocab[text]]


def test_letter_distribution_handles_missing_letters():
    variants = letter_variants(_Tokenizer())
    assert variants[0] == [1, 2]
    logprobs = {1: np.log(0.6), 2: np.log(0.1), 3: np.log(0.2), 99: np.log(0.05)}
    probs, coverage = letter_distribution(logprobs, variants, 3)
    assert abs(coverage - 0.9) < 1e-9
    assert probs[0] > probs[1] > probs[2] > 0 and abs(sum(probs) - 1) < 1e-9
