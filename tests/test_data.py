from PIL import Image

from endoexp import manifest, sources
from endoexp.fields import FIELDS, GRID, extent_bin, grid_cell
from endoexp.sources import kvasir_vqa


def test_fields_are_well_formed():
    for field in FIELDS.values():
        assert len(field.options) == len(set(field.options)) <= 10
    assert grid_cell(0.1, 0.1) == "upper-left" and grid_cell(0.99, 0.99) == "lower-right"
    assert grid_cell(0.5, 0.5) == "center" and len(GRID) == 9
    assert extent_bin(0.01) == "less than 5%" and extent_bin(0.1) == "5% to 20%" and extent_bin(0.5) == "more than 20%"


def test_question_keys_match_release_text():
    assert kvasir_vqa.question_key("Is there a green/black box artefact?") == "is there a green/black box artefact"
    key = kvasir_vqa.question_key("Are there any abnormalities in the image? Check all that are present.")
    assert key in kvasir_vqa.QUESTIONS


def test_kvasir_vqa_labels(data_root):
    records, stats = sources.load("kvasir_vqa", data_root)
    by_key = {r.key: r for r in records}
    polyp = by_key["clpolyp01"].labels
    assert polyp["paris_type"] == "0-Ip (pedunculated)"
    assert polyp["polyp_size"] == "5 to 10 mm"
    assert polyp["box_overlay"] == "no" and polyp["polyp_count"] == "1" and polyp["instrument_count"] == "0"
    assert "polyp_size" not in by_key["clnormal00"].labels          # "none" is not an option
    assert "abnormality" not in by_key["cluc00"].labels              # two abnormalities: ambiguous
    assert by_key["cluc01"].labels["abnormality"] == "ulcerative colitis"
    assert stats["skipped_answers"]["abnormality"] == 1


def test_hyperkvasir_labeled_and_segmented(data_root):
    labeled, info = sources.load("hyperkvasir", data_root)
    assert info["class_folders"] == 8
    grades = {r.labels.get("uc_grade") for r in labeled}
    assert {"grade 1 (mild)", "grade 3 (severe)"} <= grades
    assert all(r.labels["procedure"] == "gastroscopy" for r in labeled if r.meta["class"].startswith("oesophagitis"))
    segmented, info = sources.load("hyperkvasir_seg", data_root)
    assert info["images"] == 12
    record = next(r for r in segmented if r.key == "seg00")
    assert record.labels["polyp_location"] == "upper-left"
    x0, y0, x1, y1 = record.regions[0]["box"]
    assert 28 < x0 < 40 and 28 < y0 < 40 and 48 < x1 < 62   # circle at (40, 40), r=12, scaled by 1.1


def test_manifest_links_copies_of_the_same_frame(data_root):
    samples, header = manifest.build(data_root, sources.SOURCE_NAMES, seed=1)
    by_uid = {s.uid: s for s in samples}
    merged = by_uid["kvasir_vqa:clpolyp00"]
    assert sorted(m.split(":")[0] for m in merged.members) == ["hyperkvasir", "hyperkvasir_seg", "kvasir_vqa"]
    assert merged.labels["paris_type"] == "0-Is (sessile)" and merged.labels["polyp_location"] == "upper-left"
    x0, y0, x1, y1 = merged.regions[0]["box"]      # rescaled to the 200x160 Kvasir-VQA image
    assert 25 < x0 < 32 and 25 < y0 < 32 and 48 < x1 < 56
    assert header["merged_records"] == 24
    assert not any(s.source == "hyperkvasir_seg" for s in samples)
    groups = {}
    for s in samples:
        assert groups.setdefault(s.group, s.split) == s.split
    again, header2 = manifest.build(data_root, sources.SOURCE_NAMES, seed=1)
    assert header2["hash"] == header["hash"]


def test_manifest_round_trip(data_root, tmp_path):
    samples, header = manifest.build(data_root, ["kvasir_vqa"], seed=3)
    manifest.save(samples, header, tmp_path)
    loaded, loaded_header = manifest.load(tmp_path)
    assert [s.uid for s in loaded] == [s.uid for s in samples]
    assert loaded_header["hash"] == header["hash"]
    with Image.open(data_root / loaded[0].image) as img:
        assert list(img.size) == loaded[0].size
