from collections import Counter

from endoexp import manifest, sources
from endoexp.experiments import CROP_EXCLUDED, Planner, QueryBuilder, shard_items


def _samples(data_root):
    samples, _ = manifest.build(data_root, sources.SOURCE_NAMES, seed=7)
    return samples


def test_e1_pairs_conditions(data_root):
    planner = Planner(_samples(data_root), seed=7)
    items = planner.e1()
    by_pair = Counter((i.uid, i.field) for i in items)
    conditions = {i.condition for i in items}
    assert conditions == {"normal", "blind", "shuffled"}
    for item in items:
        if item.condition == "shuffled":
            normal = next(i for i in items if (i.uid, i.field, i.condition) == (item.uid, item.field, "normal"))
            assert item.image != normal.image and item.label == normal.label
        if item.condition == "blind":
            assert item.image is None
    assert all(n in (2, 3) for n in by_pair.values())
    assert all(i.split == "test" for i in items)


def test_plans_are_deterministic_and_shards_partition(data_root):
    samples = _samples(data_root)
    first = [i.key for i in Planner(samples, seed=7).e1()]
    second = [i.key for i in Planner(list(reversed(samples)), seed=7).e1()]
    assert first == second
    items = Planner(samples, seed=7).e1()
    shards = [shard_items(items, k, 3) for k in range(3)]
    assert sorted(i.key for s in shards for i in s) == sorted(i.key for i in items)
    for shard in shards:  # all questions about one sample stay in one shard
        others = {i.uid for s in shards if s is not shard for i in s}
        assert not ({i.uid for i in shard} & others)


def test_e2_e4_and_report(data_root):
    planner = Planner(_samples(data_root), seed=7, caps={"e2_samples": 100, "e4_per_view": 100})
    e2 = planner.e2()
    assert e2 and {i.view for i in e2} == {"plain", "box", "contour", "crop"}
    assert not any(i.view == "crop" and i.field in CROP_EXCLUDED for i in e2)
    assert all(i.boxes and i.masks for i in e2)
    e4 = planner.e4()
    views = Counter(i.condition for i in e4)
    assert views["plain"] > 0 and "insert_text_polyp" in views
    assert all(i.label == "none" for i in e4 if i.condition == "insert_text_polyp")
    assert len({i.key for i in e4}) == len(e4)
    report = planner.report()
    assert report and all(i.field is None for i in report)


def test_pilot_keeps_conditions_together(data_root):
    planner = Planner(_samples(data_root), seed=7, pilot_items=5)
    items = planner.e1()
    assert len({(i.uid, i.field) for i in items}) == 5
    assert {i.condition for i in items} >= {"normal", "blind"}
    assert len(planner.features()) <= 4


def test_query_builder_renders_views(data_root):
    samples = _samples(data_root)
    planner = Planner(samples, seed=7, caps={"e2_samples": 3})
    builder = QueryBuilder(data_root, max_pixels=409600)
    for item in planner.e2() + planner.e4()[:6] + planner.e1()[:6] + planner.report()[:2]:
        query = builder.build(item)
        if item.image is None:
            assert query.images == []
        else:
            width, height = query.images[0].size
            assert width % 32 == 0 and height % 32 == 0
        if item.field:
            assert query.n_options >= 2 and "Answer with the letter" in query.text
    crop = next(i for i in planner.e2() if i.view == "crop")
    assert "zoomed-in crop" in builder.build(crop).text
