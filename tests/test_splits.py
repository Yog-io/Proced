"""Group-aware splits (C.5): no scene leakage, all crops assigned."""

from collections import Counter

import pytest

from proced.config import PipelineConfig
from proced.splits import assign_splits, write_splits


def _fake_rows(n_groups=40, crops_per_group=5):
    rows = []
    labels = ["oil", "lookalike", "background"]
    for g in range(n_groups):
        lab = labels[g % 3]
        for c in range(crops_per_group):
            rows.append({
                "crop_id": f"scene{g:03d}_crop{c:03d}",
                "crop_source_scene_id": f"scene{g:03d}",
                "label": lab,
            })
    return rows


def test_every_crop_assigned_exactly_once():
    cfg = PipelineConfig(seed=1)
    splits, manifest = assign_splits(_fake_rows(), cfg)
    all_ids = splits["train"] + splits["val"] + splits["test"]
    assert len(all_ids) == len(set(all_ids)) == 200
    assert len(manifest) == 200


def test_no_group_leakage():
    cfg = PipelineConfig(seed=2)
    splits, manifest = assign_splits(_fake_rows(), cfg)
    group_split = {}
    for m in manifest:
        gid = m["crop_source_scene_id"]
        if gid in group_split:
            assert group_split[gid] == m["split"], f"LEAK: {gid}"
        group_split[gid] = m["split"]


def test_ratio_roughly_respected():
    cfg = PipelineConfig(seed=3)
    splits, _ = assign_splits(_fake_rows(n_groups=60), cfg)
    n = sum(len(v) for v in splits.values())
    assert 0.55 <= len(splits["train"]) / n <= 0.85
    assert 0.05 <= len(splits["val"]) / n <= 0.30
    assert 0.05 <= len(splits["test"]) / n <= 0.30


def test_deterministic_for_same_seed():
    a, _ = assign_splits(_fake_rows(), PipelineConfig(seed=9))
    b, _ = assign_splits(_fake_rows(), PipelineConfig(seed=9))
    assert a == b


def test_write_splits_files(tmp_path):
    cfg = PipelineConfig(seed=4, splits_dir=tmp_path)
    splits, manifest = assign_splits(_fake_rows(n_groups=12), cfg)
    paths = write_splits(tmp_path, splits, manifest)
    import pandas as pd
    for name in ("train", "val", "test"):
        df = pd.read_csv(paths[name])
        assert list(df.columns) == ["image_id"]
    man = pd.read_csv(paths["manifest"])
    assert set(man.columns) == {"image_id", "split", "crop_source_scene_id", "label"}
