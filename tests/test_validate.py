"""Validation module unit checks on a hand-built tree."""

import os
from pathlib import Path

import pandas as pd
import pytest

from proced.config import PipelineConfig
from proced.metadata import GEODATA_COLUMNS, write_geodata, write_master_metadata
from proced.validate import validate_dataset


def _good_row(cid="s1_crop001", **kw):
    r = {
        "crop_id": cid,
        "crop_source_scene_id": "s1",
        "is_calibrated": "Y",
        "has_dual_pol": "Y",
        "is_synthetic_location": "true",
        "crop_bbox_corners": "[(1.0, 2.0), (1.1, 2.0), (1.1, 1.9), (1.0, 1.9)]",
        "is_partial_object": "N",
        "parent_group_id": "abc",
        "truncated_by_scene_edge": "N",
        "db_conversion_formula_version": "v1_linear_neg30_to_0",
        "label": "oil",
        "full_extent_area_px": 100,
        "full_extent_perimeter": 40.0,
        "full_extent_boundary_complexity": 16.0,
        "full_extent_aspect_ratio": 2.0,
        "image_id": cid,
        "dataset_source": "z",
        "width": 256, "height": 256,
        "has_real_geo": "N", "lat": 2.0, "lon": 1.05,
        "timestamp_utc": "2020-01-01T00:00:00Z",
        "source_corridor_id": "c", "offset_km": 1.0,
        "crop_xoff": 0, "crop_yoff": 0, "crop_size": 256,
        "source_scene_relpath": "z/s1.tif",
        "is_synthetic_timestamp": "true",
    }
    r.update(kw)
    return r


def _build(tmp_path):
    stage = tmp_path / "stage"
    dest = tmp_path / "dest"
    (stage / "region" / "sub").mkdir(parents=True)
    (dest / "region" / "sub").mkdir(parents=True)
    # fake crop files so folder counts as crop-folder
    (dest / "region" / "sub" / "s1_crop001_VV.png").write_bytes(b"x")
    (dest / "region" / "sub" / "s1_crop001_mask.png").write_bytes(b"x")
    rows = [_good_row()]
    write_geodata(dest / "region" / "sub", rows)
    write_master_metadata(dest, rows)
    # splits consistent
    sp = tmp_path / "splits"
    sp.mkdir()
    pd.DataFrame({"image_id": ["s1_crop001"]}).to_csv(sp / "train.csv", index=False)
    pd.DataFrame({"image_id": []}).to_csv(sp / "val.csv", index=False)
    pd.DataFrame({"image_id": []}).to_csv(sp / "test.csv", index=False)
    return stage, dest, sp


def test_validation_pass(tmp_path):
    stage, dest, sp = _build(tmp_path)
    report = validate_dataset(stage, dest, cfg=PipelineConfig(splits_dir=sp), splits_dir=sp)
    assert report["ok"], report["issues"]


def test_validation_catches_missing_geodata(tmp_path):
    stage, dest, sp = _build(tmp_path)
    (dest / "region" / "sub" / "GEODATA.csv").unlink()
    report = validate_dataset(stage, dest, cfg=PipelineConfig(), splits_dir=sp)
    assert not report["ok"]
    assert any("GEODATA" in i for i in report["issues"])


def test_validation_catches_tree_mismatch(tmp_path):
    stage, dest, sp = _build(tmp_path)
    (stage / "region" / "extra_folder").mkdir()
    report = validate_dataset(stage, dest, cfg=PipelineConfig(), splits_dir=sp)
    assert not report["ok"]
    assert any("directory equivalence" in i for i in report["issues"])


def test_validation_catches_empty_required_col(tmp_path):
    stage, dest, sp = _build(tmp_path)
    gp = dest / "region" / "sub" / "GEODATA.csv"
    df = pd.read_csv(gp).fillna("")
    df.loc[0, "crop_bbox_corners"] = ""
    df.to_csv(gp, index=False)
    md = dest / "metadata.csv"
    mdf = pd.read_csv(md).fillna("")
    mdf.loc[0, "crop_bbox_corners"] = ""
    mdf.to_csv(md, index=False)
    report = validate_dataset(stage, dest, cfg=PipelineConfig(), splits_dir=sp)
    assert not report["ok"]
