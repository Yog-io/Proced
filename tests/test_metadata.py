"""Metadata: spec columns, boolean normalisation, null-freeness."""

import pandas as pd
import pytest

from proced.metadata import (
    GEODATA_COLUMNS,
    MASTER_COLUMNS,
    normalise_row,
    rows_to_frame,
    synthetic_geo_rows,
    write_geodata,
    write_master_metadata,
)


def _row(**kw):
    base = {
        "crop_id": "scene01_crop001",
        "crop_source_scene_id": "scene01",
        "is_calibrated": True,
        "has_dual_pol": False,
        "is_synthetic_location": True,
        "crop_bbox_corners": "[(60.0, 25.0), (60.1, 25.0), (60.1, 24.9), (60.0, 24.9)]",
        "is_partial_object": False,
        "parent_group_id": "abc123",
        "truncated_by_scene_edge": False,
        "db_conversion_formula_version": "v1_linear_neg30_to_0",
        "label": "oil",
        "full_extent_area_px": 500,
        "full_extent_perimeter": 90.0,
        "full_extent_boundary_complexity": 16.2,
        "full_extent_aspect_ratio": 3.1,
        "image_id": "scene01_crop001",
        "dataset_source": "zenodo",
        "width": 256,
        "height": 256,
        "has_real_geo": False,
        "lat": 24.95,
        "lon": 60.05,
        "timestamp_utc": "2021-05-01T00:00:00Z",
        "source_corridor_id": "strait_of_hormuz",
        "offset_km": 12.3,
        "crop_xoff": 10,
        "crop_yoff": 20,
        "crop_size": 256,
        "source_scene_relpath": "zenodo/scene01.tif",
        "is_synthetic_timestamp": True,
    }
    base.update(kw)
    return base


def test_geodata_required_columns_first():
    r = normalise_row(_row())
    for col in GEODATA_COLUMNS:
        assert col in r
    assert r["is_calibrated"] == "Y"
    assert r["has_dual_pol"] == "N"
    assert r["is_partial_object"] == "N"
    assert r["truncated_by_scene_edge"] == "N"
    assert r["is_synthetic_location"] == "true"


def test_nulls_never_survive_required_columns():
    r = normalise_row({
        "crop_id": "x",
        "crop_bbox_corners": None,
        "is_partial_object": float("nan"),
        "truncated_by_scene_edge": None,
        "parent_group_id": None,
        "label": None,
        "db_conversion_formula_version": None,
        "full_extent_area_px": float("nan"),
    })
    # No None/NaN may survive into the CSV row; Y/N + booleans coerce properly
    for v in r.values():
        assert v is not None
        assert not (isinstance(v, float) and v != v)  # not NaN
    assert r["crop_bbox_corners"] == ""          # None → empty string (never "nan")
    assert r["is_partial_object"] == "N"         # NaN → N
    assert r["truncated_by_scene_edge"] == "N"
    assert r["parent_group_id"] == ""
    assert r["label"] == ""
    assert r["full_extent_area_px"] == 0


def test_frame_column_order_and_no_nan(tmp_path):
    df = rows_to_frame([_row()], MASTER_COLUMNS)
    assert df["crop_id"].iloc[0] == "scene01_crop001"
    assert list(df.columns)[:len(GEODATA_COLUMNS)] == GEODATA_COLUMNS
    assert not df.isna().any().any()


def test_write_geodata_and_master(tmp_path):
    rows = [_row(), _row(crop_id="scene01_crop002", label="background",
                         full_extent_area_px=0, is_synthetic_location=True)]
    gp = write_geodata(tmp_path, rows)
    df = pd.read_csv(gp)
    assert set(GEODATA_COLUMNS).issubset(df.columns)
    assert len(df) == 2
    mp = write_master_metadata(tmp_path, rows)
    mdf = pd.read_csv(mp)
    assert len(mdf) == 2
    assert set(MASTER_COLUMNS).issubset(mdf.columns)


def test_synthetic_geo_rows_schema():
    rows = [_row(), _row(is_synthetic_location=False)]
    out = synthetic_geo_rows(rows)
    assert len(out) == 1
    assert out[0]["image_id"] == "scene01_crop001"
    assert out[0]["source_corridor_id"] == "strait_of_hormuz"
    assert out[0]["is_synthetic_location"] == "true"
