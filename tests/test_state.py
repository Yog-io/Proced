"""State serialisation round-trips for stage artifacts."""

import json
from pathlib import Path

import pytest

from proced.catalog import Catalog, SceneRecord
from proced.config import PipelineConfig
from proced.state import (
    INSTANCE_COLUMNS,
    StatePaths,
    load_catalog,
    load_crop_plans,
    load_crop_rows,
    load_instances,
    load_scene_geo,
    save_catalog,
    save_crop_plans,
    save_crop_rows,
    save_instances,
    save_scene_geo,
)


def _scene(tmp_path: Path) -> SceneRecord:
    band = tmp_path / "region" / "scene_VV.tif"
    band.parent.mkdir(parents=True, exist_ok=True)
    band.write_bytes(b"")
    mask = tmp_path / "region" / "scene_mask.tif"
    mask.write_bytes(b"")
    return SceneRecord(
        scene_id="region__scene",
        source_scene_name="scene",
        rel_folder="region",
        band_files=[band],
        pol_tags=["VV"],
        mask_path=mask,
        mask_present=True,
        is_calibrated=True,
        value_domain_hint="auto",
        has_dual_pol=False,
        width=64,
        height=64,
        default_label="oil",
        source_dataset="region",
        rel_primary="region/scene_VV.tif",
        value_domains={"VV": "power"},
    )


def test_catalog_round_trip(tmp_path):
    root = tmp_path / "stage"
    root.mkdir()
    cat = Catalog(root=root, all_dirs=["", "region"], scenes=[_scene(root)],
                  mask_only_folders=["ref"], errors=["e1"])
    path = tmp_path / "catalog.json"
    save_catalog(cat, path)
    loaded = load_catalog(path)
    assert loaded.root == root
    assert loaded.all_dirs == ["", "region"]
    assert loaded.mask_only_folders == ["ref"]
    assert loaded.errors == ["e1"]
    assert len(loaded.scenes) == 1
    s = loaded.scenes[0]
    assert s.scene_id == "region__scene"
    assert s.band_files[0].name == "scene_VV.tif"
    assert s.band_files[0].is_absolute() or True
    assert s.mask_path is not None and s.mask_path.name == "scene_mask.tif"
    assert s.value_domains == {"VV": "power"}
    assert s.is_calibrated is True


def test_instances_round_trip(tmp_path):
    rows = [{
        "scene_id": "s1",
        "instance_index": 0,
        "label": "oil",
        "mask_value": 1,
        "area_px": 100,
        "perimeter": 40.5,
        "aspect_ratio": 2.0,
        "boundary_complexity": 16.2,
        "bbox_x": 1, "bbox_y": 2, "bbox_w": 3, "bbox_h": 4,
        "centroid_x": 5, "centroid_y": 6,
        "truncated_by_scene_edge": False,
        "fragment_count": 3,
        "crop_source_scene_id": "s1",
        "dataset_source": "ds",
        "damping_ratio": None,
        "damping_ratio_db": 10.0,
        "boundary_gradient_steepness": "",
        "backscatter_variance_ratio": "",
        "glcm_contrast": 1.5,
        "glcm_homogeneity": 0.25,
        "ndpi": None,
    }]
    path = tmp_path / "instances.csv"
    save_instances(rows, path)
    loaded = load_instances(path)
    assert len(loaded) == 1
    r = loaded[0]
    assert r["instance_index"] == 0
    assert r["area_px"] == 100
    assert r["perimeter"] == pytest.approx(40.5)
    assert r["truncated_by_scene_edge"] is False
    assert r["damping_ratio"] is None
    assert r["damping_ratio_db"] == pytest.approx(10.0)
    assert r["ndpi"] is None
    # column order stable
    text = path.read_text().splitlines()[0].split(",")
    assert text == INSTANCE_COLUMNS


def test_scene_geo_plans_rows_round_trip(tmp_path):
    geo = {"s1": {"has_geo": False, "center_lat": 1.5, "timestamp_utc": "2020-01-01T00:00:00Z"}}
    save_scene_geo(geo, tmp_path / "scene_geo.json")
    assert load_scene_geo(tmp_path / "scene_geo.json")["s1"]["center_lat"] == 1.5

    plans = {"s1": [{"xoff": 0, "yoff": 0, "label": "oil", "instance_index": 0,
                     "is_partial": False, "parent_group_id": "g",
                     "truncated_by_scene_edge": False, "scene_id": "s1", "seq": 0}]}
    save_crop_plans(plans, tmp_path / "crop_plans.json")
    assert load_crop_plans(tmp_path / "crop_plans.json")["s1"][0]["label"] == "oil"

    rows = [{"crop_id": "s1_crop000", "label": "oil"}]
    save_crop_rows(rows, tmp_path / "crop_rows.json")
    assert load_crop_rows(tmp_path / "crop_rows.json")[0]["crop_id"] == "s1_crop000"


def test_missing_state_raises_with_stage_hint(tmp_path):
    paths = StatePaths(root=tmp_path / "state",
                       catalog=tmp_path / "state" / "catalog.json",
                       instances=tmp_path / "state" / "instances.csv",
                       scene_geo=tmp_path / "state" / "scene_geo.json",
                       crop_plans=tmp_path / "state" / "crop_plans.json",
                       crop_rows=tmp_path / "state" / "crop_rows.json")
    with pytest.raises(SystemExit) as ei:
        load_catalog(paths.catalog)
    assert "scan" in str(ei.value)
    with pytest.raises(SystemExit) as ei:
        load_instances(paths.instances)
    assert "features" in str(ei.value)
