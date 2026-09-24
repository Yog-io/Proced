"""Catalog: VV/VH pairing, mask discovery, tree mirror, calibration detect."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from proced.catalog import scan_tree
from proced.config import PipelineConfig


def _write_tif(path, data, dtype="float32", crs="EPSG:4326", transform=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(data, dtype=dtype)
    with rasterio.open(
        str(path), "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
        count=1, dtype=dtype, crs=crs, transform=transform or from_origin(50, 25, 0.001, 0.001),
    ) as ds:
        ds.write(arr, 1)


def _write_png(path, data):
    from proced.raster_io import write_png
    path.parent.mkdir(parents=True, exist_ok=True)
    write_png(path, np.asarray(data))


def test_dual_pol_split_files_paired_with_mask(tmp_path):
    rng = np.random.default_rng(0)
    scene = tmp_path / "region" / "subpass"
    vv = rng.random((64, 64)).astype(np.float32)
    vh = (vv * 0.5).astype(np.float32)
    mask = np.zeros((64, 64), np.uint8)
    mask[10:30, 10:30] = 1
    _write_tif(scene / "scene01_VV.tif", vv)
    _write_tif(scene / "scene01_VH.tif", vh)
    _write_tif(scene / "scene01_mask.tif", mask.astype(np.uint8), dtype="uint8")

    cfg = PipelineConfig(stage_dir=tmp_path)
    cat = scan_tree(cfg)
    assert len(cat.scenes) == 1, "VV+VH must pair into ONE logical scene"
    s = cat.scenes[0]
    assert s.has_dual_pol is True
    assert s.pol_tags == ["VV", "VH"]
    assert s.mask_path is not None and s.mask_path.name == "scene01_mask.tif"
    assert s.is_calibrated is True  # float GeoTIFF


def test_pol_split_does_not_double_process(tmp_path):
    """Regression: the archi script treated scene_VV and scene_VH as two scenes."""
    rng = np.random.default_rng(0)
    scene = tmp_path / "d"
    _write_tif(scene / "s_VV.tif", rng.random((32, 32)).astype(np.float32))
    _write_tif(scene / "s_VH.tif", rng.random((32, 32)).astype(np.float32))
    cat = scan_tree(PipelineConfig(stage_dir=tmp_path))
    assert len(cat.scenes) == 1


def test_mask_only_folder_mirrored(tmp_path):
    _write_png(tmp_path / "reference_masks" / "ref_mask.png",
               np.zeros((16, 16), np.uint8))
    cat = scan_tree(PipelineConfig(stage_dir=tmp_path))
    assert "reference_masks" in cat.all_dirs
    assert cat.scenes == []
    assert "reference_masks" in cat.mask_only_folders


def test_uncalibrated_uint8_detected(tmp_path):
    _write_png(tmp_path / "kaggle" / "img001.png",
               (np.random.rand(32, 32) * 255).astype(np.uint8))
    cat = scan_tree(PipelineConfig(stage_dir=tmp_path))
    assert len(cat.scenes) == 1
    assert cat.scenes[0].is_calibrated is False
    assert cat.scenes[0].mask_path is None


def test_lookalike_folder_default_label(tmp_path):
    _write_png(tmp_path / "zenodo_lookalike_set" / "img1.png",
               (np.random.rand(32, 32) * 255).astype(np.uint8))
    cat = scan_tree(PipelineConfig(stage_dir=tmp_path))
    assert cat.scenes[0].default_label == "lookalike"


def test_label_override(tmp_path):
    _write_png(tmp_path / "weird_set" / "img1.png",
               (np.random.rand(32, 32) * 255).astype(np.uint8))
    cfg = PipelineConfig(stage_dir=tmp_path, label_overrides={"weird_set": "lookalike"})
    cat = scan_tree(cfg)
    assert cat.scenes[0].default_label == "lookalike"


def test_calibrated_override_patterns(tmp_path):
    _write_tif(tmp_path / "zenodo" / "s.tif",
               np.full((32, 32), 100, np.uint16), dtype="uint16")
    cat = scan_tree(PipelineConfig(stage_dir=tmp_path))
    assert cat.scenes[0].is_calibrated is False  # uint heuristic
    cat2 = scan_tree(PipelineConfig(stage_dir=tmp_path, calibrated_match=("zenodo",)))
    assert cat2.scenes[0].is_calibrated is True


def test_duplicate_stems_disambiguated(tmp_path):
    for folder in ("a", "b"):
        _write_tif(tmp_path / folder / "scene.tif",
                   np.full((16, 16), 0.1, np.float32))
    cat = scan_tree(PipelineConfig(stage_dir=tmp_path))
    ids = [s.scene_id for s in cat.scenes]
    assert len(ids) == len(set(ids)) == 2


def test_all_dirs_includes_root_and_children(tmp_path):
    _write_tif(tmp_path / "x" / "y" / "s.tif", np.full((8, 8), 0.1, np.float32))
    cat = scan_tree(PipelineConfig(stage_dir=tmp_path))
    assert "" in cat.all_dirs
    assert "x" in cat.all_dirs
    assert "x/y" in cat.all_dirs
