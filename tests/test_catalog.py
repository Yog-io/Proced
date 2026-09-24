"""Catalog: VV/VH pairing, mask discovery, tree mirror, calibration detect."""

from pathlib import Path

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


def _write_zenodo_asymmetric(root: Path) -> None:
    """Sibling *_images / *_mask trees with nested subfolders (no co-located masks)."""
    rng = np.random.default_rng(3)
    img = rng.uniform(0.01, 0.3, (32, 32)).astype(np.float32)
    m = np.zeros((32, 32), np.uint8)
    m[5:15, 5:15] = 1

    pairs = [
        ("01_Train_Val_Oil_Spill_images", "01_Train_Val_Oil_Spill_mask", "pass1/sceneA"),
        ("01_Train_Val_Lookalike_images", "01_Train_Val_Lookalike_mask", "pass1/sceneB"),
        ("01_Train_Val_No_Oil_images", "01_Train_Val_No_Oil_mask", "sceneC"),
    ]
    for img_dir, mask_dir, rel in pairs:
        _write_tif(root / img_dir / f"{rel}.tif", img)
        _write_tif(root / mask_dir / f"{rel}.tif", m, dtype="uint8", crs=None)

    # test split: co-located image + mask under mixed root
    _write_tif(root / "02_Test_images_and_ground_truth" / "t1" / "img.tif", img)
    _write_tif(
        root / "02_Test_images_and_ground_truth" / "t1" / "img_mask.tif",
        m, dtype="uint8", crs=None,
    )


def test_sibling_image_mask_trees_paired(tmp_path):
    _write_zenodo_asymmetric(tmp_path)
    cat = scan_tree(PipelineConfig(stage_dir=tmp_path))
    scene_rels = {s.rel_folder for s in cat.scenes}
    assert "01_Train_Val_Oil_Spill_images/pass1" in scene_rels
    assert "01_Train_Val_Lookalike_images/pass1" in scene_rels
    assert "01_Train_Val_No_Oil_images" in scene_rels
    # mask trees must not become scenes
    assert not any(r.endswith("_mask") or "/_mask" in r for r in scene_rels)
    assert not any("_mask" in r for r in scene_rels)
    assert all(s.mask_present for s in cat.scenes), [
        (s.rel_folder, s.source_scene_name) for s in cat.scenes if not s.mask_present
    ]
    oil = next(s for s in cat.scenes if s.rel_folder.startswith("01_Train_Val_Oil_Spill"))
    assert oil.mask_path is not None
    assert "01_Train_Val_Oil_Spill_mask" in str(oil.mask_path)
    assert oil.mask_path.name == "sceneA.tif"


def test_nested_asymmetric_subfolders(tmp_path):
    _write_zenodo_asymmetric(tmp_path)
    cat = scan_tree(PipelineConfig(stage_dir=tmp_path))
    nested = [s for s in cat.scenes if "/" in Path(s.band_files[0]).name or s.rel_folder.count("/") >= 1]
    # oil/lookalike scenes live under pass1/ inside their *_images tree
    assert any(s.source_scene_name == "sceneA" for s in cat.scenes)
    sceneA = next(s for s in cat.scenes if s.source_scene_name == "sceneA")
    assert sceneA.rel_folder == "01_Train_Val_Oil_Spill_images/pass1"
    assert sceneA.mask_path is not None
    assert sceneA.mask_path.name == "sceneA.tif"


def test_folder_based_mask_not_treated_as_scene(tmp_path):
    # plain stem under a *_mask folder → mask only, not a scene
    _write_tif(
        tmp_path / "01_Train_Val_Oil_Spill_mask" / "pass2" / "plain.tif",
        np.zeros((16, 16), np.uint8), dtype="uint8", crs=None,
    )
    cat = scan_tree(PipelineConfig(stage_dir=tmp_path))
    assert cat.scenes == []
    assert any("01_Train_Val_Oil_Spill_mask" in d for d in cat.all_dirs)


def test_label_from_zenodo_folder_names(tmp_path):
    _write_zenodo_asymmetric(tmp_path)
    cat = scan_tree(PipelineConfig(stage_dir=tmp_path))
    labels = {s.rel_folder: s.default_label for s in cat.scenes}
    assert labels["01_Train_Val_Lookalike_images/pass1"] == "lookalike"
    assert labels["01_Train_Val_Oil_Spill_images/pass1"] == "oil"
    assert labels["01_Train_Val_No_Oil_images"] == "oil"


def test_tree_mirror_asymmetric_directories(tmp_path):
    _write_zenodo_asymmetric(tmp_path)
    cat = scan_tree(PipelineConfig(stage_dir=tmp_path))
    expected = {
        "01_Train_Val_Oil_Spill_images",
        "01_Train_Val_Oil_Spill_images/pass1",
        "01_Train_Val_Oil_Spill_mask",
        "01_Train_Val_Oil_Spill_mask/pass1",
        "01_Train_Val_Lookalike_images",
        "01_Train_Val_Lookalike_images/pass1",
        "01_Train_Val_Lookalike_mask",
        "01_Train_Val_Lookalike_mask/pass1",
        "01_Train_Val_No_Oil_images",
        "01_Train_Val_No_Oil_mask",
        "02_Test_images_and_ground_truth",
        "02_Test_images_and_ground_truth/t1",
    }
    assert expected.issubset(set(cat.all_dirs))
    # mask-only folders = dirs that contain masks but no scene entries (leaves)
    assert "01_Train_Val_Oil_Spill_mask/pass1" in cat.mask_only_folders
    assert "01_Train_Val_Lookalike_mask/pass1" in cat.mask_only_folders
    assert "01_Train_Val_No_Oil_mask" in cat.mask_only_folders
    assert "02_Test_images_and_ground_truth/t1" not in cat.mask_only_folders


def test_parallel_mask_folder_mapping():
    from proced.catalog import _parallel_mask_rel, _mask_search_dirs

    assert _parallel_mask_rel("01_Train_Val_Oil_Spill_images") == "01_Train_Val_Oil_Spill_mask"
    assert _parallel_mask_rel("01_Train_Val_Oil_Spill_images/pass1") == "01_Train_Val_Oil_Spill_mask/pass1"
    assert _parallel_mask_rel("images/sub") == "mask/sub"
    assert _parallel_mask_rel("kaggle") is None
    dirs = _mask_search_dirs("01_Train_Val_Oil_Spill_images/pass1")
    assert "01_Train_Val_Oil_Spill_mask/pass1" in dirs
    assert "01_Train_Val_Oil_Spill_images/pass1" in dirs
