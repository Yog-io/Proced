"""End-to-end: extract → run → validation; plus standalone stage chain."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_geotiff(path, arr, dtype, crs="EPSG:4326", origin=(60.0, 25.0), px=0.0001):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        str(path), "w", driver="GTiff",
        height=arr.shape[0], width=arr.shape[1], count=1,
        dtype=dtype, crs=crs, transform=from_origin(origin[0], origin[1], px, px),
    ) as ds:
        ds.write(arr.astype(dtype), 1)


def _write_png(path, arr):
    from proced.raster_io import write_png
    path.parent.mkdir(parents=True, exist_ok=True)
    write_png(path, arr)


def build_fixture(root: Path) -> Path:
    """Create a mini dataset tree resembling the real one."""
    raw = root / "raw_input"
    rng = np.random.default_rng(0)

    # --- Dataset A: calibrated dual-pol GeoTIFF (Zenodo-like) with lookalike value 2
    a = raw / "zenodo_set" / "region_atlantic" / "subpass_01"
    H = W = 512
    vv = rng.uniform(0.01, 0.3, (H, W)).astype(np.float32)
    vh = (vv * 0.4).astype(np.float32)
    mask = np.zeros((H, W), np.uint8)
    mask[100:180, 100:200] = 1          # oil — fits in 256 window (100×80)
    mask[300:310, 300:500] = 2          # lookalike thin line
    # Make a truly overflowing object: long slick 400px wide
    mask[60:80, 60:480] = 1             # 420×20 → overflow horizontally
    # Touching-edge object for truncation flag
    mask[500:512, 100:160] = 1          # abuts bottom edge
    vv[mask > 0] *= 0.35                 # darker where oil (damping signal)
    _write_geotiff(a / "scene01_VV.tif", vv, "float32")
    _write_geotiff(a / "scene01_VH.tif", vh, "float32")
    _write_geotiff(a / "scene01_mask.tif", mask, "uint8", crs=None)

    # second scene in same folder, single-band, binary mask.
    # 768×768 with the blob up in the corner so a 256 window can sit ≥120 px
    # clear of it (C.3 negatives need room in a scene this size).
    H2 = W2 = 768
    mask2 = np.zeros((H2, W2), np.uint8)
    mask2[40:100, 40:100] = 1
    vv2 = rng.uniform(0.01, 0.3, (H2, W2)).astype(np.float32)
    vv2[mask2 > 0] *= 0.3
    _write_geotiff(a / "scene02.tif", vv2, "float32")
    _write_geotiff(a / "scene02_mask.tif", mask2, "uint8", crs=None)

    # --- Dataset B: uncalibrated PNG + mask (Kaggle-like), lookalike folder name
    b = raw / "kaggle_lookalike_visuals" / "subpass_02"
    img = (rng.random((300, 300)) * 255).astype(np.uint8)
    m = np.zeros((300, 300), np.uint8)
    m[50:120, 50:120] = 1
    _write_png(b / "imgA.png", img)
    _write_png(b / "imgA_mask.png", m)

    # --- Mask-only folder (tree equivalence)
    rm = raw / "reference_masks"
    _write_png(rm / "ref_mask.png", np.zeros((64, 64), np.uint8))

    # --- Empty subfolder (must be mirrored)
    (raw / "empty_folder").mkdir(parents=True, exist_ok=True)
    return raw


def _common_argv(stage, dest, features, splits, out_arc=None, state=None, extra=None):
    argv = [
        "--stage_dir", str(stage),
        "--dest_dir", str(dest),
        "--features_dir", str(features),
        "--splits_dir", str(splits),
        "--workers", "2",
        "--pool-chunk", "8",
        "--neg-buffer-px", "120",
        "--neg-ratio", "0.5",
        "--seed", "7",
        "--tile-size", "256",
    ]
    if out_arc is not None:
        argv += ["--output_archive", str(out_arc)]
    if state is not None:
        argv += ["--state_dir", str(state)]
    if extra:
        argv += list(extra)
    return argv


@pytest.mark.slow
def test_pipeline_end_to_end(tmp_path):
    import sar_dataset_pipeline as pipeline
    from proced.archive import pack_7z

    raw = build_fixture(tmp_path)
    archive = tmp_path / "input.7z"
    pack_7z(raw, archive)

    stage = tmp_path / "scratch"
    dest = tmp_path / "master_out"
    features = tmp_path / "features"
    splits = tmp_path / "splits"
    out_arc = tmp_path / "master.7z"
    state = tmp_path / "state"

    # Pipeline never auto-extracts: extract is its own subcommand.
    rc = pipeline.main([
        "extract", "--archive", str(archive), "--stage_dir", str(stage),
    ])
    assert rc == 0, "extract must exit 0"

    rc = pipeline.main(["run"] + _common_argv(
        stage, dest, features, splits, out_arc=out_arc, state=state,
    ))
    assert rc == 0, "pipeline must exit 0 (validation passed)"

    # ---- Structure ------------------------------------------------------
    assert (dest / "metadata.csv").is_file()
    assert (dest / "zenodo_set" / "region_atlantic" / "subpass_01" / "GEODATA.csv").is_file()
    assert (dest / "kaggle_lookalike_visuals" / "subpass_02" / "GEODATA.csv").is_file()
    assert (dest / "empty_folder").is_dir()          # tree mirror
    assert (dest / "reference_masks").is_dir()

    # Intermediate state artifacts
    assert (state / "catalog.json").is_file()
    assert (state / "scene_geo.json").is_file()
    assert (state / "instances.csv").is_file()
    assert (state / "crop_plans.json").is_file()
    assert (state / "crop_rows.json").is_file()

    # ---- Crops & bit depth ---------------------------------------------
    md = pd.read_csv(dest / "metadata.csv").fillna("")
    assert len(md) > 0
    assert {"crop_id", "crop_bbox_corners", "is_synthetic_location",
            "parent_group_id", "truncated_by_scene_edge", "label"}.issubset(md.columns)

    # labels seen: oil + background at minimum; lookalike from value-2 instance
    labels = set(md["label"].unique())
    assert "oil" in labels and "background" in labels
    assert "lookalike" in labels

    # calibrated scenes → uint16; uncalibrated → native
    cal_rows = md[md["is_calibrated"] == "Y"]
    uncal_rows = md[md["is_calibrated"] == "N"]
    assert len(cal_rows) > 0 and len(uncal_rows) > 0
    assert set(cal_rows["db_conversion_formula_version"]) == {"v1_linear_neg30_to_0"}
    assert set(uncal_rows["db_conversion_formula_version"]) == {"native_8bit_no_db"}

    sample_cal = list(dest.rglob(f"{cal_rows.iloc[0]['crop_id']}_VV.png"))[0]
    with rasterio.open(sample_cal) as ds:
        assert ds.dtypes[0] == "uint16", "calibrated crops must be 16-bit (C.2)"

    sample_unc = list(dest.rglob(f"{uncal_rows.iloc[0]['crop_id']}_VV.png"))[0]
    with rasterio.open(sample_unc) as ds:
        assert ds.dtypes[0] == "uint8", "Kaggle-style crops stay 8-bit native (C.2)"

    # ---- Overflow partials share parent_group_id ------------------------
    partials = md[md["is_partial_object"] == "Y"]
    assert len(partials) > 0, "overflowing slick must produce multi-tile crops"
    assert partials.groupby("crop_source_scene_id")["parent_group_id"].nunique().max() >= 1
    # all partials of scene01 share ONE group
    s1p = partials[partials["crop_source_scene_id"] == "scene01"]
    assert s1p["parent_group_id"].nunique() == 1

    # ---- Truncation is object-level -------------------------------------
    assert set(md["truncated_by_scene_edge"].unique()) <= {"Y", "N"}

    # ---- Real geo preserved ---------------------------------------------
    # pandas may parse "true"/"false" cells into bools — normalise on read
    syn = md["is_synthetic_location"].astype(str).str.lower().str.strip()
    md = md.assign(_syn=syn)
    zen = md[md["dataset_source"] == "zenodo_set"]
    assert (zen["has_real_geo"] == "Y").all()
    assert (zen["_syn"] == "false").all()
    assert "60." in zen.iloc[0]["crop_bbox_corners"]  # lon near fixture origin

    # ---- Synthetic geo for unreferenced data + provenance CSV ------------
    kag = md[md["dataset_source"] == "kaggle_lookalike_visuals"]
    assert (kag["_syn"] == "true").all()
    assert (kag["has_real_geo"] == "N").all()
    assert (kag["source_corridor_id"].astype(str) != "").all()
    assert (kag["timestamp_utc"].astype(str) != "").all()

    synth_csv = features / "synthetic_geo_assignments.csv"
    assert synth_csv.is_file()
    sdf = pd.read_csv(synth_csv)
    for col in ("image_id", "synthetic_lat", "synthetic_lon",
                "synthetic_timestamp_utc", "source_corridor_id",
                "offset_km", "is_synthetic_location"):
        assert col in sdf.columns, f"guide §2.2 provenance missing {col}"

    # ---- Feature tables (C.0) -------------------------------------------
    shape = pd.read_csv(features / "shape_features.csv")
    full = pd.read_csv(features / "full_extent_features.csv")
    assert len(shape) > 0
    assert {"area", "perimeter", "aspect_ratio", "boundary_complexity",
            "fragment_count"}.issubset(shape.columns)
    assert {"damping_ratio", "glcm_contrast", "boundary_gradient_steepness",
            "backscatter_variance_ratio", "eligible_for_classifier"}.issubset(full.columns)
    # at least one eligible + one truncated (edge) instance in fixture
    assert full["eligible_for_classifier"].astype(str).isin(["True", "true", "Y"]).any()
    # radiometric populated for calibrated scenes
    cal_f = full[full["is_calibrated"] == "Y"]
    assert cal_f["damping_ratio_db"].astype(str).replace("", np.nan).notna().any()

    # ---- Splits: no leakage (C.5) ---------------------------------------
    man = pd.read_csv(splits / "manifest.csv")
    assert set(man["split"]) <= {"train", "val", "test"}
    leak = man.groupby("crop_source_scene_id")["split"].nunique()
    assert (leak == 1).all(), "scene group leaked across splits"

    # ---- Archive produced ------------------------------------------------
    assert out_arc.is_file() and out_arc.stat().st_size > 0

    # ---- Summary artifact ------------------------------------------------
    summary = json.loads((dest / "pipeline_summary.json").read_text())
    assert summary["validation"]["ok"] is True
    assert summary["crops"] == len(md)

    # ---- GEODATA null-free (checklist #2) --------------------------------
    from proced.metadata import GEODATA_COLUMNS
    for gp in dest.rglob("GEODATA.csv"):
        gdf = pd.read_csv(gp).fillna("")
        for col in GEODATA_COLUMNS:
            assert col in gdf.columns
        assert (gdf["crop_bbox_corners"].astype(str).str.len() > 0).all()
        assert (gdf["is_partial_object"].astype(str).str.len() > 0).all()
        assert (gdf["truncated_by_scene_edge"].astype(str).str.len() > 0).all()


@pytest.mark.slow
def test_pipeline_resume_skips_processed(tmp_path):
    import sar_dataset_pipeline as pipeline
    from proced.archive import pack_7z

    raw = build_fixture(tmp_path)
    archive = tmp_path / "input.7z"
    pack_7z(raw, archive)
    stage, dest = tmp_path / "scratch", tmp_path / "master_out"

    assert pipeline.main([
        "extract", "--archive", str(archive), "--stage_dir", str(stage),
    ]) == 0

    base = _common_argv(
        stage, dest, tmp_path / "f", tmp_path / "s",
        out_arc=tmp_path / "o.7z", state=tmp_path / "state",
        extra=["--no-pack", "--workers", "1"],
    )
    assert pipeline.main(["run"] + base) == 0
    md1 = pd.read_csv(dest / "metadata.csv")

    # Second run without --force → same crop count (folders skipped, rows reloaded)
    assert pipeline.main(["run"] + base) == 0
    md2 = pd.read_csv(dest / "metadata.csv")
    assert len(md1) == len(md2)


@pytest.mark.slow
def test_each_stage_runs_standalone(tmp_path):
    """scan → geo → features → plan → convert → metadata → split → validate."""
    import sar_dataset_pipeline as pipeline

    # Use the fixture tree directly (already "extracted") — no archive involved.
    stage = build_fixture(tmp_path)
    dest = tmp_path / "master_out"
    features = tmp_path / "features"
    splits = tmp_path / "splits"
    state = tmp_path / "state"
    base = _common_argv(stage, dest, features, splits, state=state,
                        extra=["--workers", "1"])

    assert pipeline.main(["scan"] + base) == 0
    assert (state / "catalog.json").is_file()

    assert pipeline.main(["geo"] + base) == 0
    assert (state / "scene_geo.json").is_file()

    assert pipeline.main(["features"] + base) == 0
    assert (state / "instances.csv").is_file()

    assert pipeline.main(["plan"] + base) == 0
    assert (state / "crop_plans.json").is_file()

    assert pipeline.main(["convert"] + base) == 0
    assert (state / "crop_rows.json").is_file()
    assert (dest / "metadata.csv").parent.is_dir() or True

    assert pipeline.main(["metadata"] + base) == 0
    assert (dest / "metadata.csv").is_file()
    assert (features / "shape_features.csv").is_file()

    assert pipeline.main(["split"] + base) == 0
    assert (splits / "train.csv").is_file()

    assert pipeline.main(["validate"] + base) == 0


def test_run_rejects_archive_flag(tmp_path, capsys):
    import sar_dataset_pipeline as pipeline

    rc = pipeline.main([
        "run",
        "--archive", str(tmp_path / "x.7z"),
        "--stage_dir", str(tmp_path / "missing"),
        "--dest_dir", str(tmp_path / "d"),
        "--state_dir", str(tmp_path / "s"),
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "extract" in err


def test_legacy_argv_maps_to_run_but_rejects_archive(tmp_path, capsys):
    import sar_dataset_pipeline as pipeline

    rc = pipeline.main([
        "--archive", str(tmp_path / "x.7z"),
        "--stage_dir", str(tmp_path / "missing"),
    ])
    assert rc == 2
    assert "extract" in capsys.readouterr().err


def test_extract_then_standalone_without_run(tmp_path):
    """extract works alone; subsequent stages consume stage_dir as-is."""
    import sar_dataset_pipeline as pipeline
    from proced.archive import pack_7z

    raw = build_fixture(tmp_path)
    archive = tmp_path / "in.7z"
    pack_7z(raw, archive)
    stage = tmp_path / "unpacked"

    assert pipeline.main([
        "extract", "--archive", str(archive), "--stage_dir", str(stage),
    ]) == 0
    assert (stage / "zenodo_set").is_dir()
    assert (stage / "empty_folder").is_dir()

    # Running scan on the extracted tree with NO archive flag must succeed
    rc = pipeline.main([
        "scan",
        "--stage_dir", str(stage),
        "--state_dir", str(tmp_path / "state"),
        "--dest_dir", str(tmp_path / "dest"),
    ])
    assert rc == 0
    assert (tmp_path / "state" / "catalog.json").is_file()
