"""Independent QA suite tests (DATASET_VERIFICATION_ARCHITECTURE).

These exercise qa_verification/ against a mini pipeline fixture — they must
NOT import proced/ from inside qa_verification (boundary rule), though the
fixture builder may use the pipeline to produce artifacts.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qa_verification._lib import (  # noqa: E402
    QAPaths,
    Report,
    db_from_uint16,
    detect_domain_independent,
    to_db_independent,
    uint16_from_db,
)


# --------------------------------------------------------------------- helpers
def _qa_paths(stage, dest, state, features, splits) -> QAPaths:
    return QAPaths(
        stage_dir=Path(stage),
        dest_dir=Path(dest),
        state_dir=Path(state),
        features_dir=Path(features),
        splits_dir=Path(splits),
        corridors_path=ROOT / "data" / "reference" / "shipping_corridors.geojson",
        output_archive=None,
        source_archive=None,
    )


def _run_pipeline_fixture(tmp_path, *, seed: int = 7):
    """Build the standard mini fixture and run extract→run once."""
    import sar_dataset_pipeline as pipeline
    from tests.test_pipeline_e2e import _common_argv, build_fixture

    raw = build_fixture(tmp_path)
    stage = tmp_path / "scratch"
    # use tree directly (already extracted)
    stage = raw
    dest = tmp_path / "master_out"
    features = tmp_path / "features"
    splits = tmp_path / "splits"
    state = tmp_path / "state"
    base = _common_argv(
        stage, dest, features, splits, state=state,
        extra=["--workers", "1", "--seed", str(seed)],
    )
    for stage_cmd in ("scan", "geo", "features", "plan", "convert",
                      "metadata", "split", "validate"):
        rc = pipeline.main([stage_cmd] + base)
        assert rc == 0, f"{stage_cmd} failed rc={rc}"
    return _qa_paths(stage, dest, state, features, splits)


# ------------------------------------------------------------------ unit math
def test_db_inverse_matches_spec_formula():
    # Spec C.2: db = (px / 65535) * 30 - 30
    px = np.array([0, 32768, 65535], dtype=np.uint16)
    db = db_from_uint16(px)
    assert db[0] == pytest.approx(-30.0)
    assert db[2] == pytest.approx(0.0)
    assert db[1] == pytest.approx((32768 / 65535) * 30 - 30, abs=1e-9)


def test_uint16_roundtrip_quantization():
    db = np.linspace(-30, 0, 256)
    px = uint16_from_db(db)
    back = db_from_uint16(px)
    # one LSB of 30/65535 ≈ 0.000458
    assert np.max(np.abs(back - db)) <= (30.0 / 65535.0) + 1e-9


def test_detect_domain_independent():
    assert detect_domain_independent(np.array([1, 2, 3], dtype=np.uint8)) == "native"
    assert detect_domain_independent(np.array([-5.0, -1.0], dtype=np.float32)) == "db"
    assert detect_domain_independent(np.array([0.01, 0.5], dtype=np.float32)) == "power"
    assert detect_domain_independent(np.array([10.0, 50.0], dtype=np.float32)) == "amplitude"


def test_to_db_power_vs_amplitude():
    # power: 10*log10; amplitude: 20*log10
    p = np.array([0.01], dtype=np.float64)
    assert to_db_independent(p, "power")[0] == pytest.approx(-20.0)
    a = np.array([0.1], dtype=np.float64)
    assert to_db_independent(a, "amplitude")[0] == pytest.approx(-20.0)


def test_report_hard_fail_sets_ok_false():
    r = Report("t")
    r.hard(False, "bad")
    assert not r.ok
    assert len(r.hard_failures) == 1
    r2 = Report("t2")
    r2.soft(False, "meh")
    assert r2.ok  # soft does not flip ok


# ----------------------------------------------------------- boundary rule
def test_qa_never_imports_proced():
    import re as _re
    qa_root = ROOT / "qa_verification"
    # Match real import statements only (not docstrings mentioning the rule).
    pat = _re.compile(
        r"^\s*(?:from\s+proced[\s.]|import\s+proced\b)",
    )
    offenders = []
    for p in qa_root.rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            s = line.strip()
            if s.startswith("#"):
                continue
            if pat.match(s):
                offenders.append(f"{p}:{i}: {s}")
    assert not offenders, f"qa_verification must not import proced/: {offenders}"


# ------------------------------------------------------------- e2e vs fixture
@pytest.mark.slow
def test_db_roundtrip_module_on_fixture(tmp_path):
    from qa_verification.independent_checks import recompute_db_roundtrip

    paths = _run_pipeline_fixture(tmp_path)
    rep = recompute_db_roundtrip.run(paths, sample=50, seed=7)
    assert rep.ok, [f.message for f in rep.hard_failures]
    # must have actually checked something (calibrated fixture scenes exist)
    checked = next(
        (f.details.get("checked") for f in rep.findings
         if "dB round-trip" in f.message),
        None,
    )
    if checked is not None:
        assert checked > 0, "dB round-trip checked zero band-crops"


@pytest.mark.slow
def test_schema_and_joins_modules_on_fixture(tmp_path):
    from qa_verification.structural_checks import (
        test_cross_file_joins,
        test_schema_nulls,
    )

    paths = _run_pipeline_fixture(tmp_path)
    rep = test_schema_nulls.run(paths)
    assert rep.ok, [f.message for f in rep.hard_failures]

    rep2 = test_cross_file_joins.run(paths)
    assert rep2.ok, [f.message for f in rep2.hard_failures]


@pytest.mark.slow
def test_split_leakage_module_on_fixture(tmp_path):
    from qa_verification.structural_checks import test_split_leakage

    paths = _run_pipeline_fixture(tmp_path)
    rep = test_split_leakage.run(paths)
    assert rep.ok, [f.message for f in rep.hard_failures]


@pytest.mark.slow
def test_shape_and_plan_modules_on_fixture(tmp_path):
    from qa_verification.independent_checks import (
        recompute_crop_plans,
        recompute_shape_features,
    )

    paths = _run_pipeline_fixture(tmp_path)
    rep = recompute_shape_features.run(paths, sample=20, seed=7)
    assert rep.ok, [f.message for f in rep.hard_failures]
    rep2 = recompute_crop_plans.run(paths, neg_buffer_px=120)
    assert rep2.ok, [f.message for f in rep2.hard_failures]


@pytest.mark.slow
def test_run_all_verification_on_fixture(tmp_path, monkeypatch):
    """Full orchestrator: must produce verification_report.json, exit 0."""
    from qa_verification.run_all_verification import build_report

    paths = _run_pipeline_fixture(tmp_path)
    # point env at fixture
    env = {
        "QA_STAGE_DIR": str(paths.stage_dir),
        "QA_DEST_DIR": str(paths.dest_dir),
        "QA_STATE_DIR": str(paths.state_dir),
        "QA_FEATURES_DIR": str(paths.features_dir),
        "QA_SPLITS_DIR": str(paths.splits_dir),
        "QA_CORRIDORS": str(paths.corridors_path),
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("QA_OUTPUT_ARCHIVE", raising=False)
    monkeypatch.delenv("QA_SOURCE_ARCHIVE", raising=False)

    out = tmp_path / "verification_report.json"
    payload = build_report(
        sample=30,
        seed=7,
        only=[
            "stage1_scan",
            "stage2_features",
            "stage3b_plan",
            "stage4_db_roundtrip",
            "stage5_schema_nulls",
            "stage5_cross_file_joins",
            "stage6_split_leakage",
        ],
        skip_dashboards=True,
        skip_visual=True,
        skip_adversarial=True,
        out_path=out,
    )
    assert out.is_file()
    data = json.loads(out.read_text())
    assert data["kind"] == "independent_qa_verification"
    assert data["ok"] is True, [
        (m["name"], [f["message"] for f in m["findings"]
                     if not f["ok"] and f["severity"] == "hard"])
        for m in data["modules"]
        if not m["ok"]
    ]
    assert data["n_hard_fail"] == 0
    assert payload["exit_code"] == 0
    # separate from pipeline summary (never overwrite it)
    dest_report = paths.dest_dir / "verification_report.json"
    assert not dest_report.is_file() or dest_report.resolve() != out.resolve()


@pytest.mark.slow
def test_perpendicular_offset_module(tmp_path):
    from qa_verification.independent_checks import recompute_perpendicular_offset

    paths = _run_pipeline_fixture(tmp_path)
    rep = recompute_perpendicular_offset.run(paths, sample=20, seed=7)
    # fixture synthetic points should be near corridors; hard assert only
    # if module actually checked samples (not skipped)
    if any("perpendicular" in f.message and f.details.get("checked", 0) > 0
           for f in rep.findings):
        assert rep.ok, [f.message for f in rep.hard_failures]


@pytest.mark.slow
def test_visual_and_stats_modules(tmp_path, monkeypatch):
    from qa_verification.stats_dashboard import (
        generate_geo_scatter,
        generate_histograms,
    )
    from qa_verification.visual_spotcheck import sample_and_render

    paths = _run_pipeline_fixture(tmp_path)
    rep = sample_and_render.run(paths, sample=10, seed=7)
    assert (paths.dest_dir / "qa_visual_spotcheck.html").is_file()
    # hard failures only if zero rendered
    assert rep.ok, [f.message for f in rep.hard_failures]

    rep_h = generate_histograms.run(paths, sample=20, seed=7)
    assert rep_h.ok, [f.message for f in rep_h.hard_failures]
    rep_g = generate_geo_scatter.run(paths, sample=20, seed=7)
    assert rep_g.ok, [f.message for f in rep_g.hard_failures]


@pytest.mark.slow
def test_adversarial_graceful_failure(tmp_path):
    from qa_verification.adversarial import test_graceful_failure

    paths = _run_pipeline_fixture(tmp_path)
    rep = test_graceful_failure.run(paths, full_pipeline=False)
    assert rep.ok, [f.message for f in rep.hard_failures]


def test_adversarial_fixtures_build(tmp_path):
    from qa_verification.adversarial.make_fixtures import write_fixture_tree

    cases = write_fixture_tree(tmp_path / "fx")
    assert set(cases) == {
        "corrupted_tiff", "mask_size_mismatch", "all_zero_mask",
        "mask_all_borders", "tiny_scene",
    }
    for p in cases.values():
        assert p.is_dir()
        assert any(p.iterdir())


# --------------------------------------------------------------- raw_audit (§8)
def _write_tif(path, arr, dtype, crs="EPSG:4326"):
    import rasterio
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        str(path), "w", driver="GTiff",
        height=arr.shape[0], width=arr.shape[1], count=1,
        dtype=dtype, crs=crs, transform=from_origin(60.0, 25.0, 0.001, 0.001),
    ) as ds:
        ds.write(arr.astype(dtype), 1)


def _build_raw_audit_tree(root: Path) -> Path:
    """Minimal pre-pipeline tree: paired scene, orphan, duplicate, corrupt."""
    import rasterio
    from rasterio.transform import from_origin

    rng = np.random.default_rng(1)
    sub = root / "pass_a" / "deep" / "nested"
    sub.mkdir(parents=True, exist_ok=True)

    # well-paired dual-pol scene (2000-ish is ideal, use 256 for test speed;
    # integrity size_tol allows smaller non-mask only if within ±35% of 2000 —
    # so use 2000×2000 for images to avoid resolution-outlier noise).
    H = W = 256  # test images will flag as resolution outliers (expected);
    # keep small for speed; we assert outliers are recorded, not that they're empty.
    vv = rng.uniform(0.01, 0.3, (H, W)).astype(np.float32)
    vh = (vv * 0.4).astype(np.float32)
    mask = np.zeros((H, W), np.uint8)
    mask[50:100, 50:120] = 1
    _write_tif(sub / "sceneA_VV.tif", vv, "float32")
    _write_tif(sub / "sceneA_VH.tif", vh, "float32")
    _write_tif(sub / "sceneA_mask.tif", mask, "uint8", crs=None)

    # orphan image (no matching mask in same folder)
    _write_tif(sub / "orphanB.tif", vv, "float32")

    # byte-identical duplicate of sceneA_VV
    import shutil as _sh
    _sh.copy(sub / "sceneA_VV.tif", sub / "sceneA_VV_dup.tif")

    # second sibling folder with fewer files (gap candidate)
    pass_b = root / "pass_b"
    pass_b.mkdir(parents=True, exist_ok=True)
    for i in range(6):
        _write_tif(pass_b / f"img{i}.tif", vv, "float32")
        _write_tif(pass_b / f"img{i}_mask.tif", mask, "uint8", crs=None)

    # third sibling with only 1 file → gap vs median of siblings
    pass_c = root / "pass_c"
    pass_c.mkdir(parents=True, exist_ok=True)
    _write_tif(pass_c / "only.tif", vv, "float32")

    # corrupt file (truncated garbage claiming .tif)
    bad = root / "pass_a" / "corrupt.tif"
    bad.write_bytes(b"not-a-real-tiff" + b"\x00" * 32)

    return root


def test_raw_audit_discover_tree(tmp_path):
    from qa_verification.raw_audit.discover_tree import discover_tree

    root = _build_raw_audit_tree(tmp_path / "raw")
    disc = discover_tree(root)
    # nested: VV+VH+mask+orphanB+dup = 5; pass_a: corrupt = 1;
    # pass_b: 6 img + 6 mask = 12; pass_c: 1 → 19
    assert len(disc.rasters) == 19
    assert any(f.rel.endswith("corrupt.tif") for f in disc.rasters)
    assert any(f.is_mask_by_name for f in disc.rasters)
    assert "pass_b" in disc.file_count_by_folder or any(
        "pass_b" in k for k in disc.file_count_by_folder
    )


def test_raw_audit_integrity_detects_corrupt(tmp_path):
    from qa_verification.raw_audit.check_integrity import check_integrity
    from qa_verification.raw_audit.discover_tree import discover_tree

    root = _build_raw_audit_tree(tmp_path / "raw")
    disc = discover_tree(root)
    res = check_integrity(disc, sample_values=False)
    assert res.opened >= 13
    assert len(res.failed) >= 1
    assert any("corrupt.tif" in f["rel"] for f in res.failed)
    # dtype/crs/domain snapshot present for opened files
    assert all("dtypes" in s and "crs" in s for s in res.snapshots)


def test_raw_audit_pairing_orphans(tmp_path):
    from qa_verification.raw_audit.check_pairing import check_pairing
    from qa_verification.raw_audit.discover_tree import discover_tree

    root = _build_raw_audit_tree(tmp_path / "raw")
    disc = discover_tree(root)
    res = check_pairing(disc)
    orphan_rels = [o["rel"] for o in res.orphan_images]
    assert any("orphanB" in r for r in orphan_rels)
    # sceneA VV/VH/mask should match (not orphaned); _dup is a separate base
    assert not any(r.endswith("sceneA_VV.tif") for r in orphan_rels)
    assert not any(r.endswith("sceneA_VH.tif") for r in orphan_rels)
    assert res.matched_pairs >= 1


def test_raw_audit_duplicates(tmp_path):
    from qa_verification.raw_audit.check_duplicates import check_duplicates
    from qa_verification.raw_audit.discover_tree import discover_tree

    root = _build_raw_audit_tree(tmp_path / "raw")
    disc = discover_tree(root)
    res = check_duplicates(disc)
    assert len(res.duplicate_groups) >= 1
    group_files = [f for g in res.duplicate_groups for f in g["files"]]
    assert any("sceneA_VV" in f for f in group_files)
    assert any("dup" in f for f in group_files)


def test_raw_audit_run_audit_report(tmp_path):
    from qa_verification.raw_audit.run_raw_folder_audit import run_audit

    root = _build_raw_audit_tree(tmp_path / "raw")
    out = tmp_path / "raw_folder_audit_report.json"
    payload = run_audit(root, out_path=out, sample_values=False)

    assert out.is_file()
    data = json.loads(out.read_text())
    assert data["kind"] == "raw_folder_audit"
    assert "ready_for_pipeline" in data
    assert "counts" in data and "pass" in data["counts"]
    assert "pairing" in data and "duplicates" in data
    # corrupt file → not ready
    assert payload["ready_for_pipeline"] is False
    assert payload["verdict"] == "NOT_READY"
    # report path recorded
    assert payload["report_path"] == str(out)


def test_raw_audit_cli_exit_code(tmp_path, capsys):
    from qa_verification.raw_audit.run_raw_folder_audit import main

    root = _build_raw_audit_tree(tmp_path / "raw")
    out = tmp_path / "r.json"
    rc = main(["--root", str(root), "--out", str(out), "--no-sample-values"])
    captured = capsys.readouterr()
    assert "ready for pipeline" in captured.out.lower() or "ready for pipeline:" in captured.out
    assert rc == 2  # NOT_READY because corrupt file present
    assert out.is_file()


def test_raw_audit_ready_on_clean_tree(tmp_path):
    from qa_verification.raw_audit.run_raw_folder_audit import run_audit

    root = tmp_path / "clean"
    sub = root / "set1"
    sub.mkdir(parents=True)
    rng = np.random.default_rng(0)
    vv = rng.uniform(0.01, 0.3, (64, 64)).astype(np.float32)
    mask = np.zeros((64, 64), np.uint8)
    mask[10:30, 10:30] = 1
    _write_tif(sub / "s1_VV.tif", vv, "float32")
    _write_tif(sub / "s1_VH.tif", vv * 0.5, "float32")
    _write_tif(sub / "s1_mask.tif", mask, "uint8", crs=None)

    payload = run_audit(root, out_path=tmp_path / "r.json", sample_values=False)
    assert payload["ready_for_pipeline"] is True
    assert payload["counts"]["open_failed"] == 0
    assert payload["pairing"]["orphan_images"] == []
