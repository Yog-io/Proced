"""Subcommand CLI for the SAR dataset pipeline.

Stages are independently runnable; intermediate artifacts live under
``--state_dir`` (default ``data/state/``). The pipeline **never** extracts a
.7z for you — use the ``extract`` subcommand (or unpack yourself) and point
``--stage_dir`` at the structurally extracted folder.

    extract → scan → geo → features → plan → convert → metadata → split → archive → validate
    run = scan … validate (never extracts)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .archive import unpack_7z
from .config import PipelineConfig
from .stage_archive import run_archive
from .stage_convert import run_convert
from .stage_features import run_features
from .stage_geo import run_geo
from .stage_metadata import run_metadata
from .stage_plan import run_plan
from .stage_scan import run_scan
from .stage_split import run_split
from .stage_validate import run_validate

log = logging.getLogger("proced.cli")

SUBCOMMANDS = (
    "extract", "scan", "geo", "features", "plan", "convert",
    "metadata", "split", "archive", "validate", "run",
)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(processName)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def add_common_args(p: argparse.ArgumentParser) -> None:
    """Path / concurrency / tunable flags shared by run and every stage."""
    d = ROOT / "data"
    p.add_argument("--stage_dir", type=str, default=str(d / "scratch_unpacked"),
                   help="Structurally extracted input tree (already unpacked for you)")
    p.add_argument("--dest_dir", type=str, default=str(d / "master_dataset_processed"),
                   help="Directory for cropped master output")
    p.add_argument("--output_archive", type=str, default=str(d / "master_dataset_v1.7z"),
                   help="Output .7z (archive stage / run)")
    p.add_argument("--features_dir", type=str, default=str(d / "features"))
    p.add_argument("--splits_dir", type=str, default=str(d / "splits"))
    p.add_argument("--state_dir", type=str, default=str(d / "state"),
                   help="Intermediate stage artifacts (catalog/instances/geo/plans/rows)")
    p.add_argument("--corridors", type=str,
                   default=str(d / "reference" / "shipping_corridors.geojson"))

    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 2),
                   help="Parallel worker processes")
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--jitter", type=int, default=25, help="Compact-crop jitter ±px (spec 20-30)")
    p.add_argument("--stride", type=int, default=0, help="Overflow stride (0 => 50%% overlap)")
    p.add_argument("--neg-buffer-px", type=int, default=300)
    p.add_argument("--neg-ratio", type=float, default=0.4)
    p.add_argument("--max-tiles-per-object", type=int, default=48)
    p.add_argument("--value-domain", choices=["auto", "db", "power", "amplitude"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gdal-cache-mb", type=int, default=256,
                   help="GDAL cache per WORKER process (archi §2)")
    p.add_argument("--pool-chunk", type=int, default=16,
                   help="Jobs per executor before recycling workers (memory guardrail)")
    p.add_argument("--calibrated-match", nargs="*", default=[],
                   help="Substring patterns forcing is_calibrated=Y")
    p.add_argument("--uncalibrated-match", nargs="*", default=[],
                   help="Substring patterns forcing is_calibrated=N")
    p.add_argument("--label-override", nargs="*", default=[], metavar="KEY=LABEL",
                   help="e.g. kaggle=oil  (KEY substring of rel path → default label)")
    p.add_argument("--negatives-from-unlabeled", action="store_true",
                   help="Allow background patches from scenes WITHOUT masks")
    p.add_argument("--no-pack", action="store_true", help="Skip archival (run)")
    p.add_argument("--no-validate", action="store_true", help="Skip validation (run)")
    p.add_argument("--force", action="store_true",
                   help="Reprocess folders that already contain GEODATA.csv")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sar_dataset_pipeline",
        description="SAR Master Dataset Processing Pipeline — stage subcommands "
                    "(extract is separate; run never auto-extracts)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    pe = sub.add_parser("extract", help="Stage 0: unpack a raw .7z into --stage_dir")
    pe.add_argument("--archive", type=str, required=True, help="Path to raw source .7z")
    pe.add_argument("--stage_dir", type=str, default=str(ROOT / "data" / "scratch_unpacked"))

    for name, help_text in (
        ("scan", "Stage 1: tree parse → catalog.json (+ value domains, tree mirror)"),
        ("geo", "Stage 3a: scene geolocation/timestamps → scene_geo.json"),
        ("features", "Stage 2: full-extent C.0 features → instances.csv"),
        ("plan", "Stage 3b: crop geometry → crop_plans.json"),
        ("convert", "Stage 4: windowed tiling + PNGs + GEODATA → crop_rows.json"),
        ("metadata", "Stage 5: master metadata.csv + feature/provenance CSVs"),
        ("split", "Stage 6a: group-safe train/val/test splits (C.5)"),
        ("archive", "Stage 6b: pack dest_dir into output .7z"),
        ("validate", "Stage 6c: validation checklist (archi §5)"),
        ("run", "Full pipeline scan→validate (never extracts — use extract first)"),
    ):
        sp = sub.add_parser(name, help=help_text,
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        add_common_args(sp)
        if name == "run":
            sp.add_argument("--archive", type=str, default=None,
                            help="Rejected — use the extract subcommand first")
        if name == "validate":
            sp.add_argument("--json", action="store_true", help="Print full report as JSON")

    return p


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    overrides = {}
    for key in ("archive", "stage_dir", "dest_dir", "output_archive",
                "features_dir", "splits_dir", "state_dir"):
        val = getattr(args, key, None)
        if val is not None:
            overrides[key] = Path(val)
    label_overrides = {}
    for item in getattr(args, "label_override", None) or []:
        if "=" in item:
            k, v = item.split("=", 1)
            label_overrides[k.strip()] = v.strip()
    corridors = getattr(args, "corridors", None)
    cfg = PipelineConfig(
        corridors_path=Path(corridors) if corridors else None,
        workers=max(1, int(getattr(args, "workers", 4))),
        gdal_cache_mb_per_worker=int(getattr(args, "gdal_cache_mb", 256)),
        tile_size=int(getattr(args, "tile_size", 256)),
        jitter_px=int(getattr(args, "jitter", 25)),
        stride=int(getattr(args, "stride", 0) or 0),
        neg_buffer_px=int(getattr(args, "neg_buffer_px", 300)),
        neg_ratio=float(getattr(args, "neg_ratio", 0.4)),
        max_tiles_per_object=int(getattr(args, "max_tiles_per_object", 48)),
        value_domain=getattr(args, "value_domain", "auto"),
        seed=int(getattr(args, "seed", 42)),
        pool_chunk=int(getattr(args, "pool_chunk", 16)),
        calibrated_match=tuple(getattr(args, "calibrated_match", None) or ()),
        uncalibrated_match=tuple(getattr(args, "uncalibrated_match", None) or ()),
        label_overrides=label_overrides,
        negatives_from_unlabeled=bool(getattr(args, "negatives_from_unlabeled", False)),
        no_pack=bool(getattr(args, "no_pack", False)),
        no_validate=bool(getattr(args, "no_validate", False)),
        force=bool(getattr(args, "force", False)),
        **overrides,
    )
    if getattr(args, "archive", None) is None:
        cfg.archive = None
    return cfg


def run_pipeline(cfg: PipelineConfig) -> Dict:
    """scan → geo → features → plan → convert → metadata → split → archive → validate.

    Never extracts an archive. ``cfg.archive`` must be None here.
    """
    t_start = time.time()
    if cfg.archive:
        raise SystemExit(
            "--archive is only accepted by the 'extract' subcommand.\n"
            "  1) python sar_dataset_pipeline.py extract --archive <file.7z> --stage_dir <dir>\n"
            "  2) python sar_dataset_pipeline.py run --stage_dir <dir> ..."
        )

    stage_dir = Path(cfg.stage_dir)
    if not stage_dir.is_dir():
        raise SystemExit(
            f"stage_dir not found: {stage_dir}\n"
            "Extract first: python sar_dataset_pipeline.py extract "
            f"--archive <raw.7z> --stage_dir {stage_dir}"
        )

    run_scan(cfg)
    run_geo(cfg)
    run_features(cfg)
    run_plan(cfg)
    rows = run_convert(cfg)
    stats = run_metadata(cfg)

    splits_paths: Dict[str, Path] = {}
    if rows:
        splits_paths = run_split(cfg)

    if not cfg.no_pack:
        run_archive(cfg)

    report = None
    if not cfg.no_validate:
        report = run_validate(cfg)

    summary = {
        "scenes": stats.get("full_rows", 0),  # replaced below
        "crops": len(rows),
        "feature_rows": stats.get("full_rows", 0),
        "splits": {k: str(v) for k, v in splits_paths.items()},
        "output_archive": str(cfg.output_archive) if not cfg.no_pack else None,
        "validation": report,
        "elapsed_s": round(time.time() - t_start, 1),
        "config": cfg.to_dict(),
    }
    # Preserve prior summary keys used by tests / operators
    from .state import StatePaths, load_catalog
    try:
        cat = load_catalog(StatePaths.from_config(cfg).catalog)
        summary["scenes"] = len(cat.scenes)
        summary["catalog_errors"] = cat.errors
        summary["folders"] = len(cat.folder_to_scenes)
    except SystemExit:
        pass

    summary_path = Path(cfg.dest_dir) / "pipeline_summary.json"
    Path(cfg.dest_dir).mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log.info("Pipeline complete in %.1fs → %s", summary["elapsed_s"], summary_path)
    if report and not report["ok"]:
        log.error("VALIDATION FAILED: %s", report["issues"])
        raise SystemExit(2)
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Legacy invocation without a subcommand → treat as `run`
    # (but reject --archive with a hint to use `extract`).
    if argv and argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
        argv = ["run"] + argv

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse already printed the error
        return int(exc.code or 0)

    _setup_logging()
    cmd = args.command
    cfg = config_from_args(args)

    # Persist run knobs for independent QA (qa_verification recompute_crop_plans).
    # Written for every stage so stage-by-stage chains are as inspectable as `run`.
    if cmd != "extract":
        try:
            state_root = Path(cfg.state_dir)
            state_root.mkdir(parents=True, exist_ok=True)
            with open(state_root / "run_config.json", "w") as fh:
                json.dump(cfg.to_dict(), fh, indent=2, default=str)
        except OSError:
            pass

    if cmd == "extract":
        unpack_7z(Path(args.archive), Path(args.stage_dir))
        log.info("extract: done → %s", args.stage_dir)
        return 0

    if cmd == "run":
        if getattr(args, "archive", None):
            print(
                "error: --archive is only accepted by the 'extract' subcommand.\n"
                "  python sar_dataset_pipeline.py extract --archive <file.7z> --stage_dir <dir>\n"
                "  python sar_dataset_pipeline.py run --stage_dir <dir> ...",
                file=sys.stderr,
            )
            return 2
        run_pipeline(cfg)
        return 0

    if cmd == "scan":
        run_scan(cfg)
        return 0
    if cmd == "geo":
        run_geo(cfg)
        return 0
    if cmd == "features":
        run_features(cfg)
        return 0
    if cmd == "plan":
        run_plan(cfg)
        return 0
    if cmd == "convert":
        rows = run_convert(cfg)
        log.info("convert: %d crop rows", len(rows))
        return 0
    if cmd == "metadata":
        run_metadata(cfg)
        return 0
    if cmd == "split":
        run_split(cfg)
        return 0
    if cmd == "archive":
        run_archive(cfg)
        return 0
    if cmd == "validate":
        report = run_validate(cfg)
        if getattr(args, "json", False):
            print(json.dumps(report, indent=2, default=str))
        else:
            status = "PASS" if report["ok"] else "FAIL"
            print(f"VALIDATION: {status}")
            if report["issues"]:
                for issue in report["issues"]:
                    print(f"  - {issue}")
        return 0 if report["ok"] else 2

    parser.error(f"unknown command: {cmd}")
    return 2
