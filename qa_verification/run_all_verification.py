#!/usr/bin/env python3
"""Aggregate every independent QA module → verification_report.json.

Architecture §6: this report is intentionally SEPARATE from the pipeline's
own pipeline_summary.json so it's obvious which findings came from
independent recomputation vs. the pipeline grading its own homework.

Exit codes:
  0 — no hard failures (soft findings may exist)
  2 — at least one hard-fail (blocks handoff to Block 2/Backend)
  1 — harness error

Usage:
  python qa_verification/run_all_verification.py
  python qa_verification/run_all_verification.py --sample-size 200 --seed 7
  python qa_verification/run_all_verification.py --zenodo-expected
  python qa_verification/run_all_verification.py --only stage4_db_roundtrip,stage5_schema_nulls
  python qa_verification/run_all_verification.py --skip-dashboards --skip-visual
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qa_verification._lib import (  # noqa: E402
    QAPaths,
    Report,
    write_report,
)
from qa_verification._progress import bar as progress_bar  # noqa: E402


# --------------------------------------------------------------------- registry
def _modules(sample: int, seed: int) -> Dict[str, Callable[[], Report]]:
    """name → zero-arg thunk (each closes over paths/sample/seed via _paths())."""
    from qa_verification.independent_checks import (
        recompute_crop_plans,
        recompute_db_roundtrip,
        recompute_extract,
        recompute_geo_transform,
        recompute_perpendicular_offset,
        recompute_scan,
        recompute_shape_features,
        recompute_splits,
    )
    from qa_verification.structural_checks import (
        test_cross_file_joins,
        test_schema_nulls,
        test_split_leakage,
    )

    def P() -> QAPaths:
        return QAPaths.from_env()

    return {
        # --- independent recomputes ---------------------------------------
        "stage0_extract": lambda: recompute_extract.run(P(), sample=sample, seed=seed),
        "stage1_scan": lambda: recompute_scan.run(P(), sample=sample, seed=seed),
        "stage2_features": lambda: recompute_shape_features.run(
            P(), sample=sample, seed=seed
        ),
        "stage3a_geo": lambda: recompute_geo_transform.run(
            P(), sample=sample, seed=seed
        ),
        "perpendicular_offset": lambda: recompute_perpendicular_offset.run(
            P(), sample=min(sample, 20), seed=seed
        ),
        "stage3b_plan": lambda: recompute_crop_plans.run(P()),
        "stage4_db_roundtrip": lambda: recompute_db_roundtrip.run(
            P(), sample=max(sample, 100), seed=seed
        ),
        "stage6_splits_independent": lambda: recompute_splits.run(P()),
        # --- structural / schema ------------------------------------------
        "stage5_schema_nulls": lambda: test_schema_nulls.run(P()),
        "stage5_cross_file_joins": lambda: test_cross_file_joins.run(P()),
        "stage6_split_leakage": lambda: test_split_leakage.run(P()),
    }


def _optional_modules(sample: int, seed: int) -> Dict[str, Callable[[], Report]]:
    from qa_verification.visual_spotcheck import sample_and_render
    from qa_verification.stats_dashboard import generate_geo_scatter, generate_histograms
    from qa_verification.adversarial import test_determinism, test_graceful_failure

    def P() -> QAPaths:
        return QAPaths.from_env()

    return {
        "visual_spotcheck": lambda: sample_and_render.run(
            P(), sample=max(sample, 20), seed=seed
        ),
        "stats_histograms": lambda: generate_histograms.run(P(), sample=sample, seed=seed),
        "stats_geo_scatter": lambda: generate_geo_scatter.run(P(), sample=sample, seed=seed),
        "adversarial_graceful_failure": lambda: test_graceful_failure.run(P()),
        "adversarial_determinism": lambda: test_determinism.run(P(), seed=seed),
    }


# ------------------------------------------------------------------------ main
def build_report(
    *,
    sample: int = 100,
    seed: int = 42,
    only: Optional[List[str]] = None,
    skip_dashboards: bool = False,
    skip_visual: bool = False,
    skip_adversarial: bool = False,
    zenodo_expected: bool = False,
    out_path: Optional[Path] = None,
) -> dict:
    t0 = time.time()
    paths = QAPaths.from_env()

    mods = _modules(sample, seed)
    optional = _optional_modules(sample, seed)

    selected: Dict[str, Callable[[], Report]] = {}
    for name, fn in list(mods.items()) + list(optional.items()):
        if only and name not in only:
            continue
        if only:
            selected[name] = fn
            continue
        if skip_dashboards and name.startswith("stats_"):
            continue
        if skip_visual and name == "visual_spotcheck":
            continue
        if skip_adversarial and name.startswith("adversarial_"):
            continue
        selected[name] = fn

    if only:
        # allow only= to reference optional names not in mods
        for name in only:
            if name not in selected and name in optional:
                selected[name] = optional[name]

    reports: List[dict] = []
    errors: List[dict] = []
    names = sorted(selected)
    pb = progress_bar(len(names), "qa modules", unit="module")
    try:
        for name in names:
            fn = selected[name]
            pb.set_desc(f"qa {name}")
            try:
                rep = fn()
                reports.append(rep.to_dict())
            except Exception as exc:
                errors.append({
                    "module": name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=8),
                })
                # surface as hard failure so the suite never "passes" by crashing
                fail = Report(name)
                fail.hard(False, f"module crashed: {type(exc).__name__}: {exc}")
                reports.append(fail.to_dict())
            pb.update()
    finally:
        pb.close()

    n_hard = sum(r["n_hard_fail"] for r in reports)
    n_soft = sum(r["n_soft_fail"] for r in reports)
    n_findings = sum(len(r["findings"]) for r in reports)
    ok = n_hard == 0 and not errors

    payload = {
        "kind": "independent_qa_verification",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ok": ok,
        "exit_code": 0 if ok else 2,
        "n_modules": len(reports),
        "n_hard_fail": n_hard,
        "n_soft_fail": n_soft,
        "n_findings": n_findings,
        "sample_size": sample,
        "seed": seed,
        "zenodo_expected": zenodo_expected,
        "paths": {
            "stage_dir": str(paths.stage_dir),
            "dest_dir": str(paths.dest_dir),
            "state_dir": str(paths.state_dir),
            "features_dir": str(paths.features_dir),
            "splits_dir": str(paths.splits_dir),
            "corridors": str(paths.corridors_path),
            "output_archive": str(paths.output_archive) if paths.output_archive else None,
            "source_archive": str(paths.source_archive) if paths.source_archive else None,
        },
        "modules": reports,
        "module_errors": errors,
        "elapsed_s": round(time.time() - t0, 2),
        "criteria": {
            "hard_fail_blocks_handoff": [
                "null in required column",
                "orphaned crop/metadata row (either direction)",
                "dB round-trip mismatch beyond tolerance",
                "train/val/test group leakage",
                "non-binary mask pixel",
                "perpendicular-offset bearing error",
            ],
            "soft_fail_flagged": [
                "label class imbalance beyond band",
                "visual spot-check reviewer flag",
                "mock-wind rows without visible flag",
            ],
        },
    }

    dest = out_path or (paths.dest_dir / "verification_report.json")
    try:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_report(dest, payload)
        payload["report_path"] = str(dest)
    except Exception as exc:
        payload["report_path_error"] = str(exc)
        # always try state dir / cwd fallback
        try:
            fallback = Path.cwd() / "verification_report.json"
            write_report(fallback, payload)
            payload["report_path"] = str(fallback)
        except Exception:
            pass

    return payload


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Independent QA verification suite "
                    "(DATASET_VERIFICATION_ARCHITECTURE)",
    )
    ap.add_argument("--sample-size", type=int, default=100,
                    help="stochastic sample size for recompute checks")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated module names to run exclusively")
    ap.add_argument("--skip-dashboards", action="store_true")
    ap.add_argument("--skip-visual", action="store_true")
    ap.add_argument("--skip-adversarial", action="store_true",
                    help="skip malformed-input + determinism (slow)")
    ap.add_argument("--zenodo-expected", action="store_true",
                    help="enforce exact Zenodo label counts (1400/700 band)")
    ap.add_argument("--out", type=str, default=None,
                    help="path for verification_report.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    only = [s.strip() for s in args.only.split(",")] if args.only else None
    out = Path(args.out) if args.out else None

    payload = build_report(
        sample=args.sample_size,
        seed=args.seed,
        only=only,
        skip_dashboards=args.skip_dashboards,
        skip_visual=args.skip_visual,
        skip_adversarial=args.skip_adversarial,
        zenodo_expected=args.zenodo_expected,
        out_path=out,
    )

    if not args.quiet:
        print()
        print("=" * 72)
        print("INDEPENDENT QA VERIFICATION")
        print("=" * 72)
        for m in payload["modules"]:
            status = "OK  " if m["ok"] else "FAIL"
            print(
                f"  [{status}] {m['name']:<32} "
                f"hard={m['n_hard_fail']} soft={m['n_soft_fail']} "
                f"findings={len(m['findings'])}"
            )
            for f in m["findings"]:
                if not f["ok"] and f["severity"] in ("hard", "soft"):
                    tag = "HARD" if f["severity"] == "hard" else "soft"
                    print(f"         · [{tag}] {f['message']}")
        for e in payload["module_errors"]:
            print(f"  [ERR ] {e['module']}: {e['error']}")
        print("-" * 72)
        print(
            f"  modules={payload['n_modules']} "
            f"hard_fail={payload['n_hard_fail']} "
            f"soft_fail={payload['n_soft_fail']} "
            f"elapsed={payload['elapsed_s']}s"
        )
        print(f"  report → {payload.get('report_path', '(unwritten)')}")
        print(f"  RESULT: {'PASS' if payload['ok'] else 'HARD FAIL (exit 2)'}")
        print("=" * 72)

    if payload["module_errors"] and payload["n_hard_fail"] == 0:
        return 1
    return int(payload["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
