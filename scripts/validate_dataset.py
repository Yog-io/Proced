#!/usr/bin/env python3
"""Verification & Validation Checklist (archi.md §5) as a standalone command.

Checks
------
1. **Directory equivalence** — ``find stage -type d`` ≡ ``find dest -type d``
2. **Metadata integrity** — every crop folder has GEODATA.csv; required
   columns null-free
3. **Bit-depth accuracy** — calibrated PNGs report Type=UInt16
4. **Split hygiene (C.5)** — no ``crop_source_scene_id`` spans splits
5. **Provenance** — ``is_synthetic_location`` present/valid on every row

Usage::

    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --stage_dir ... --dest_dir ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from proced.config import PipelineConfig
from proced.validate import validate_dataset


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage_dir", default=str(ROOT / "data" / "scratch_unpacked"))
    ap.add_argument("--dest_dir", default=str(ROOT / "data" / "master_dataset_processed"))
    ap.add_argument("--splits_dir", default=str(ROOT / "data" / "splits"))
    ap.add_argument("--json", action="store_true", help="print full report as JSON")
    args = ap.parse_args(argv)

    cfg = PipelineConfig()
    report = validate_dataset(
        Path(args.stage_dir), Path(args.dest_dir),
        cfg=cfg, splits_dir=Path(args.splits_dir),
    )

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print("=" * 60)
        print("VALIDATION:", "PASS ✓" if report["ok"] else "FAIL ✗")
        print("=" * 60)
        for name, chk in report["checks"].items():
            print(f"  [{name}] {json.dumps(chk, default=str)[:200]}")
        if report["issues"]:
            print("\nIssues:")
            for issue in report["issues"]:
                print(f"  - {issue}")
        print(f"\nSummary: {report['summary']}")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
