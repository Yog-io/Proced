#!/usr/bin/env python3
"""Standalone raw-folder pre-pipeline audit CLI (architecture §8).

No ``--stage_dir``/``--dest_dir``/pipeline state required. Fully standalone —
runs before any pipeline stage.

Usage:
  python qa_verification/raw_audit/run_raw_folder_audit.py --root <extracted_tree>
  python qa_verification/raw_audit/run_raw_folder_audit.py --root <dir> --out report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qa_verification.raw_audit.check_duplicates import check_duplicates  # noqa: E402
from qa_verification.raw_audit.check_integrity import check_integrity  # noqa: E402
from qa_verification.raw_audit.check_pairing import check_pairing  # noqa: E402
from qa_verification.raw_audit.discover_tree import discover_tree  # noqa: E402

# Soft threshold: flag folder if its file count is < this fraction of the
# median non-empty sibling count (classic multi-part extraction gap).
FOLDER_GAP_FRACTION = 0.25
EXPECTED_SIZE = (2000, 2000)


def _folder_gap_flags(discovery) -> list:
    counts = [c for c in discovery.file_count_by_folder.values() if c > 0]
    if len(counts) < 2:
        return []
    counts_sorted = sorted(counts)
    median = counts_sorted[len(counts_sorted) // 2]
    if median <= 0:
        return []
    flags = []
    for folder, n in sorted(discovery.file_count_by_folder.items()):
        if folder == "":
            continue
        if 0 < n < median * FOLDER_GAP_FRACTION:
            flags.append({
                "folder": folder,
                "file_count": n,
                "median_sibling_count": median,
                "note": "suspiciously low vs siblings (possible failed .7z part)",
            })
        elif n == 0:
            continue
    return flags


def run_audit(
    root: Path,
    *,
    out_path: Optional[Path] = None,
    sample_values: bool = True,
) -> dict:
    root = Path(root)
    discovery = discover_tree(root)
    integrity = check_integrity(discovery, sample_values=sample_values)
    pairing = check_pairing(discovery)
    duplicates = check_duplicates(discovery)
    folder_gaps = _folder_gap_flags(discovery)

    n_fail = len(integrity.failed)
    n_warn = (
        len(pairing.orphan_images)
        + len(pairing.orphan_masks)
        + len(integrity.resolution_outliers)
        + len(folder_gaps)
        + len(duplicates.duplicate_groups)
    )
    n_pass = integrity.opened

    hard_problems = list(integrity.failed)
    ready = n_fail == 0 and not discovery.errors

    payload = {
        "kind": "raw_folder_audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "ready_for_pipeline": ready,
        "counts": {
            "files_discovered": len(discovery.rasters),
            "images": len(discovery.images),
            "masks": len(discovery.masks),
            "total_bytes": discovery.total_bytes,
            "opened_ok": n_pass,
            "open_failed": n_fail,
            "pass": n_pass,
            "warn": n_warn,
            "fail": n_fail,
        },
        "integrity": {
            "failed": integrity.failed,
            "resolution_outliers": integrity.resolution_outliers,
            "snapshots": integrity.snapshots,
            "expected_size": list(EXPECTED_SIZE),
        },
        "pairing": {
            "orphan_images": pairing.orphan_images,
            "orphan_masks": pairing.orphan_masks,
            "pol_pair_gaps": pairing.pol_pair_gaps,
            "matched_pairs": pairing.matched_pairs,
            "folders_checked": pairing.folders_checked,
        },
        "folder_gaps": folder_gaps,
        "duplicates": {
            "groups": duplicates.duplicate_groups,
            "hashed": duplicates.hashed,
            "errors": duplicates.errors,
        },
        "discovery_errors": discovery.errors,
        "verdict": (
            "READY" if ready and n_warn == 0
            else ("READY_WITH_WARNINGS" if ready else "NOT_READY")
        ),
    }

    if out_path is None:
        out_path = root / "raw_folder_audit_report.json"
    out_path = Path(out_path)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, default=str))
        payload["report_path"] = str(out_path)
    except OSError as exc:
        payload["report_path_error"] = str(exc)

    return payload


def _print_summary(payload: dict) -> None:
    c = payload["counts"]
    print("RAW FOLDER AUDIT")
    print(f"  root:               {payload['root']}")
    print(f"  files discovered:   {c['files_discovered']} "
          f"({c['images']} images, {c['masks']} masks)")
    print(f"  opened OK / failed: {c['opened_ok']} / {c['open_failed']}")
    print(f"  pass / warn / fail: {c['pass']} / {c['warn']} / {c['fail']}")
    print(f"  matched pairs:      {payload['pairing']['matched_pairs']}")
    print(f"  orphan images:      {len(payload['pairing']['orphan_images'])}")
    print(f"  orphan masks:       {len(payload['pairing']['orphan_masks'])}")
    print(f"  folder gaps:        {len(payload['folder_gaps'])}")
    print(f"  duplicate groups:   {len(payload['duplicates']['groups'])}")
    print(f"  resolution outliers:{len(payload['integrity']['resolution_outliers'])}")
    print(f"  verdict:            {payload['verdict']}")
    print(f"  ready for pipeline: "
          f"{'yes' if payload['ready_for_pipeline'] else 'NO'}")
    if payload.get("report_path"):
        print(f"  report:             {payload['report_path']}")
    for f in payload["integrity"]["failed"][:10]:
        print(f"  FAIL open: {f['rel']}: {f['error']}")
    for o in payload["pairing"]["orphan_images"][:5]:
        print(f"  WARN orphan image: {o['rel']}")
    for o in payload["pairing"]["orphan_masks"][:5]:
        print(f"  WARN orphan mask:  {o['rel']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Standalone raw-folder pre-pipeline audit (architecture §8)",
    )
    ap.add_argument("--root", required=True,
                    help="path to freshly-extracted dataset tree")
    ap.add_argument("--out", default=None,
                    help="report path (default: <root>/raw_folder_audit_report.json)")
    ap.add_argument("--no-sample-values", action="store_true",
                    help="skip center-window domain sampling (header-only open)")
    args = ap.parse_args(argv)

    payload = run_audit(
        Path(args.root),
        out_path=Path(args.out) if args.out else None,
        sample_values=not args.no_sample_values,
    )
    _print_summary(payload)
    if not payload["ready_for_pipeline"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
