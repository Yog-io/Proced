"""Verification & validation checklist (archi.md §5) + refinement invariants.

1. Directory equivalence — raw tree ≡ processed tree
2. Metadata integrity — every terminal folder with crops has a GEODATA.csv,
   required columns null-free
3. Bit-depth accuracy — calibrated VV/VH PNGs are UInt16
4. Split hygiene (C.5) — no crop_source_scene_id spans splits
5. Label / provenance sanity — is_synthetic_location present on every row
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .metadata import GEODATA_COLUMNS
from .progress import bar as progress_bar

REQUIRED_NULL_FREE = ("crop_bbox_corners", "is_partial_object", "truncated_by_scene_edge",
                      "crop_id", "label")


def _dir_set(root: Path) -> set:
    out = set()
    for dirpath, _dirnames, _filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        out.add("" if rel == "." else rel.replace(os.sep, "/"))
    return out


def _folders_with_crops(dest: Path) -> List[Path]:
    folders = []
    for dirpath, _d, files in os.walk(dest):
        if any(f.endswith("_mask.png") or f.endswith("_VV.png") for f in files):
            folders.append(Path(dirpath))
    return folders


def _gdalinfo_dtype(path: Path) -> Optional[str]:
    exe = shutil.which("gdalinfo")
    if not exe:
        return None
    try:
        out = subprocess.check_output([exe, str(path)], stderr=subprocess.DEVNULL, text=True)
    except Exception:
        return None
    for line in out.splitlines():
        if "Type=" in line:
            return line.split("Type=")[1].split(",")[0].strip()
    return None


def _rasterio_dtype(path: Path) -> Optional[str]:
    try:
        import rasterio
        with rasterio.open(str(path)) as ds:
            return ds.dtypes[0]
    except Exception:
        return None


def validate_dataset(
    stage_dir: Path,
    dest_dir: Path,
    cfg: Optional[PipelineConfig] = None,
    splits_dir: Optional[Path] = None,
    sample_bitdepth: int = 12,
) -> Dict[str, Any]:
    stage_dir, dest_dir = Path(stage_dir), Path(dest_dir)
    report: Dict[str, Any] = {"ok": True, "checks": {}, "issues": []}

    def fail(msg: str) -> None:
        report["ok"] = False
        report["issues"].append(msg)

    # --- 1. Directory equivalence ----------------------------------------
    if stage_dir.is_dir() and dest_dir.is_dir():
        src_dirs = _dir_set(stage_dir)
        dst_dirs = _dir_set(dest_dir)
        missing = sorted(src_dirs - dst_dirs)
        extra = sorted(dst_dirs - src_dirs)
        check1 = {"match": not missing, "missing_in_dest": missing, "extra_in_dest": extra}
        report["checks"]["directory_equivalence"] = check1
        if missing:
            fail(f"directory equivalence: {len(missing)} source folders missing in dest e.g. {missing[:3]}")
    else:
        report["checks"]["directory_equivalence"] = {"match": False, "error": "stage/dest missing"}
        fail("directory equivalence: stage_dir or dest_dir does not exist")

    # --- 2. Metadata integrity -------------------------------------------
    master_path = dest_dir / "metadata.csv"
    master_df = None
    if master_path.is_file():
        master_df = pd.read_csv(master_path).fillna("")
    geodata_problems: List[str] = []
    crop_folders = _folders_with_crops(dest_dir)
    pb = progress_bar(len(crop_folders), "validate folders", unit="folder")
    try:
        for folder in crop_folders:
            pb.update()  # count every folder as it starts (incl. early continues)
            gp = folder / "GEODATA.csv"
            if not gp.is_file():
                geodata_problems.append(f"{folder}: missing GEODATA.csv")
                continue
            try:
                df = pd.read_csv(gp).fillna("")
            except Exception as exc:
                geodata_problems.append(f"{folder}: unreadable GEODATA.csv ({exc})")
                continue
            for col in GEODATA_COLUMNS:
                if col not in df.columns:
                    geodata_problems.append(f"{folder}: missing column {col}")
            for col in REQUIRED_NULL_FREE:
                if col in df.columns:
                    empties = df[col].astype(str).str.strip() == ""
                    # parent_group_id may be blank only if absent entirely; required 3 must not be
                    if empties.any() and col in REQUIRED_NULL_FREE:
                        geodata_problems.append(f"{folder}: {int(empties.sum())} null/empty '{col}'")
    finally:
        pb.close()
    report["checks"]["geodata_integrity"] = {
        "folders_checked": len(crop_folders),
        "problems": geodata_problems[:20],
        "n_problems": len(geodata_problems),
    }
    if geodata_problems:
        fail(f"GEODATA integrity: {len(geodata_problems)} problem(s)")

    if master_df is None:
        fail("master metadata.csv missing at dest root")
        report["checks"]["master_metadata"] = {"present": False}
    else:
        null_issues = []
        for col in ("crop_id", "crop_bbox_corners", "is_partial_object",
                    "truncated_by_scene_edge", "is_synthetic_location"):
            if col not in master_df.columns:
                null_issues.append(f"missing column {col}")
                continue
            empty = master_df[col].astype(str).str.strip() == ""
            if empty.any():
                null_issues.append(f"{col}: {int(empty.sum())} empty")
        report["checks"]["master_metadata"] = {
            "present": True, "rows": int(len(master_df)), "problems": null_issues,
        }
        if null_issues:
            fail(f"master metadata problems: {null_issues}")

    # --- 3. Bit-depth accuracy -------------------------------------------
    bitdepth_problems: List[str] = []
    checked = 0
    if master_df is not None:
        cal = master_df[master_df.get("is_calibrated", pd.Series(dtype=str)).astype(str) == "Y"]
        for _, r in cal.head(sample_bitdepth).iterrows():
            cid = r.get("crop_id", "")
            folder_rel = None
            # Find the file anywhere under dest (cheap walk limited by count)
            matches = list(dest_dir.rglob(f"{cid}_VV.png")) + list(dest_dir.rglob(f"{cid}_VH.png"))
            if not matches:
                bitdepth_problems.append(f"{cid}: PNG not found")
                continue
            for m in matches[:1]:
                dtype = _gdalinfo_dtype(m) or _rasterio_dtype(m)
                checked += 1
                if dtype and dtype.lower() not in ("uint16", "u16"):
                    bitdepth_problems.append(f"{m.name}: Type={dtype}, expected UInt16")
            _ = folder_rel
    report["checks"]["bit_depth"] = {
        "checked": checked, "problems": bitdepth_problems,
        "tool": "gdalinfo" if shutil.which("gdalinfo") else "rasterio",
    }
    if bitdepth_problems:
        fail(f"bit-depth: {len(bitdepth_problems)} problem(s)")

    # --- 4. Split hygiene (C.5) ------------------------------------------
    splits_dir = Path(splits_dir) if splits_dir else (cfg.splits_dir if cfg else None)
    if splits_dir and Path(splits_dir).is_dir():
        seen_group: Dict[str, str] = {}
        leakage: List[str] = []
        split_counts: Dict[str, int] = {}
        for name in ("train", "val", "test"):
            p = Path(splits_dir) / f"{name}.csv"
            if not p.is_file():
                leakage.append(f"missing {p.name}")
                continue
            ids = set(pd.read_csv(p)["image_id"].astype(str))
            split_counts[name] = len(ids)
            if master_df is not None:
                sub = master_df[master_df["crop_id"].astype(str).isin(ids)]
                for gid, grp in sub.groupby("crop_source_scene_id"):
                    gid = str(gid)
                    if gid in seen_group and seen_group[gid] != name:
                        leakage.append(f"group {gid} in {seen_group[gid]} and {name}")
                    seen_group[gid] = name
        report["checks"]["splits"] = {"counts": split_counts, "leakage": leakage[:20]}
        if leakage:
            fail(f"split leakage/problems: {leakage[:3]}")
    else:
        report["checks"]["splits"] = {"skipped": True}

    # --- 5. Provenance sanity --------------------------------------------
    if master_df is not None and "is_synthetic_location" in master_df.columns:
        vals = set(master_df["is_synthetic_location"].astype(str).str.lower().unique())
        bad = vals - {"true", "false"}
        report["checks"]["provenance_flag"] = {"values": sorted(vals)}
        if bad:
            fail(f"is_synthetic_location has unexpected values: {bad}")

    report["summary"] = {
        "crop_folders": len(crop_folders),
        "master_rows": int(len(master_df)) if master_df is not None else 0,
        "ok": report["ok"],
    }
    return report
