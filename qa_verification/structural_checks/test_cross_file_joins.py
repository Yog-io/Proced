"""Structural — bidirectional join integrity (architecture §2 Stage 5).

metadata.csv ↔ PNGs on disk ↔ per-folder GEODATA.csv — no orphans either
direction. Recomputes from the filesystem with pure pandas/os, no proced/.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

from .._lib import QAPaths, Report, iter_geodata, load_metadata

_POL_RE = re.compile(r"_(VV|VH|mask)\.png$", re.IGNORECASE)


def _crop_ids_from_pngs(dest: Path) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    for p in dest.rglob("*.png"):
        m = _POL_RE.search(p.name)
        if not m:
            # also accept arbitrary suffix _mask.png already handled; skip non-crop
            stem = p.stem
            if stem.endswith("_VV") or stem.endswith("_VH") or stem.endswith("_mask"):
                cid = stem.rsplit("_", 1)[0]
            else:
                continue
        else:
            cid = p.name[: m.start()]
        out.setdefault(cid, []).append(p)
    return out


def run(paths: QAPaths) -> Report:
    rep = Report("stage5_cross_file_joins")
    skip = paths.require("dest")
    if skip:
        rep.hard(False, f"skip: {skip}")
        return rep

    md_path = paths.dest_dir / "metadata.csv"
    if not md_path.is_file():
        rep.hard(False, f"metadata.csv missing: {md_path}")
        return rep

    md = load_metadata(paths)
    if len(md) == 0:
        rep.hard(False, "metadata.csv empty")
        return rep

    id_col = "crop_id" if "crop_id" in md.columns else "image_id"
    md_ids = set(md[id_col].astype(str))

    # --- metadata → PNG ---------------------------------------------------
    png_map = _crop_ids_from_pngs(paths.dest_dir)
    png_ids = set(png_map)
    meta_no_png = sorted(md_ids - png_ids)
    rep.hard(
        not meta_no_png,
        f"every metadata crop has ≥1 PNG ({len(md_ids)} ids)"
        if not meta_no_png else f"{len(meta_no_png)} metadata rows with no PNG on disk",
        examples=meta_no_png[:25],
        n_metadata=len(md_ids),
        n_png_ids=len(png_ids),
    )

    # --- PNG → metadata (orphaned files) ----------------------------------
    png_no_meta = sorted(png_ids - md_ids)
    rep.hard(
        not png_no_meta,
        f"every PNG has a metadata row ({len(png_ids)} ids)"
        if not png_no_meta else f"{len(png_no_meta)} orphaned PNGs without metadata",
        examples=png_no_meta[:25],
    )

    # --- metadata ↔ GEODATA -----------------------------------------------
    geodata_ids: Set[str] = set()
    gd_unreadable: List[str] = []
    for p, gdf in iter_geodata(paths):
        idc = "crop_id" if "crop_id" in gdf.columns else None
        if idc is None:
            gd_unreadable.append(str(p))
            continue
        geodata_ids.update(gdf[idc].astype(str))

    if not geodata_ids and not gd_unreadable:
        rep.soft(False, "no GEODATA.csv rows found under dest_dir")
    else:
        md_not_in_gd = sorted(md_ids - geodata_ids)
        gd_not_in_md = sorted(geodata_ids - md_ids)
        # Negatives are in metadata but should also be in GEODATA (same rows).
        rep.hard(
            not md_not_in_gd,
            f"every metadata crop_id appears in some GEODATA.csv ({len(md_ids)})"
            if not md_not_in_gd else f"{len(md_not_in_gd)} metadata ids missing from GEODATA",
            examples=md_not_in_gd[:25],
        )
        rep.hard(
            not gd_not_in_md,
            f"every GEODATA crop_id appears in metadata ({len(geodata_ids)})"
            if not gd_not_in_md else f"{len(gd_not_in_md)} GEODATA ids missing from metadata",
            examples=gd_not_in_md[:25],
        )

    # --- PNG folder co-location: each crop's PNGs share a folder with GEODATA
    orphan_folder: List[str] = []
    for cid, files in png_map.items():
        if not files:
            continue
        folder = files[0].parent
        if not (folder / "GEODATA.csv").is_file():
            orphan_folder.append(f"{cid} @ {folder}")
    rep.hard(
        not orphan_folder,
        "every PNG folder contains GEODATA.csv"
        if not orphan_folder else f"{len(orphan_folder)} PNG folders lack GEODATA.csv",
        examples=orphan_folder[:20],
    )

    # --- crop_rows.json ↔ metadata row count ------------------------------
    cr_path = paths.state_dir / "crop_rows.json"
    if cr_path.is_file():
        try:
            import json
            rows = json.loads(cr_path.read_text())
            n_cr = len(rows)
            rep.hard(
                n_cr == len(md),
                f"crop_rows.json ({n_cr}) ↔ metadata.csv ({len(md)}) counts match"
                if n_cr == len(md) else f"row count mismatch: {n_cr} vs {len(md)}",
            )
            if rows:
                cr_ids = {str(r.get("crop_id") or r.get("image_id")) for r in rows}
                only_md = sorted(md_ids - cr_ids)
                only_cr = sorted(cr_ids - md_ids)
                rep.hard(
                    not only_md and not only_cr,
                    "crop_rows ↔ metadata id sets identical"
                    if not only_md and not only_cr
                    else "crop_rows ↔ metadata id set mismatch",
                    only_metadata=only_md[:20],
                    only_crop_rows=only_cr[:20],
                )
        except Exception as exc:
            rep.soft(False, f"crop_rows.json unreadable: {exc}")

    return rep
