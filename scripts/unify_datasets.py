#!/usr/bin/env python3
"""Task 1.3 — Unify supplementary datasets (e.g. Kaggle sets) into one format.

* masks → binary PNG (0 background / 255 foreground)
* images → consistent rasters, PADDED or TILED to 256×256 (never squashed —
  resizing distorts the shape features the look-alike classifier needs)
* appends rows to ``data/processed/metadata.csv`` with the guide Task 1.3
  columns (+ the refinement C.4 columns, defaulting safely)

NOTE: Kaggle sets stay 8-bit native (refinement C.2 — pretraining only).
Real geodata is left empty here; run ``scripts/assign_synthetic_geo.py`` after.

Usage::

    python scripts/unify_datasets.py data/raw/<kaggle_set> \\
        --source-name kaggle_sar_train --default-label oil
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd

from proced.config import MASK_NAME_TOKENS, NATIVE_FORMULA_VERSION, RASTER_EXTENSIONS
from proced.raster_io import open_any, write_png

MASTER_COLUMNS = [
    "crop_id", "crop_source_scene_id", "is_calibrated", "has_dual_pol",
    "is_synthetic_location", "crop_bbox_corners", "is_partial_object",
    "parent_group_id", "truncated_by_scene_edge", "db_conversion_formula_version",
    "label", "full_extent_area_px", "full_extent_perimeter",
    "full_extent_boundary_complexity", "full_extent_aspect_ratio",
    "image_id", "dataset_source", "width", "height", "has_real_geo",
    "lat", "lon", "timestamp_utc",
]


def _is_mask(p: Path) -> bool:
    toks = [t for t in p.stem.lower().replace("-", "_").split("_") if t]
    return any(t in MASK_NAME_TOKENS for t in toks)


def pad_or_tile(arr: np.ndarray, size: int):
    """Yield size×size tiles: pad small arrays, tile large ones (never resize)."""
    h, w = arr.shape[:2]
    if h <= size and w <= size:
        out = np.zeros((size, size) + arr.shape[2:], dtype=arr.dtype)
        out[:h, :w] = arr
        yield out, 0, 0
        return
    for y in range(0, max(1, h - size + 1), size):
        for x in range(0, max(1, w - size + 1), size):
            tile = arr[y:y + size, x:x + size]
            if tile.shape[0] < size or tile.shape[1] < size:
                padded = np.zeros((size, size) + arr.shape[2:], dtype=arr.dtype)
                padded[: tile.shape[0], : tile.shape[1]] = tile
                tile = padded
            yield tile, x, y


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="Dataset folder (images + optional masks)")
    ap.add_argument("--source-name", required=True, help="dataset_source tag, e.g. kaggle_sar_train")
    ap.add_argument("--out", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--default-label", default="oil", choices=["oil", "lookalike"])
    ap.add_argument("--tile-size", type=int, default=256)
    args = ap.parse_args(argv)

    src = Path(args.src)
    out = Path(args.out)
    img_dir = out / "unified_images"
    mask_dir = out / "unified_masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rasters = [p for p in sorted(src.rglob("*"))
               if p.is_file() and p.suffix.lower() in RASTER_EXTENSIONS]
    images = [p for p in rasters if not _is_mask(p)]
    masks = [p for p in rasters if _is_mask(p)]

    # Pair masks by stem
    mask_by_stem = {}
    for m in masks:
        stem = re.sub(r"(_mask|_label|_gt|mask|label)$", "", m.stem, flags=re.I)
        mask_by_stem[stem.lower()] = m

    rows = []
    n = 0
    for ip in images:
        try:
            with open_any(ip) as ds:
                img = ds.read()
                if img.ndim == 3:
                    img = img[0]  # unified single-channel per guide Task 1.3
                has_dual = ds.count >= 2
        except Exception as exc:
            print(f"skip {ip.name}: {exc}", file=sys.stderr)
            continue
        if img.dtype != np.uint8:
            # Keep native range for uncalibrated exports; scale only if clearly 16-bit visual
            if img.dtype == np.uint16:
                img = (img / 257).astype(np.uint8) if img.max() > 255 else img.astype(np.uint8)
            else:
                lo, hi = float(np.min(img)), float(np.max(img))
                rng = (hi - lo) or 1.0
                img = np.clip((img - lo) / rng * 255.0, 0, 255).astype(np.uint8)

        mask_full = None
        ms = mask_by_stem.get(ip.stem.lower()) or mask_by_stem.get(
            ip.stem.lower().replace("_vv", "").replace("_vh", "")
        )
        if ms is not None:
            try:
                with open_any(ms) as ds:
                    mask_full = ds.read(1)
                mask_full = (mask_full > 0).astype(np.uint8) * 255
            except Exception:
                mask_full = None

        for tile_i, (tile, x, y) in enumerate(pad_or_tile(img, args.tile_size)):
            image_id = f"{args.source_name}__{ip.stem}_t{tile_i:03d}"
            write_png(img_dir / f"{image_id}.png", tile)
            if mask_full is not None:
                m = np.zeros((args.tile_size, args.tile_size), np.uint8)
                h = min(args.tile_size, mask_full.shape[0] - y)
                w = min(args.tile_size, mask_full.shape[1] - x)
                if h > 0 and w > 0:
                    m[:h, :w] = mask_full[y:y + h, x:x + w]
                write_png(mask_dir / f"{image_id}.png", m)
            rows.append({
                "crop_id": image_id,
                "crop_source_scene_id": ip.stem,
                "is_calibrated": "N",
                "has_dual_pol": "Y" if has_dual else "N",
                "is_synthetic_location": "true",
                "crop_bbox_corners": "",
                "is_partial_object": "N",
                "parent_group_id": "",
                "truncated_by_scene_edge": "N",
                "db_conversion_formula_version": NATIVE_FORMULA_VERSION,
                "label": args.default_label,
                "full_extent_area_px": 0,
                "full_extent_perimeter": 0.0,
                "full_extent_boundary_complexity": 0.0,
                "full_extent_aspect_ratio": 0.0,
                "image_id": image_id,
                "dataset_source": args.source_name,
                "width": args.tile_size,
                "height": args.tile_size,
                "has_real_geo": "N",
                "lat": "", "lon": "", "timestamp_utc": "",
            })
            n += 1

    # Append to (or create) unified metadata.csv
    meta_path = out / "metadata.csv"
    new_df = pd.DataFrame(rows, columns=MASTER_COLUMNS)
    if meta_path.is_file():
        old = pd.read_csv(meta_path)
        for c in MASTER_COLUMNS:
            if c not in old.columns:
                old[c] = ""
        new_df = pd.concat([old[MASTER_COLUMNS], new_df], ignore_index=True)
    new_df.fillna("").to_csv(meta_path, index=False)

    print(f"Unified {len(images)} image(s) → {n} tile(s)")
    print(f"  images: {img_dir}")
    print(f"  masks:  {mask_dir} ({'present' if masks else 'NONE'})")
    print(f"  metadata: {meta_path}")
    print("Next: python scripts/assign_synthetic_geo.py --metadata "
          f"{meta_path} --update-metadata")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
