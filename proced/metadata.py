"""Stage 5 — metadata synthesis.

* per-folder ``GEODATA.csv`` (archi.md spec columns) written by the owning
  worker — no two processes ever touch the same CSV (race fix vs a naive
  shared-writer design)
* master ``metadata.csv`` (guide Task 1.3 / refinement C.4 superset) written
  once by the driver
* feature CSVs for the look-alike classifier (C.0 → Task 1.7 chain)
* synthetic-geo provenance CSV (guide §2.2 Step 4)

All booleans are normalised to lowercase ``true``/``false`` (spec wording),
and required columns are never NaN/None (validation checklist).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

# archi.md required GEODATA columns, in spec order
GEODATA_COLUMNS = [
    "crop_id",
    "crop_source_scene_id",
    "is_calibrated",
    "has_dual_pol",
    "is_synthetic_location",
    "crop_bbox_corners",
    "is_partial_object",
    "parent_group_id",
    "truncated_by_scene_edge",
    "db_conversion_formula_version",
    "label",
    "full_extent_area_px",
    "full_extent_perimeter",
    "full_extent_boundary_complexity",
    "full_extent_aspect_ratio",
]

# guide Task 1.3 + refinement C.4 extras (master metadata.csv superset)
MASTER_EXTRA_COLUMNS = [
    "image_id",
    "dataset_source",
    "width",
    "height",
    "has_real_geo",
    "lat",
    "lon",
    "timestamp_utc",
    "source_corridor_id",
    "offset_km",
    "crop_xoff",
    "crop_yoff",
    "crop_size",
    "source_scene_relpath",
    "is_synthetic_timestamp",
]

MASTER_COLUMNS = GEODATA_COLUMNS + [c for c in MASTER_EXTRA_COLUMNS if c not in GEODATA_COLUMNS]

SHAPE_FEATURE_COLUMNS = [
    "feature_id", "image_id", "instance_index", "label",
    "area", "perimeter", "aspect_ratio", "boundary_complexity",
    "fragment_count", "truncated_by_scene_edge", "eligible_for_classifier",
    "crop_source_scene_id", "dataset_source",
]

FULL_FEATURE_COLUMNS = SHAPE_FEATURE_COLUMNS + [
    "is_calibrated", "has_dual_pol",
    "damping_ratio", "damping_ratio_db",
    "boundary_gradient_steepness", "backscatter_variance_ratio",
    "glcm_contrast", "glcm_homogeneity", "ndpi",
    "lat", "lon", "timestamp_utc", "is_synthetic_location",
    "source_corridor_id",
]

SYNTHETIC_GEO_COLUMNS = [
    "image_id", "synthetic_lat", "synthetic_lon", "synthetic_timestamp_utc",
    "source_corridor_id", "offset_km", "is_synthetic_location",
]


def _is_nan(v) -> bool:
    try:
        return isinstance(v, float) and v != v
    except Exception:
        return False


def norm_bool(v: Any) -> str:
    """Spec wants true/false (and Y/N for the Y/N columns).

    NOTE: bool(nan) is True in Python — NaN must be treated as missing first.
    """
    if _is_nan(v) or v is None:
        return "false"
    if isinstance(v, str):
        low = v.strip().lower()
        if low in ("y", "yes", "true", "1"):
            return "true" if low in ("true", "1") else "Y"
        if low in ("n", "no", "false", "0"):
            return "false" if low in ("false", "0") else "N"
        return v
    if isinstance(v, (bool, np.bool_)):
        return "true" if bool(v) else "false"
    return "true" if v else "false"


def _yn(v: Any) -> str:
    if _is_nan(v) or v is None:
        return "N"
    if isinstance(v, str):
        return v if v in ("Y", "N") else ("Y" if v.lower() in ("y", "yes", "true", "1") else "N")
    return "Y" if v else "N"


def normalise_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce one metadata row into spec-valid strings."""
    out = dict(row)
    for key in ("is_synthetic_location",):
        if key in out:
            out[key] = norm_bool(out[key])
    for key in ("is_calibrated", "has_dual_pol"):
        if key in out:
            out[key] = _yn(out[key])
    for key in ("is_partial_object", "truncated_by_scene_edge"):
        if key in out:
            out[key] = _yn(out[key])
    for key in ("has_real_geo",):
        if key in out:
            out[key] = _yn(out[key])
    for key in ("is_synthetic_timestamp",):
        if key in out:
            out[key] = norm_bool(out[key])
    # Required columns must never be null
    for key in ("crop_bbox_corners", "is_partial_object", "truncated_by_scene_edge",
                "crop_id", "parent_group_id", "label", "db_conversion_formula_version"):
        if out.get(key) is None or (isinstance(out.get(key), float) and np.isnan(out[key])):
            out[key] = ""
    for key in ("full_extent_area_px", "full_extent_perimeter",
                "full_extent_boundary_complexity", "full_extent_aspect_ratio"):
        v = out.get(key, 0)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out[key] = 0 if key.endswith("area_px") else 0.0
    return out


def rows_to_frame(rows: Sequence[Dict[str, Any]], columns: Optional[List[str]] = None) -> pd.DataFrame:
    normed = [normalise_row(r) for r in rows]
    df = pd.DataFrame(normed)
    if columns:
        for c in columns:
            if c not in df.columns:
                df[c] = ""
        ordered = [c for c in columns if c in df.columns]
        rest = [c for c in df.columns if c not in ordered]
        df = df[ordered + rest]
    return df.fillna("")


def write_geodata(folder: Path, rows: Sequence[Dict[str, Any]]) -> Path:
    """Emit GEODATA.csv inside one terminal subfolder (archi Stage 5)."""
    df = rows_to_frame(rows, GEODATA_COLUMNS)
    # Keep the spec columns strictly first, extras after
    path = Path(folder) / "GEODATA.csv"
    df.to_csv(path, index=False)
    return path


def write_master_metadata(dest: Path, rows: Sequence[Dict[str, Any]]) -> Path:
    df = rows_to_frame(rows, MASTER_COLUMNS)
    path = Path(dest) / "metadata.csv"
    df.to_csv(path, index=False)
    return path


def write_feature_csv(path: Path, rows: Sequence[Dict[str, Any]], columns: List[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = rows_to_frame(rows, columns)
    df.to_csv(path, index=False)
    return path


def synthetic_geo_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Guide §2.2 Step-4 provenance rows for every synthetic-location image."""
    out: List[Dict[str, Any]] = []
    for r in rows:
        if str(r.get("is_synthetic_location", "")).lower() in ("true", "1", "y", "yes"):
            out.append({
                "image_id": r.get("crop_id") or r.get("image_id", ""),
                "synthetic_lat": r.get("lat", ""),
                "synthetic_lon": r.get("lon", ""),
                "synthetic_timestamp_utc": r.get("timestamp_utc", ""),
                "source_corridor_id": r.get("source_corridor_id", ""),
                "offset_km": r.get("offset_km", ""),
                "is_synthetic_location": "true",
            })
    return out


def write_synthetic_assignments(path: Path, rows: Sequence[Dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = rows_to_frame(rows, SYNTHETIC_GEO_COLUMNS)
    df.to_csv(path, index=False)
    return path
