"""Stage 5 — master metadata.csv + feature/provenance CSVs from pipeline state."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from .config import PipelineConfig
from .metadata import (
    FULL_FEATURE_COLUMNS,
    SHAPE_FEATURE_COLUMNS,
    synthetic_geo_rows,
    write_feature_csv,
    write_master_metadata,
    write_synthetic_assignments,
)
from .state import (
    StatePaths,
    load_catalog,
    load_crop_rows,
    load_instances,
    load_scene_geo,
)

log = logging.getLogger("proced.metadata")


def _yn(v) -> str:
    return "Y" if v else "N"


def _f6(v):
    if v is None or v == "":
        return ""
    try:
        return round(float(v), 6)
    except (TypeError, ValueError):
        return ""


def build_feature_rows(instances: List[dict], scenes: dict, geo: dict):
    """Shape-only and full (radiometric + geo-joined) classifier rows (C.0)."""
    shape_rows: List[dict] = []
    full_rows: List[dict] = []
    for inst in instances:
        sid = inst["scene_id"]
        scene = scenes.get(sid)
        g = geo.get(sid) or {}
        base = {
            "feature_id": f"{sid}_inst{int(inst['instance_index']):03d}",
            "image_id": sid,
            "instance_index": int(inst["instance_index"]),
            "label": inst.get("label", ""),
            "area": int(inst.get("area_px") or 0),
            "perimeter": _f6(inst.get("perimeter")),
            "aspect_ratio": _f6(inst.get("aspect_ratio")),
            "boundary_complexity": _f6(inst.get("boundary_complexity")),
            "fragment_count": int(inst.get("fragment_count") or 0),
            "truncated_by_scene_edge": bool(inst.get("truncated_by_scene_edge")),
            "eligible_for_classifier": not bool(inst.get("truncated_by_scene_edge")),
            "crop_source_scene_id": inst.get("crop_source_scene_id", ""),
            "dataset_source": inst.get("dataset_source", ""),
        }
        shape_rows.append(dict(base))
        is_cal = bool(scene.is_calibrated) if scene else False
        has_dual = bool(scene.has_dual_pol) if scene else False
        full_rows.append({
            **base,
            "is_calibrated": _yn(is_cal),
            "has_dual_pol": _yn(has_dual),
            "damping_ratio": _f6(inst.get("damping_ratio")),
            "damping_ratio_db": _f6(inst.get("damping_ratio_db")),
            "boundary_gradient_steepness": _f6(inst.get("boundary_gradient_steepness")),
            "backscatter_variance_ratio": _f6(inst.get("backscatter_variance_ratio")),
            "glcm_contrast": _f6(inst.get("glcm_contrast")),
            "glcm_homogeneity": _f6(inst.get("glcm_homogeneity")),
            "ndpi": _f6(inst.get("ndpi")),
            "lat": g.get("center_lat", ""),
            "lon": g.get("center_lon", ""),
            "timestamp_utc": g.get("timestamp_utc", ""),
            "is_synthetic_location": bool(g.get("is_synthetic_location")),
            "source_corridor_id": g.get("source_corridor_id", ""),
        })
    return shape_rows, full_rows


def run_metadata(cfg: PipelineConfig) -> Dict[str, int]:
    paths = StatePaths.from_config(cfg)
    cat = load_catalog(paths.catalog)
    geo = load_scene_geo(paths.scene_geo)
    instances = load_instances(paths.instances)
    rows = load_crop_rows(paths.crop_rows)

    dest = Path(cfg.dest_dir)
    features_dir = Path(cfg.features_dir)
    dest.mkdir(parents=True, exist_ok=True)

    if rows:
        write_master_metadata(dest, rows)
        write_synthetic_assignments(
            features_dir / "synthetic_geo_assignments.csv",
            synthetic_geo_rows(rows),
        )
        log.info("metadata: master metadata.csv (%d rows) + synthetic provenance", len(rows))
    else:
        log.warning("metadata: no crop rows — master metadata not written")

    scenes = {s.scene_id: s for s in cat.scenes}
    shape_rows, full_rows = build_feature_rows(instances, scenes, geo)
    write_feature_csv(features_dir / "shape_features.csv", shape_rows, SHAPE_FEATURE_COLUMNS)
    write_feature_csv(features_dir / "full_extent_features.csv", full_rows, FULL_FEATURE_COLUMNS)
    log.info("metadata: %d shape / %d full-extent feature rows → %s",
             len(shape_rows), len(full_rows), features_dir)

    return {
        "master_rows": len(rows),
        "shape_rows": len(shape_rows),
        "full_rows": len(full_rows),
    }
