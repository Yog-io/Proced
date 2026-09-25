"""Stage 3b — crop geometry planning → crop_plans.json (guide A.2 / C.1 / C.3).

Decides WHERE every 256×256 window goes before any pixel is read for conversion.
Reads the mask only for overflow-tile filtering and negative placement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .catalog import SceneRecord
from .config import PipelineConfig
from .crops import CropPlan, plan_crops
from .features import Instance
from .pool import numpy_rng, run_serial_or_pool
from .scene_io import read_mask
from .state import StatePaths, load_catalog, load_instances, save_crop_plans

log = logging.getLogger("proced.plan")


@dataclass
class PlanJob:
    scene: SceneRecord
    instances: List[Instance] = field(default_factory=list)
    cfg: Optional[PipelineConfig] = None


def instance_from_row(r: dict) -> Instance:
    def _f(v, default=None):
        if v is None or v == "":
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    return Instance(
        instance_index=int(r.get("instance_index", 0)),
        label=str(r.get("label", "oil")),
        mask_value=int(r.get("mask_value", 1) or 1),
        area_px=int(r.get("area_px", 0) or 0),
        perimeter=_f(r.get("perimeter"), 0.0) or 0.0,
        aspect_ratio=_f(r.get("aspect_ratio"), 1.0) or 1.0,
        boundary_complexity=_f(r.get("boundary_complexity"), 0.0) or 0.0,
        bbox=(
            int(r.get("bbox_x", 0) or 0), int(r.get("bbox_y", 0) or 0),
            int(r.get("bbox_w", 0) or 0), int(r.get("bbox_h", 0) or 0),
        ),
        centroid=(
            int(r.get("centroid_x", 0) or 0), int(r.get("centroid_y", 0) or 0),
        ),
        truncated_by_scene_edge=bool(r.get("truncated_by_scene_edge")),
        contour=None,
        comp_mask=None,
        damping_ratio=_f(r.get("damping_ratio")),
        damping_ratio_db=_f(r.get("damping_ratio_db")),
        boundary_gradient_steepness=_f(r.get("boundary_gradient_steepness")),
        backscatter_variance_ratio=_f(r.get("backscatter_variance_ratio")),
        glcm_contrast=_f(r.get("glcm_contrast")),
        glcm_homogeneity=_f(r.get("glcm_homogeneity")),
        ndpi=_f(r.get("ndpi")),
    )


def plan_to_dict(p: CropPlan) -> dict:
    return {
        "xoff": p.xoff,
        "yoff": p.yoff,
        "label": p.label,
        "instance_index": p.instance_index,
        "is_partial": p.is_partial,
        "parent_group_id": p.parent_group_id,
        "truncated_by_scene_edge": p.truncated_by_scene_edge,
        "scene_id": p.scene_id,
        "seq": p.seq,
    }


def plan_from_dict(d: dict) -> CropPlan:
    return CropPlan(
        xoff=int(d["xoff"]),
        yoff=int(d["yoff"]),
        label=str(d["label"]),
        instance_index=None if d.get("instance_index") is None else int(d["instance_index"]),
        is_partial=bool(d.get("is_partial")),
        parent_group_id=str(d.get("parent_group_id", "")),
        truncated_by_scene_edge=bool(d.get("truncated_by_scene_edge")),
        scene_id=str(d["scene_id"]),
        seq=int(d.get("seq", 0)),
    )


def process_plan_job(job: PlanJob):  # pragma: no cover - child
    import gc

    scene, cfg = job.scene, job.cfg
    assert cfg is not None
    width, height = scene.width, scene.height
    has_mask = scene.mask_path is not None

    mask = None
    if has_mask:
        mask, ok = read_mask(scene, width, height)
        if not ok:
            mask = None
            has_mask = False
    if mask is None:
        mask = np.zeros((height, width), dtype=np.uint8)

    rng = numpy_rng(cfg.seed, scene.rel_folder, scene.scene_id)
    plans = plan_crops(
        width, height, job.instances, mask, cfg, scene.scene_id, rng, has_mask=has_mask,
    )
    mask = None  # scene-sized array freed immediately after planning
    gc.collect()
    return scene.scene_id, [plan_to_dict(p) for p in plans]


def run_plan(cfg: PipelineConfig) -> Dict[str, List[dict]]:
    paths = StatePaths.from_config(cfg)
    cat = load_catalog(paths.catalog)
    instances = load_instances(paths.instances)

    by_scene: Dict[str, List[dict]] = {}
    for row in instances:
        by_scene.setdefault(row["scene_id"], []).append(row)

    jobs: List[PlanJob] = []
    empty: Dict[str, List[dict]] = {}
    for scene in cat.scenes:
        rows = by_scene.get(scene.scene_id, [])
        insts = [instance_from_row(r) for r in rows]
        # Nothing can come out of an unlabeled scene unless negatives are enabled
        if not insts and scene.mask_path is None and not cfg.negatives_from_unlabeled:
            empty[scene.scene_id] = []
            continue
        jobs.append(PlanJob(scene=scene, instances=insts, cfg=cfg))

    log.info("plan: planning crops for %d scenes (%d trivial-empty) …",
             len(jobs), len(empty))

    results = run_serial_or_pool(cfg, jobs, process_plan_job, label="scenes")
    plans: Dict[str, List[dict]] = dict(empty)
    for scene_id, plist in results:
        plans[scene_id] = plist

    # Stable key order
    ordered = {sid: plans[sid] for sid in sorted(plans)}
    n_tiles = sum(len(v) for v in ordered.values())

    paths.ensure_root()
    save_crop_plans(ordered, paths.crop_plans)
    log.info("plan: %d tiles across %d scenes → %s",
             n_tiles, len(ordered), paths.crop_plans)
    return ordered
