"""Stage 2 — full-extent feature extraction → instances.csv (refinement C.0).

Shape + radiometric features are computed from the FULL-SCENE mask BEFORE any
tiling decision. Contours/component masks stay in-process only; the CSV holds
the serialisable scalar fields plan/convert/metadata need.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .catalog import SceneRecord
from .config import PipelineConfig
from .features import SceneFeatures, extract_full_extent_features, prepare_band_fields
from .pool import run_serial_or_pool
from .scene_io import band_slot, load_scene_geometry, read_mask
from .raster_io import read_full
from .state import StatePaths, load_catalog, save_instances

log = logging.getLogger("proced.features")


@dataclass
class FeatureJob:
    scene: SceneRecord
    cfg: PipelineConfig


@dataclass
class FeatureResult:
    scene_id: str
    rows: List[dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def instance_row(inst, scene: SceneRecord, fragment_count: int) -> dict:
    return {
        "scene_id": scene.scene_id,
        "instance_index": inst.instance_index,
        "label": inst.label,
        "mask_value": inst.mask_value,
        "area_px": inst.area_px,
        "perimeter": round(inst.perimeter, 6),
        "aspect_ratio": round(inst.aspect_ratio, 6),
        "boundary_complexity": round(inst.boundary_complexity, 6),
        "bbox_x": inst.bbox[0], "bbox_y": inst.bbox[1],
        "bbox_w": inst.bbox[2], "bbox_h": inst.bbox[3],
        "centroid_x": inst.centroid[0], "centroid_y": inst.centroid[1],
        "truncated_by_scene_edge": inst.truncated_by_scene_edge,
        "fragment_count": fragment_count,
        "crop_source_scene_id": scene.source_scene_name,
        "dataset_source": scene.source_dataset,
        "damping_ratio": _r(inst.damping_ratio),
        "damping_ratio_db": _r(inst.damping_ratio_db),
        "boundary_gradient_steepness": _r(inst.boundary_gradient_steepness),
        "backscatter_variance_ratio": _r(inst.backscatter_variance_ratio),
        "glcm_contrast": _r(inst.glcm_contrast),
        "glcm_homogeneity": _r(inst.glcm_homogeneity),
        "ndpi": _r(inst.ndpi),
    }


def _r(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return round(float(v), 6)
    except (TypeError, ValueError):
        return None


def process_feature_job(job: FeatureJob) -> FeatureResult:  # pragma: no cover - child
    import gc

    scene, cfg = job.scene, job.cfg
    res = FeatureResult(scene_id=scene.scene_id)
    if scene.mask_path is None:
        res.notes.append("no mask — unlabeled scene, no feature rows")
        return res

    datasets, meta = load_scene_geometry(scene)
    band_arrays = band_db = band_lin = mask = None
    try:
        width, height = int(meta["width"]), int(meta["height"])
        mask, has_mask = read_mask(scene, width, height)
        if mask is None or not has_mask:
            res.notes.append("mask missing or size-mismatched — skipped")
            return res
        if not np.any(mask > 0):
            res.notes.append("empty mask — no positive instances")
            return res

        if scene.is_calibrated:
            band_arrays = {}
            for i, tag in enumerate(scene.pol_tags):
                if tag not in ("VV", "VH"):
                    continue
                domain = scene.value_domains.get(tag, "native")
                if domain == "native":
                    continue
                ds_i, band_i = band_slot(scene, i)
                band_arrays[tag] = read_full(datasets[ds_i], band_i)
            band_db, band_lin, notes = prepare_band_fields(
                band_arrays, scene.value_domains, cfg
            )
            res.notes.extend(notes)
            # Immediate RAM saving: the raw full-scene bands are redundant the
            # moment dB/linear copies exist — drop them before feature work.
            band_arrays = None

        feats: SceneFeatures = extract_full_extent_features(
            mask, cfg,
            band_db=band_db,
            band_linear=band_lin,
            default_label=scene.default_label,
            has_dual_pol=scene.has_dual_pol,
        )
        res.rows = [instance_row(inst, scene, feats.fragment_count)
                    for inst in feats.instances]
        res.notes.extend(feats.notes)
        return res
    except Exception as exc:
        res.errors.append(f"{scene.scene_id}: {type(exc).__name__}: {exc}")
        return res
    finally:
        for ds in datasets:
            try:
                ds.close()
            except Exception:
                pass
        # Free every full-scene array NOW (not when the frame exits) so a long
        # worker run never stacks scene-sized buffers.
        band_arrays = band_db = band_lin = mask = None
        gc.collect()


def run_features(cfg: PipelineConfig) -> List[dict]:
    paths = StatePaths.from_config(cfg)
    cat = load_catalog(paths.catalog)

    jobs = [FeatureJob(scene=s, cfg=cfg) for s in cat.scenes if s.mask_path is not None]
    log.info("features: extracting C.0 features from %d masked scenes …", len(jobs))

    results = run_serial_or_pool(cfg, jobs, process_feature_job, label="scenes")

    rows: List[dict] = []
    errors: List[str] = []
    for res in results:
        rows.extend(res.rows)
        errors.extend(res.errors)
        res.rows.clear()  # immediate: rows now owned by `rows`, not the result
        for n in res.notes:
            log.debug("features %s: %s", res.scene_id, n)
    results.clear()
    import gc
    gc.collect()

    # Deterministic order regardless of pool completion order
    rows.sort(key=lambda r: (r["scene_id"], r["instance_index"]))

    paths.ensure_root()
    save_instances(rows, paths.instances)
    log.info("features: %d instances, %d errors → %s",
             len(rows), len(errors), paths.instances)
    if errors:
        for e in errors[:20]:
            log.error("features: %s", e)
    return rows
