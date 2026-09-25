"""Stage 4 — windowed tiling, calibration, PNG writes, per-folder GEODATA → crop_rows.json.

Heavy parallel stage: each worker owns entire subfolders so GEODATA.csv writes
never race. Resume: folders that already have GEODATA.csv are loaded (not
rewritten) unless ``cfg.force`` — so master metadata stays complete.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .catalog import SceneRecord
from .config import PipelineConfig
from .crops import CropPlan
from .geo import format_corners, synthetic_window_corners, window_wgs84_corners
from .metadata import write_geodata
from .pool import init_worker, run_serial_or_pool
from .radiometry import calibrate_window
from .raster_io import open_any, read_window, write_png
from .scene_io import band_slot, load_scene_geometry, read_mask
from .stage_plan import plan_from_dict
from .state import StatePaths, load_catalog, load_crop_plans, load_instances, load_scene_geo, save_crop_rows

log = logging.getLogger("proced.convert")


@dataclass
class ConvertJob:
    rel_folder: str
    scenes: List[SceneRecord]
    cfg: PipelineConfig


@dataclass
class ConvertResult:
    rel_folder: str
    rows: List[dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    n_crops: int = 0
    skipped: bool = False
    elapsed_s: float = 0.0


# Loaded once per worker process (spawn initializer)
_WORKER_STATE: dict = {}


def _init_convert_worker(cfg: PipelineConfig) -> None:  # pragma: no cover - child
    init_worker(cfg.gdal_cache_mb_per_worker, str(cfg.gdal_num_threads))
    paths = StatePaths.from_config(cfg)
    geo = load_scene_geo(paths.scene_geo)
    raw_instances = load_instances(paths.instances)
    raw_plans = load_crop_plans(paths.crop_plans)
    instances: Dict[str, List[dict]] = {}
    for row in raw_instances:
        instances.setdefault(row["scene_id"], []).append(row)
    plans: Dict[str, List[CropPlan]] = {
        sid: [plan_from_dict(d) for d in plist]
        for sid, plist in raw_plans.items()
    }
    _WORKER_STATE.clear()
    _WORKER_STATE.update({"geo": geo, "instances": instances, "plans": plans})


def _build_row(
    scene: SceneRecord,
    plan: CropPlan,
    cfg: PipelineConfig,
    geo: dict,
    inst_row: Optional[dict],
    corners: List[tuple],
    is_synth: bool,
    corridor_id: str,
    offset_km,
    scene_ts: str,
    is_synth_ts: bool,
    formula_version: str,
) -> dict:
    lat_c = round((corners[0][1] + corners[2][1]) / 2.0, 6)
    lon_c = round((corners[0][0] + corners[2][0]) / 2.0, 6)
    inst = inst_row or {}
    return {
        "crop_id": plan.crop_id,
        "crop_source_scene_id": scene.source_scene_name,
        "is_calibrated": "Y" if scene.is_calibrated else "N",
        "has_dual_pol": "Y" if scene.has_dual_pol else "N",
        "is_synthetic_location": is_synth,
        "crop_bbox_corners": format_corners(corners),
        "is_partial_object": plan.is_partial,
        "parent_group_id": plan.parent_group_id,
        "truncated_by_scene_edge": plan.truncated_by_scene_edge,
        "db_conversion_formula_version": formula_version,
        "label": plan.label,
        "full_extent_area_px": inst.get("area_px", 0),
        "full_extent_perimeter": round(float(inst.get("perimeter") or 0.0), 4),
        "full_extent_boundary_complexity": round(float(inst.get("boundary_complexity") or 0.0), 6),
        "full_extent_aspect_ratio": round(float(inst.get("aspect_ratio") or 0.0), 6),
        # master extras
        "image_id": plan.crop_id,
        "dataset_source": scene.source_dataset,
        "width": cfg.tile_size,
        "height": cfg.tile_size,
        "has_real_geo": not is_synth,
        "lat": lat_c,
        "lon": lon_c,
        "timestamp_utc": scene_ts,
        "source_corridor_id": corridor_id,
        "offset_km": offset_km,
        "crop_xoff": plan.xoff,
        "crop_yoff": plan.yoff,
        "crop_size": cfg.tile_size,
        "source_scene_relpath": scene.rel_primary,
        "is_synthetic_timestamp": is_synth_ts,
    }


def process_convert_job(job: ConvertJob) -> ConvertResult:  # pragma: no cover - child
    import gc
    import time
    t0 = time.time()
    cfg = job.cfg
    geo_all: Dict[str, dict] = _WORKER_STATE["geo"]
    instances_all: Dict[str, List[dict]] = _WORKER_STATE["instances"]
    plans_all: Dict[str, List[CropPlan]] = _WORKER_STATE["plans"]

    res = ConvertResult(rel_folder=job.rel_folder)
    dest_folder = Path(cfg.dest_dir) / job.rel_folder if job.rel_folder else Path(cfg.dest_dir)
    dest_folder.mkdir(parents=True, exist_ok=True)

    for scene in job.scenes:
        try:
            geo = geo_all.get(scene.scene_id) or {}
            scene_plans = plans_all.get(scene.scene_id) or []
            if not scene_plans:
                continue

            inst_rows = instances_all.get(scene.scene_id) or []
            inst_by_idx = {int(r["instance_index"]): r for r in inst_rows}

            datasets, meta = load_scene_geometry(scene)
            try:
                width, height = int(meta["width"]), int(meta["height"])
                transform, crs, has_geo = meta["transform"], meta["crs"], meta["has_geo"]

                mask, has_mask = read_mask(scene, width, height)
                if mask is None:
                    mask = np.zeros((height, width), dtype=np.uint8)

                synthetic_geo = geo.get("synthetic_geo")
                scene_ts = geo.get("timestamp_utc", "")
                is_synth_ts = bool(geo.get("is_synthetic_timestamp"))
                corridor_id = geo.get("source_corridor_id", "")
                offset_km = geo.get("offset_km", "")
                is_synth_scene = bool(geo.get("is_synthetic_location"))

                scene_rows: List[dict] = []
                for plan in scene_plans:
                    crop_base = plan.crop_id
                    formula_version = "v1_linear_neg30_to_0"
                    native_write = True

                    for i, tag in enumerate(scene.pol_tags):
                        if tag not in ("VV", "VH"):
                            continue
                        domain = scene.value_domains.get(tag, "native")
                        ds_i, band_i = band_slot(scene, i)
                        window = read_window(
                            datasets[ds_i], plan.xoff, plan.yoff,
                            cfg.tile_size, cfg.tile_size, band_i,
                        )
                        out, formula_version = calibrate_window(
                            window, domain, lo=cfg.db_min, hi=cfg.db_max,
                            eps=cfg.epsilon, use_gpu=cfg.gpu,
                        )
                        if domain != "native":
                            native_write = False
                        write_png(dest_folder / f"{crop_base}_{tag}.png", out)

                    # Mask window → binary 0/255 PNG (Task 1.3)
                    mw = mask[plan.yoff:plan.yoff + cfg.tile_size,
                              plan.xoff:plan.xoff + cfg.tile_size]
                    if mw.shape[0] != cfg.tile_size or mw.shape[1] != cfg.tile_size:
                        padded = np.zeros((cfg.tile_size, cfg.tile_size), dtype=mask.dtype)
                        padded[: mw.shape[0], : mw.shape[1]] = mw
                        mw = padded
                    write_png(dest_folder / f"{crop_base}_mask.png",
                              (mw > 0).astype(np.uint8) * 255)

                    if has_geo:
                        corners = window_wgs84_corners(
                            transform, crs, plan.xoff, plan.yoff,
                            cfg.tile_size, cfg.tile_size,
                        )
                        is_synth = False
                        c_id, o_km = "", ""
                    else:
                        corners = synthetic_window_corners(
                            synthetic_geo, plan.xoff, plan.yoff,
                            cfg.tile_size, cfg.tile_size,
                        )
                        is_synth = True
                        c_id, o_km = corridor_id, offset_km

                    if not scene.is_calibrated:
                        formula_version = "native_8bit_no_db"
                    elif native_write and scene.is_calibrated:
                        pass  # mixed domain — keep honest tag from last band

                    inst_row = (inst_by_idx.get(plan.instance_index)
                                if plan.instance_index is not None else None)
                    row = _build_row(
                        scene, plan, cfg, geo, inst_row, corners,
                        is_synth, c_id, o_km, scene_ts, is_synth_ts,
                        formula_version,
                    )
                    scene_rows.append(row)

                if scene_rows:
                    write_geodata(dest_folder, scene_rows)
                    res.rows.extend(scene_rows)
                    res.n_crops += len(scene_rows)
            finally:
                for ds in datasets:
                    try:
                        ds.close()
                    except Exception:
                        pass
                # Immediate RAM saving: drop the scene-sized mask/band window
                # before the next scene of this folder is processed.
                mask = None
                gc.collect()
        except Exception as exc:
            res.errors.append(f"{scene.scene_id}: {type(exc).__name__}: {exc}")
            log.exception("convert failed for scene %s", scene.scene_id)

    # A folder may have multiple scenes; GEODATA is written once per scene above
    # into the same file — rewrite the combined set for consistency.
    if res.rows:
        write_geodata(dest_folder, res.rows)

    res.elapsed_s = time.time() - t0
    return res


def _load_existing_geodata(dest_folder: Path) -> List[dict]:
    gp = dest_folder / "GEODATA.csv"
    if not gp.is_file():
        return []
    try:
        import pandas as pd
        df = pd.read_csv(gp).fillna("")
        return df.to_dict("records")
    except Exception:
        return []


def run_convert(cfg: PipelineConfig) -> List[dict]:
    paths = StatePaths.from_config(cfg)
    cat = load_catalog(paths.catalog)
    load_scene_geo(paths.scene_geo)  # prerequisite check
    load_instances(paths.instances)
    load_crop_plans(paths.crop_plans)

    from .stage_scan import ensure_output_tree
    ensure_output_tree(cfg, cat)

    folder_map = cat.folder_to_scenes
    jobs: List[ConvertJob] = []
    all_rows: List[dict] = []
    skipped = 0

    for rel, scenes in folder_map.items():
        dest_folder = Path(cfg.dest_dir) / rel if rel else Path(cfg.dest_dir)
        if not cfg.force and (dest_folder / "GEODATA.csv").is_file():
            skipped += 1
            all_rows.extend(_load_existing_geodata(dest_folder))
            continue
        jobs.append(ConvertJob(rel_folder=rel, scenes=scenes, cfg=cfg))

    if skipped:
        log.info("convert: resume — %d folders already done (use --force to redo); "
                 "loaded %d existing rows", skipped, len(all_rows))

    log.info("convert: %d folders across %d workers (GDAL cache %d MB/worker) …",
             len(jobs), cfg.workers, cfg.gdal_cache_mb_per_worker)

    results = run_serial_or_pool(
        cfg, jobs, process_convert_job,
        init=_init_convert_worker, initargs=(cfg,),
        label="folders",
    )

    errors: List[str] = []
    for res in results:
        all_rows.extend(res.rows)
        errors.extend(res.errors)
        res.rows.clear()  # immediate: rows now owned by `all_rows` only
        res.errors.clear()
    results.clear()
    import gc
    gc.collect()

    # Deterministic master order: by crop_id
    all_rows.sort(key=lambda r: str(r.get("crop_id", "")))

    paths.ensure_root()
    save_crop_rows(all_rows, paths.crop_rows)
    log.info("convert: %d crops from %d folders (%d skipped-resume, %d errors) → %s",
             len(all_rows), len(jobs), skipped, len(errors), paths.crop_rows)
    for e in errors:
        log.error("convert: %s", e)
    return all_rows
