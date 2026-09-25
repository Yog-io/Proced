"""Stage 3a — scene geolocation & timestamps → scene_geo.json.

Real CRS/affine preserved when present; otherwise a corridor-based synthetic
location (guide §2.2) with full provenance. Runs BEFORE features so classifier
rows can carry lat/lon/timestamp without re-opening scenes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from .config import PipelineConfig
from .geo import (
    load_corridors,
    sample_synthetic_location,
    synthetic_scene_georef,
    synthetic_timestamp,
)
from .pool import stable_rng
from .progress import bar as progress_bar
from .scene_io import load_scene_geometry, scene_timestamp_from_tags
from .state import StatePaths, load_catalog, save_scene_geo

log = logging.getLogger("proced.geo")


def _transform_list(transform) -> Optional[list]:
    if transform is None:
        return None
    try:
        return [float(transform.a), float(transform.b), float(transform.c),
                float(transform.d), float(transform.e), float(transform.f)]
    except Exception:
        return None


def _center_wgs84(transform, crs, width: int, height: int):
    from rasterio.warp import transform as warp_transform
    try:
        cx, cy = transform * (width / 2.0, height / 2.0)
        lons, lats = warp_transform(crs, "EPSG:4326", [cx], [cy])
        return round(float(lats[0]), 6), round(float(lons[0]), 6)
    except Exception:
        return None, None


def run_geo(cfg: PipelineConfig) -> Dict[str, dict]:
    paths = StatePaths.from_config(cfg)
    cat = load_catalog(paths.catalog)
    corridors = load_corridors(cfg.corridors_path)
    geo: Dict[str, dict] = {}
    pb = progress_bar(len(cat.scenes), "geo", unit="scene")

    try:
        for scene in cat.scenes:
            rng = stable_rng(cfg.seed, scene.rel_folder, scene.scene_id)
            datasets, meta = load_scene_geometry(scene)
            try:
                width, height = int(meta["width"]), int(meta["height"])
                transform, crs, has_geo = meta["transform"], meta["crs"], meta["has_geo"]

                entry: dict = {
                    "has_geo": bool(has_geo),
                    "width": width,
                    "height": height,
                    "transform": _transform_list(transform) if has_geo else None,
                    "crs": str(crs) if (has_geo and crs is not None) else None,
                    "synthetic_geo": None,
                    "source_corridor_id": "",
                    "offset_km": "",
                    "synthetic_lat": None,
                    "synthetic_lon": None,
                }

                if not has_geo:
                    loc = sample_synthetic_location(
                        corridors, rng,
                        offset_scale_km=cfg.offset_scale_km,
                        retry_attempts=cfg.land_retry_attempts,
                    )
                    entry["synthetic_geo"] = synthetic_scene_georef(
                        loc["synthetic_lat"], loc["synthetic_lon"], width, height,
                        pixel_size_m=cfg.synthetic_pixel_size_m,
                    )
                    entry["is_synthetic_location"] = True
                    entry["source_corridor_id"] = loc["source_corridor_id"]
                    entry["offset_km"] = loc["offset_km"]
                    entry["synthetic_lat"] = loc["synthetic_lat"]
                    entry["synthetic_lon"] = loc["synthetic_lon"]
                    entry["center_lat"] = loc["synthetic_lat"]
                    entry["center_lon"] = loc["synthetic_lon"]
                    entry["timestamp_utc"] = synthetic_timestamp(
                        rng, cfg.ts_start_dt(), cfg.ts_end_dt()
                    )
                    entry["is_synthetic_timestamp"] = True
                else:
                    lat, lon = _center_wgs84(transform, crs, width, height)
                    entry["is_synthetic_location"] = False
                    entry["center_lat"] = lat
                    entry["center_lon"] = lon
                    ts = scene_timestamp_from_tags(datasets[0]) if datasets else None
                    if ts is None:
                        ts = synthetic_timestamp(rng, cfg.ts_start_dt(), cfg.ts_end_dt())
                        entry["is_synthetic_timestamp"] = True
                    else:
                        entry["is_synthetic_timestamp"] = False
                    entry["timestamp_utc"] = ts

                geo[scene.scene_id] = entry
            finally:
                for ds in datasets:
                    try:
                        ds.close()
                    except Exception:
                        pass
                pb.update()  # every scene counts, even on failure

    finally:
        pb.close()
    paths.ensure_root()
    save_scene_geo(geo, paths.scene_geo)
    n_syn = sum(1 for g in geo.values() if g.get("is_synthetic_location"))
    log.info("geo: %d scenes (%d synthetic locations) → %s",
             len(geo), n_syn, paths.scene_geo)
    return geo
