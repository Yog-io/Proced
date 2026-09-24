"""Stage 3a — independent geo checks: real-geo corners + synthetic corridors."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from .._lib import (
    QAPaths,
    Report,
    apply_affine,
    load_catalog,
    load_crop_rows,
    load_corridors_independent,
    load_scene_geo,
    parse_corners,
    sample_indices,
)


def _wgs84_from_affine(transform6, crs_str: str, col: float, row: float):
    from rasterio.warp import transform as warp_transform
    import rasterio.crs
    x, y = apply_affine(transform6, col, row)
    crs = rasterio.crs.CRS.from_user_input(crs_str)
    lons, lats = warp_transform(crs, "EPSG:4326", [x], [y])
    return float(lats[0]), float(lons[0])


def run(paths: QAPaths, *, sample: int = 30, seed: int = 42) -> Report:
    rep = Report("stage3a_geo")
    skip = paths.require("scene_geo", "catalog")
    if skip:
        rep.hard(False, f"skip: {skip}")
        return rep

    geo = load_scene_geo(paths)
    cat = load_catalog(paths)
    scenes = {s["scene_id"]: s for s in cat.get("scenes") or []}

    # --- synthetic flags + timestamp window -------------------------------
    ts_bad = []
    synth_bad = []
    for sid, g in geo.items():
        if g.get("is_synthetic_location"):
            if not g.get("synthetic_lat") and not g.get("center_lat"):
                synth_bad.append({"scene_id": sid, "error": "no synthetic coords"})
            ts = str(g.get("timestamp_utc") or "")
            try:
                dt = datetime.strptime(ts[:19].replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
                if not (2016 <= dt.year <= 2025):
                    ts_bad.append({"scene_id": sid, "ts": ts})
            except Exception:
                ts_bad.append({"scene_id": sid, "ts": ts, "error": "unparseable"})
    rep.hard(
        not synth_bad,
        "synthetic scenes carry coordinates" if not synth_bad
        else f"{len(synth_bad)} synthetic scenes missing coords",
        examples=synth_bad[:10],
    )
    rep.hard(
        not ts_bad,
        "synthetic timestamps in 2016–2025" if not ts_bad
        else f"{len(ts_bad)} timestamps outside 2016–2025 or bad",
        examples=ts_bad[:10],
    )

    # --- corridors distance (don't trust stored offset_km) ----------------
    corridors = []
    if paths.corridors_path.is_file():
        corridors = load_corridors_independent(paths.corridors_path)
    rep.hard(
        len(corridors) >= 1,
        f"loaded {len(corridors)} corridors from GeoJSON" if corridors
        else "corridors GeoJSON empty/unreadable",
    )

    if corridors and geo:
        # sample synthetic scene centres
        synth_items = [(sid, g) for sid, g in geo.items() if g.get("is_synthetic_location")]
        idxs = sample_indices(len(synth_items), sample, seed, "synth_geo")
        far = []
        for i in idxs:
            sid, g = synth_items[i]
            lat = float(g.get("synthetic_lat") or g.get("center_lat"))
            lon = float(g.get("synthetic_lon") or g.get("center_lon"))
            stored_off = g.get("offset_km")
            # min distance to any corridor
            best = min(
                (_min_poly_km(lat, lon, c["waypoints"]) for c in corridors),
                key=lambda t: t[0],
            )
            dist_km, closest = best
            # allow generous band: half-normal offset scale ~50km + corridor width
            # Architecture: within offset_km of a corridor. Stored offset is the
            # *applied* offset; point should be ≈ that far from the line.
            try:
                allow = float(stored_off) + 15.0 if stored_off not in ("", None) else 80.0
            except (TypeError, ValueError):
                allow = 80.0
            allow = max(allow, 15.0)
            if dist_km > allow + 5.0:
                far.append({
                    "scene_id": sid, "dist_km": round(dist_km, 2),
                    "stored_offset_km": stored_off, "allow_km": allow,
                })
        rep.hard(
            not far,
            f"synthetic points within corridor offset band ({len(idxs)} sampled)"
            if not far else f"{len(far)} synthetic points far from every corridor",
            far=far[:15],
            sampled=len(idxs),
        )

        # --- perpendicular-offset regression (bug #1): bearing ≈ 90° -------
        perp_bad = []
        idxs2 = sample_indices(len(synth_items), min(20, sample), seed, "perp")
        for i in idxs2:
            sid, g = synth_items[i]
            lat = float(g.get("synthetic_lat") or g.get("center_lat"))
            lon = float(g.get("synthetic_lon") or g.get("center_lon"))
            # nearest corridor + nearest point
            best = None
            for c in corridors:
                d, cl = _min_poly_km(lat, lon, c["waypoints"])
                # find local segment heading at closest point
                heading = _heading_at_closest(lat, lon, c["waypoints"])
                if best is None or d < best[0]:
                    best = (d, cl, heading, c["id"])
            if best is None:
                continue
            _d, (cla, clo), heading, cid = best
            # bearing from corridor point → assigned point should be ±90° from heading
            offset_bearing = bearing(cla, clo, lat, lon)
            diff = _ang_diff(offset_bearing, heading)
            # diff should be near 90 (or 270 → also 90)
            err = min(abs(diff - 90.0), abs(diff - 270.0), abs(diff - 90.0))
            err = abs(diff - 90.0) if diff <= 180 else abs(270.0 - diff)
            err = min(abs(diff - 90.0), abs(diff - 270.0))
            if err > 15.0:  # few degrees tolerance → architecture says ~90 ± few
                # very short offsets can be noisy
                try:
                    off_km = float(g.get("offset_km") or 0)
                except (TypeError, ValueError):
                    off_km = 0.0
                if off_km > 2.0:
                    perp_bad.append({
                        "scene_id": sid, "corridor": cid,
                        "heading_deg": round(heading, 1),
                        "offset_bearing_deg": round(offset_bearing, 1),
                        "err_deg": round(err, 1),
                        "offset_km": off_km,
                    })
        rep.hard(
            not perp_bad,
            f"perpendicular offset ≈ 90°±15° on {len(idxs2)} samples (bug #1)"
            if not perp_bad else f"{len(perp_bad)} perpendicular-offset bearing errors",
            bad=perp_bad[:20],
            sampled=len(idxs2),
        )

    # --- real-geo crop corners vs independent affine (sample) -------------
    if paths.dest_dir.is_dir() and paths.state_dir.joinpath("crop_rows.json").is_file():
        rows = load_crop_rows(paths)
        real = rows[rows["has_real_geo"].map(lambda v: str(v).lower() in ("y", "true", "1", "yes")
                                             or v is True)]
        if len(real) == 0:
            # has_real_geo may be bool already in json→df
            real = rows[rows["is_synthetic_location"].map(
                lambda v: str(v).lower() in ("n", "false", "0", "no") or v is False)]
        idxs = sample_indices(len(real), sample, seed, "corners")
        corner_bad = []
        for i in idxs:
            r = real.iloc[i]
            try:
                stored = parse_corners(r["crop_bbox_corners"])
            except Exception as exc:
                corner_bad.append({"crop_id": r.get("crop_id"), "error": str(exc)})
                continue
            # recompute from source scene transform
            src_rel = str(r.get("source_scene_relpath") or "")
            # find scene via catalog by rel_primary
            scene = None
            for s in cat.get("scenes") or []:
                if s.get("rel_primary") == src_rel or Path(str(s.get("rel_primary", ""))).name == Path(src_rel).name:
                    scene = s
                    break
            if scene is None or not scene.get("band_files"):
                continue
            try:
                import rasterio
                from .._lib import resolve_stage_path
                root = cat.get("root") or str(paths.stage_dir)
                path = resolve_stage_path(scene["band_files"][0], root, paths.stage_dir)
                with rasterio.open(str(path)) as ds:
                    if ds.crs is None or ds.transform.is_identity:
                        continue
                    t = [ds.transform.a, ds.transform.b, ds.transform.c,
                         ds.transform.d, ds.transform.e, ds.transform.f]
                    crs = str(ds.crs)
                xoff = int(r["crop_xoff"])
                yoff = int(r["crop_yoff"])
                size = int(r["crop_size"])
                pts_px = [(xoff, yoff), (xoff + size, yoff),
                          (xoff + size, yoff + size), (xoff, yoff + size)]
                recomp = []
                for col, row_px in pts_px:
                    la, lo = _wgs84_from_affine(t, crs, col, row_px)
                    recomp.append((round(lo, 5), round(la, 5)))
                # compare TL within ~0.001° (rounding in pipeline is 6dp)
                if abs(recomp[0][0] - stored[0][0]) > 0.002 or abs(recomp[0][1] - stored[0][1]) > 0.002:
                    corner_bad.append({
                        "crop_id": r.get("crop_id"),
                        "stored_tl": stored[0], "recomputed_tl": recomp[0],
                    })
            except Exception as exc:
                corner_bad.append({"crop_id": r.get("crop_id"), "error": str(exc)})

        rep.hard(
            not corner_bad,
            f"crop_bbox_corners match independent affine ({len(idxs)} sampled)"
            if not corner_bad else f"{len(corner_bad)} corner mismatches",
            bad=corner_bad[:15],
            sampled=len(idxs),
        )
    else:
        rep.info("crop_rows/dest missing — skipped corner recompute")
    return rep


def _min_poly_km(lat, lon, waypoints):
    from .._lib import min_distance_to_polyline_km
    return min_distance_to_polyline_km(lat, lon, waypoints)


def _heading_at_closest(lat, lon, waypoints) -> float:
    """Bearing of the polyline segment nearest to (lat, lon)."""
    from .._lib import point_segment_distance_km, bearing
    best = None
    for i in range(len(waypoints) - 1):
        la1, lo1 = waypoints[i]
        la2, lo2 = waypoints[i + 1]
        d, cl = point_segment_distance_km(lat, lon, la1, lo1, la2, lo2)
        if best is None or d < best[0]:
            best = (d, la1, lo1, la2, lo2)
    _d, la1, lo1, la2, lo2 = best
    return bearing(la1, lo1, la2, lo2)


def bearing(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def _ang_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d
