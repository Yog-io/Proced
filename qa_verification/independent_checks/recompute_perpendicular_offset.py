"""Stage 3a — focused perpendicular-offset regression (bug #1).

Architecture calls this out as the single most important geo check: the
assigned synthetic point must sit along the true normal of the corridor
heading (bearing difference ≈ 90° ± 15°) in a local metric frame — not a
fixed ±π/2 applied in raw lat/lon.
"""

from __future__ import annotations

from typing import List

from .._lib import (
    QAPaths,
    Report,
    angular_diff_deg,
    bearing_deg,
    load_corridors_independent,
    load_scene_geo,
    min_distance_to_polyline_km,
    point_segment_distance_km,
    sample_indices,
)


def _heading_at_closest(lat: float, lon: float, waypoints) -> float:
    best = None
    for i in range(len(waypoints) - 1):
        la1, lo1 = waypoints[i]
        la2, lo2 = waypoints[i + 1]
        d, cl = point_segment_distance_km(lat, lon, la1, lo1, la2, lo2)
        if best is None or d < best[0]:
            best = (d, la1, lo1, la2, lo2)
    _d, la1, lo1, la2, lo2 = best
    return bearing_deg(la1, lo1, la2, lo2)


def run(paths: QAPaths, *, sample: int = 20, seed: int = 42,
        tol_deg: float = 15.0, min_offset_km: float = 2.0) -> Report:
    rep = Report("perpendicular_offset")
    skip = paths.require("scene_geo", "corridors")
    if skip:
        rep.hard(False, f"skip: {skip}")
        return rep

    geo = load_scene_geo(paths)
    corridors = load_corridors_independent(paths.corridors_path)
    if not corridors:
        rep.hard(False, "corridors GeoJSON has no LineString features")
        return rep

    synth = [(sid, g) for sid, g in geo.items() if g.get("is_synthetic_location")]
    if not synth:
        rep.info("no synthetic-geo scenes — perpendicular check skipped")
        return rep

    idxs = sample_indices(len(synth), sample, seed, "perp_offset")
    bad: List[dict] = []
    checked = 0

    for i in idxs:
        sid, g = synth[i]
        try:
            lat = float(g.get("synthetic_lat") or g.get("center_lat"))
            lon = float(g.get("synthetic_lon") or g.get("center_lon"))
        except (TypeError, ValueError):
            bad.append({"scene_id": sid, "error": "missing/invalid coords"})
            continue
        try:
            off_km = float(g.get("offset_km") or 0)
        except (TypeError, ValueError):
            off_km = 0.0
        if off_km < min_offset_km:
            continue  # too short for a stable bearing

        best = None
        for c in corridors:
            d, closest = min_distance_to_polyline_km(lat, lon, c["waypoints"])
            heading = _heading_at_closest(lat, lon, c["waypoints"])
            if best is None or d < best[0]:
                best = (d, closest, heading, c["id"])
        if best is None:
            continue
        _d, (cla, clo), heading, cid = best
        offset_bearing = bearing_deg(cla, clo, lat, lon)
        diff = angular_diff_deg(offset_bearing, heading)
        # distance to heading line should be near 90 (or 270 ≡ 90)
        err = min(abs(diff - 90.0), abs(diff - 270.0))
        checked += 1
        if err > tol_deg:
            bad.append({
                "scene_id": sid,
                "corridor": cid,
                "heading_deg": round(heading, 2),
                "offset_bearing_deg": round(offset_bearing, 2),
                "err_deg": round(err, 2),
                "offset_km": off_km,
                "dist_to_corridor_km": round(_d, 3),
            })

    rep.hard(
        not bad and checked > 0,
        f"perpendicular offset ≈90°±{tol_deg}° on {checked} samples (bug #1)"
        if not bad and checked > 0
        else (
            "no offsets ≥2 km to check" if checked == 0
            else f"{len(bad)}/{checked} perpendicular-offset bearing errors"
        ),
        failures=bad[:20],
        checked=checked,
        tol_deg=tol_deg,
    )
    return rep
