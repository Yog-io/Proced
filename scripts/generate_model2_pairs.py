#!/usr/bin/env python3
"""Task 1.10 — OpenDrift synthetic drift-pair dataset for Model 2.

Generates ≥2,000 paired sequences ``{initial_shape → resulting_shape}``:

* point-source releases: single blob at t=0 → 24 h drift, 30-min cadence
* line-source releases: moving-vessel discharge along a corridor segment
  (1-4 h seeding), then drift forward

Backends
--------
* ``opendrift`` — real physics (OpenOil + CMEMS/ERA5 forcing files). Used
  automatically when the package AND forcing paths are available.
* ``simple`` (default fallback) — documented physics-lite advection:
  uniform current + windage (3 %) + random-walk diffusion, honest provenance
  via ``current_field_ref``/``wind_field_ref`` = ``synthetic_*``. This is a
  clearly-labelled stand-in so the MVP can ship 2k-5k pairs before forcing
  data is wired up — swap in opendrift for production.

Output per pair (``data/synthetic/model2_pairs/pair_XXXX.geojson``)::

    {initial_shape, wind_field_ref, current_field_ref,
     elapsed_hours, resulting_shape, release_type: "point"|"line"}

Usage::

    python scripts/generate_model2_pairs.py --pairs 2500
    python scripts/generate_model2_pairs.py --backend opendrift \\
        --currents-nc data/cache/currents/x.nc --winds-nc data/cache/winds/y.nc
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shapely.geometry import MultiPoint, Point, Polygon, mapping

from proced.geo import is_ocean, load_corridors

Point2 = Tuple[float, float]  # (lat, lon)


# ---------------------------------------------------------------- field refs
def _load_uniform_fields(nc_path: Optional[str], kind: str):
    """Read a constant (u, v) from a NetCDF if given; else synthetic."""
    if not nc_path:
        return None
    try:
        import netCDF4
        import numpy as np
        with netCDF4.Dataset(nc_path) as ds:
            names = {"current": ("uo", "vo"), "wind": ("u10", "v10")}[kind]
            u = np.nanmean(np.asarray(ds.variables[names[0]][:], dtype=float))
            v = np.nanmean(np.asarray(ds.variables[names[1]][:], dtype=float))
            if np.isnan(u):
                u, v = 0.0, 0.0
            return float(u), float(v), str(nc_path)
    except Exception as exc:
        print(f"WARN: could not read {nc_path} ({exc}); using synthetic field", file=sys.stderr)
    return None


def _synthetic_current(rng: random.Random):
    speed = rng.uniform(0.05, 0.30)  # m/s — typical shelf/open-ocean slow flow
    bearing = rng.uniform(0, 2 * math.pi)
    return speed * math.sin(bearing), speed * math.cos(bearing), "synthetic_uniform_current"


def _synthetic_wind(rng: random.Random):
    speed = rng.uniform(3.0, 12.0)  # m/s
    bearing = rng.uniform(0, 2 * math.pi)
    return speed * math.sin(bearing), speed * math.cos(bearing), "synthetic_uniform_wind"


# ---------------------------------------------------------------- particles
def _seed_point_blob(center: Point2, radius_m: float, n: int, rng: random.Random) -> List[Point2]:
    pts = []
    for _ in range(n):
        r = radius_m * math.sqrt(rng.random())
        th = rng.uniform(0, 2 * math.pi)
        dlat = (r * math.sin(th)) / 110_570.0
        dlon = (r * math.cos(th)) / (111_320.0 * max(math.cos(math.radians(center[0])), 1e-6))
        pts.append((center[0] + dlat, center[1] + dlon))
    return pts


def _seed_line(waypoints: List[Point2], duration_h: float, rate_per_h: int,
               rng: random.Random) -> List[Point2]:
    """Vessel sails the segment while continuously releasing particles."""
    pts: List[Point2] = []
    n_steps = max(2, int(duration_h * 4))  # 15-min seeding steps
    total = _polyline_len_km(waypoints)
    for k in range(n_steps):
        t = k / max(1, n_steps - 1)
        lat, lon, _ = _point_at_fraction(waypoints, t)
        for _ in range(max(1, rate_per_h // 4)):
            jlat = lat + rng.gauss(0, 0.002)
            jlon = lon + rng.gauss(0, 0.002)
            pts.append((jlat, jlon))
    _ = total
    return pts


def _polyline_len_km(waypoints: List[Point2]) -> float:
    total = 0.0
    for i in range(len(waypoints) - 1):
        (la1, lo1), (la2, lo2) = waypoints[i], waypoints[i + 1]
        lat0 = math.radians((la1 + la2) / 2)
        dx = (lo2 - lo1) * 111.32 * math.cos(lat0)
        dy = (la2 - la1) * 110.57
        total += math.hypot(dx, dy)
    return total


def _point_at_fraction(waypoints: List[Point2], t: float) -> Tuple[float, float, float]:
    segs = []
    total = 0.0
    for i in range(len(waypoints) - 1):
        (la1, lo1), (la2, lo2) = waypoints[i], waypoints[i + 1]
        lat0 = math.radians((la1 + la2) / 2)
        dx = (lo2 - lo1) * 111.32 * math.cos(lat0)
        dy = (la2 - la1) * 110.57
        seg_len = math.hypot(dx, dy)
        segs.append((la1, lo1, la2, lo2, seg_len, dx, dy))
        total += seg_len
    if total <= 0:
        return waypoints[0][0], waypoints[0][1], 0.0
    target = t * total
    acc = 0.0
    for la1, lo1, la2, lo2, seg_len, dx, dy in segs:
        if acc + seg_len >= target:
            u = (target - acc) / seg_len if seg_len else 0.0
            heading = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
            return la1 + u * (la2 - la1), lo1 + u * (lo2 - lo1), heading
        acc += seg_len
    return waypoints[-1][0], waypoints[-1][1], 0.0


def _advect(particles: List[Point2], u: float, v: float, wind_u: float, wind_v: float,
            hours: float, step_min: int, diffusion_m: float, rng: random.Random,
            windage: float = 0.03) -> List[Point2]:
    """Forward Euler: current + windage + Gaussian walk (simple backend)."""
    dt_s = step_min * 60.0
    n_steps = int(hours * 60 / step_min)
    parts = list(particles)
    for _ in range(n_steps):
        new_parts = []
        for lat, lon in parts:
            # current + windage displacements (m → deg)
            du = (u + windage * wind_u) * dt_s
            dv = (v + windage * wind_v) * dt_s
            # diffusion (isotropic random walk, std in metres per step)
            du += rng.gauss(0, diffusion_m)
            dv += rng.gauss(0, diffusion_m)
            dlat = dv / 110_570.0
            dlon = du / (111_320.0 * max(math.cos(math.radians(lat)), 1e-6))
            new_parts.append((lat + dlat, lon + dlon))
        parts = new_parts
    return parts


def _polygon(points: List[Point2]) -> Optional[dict]:
    """Particle cloud → GeoJSON Polygon (convex hull per guide)."""
    if len(points) < 3:
        if not points:
            return None
        p = points[0]
        # Degenerate: tiny square so Model 2 always receives a polygon
        e = 0.0005
        return mapping(Polygon([
            (p[0] - e, p[1] - e), (p[0] - e, p[1] + e),
            (p[0] + e, p[1] + e), (p[0] + e, p[1] - e),
        ]))
    hull = MultiPoint(points).convex_hull
    if hull.geom_type == "Point":
        hull = hull.buffer(0.001)
    if hull.geom_type != "Polygon" or hull.is_empty:
        return None
    return mapping(hull)


# ------------------------------------------------------------- opendrift path
def _opendrift_pair(release_type, center, waypoints, rng, currents_nc, winds_nc,
                    duration_h) -> Optional[dict]:
    try:
        from opendrift.models.openoil import OpenOil
    except ImportError:
        return None
    try:
        o = OpenOil(loglevel=20)
        if release_type == "point":
            o.seed_elements(lon=[center[1]], lat=[center[0]], number=500,
                            radius=200, time="2023-06-01 00:00:00")
        else:
            lons = [wp[1] for wp in waypoints]
            lats = [wp[0] for wp in waypoints]
            o.seed_line(lon=lons, lat=lats, number=500,
                        time="2023-06-01 00:00:00")
        if currents_nc:
            o.add_readers_from_files([currents_nc])
        if winds_nc:
            o.add_readers_from_files([winds_nc])
        o.run(duration=duration_h / 24.0, time_step=1800)
        final_lats = o.history["lat"][:, -1]
        final_lons = o.history["lon"][:, -1]
        init_lats = o.history["lat"][:, 0]
        init_lons = o.history["lon"][:, 0]
        initial = _polygon(list(zip(init_lats, init_lons)))
        resulting = _polygon(list(zip(final_lats, final_lons)))
        if initial is None or resulting is None:
            return None
        return {
            "initial_shape": initial,
            "current_field_ref": currents_nc or "none",
            "wind_field_ref": winds_nc or "none",
            "elapsed_hours": duration_h,
            "resulting_shape": resulting,
            "release_type": release_type,
        }
    except Exception as exc:  # pragma: no cover - requires forcings
        print(f"WARN: opendrift backend failed ({exc}); falling back to simple",
              file=sys.stderr)
        return None


# ------------------------------------------------------------------ main gen
def generate_pairs(n_pairs: int, corridors, cfg, backend: str,
                   currents_nc: Optional[str], winds_nc: Optional[str],
                   out_dir: Path, seed: int) -> dict:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    curr = _load_uniform_fields(currents_nc, "current") if currents_nc else None
    wind = _load_uniform_fields(winds_nc, "wind") if winds_nc else None

    made = 0
    counts = {"point": 0, "line": 0, "opendrift": 0, "simple": 0}
    for i in range(n_pairs * 3):  # bounded attempts
        if made >= n_pairs:
            break
        corridor = rng.choice(corridors)
        release_type = "point" if rng.random() < 0.6 else "line"
        duration_h = 24.0
        used_backend = "simple"

        if release_type == "point":
            t = rng.random()
            # pick a random segment
            idx = rng.randrange(len(corridor["waypoints"]) - 1)
            (la1, lo1), (la2, lo2) = corridor["waypoints"][idx], corridor["waypoints"][idx + 1]
            center = (la1 + t * (la2 - la1), lo1 + t * (lo2 - lo1))
            if not is_ocean(*center):
                continue
            radius_m = rng.uniform(50.0, 400.0)
            particles = _seed_point_blob(center, radius_m, n=80, rng=rng)
            waypoints = None
        else:
            # take up to 3 consecutive corridor waypoints as the vessel track
            start_idx = rng.randrange(max(1, len(corridor["waypoints"]) - 2))
            waypoints = corridor["waypoints"][start_idx:start_idx + 3]
            if len(waypoints) < 2:
                continue
            line_hours = rng.uniform(1.0, 4.0)
            particles = _seed_line(waypoints, line_hours, rate_per_h=200, rng=rng)
            center = waypoints[0]

        if backend == "opendrift":
            pair = _opendrift_pair(release_type, center, waypoints, rng,
                                   currents_nc, winds_nc, duration_h)
            if pair is not None:
                used_backend = "opendrift"
            else:
                pair = None
        else:
            pair = None

        if pair is None:
            used_backend = "simple"
            cu, cv, cref = curr if curr else _synthetic_current(rng)
            wu, wv, wref = wind if wind else _synthetic_wind(rng)
            diff = rng.uniform(15.0, 60.0)  # m per 30-min step
            final = _advect(particles, cu, cv, wu, wv, duration_h, 30, diff, rng)
            initial_poly = _polygon(particles)
            final_poly = _polygon(final)
            if initial_poly is None or final_poly is None:
                continue
            pair = {
                "initial_shape": initial_poly,
                "current_field_ref": cref,
                "wind_field_ref": wref,
                "elapsed_hours": duration_h,
                "resulting_shape": final_poly,
                "release_type": release_type,
            }

        pair["release_type"] = release_type
        pair["corridor_id"] = corridor["id"]
        pair["backend"] = used_backend
        out_path = out_dir / f"pair_{made:05d}.geojson"
        with open(out_path, "w") as fh:
            json.dump(pair, fh)
        made += 1
        counts[release_type] += 1
        counts[used_backend] += 1

    return {"pairs": made, **counts}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=int, default=2500)
    ap.add_argument("--out", default=str(ROOT / "data" / "synthetic" / "model2_pairs"))
    ap.add_argument("--corridors", default=str(ROOT / "data" / "reference" / "shipping_corridors.geojson"))
    ap.add_argument("--backend", choices=["auto", "opendrift", "simple"], default="auto")
    ap.add_argument("--currents-nc", default=None, help="CMEMS NetCDF (opendrift/simple)")
    ap.add_argument("--winds-nc", default=None, help="ERA5 NetCDF (opendrift/simple)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    backend = args.backend
    if backend == "auto":
        try:
            import opendrift  # noqa: F401
            backend = "opendrift" if (args.currents_nc and args.winds_nc) else "simple"
        except ImportError:
            backend = "simple"

    corridors = load_corridors(Path(args.corridors))
    stats = generate_pairs(args.pairs, corridors, None, backend,
                           args.currents_nc, args.winds_nc, Path(args.out), args.seed)
    print(f"Generated {stats['pairs']} Model-2 pairs → {args.out}")
    print(f"  release types: point={stats['point']} line={stats['line']}")
    print(f"  backends: opendrift={stats['opendrift']} simple={stats['simple']}")
    if stats["simple"] and not (args.currents_nc and args.winds_nc):
        print("  NOTE: synthetic wind/current fields flagged in wind_field_ref/"
              "current_field_ref — regenerate with real forcings for production.")
    if stats["pairs"] < 2000:
        print(f"  WARNING: guide wants ≥2000 pairs (got {stats['pairs']})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
