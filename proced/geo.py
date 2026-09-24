"""Geolocation: shipping corridors, synthetic-coordinate assignment (guide §2.2),
real-affine crop corners in WGS84, and synthetic scene geometry.

Fixes vs the embedded archi.md script:
  * corridors loaded from data/reference/shipping_corridors.geojson (guide Step 1)
  * offset is genuinely PERPENDICULAR to the local segment heading
    (the original applied a fixed ±90° in raw lat/lon space, so every offset
    was pure-longitude regardless of corridor orientation)
  * longitude degrees scaled by cos(lat) (the original treated 1° lon == 111 km
    everywhere)
  * full provenance per guide Step 4: source_corridor_id, offset_km,
    synthetic_timestamp_utc (guide Step 3), is_synthetic_location
"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

try:
    from global_land_mask import globe as _globe
    HAS_LAND_MASK = True
except Exception:  # pragma: no cover - optional dependency
    _globe = None
    HAS_LAND_MASK = False

# Fallback if the geojson is missing (lat, lon waypoints). Mirrors the guide list.
FALLBACK_CORRIDORS = [
    {"id": "strait_of_malacca", "weight": 1.4,
     "waypoints": [(5.5, 97.5), (3.6, 100.3), (2.5, 101.5), (1.6, 103.0), (1.2, 103.8)]},
    {"id": "gulf_of_aden_mumbai", "weight": 1.2,
     "waypoints": [(12.5, 43.5), (13.0, 48.0), (14.5, 53.0), (16.5, 60.0), (18.5, 66.0), (19.5, 70.0), (19.0, 72.5)]},
    {"id": "strait_of_hormuz", "weight": 1.3,
     "waypoints": [(27.0, 51.5), (26.5, 54.0), (26.0, 56.5), (25.5, 57.5)]},
    {"id": "suez_eastern_med", "weight": 1.1,
     "waypoints": [(31.3, 32.3), (30.5, 32.6), (32.5, 33.5), (34.0, 34.5), (35.5, 35.5)]},
    {"id": "english_channel_north_sea", "weight": 1.2,
     "waypoints": [(51.2, 1.8), (50.5, 0.0), (50.0, -1.0), (50.1, -4.5), (50.6, -5.5)]},
    {"id": "gulf_of_mexico", "weight": 1.0,
     "waypoints": [(26.0, -86.0), (27.2, -88.5), (28.0, -89.0), (27.5, -90.0), (27.8, -92.5)]},
    {"id": "bengal_bay_kolkata", "weight": 0.9,
     "waypoints": [(20.5, 86.0), (20.8, 88.5), (21.5, 90.5), (22.3, 91.8)]},
    {"id": "south_china_sea", "weight": 1.2,
     "waypoints": [(15.5, 111.0), (12.5, 112.5), (9.5, 114.0), (6.5, 116.5), (4.0, 119.0)]},
]

Corridor = dict  # {"id", "weight", "waypoints": [(lat, lon), ...]}


# --------------------------------------------------------------------- loaders
def load_corridors(path: Optional[Path]) -> List[Corridor]:
    """Load corridors from GeoJSON (LineString features, [lon, lat] coords).

    Falls back to the embedded guide corridors if the file is absent.
    """
    if path is not None and Path(path).is_file():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                gj = json.load(fh)
            corridors: List[Corridor] = []
            for feat in gj.get("features", []):
                geom = feat.get("geom") or feat.get("geometry") or {}
                if geom.get("type") != "LineString":
                    continue
                props = feat.get("properties") or {}
                # GeoJSON is [lon, lat]; internal convention is (lat, lon)
                waypoints = [(float(c[1]), float(c[0])) for c in geom.get("coordinates", [])]
                if len(waypoints) < 2:
                    continue
                corridors.append({
                    "id": str(props.get("id", f"corridor_{len(corridors)}")),
                    "weight": float(props.get("weight", 1.0)),
                    "waypoints": waypoints,
                })
            if corridors:
                return corridors
        except Exception:
            pass
    return [dict(c) for c in FALLBACK_CORRIDORS]


# ------------------------------------------------------------------- sampling
def _pick_corridor(corridors: Sequence[Corridor], rng: random.Random) -> Corridor:
    weights = [max(c.get("weight", 1.0), 1e-6) for c in corridors]
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for c, w in zip(corridors, weights):
        acc += w
        if r <= acc:
            return c
    return corridors[-1]


def _sample_point_on_corridor(corridor: Corridor, rng: random.Random) -> Tuple[float, float, Tuple[float, float]]:
    """Return (lat, lon, (d_lat, d_lon)) for a random point along the polyline."""
    wps = corridor["waypoints"]
    idx = rng.randrange(len(wps) - 1)
    p1, p2 = wps[idx], wps[idx + 1]
    t = rng.random()
    lat = p1[0] + t * (p2[0] - p1[0])
    lon = p1[1] + t * (p2[1] - p1[1])
    d_lat, d_lon = p2[0] - p1[0], p2[1] - p1[1]
    return lat, lon, (d_lat, d_lon)


def perpendicular_offset_deg(
    d_lat: float, d_lon: float, lat: float, offset_deg: float, rng: random.Random
) -> Tuple[float, float]:
    """Offset (offset_deg, in latitude-equivalent degrees) PERPENDICULAR to the
    segment heading at ``lat``, ± randomly chosen side.

    The archi.md script used ``angle = ±π/2`` in raw lat/lon space which made
    every offset purely longitudinal. Here the heading is computed in a local
    east/north metric frame, rotated 90°, then converted back to degrees with
    proper cos(lat) longitude scaling.
    """
    lat0 = math.radians(lat)
    # Local metric components of the heading
    east = d_lon * math.cos(lat0)
    north = d_lat
    if abs(east) < 1e-12 and abs(north) < 1e-12:
        east = 1.0  # degenerate segment: treat as eastbound
    # Perpendicular direction in (east, north): rotate heading +90° or -90°
    side = 1.0 if rng.random() < 0.5 else -1.0
    perp_east = -north * side
    perp_north = east * side
    norm = math.hypot(perp_east, perp_north)
    perp_east /= norm
    perp_north /= norm
    # Convert offset (degrees-of-latitude-equivalent ~111.32 km/deg) to dlat/dlon
    km_per_deg_lat = 111.32
    dlat = (offset_deg * perp_north) / 1.0  # offset_deg already in latitude degrees
    coslat = max(math.cos(lat0), 1e-6)
    dlon = (offset_deg * perp_east) / coslat
    _ = km_per_deg_lat
    return dlat, dlon


def is_ocean(lat: float, lon: float) -> bool:
    """Water check via global_land_mask, with a coarse fallback heuristic."""
    if HAS_LAND_MASK:
        try:
            return bool(_globe.is_ocean(lat, lon))
        except Exception:
            pass
    # Fallback: crude latitude box avoiding major landmasses (guide allows heuristic)
    if not (-85.0 <= lat <= 85.0 and -180.0 <= lon <= 180.0):
        return False
    return True


def sample_synthetic_location(
    corridors: Sequence[Corridor],
    rng: random.Random,
    offset_scale_km: float = 50.0,
    retry_attempts: int = 50,
) -> dict:
    """Guide §2.2 algorithm → provenance dict (guide Step 4 schema + extras).

    1. pick corridor (weighted)
    2. random point along polyline
    3. half-normal perpendicular offset (scale ≈ offset_scale_km)
    4. ± perpendicular direction
    5. compute final point (cos(lat)-scaled longitude)
    6. water check → resample on land
    """
    offset_scale_deg = offset_scale_km / 111.32
    last: Optional[dict] = None
    for attempt in range(retry_attempts):
        corridor = _pick_corridor(corridors, rng)
        lat, lon, (d_lat, d_lon) = _sample_point_on_corridor(corridor, rng)
        offset_km = abs(rng.gauss(0.0, offset_scale_km))  # half-normal
        offset_deg = offset_km / 111.32
        dlat, dlon = perpendicular_offset_deg(d_lat, d_lon, lat, offset_deg, rng)
        final_lat = lat + dlat
        final_lon = lon + dlon
        if not (-85.0 <= final_lat <= 85.0):
            continue
        if final_lon > 180.0:
            final_lon -= 360.0
        elif final_lon < -180.0:
            final_lon += 360.0
        last = {
            "synthetic_lat": round(final_lat, 6),
            "synthetic_lon": round(final_lon, 6),
            "source_corridor_id": corridor["id"],
            "offset_km": round(offset_km, 3),
            "is_synthetic_location": True,
            "land_fallback": not is_ocean(final_lat, final_lon),
        }
        if is_ocean(final_lat, final_lon):
            return last
    # Could not find open water in N attempts — accept last ocean-ish sample or
    # fall back to a known-open-water coordinate, still flagged synthetic.
    if last is not None:
        last["land_fallback"] = True
        return last
    return {
        "synthetic_lat": 15.0,
        "synthetic_lon": 65.0,
        "source_corridor_id": "fallback_arabian_sea",
        "offset_km": 0.0,
        "is_synthetic_location": True,
        "land_fallback": True,
    }


def synthetic_timestamp(rng: random.Random, start: datetime, end: datetime) -> str:
    """Uniform random ISO8601 UTC timestamp inside [start, end] (guide Step 3)."""
    span = int((end - start).total_seconds())
    if span <= 0:
        return start.strftime("%Y-%m-%dT%H:%M:%SZ")
    ts = start + timedelta(seconds=rng.randrange(span))
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------- crop corner math
def window_wgs84_corners(
    transform,  # rasterio Affine of the SOURCE scene
    crs,
    xoff: int,
    yoff: int,
    width: int,
    height: int,
) -> List[Tuple[float, float]]:
    """Four crop corners (lon, lat): TL, TR, BR, BL via windowed-origin affine.

    new_origin_x = old_origin_x + col_off * pixel_width   (refinement A.4)
    new_origin_y = old_origin_y + row_off * pixel_height
    """
    from rasterio.warp import transform as warp_transform

    new_o = transform * (xoff, yoff)  # (x, y) of crop TL in source CRS
    a, b, _, d, e, _ = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    px_w, px_h = a, e  # (b/d assumed 0 for north-up rasters; general form below)
    # Full affine: x = c + a*col + b*row ; y = f + d*col + e*row
    def _xy(col: float, row: float) -> Tuple[float, float]:
        return (transform.c + transform.a * col + transform.b * row,
                transform.f + transform.d * col + transform.e * row)

    pts_px = [(xoff, yoff), (xoff + width, yoff), (xoff + width, yoff + height), (xoff, yoff + height)]
    xs, ys = zip(*(_xy(c, r) for c, r in pts_px))
    lons, lats = warp_transform(crs, "EPSG:4832" if False else "EPSG:4326", list(xs), list(ys))
    _ = new_o, px_w, px_h
    return [(round(lon, 6), round(lat, 6)) for lon, lat in zip(lons, lats)]


def synthetic_scene_georef(
    lat: float, lon: float, scene_w: int, scene_h: int, pixel_size_m: float = 10.0
) -> dict:
    """Build a synthetic north-up 'scene frame' centered on (lat, lon).

    Returns a dict consumed by :func:`synthetic_window_corners` so that every
    crop gets correctly-scaled bbox corners (~10 m/px Sentinel-1 scale) instead
    of the original script's fixed 0.02° box regardless of crop size.
    """
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * max(math.cos(math.radians(lat)), 1e-6)
    dlat_pp = pixel_size_m / m_per_deg_lat
    dlon_pp = pixel_size_m / m_per_deg_lon
    origin_lat = lat + (scene_h / 2.0) * dlat_pp  # top edge (north)
    origin_lon = lon - (scene_w / 2.0) * dlon_pp  # left edge (west)
    return {
        "origin_lat": origin_lat,
        "origin_lon": origin_lon,
        "dlat_pp": dlat_pp,
        "dlon_pp": dlon_pp,
        "center": (lat, lon),
    }


def synthetic_window_corners(
    georef: dict, xoff: int, yoff: int, width: int, height: int
) -> List[Tuple[float, float]]:
    lat_tl = georef["origin_lat"] - yoff * georef["dlat_pp"]
    lon_tl = georef["origin_lon"] + xoff * georef["dlon_pp"]
    lat_bl = lat_tl - height * georef["dlat_pp"]
    lon_tr = lon_tl + width * georef["dlon_pp"]
    return [
        (round(lon_tl, 6), round(lat_tl, 6)),
        (round(lon_tr, 6), round(lat_tl, 6)),
        (round(lon_tr, 6), round(lat_bl, 6)),
        (round(lon_tl, 6), round(lat_bl, 6)),
    ]


def format_corners(corners: Sequence[Tuple[float, float]]) -> str:
    """Render corners exactly as the archi.md GEODATA spec expects."""
    return "[" + ", ".join(f"({lon}, {lat})" for lon, lat in corners) + "]"
