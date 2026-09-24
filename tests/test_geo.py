"""Geo: corridors, perpendicular offset, provenance, crop corners."""

import math
import random
from datetime import datetime
from pathlib import Path

import pytest

from proced.geo import (
    FALLBACK_CORRIDORS,
    format_corners,
    load_corridors,
    perpendicular_offset_deg,
    sample_synthetic_location,
    synthetic_scene_georef,
    synthetic_timestamp,
    synthetic_window_corners,
)

ROOT = Path(__file__).resolve().parents[1]
CORRIDORS = ROOT / "data" / "reference" / "shipping_corridors.geojson"


def test_load_corridors_from_geojson():
    corridors = load_corridors(CORRIDORS)
    assert len(corridors) >= 6  # guide requires 6-10
    ids = {c["id"] for c in corridors}
    assert "strait_of_malacca" in ids
    assert "gulf_of_aden_mumbai" in ids
    # GeoJSON [lon,lat] must be converted to internal (lat,lon) —
    # Malacca first waypoint is lon=97.5, lat=5.5
    malacca = next(c for c in corridors if c["id"] == "strait_of_malacca")
    lat, lon = malacca["waypoints"][0]
    assert lat == pytest.approx(5.5, abs=0.01)
    assert lon == pytest.approx(97.5, abs=0.01)


def test_load_corridors_fallback_when_missing():
    corridors = load_corridors(Path("/nonexistent/x.geojson"))
    assert len(corridors) == len(FALLBACK_CORRIDORS)


def test_perpendicular_offset_northbound_gives_east_west_only():
    """Regression: archi's fixed ±90° always offset in longitude only.
    A NORTH-bound segment must offset EAST/WEST (dlon≠0, dlat≈0)."""
    rng = random.Random(0)
    dlat, dlon = perpendicular_offset_deg(d_lat=1.0, d_lon=0.0, lat=10.0,
                                          offset_deg=0.5, rng=rng)
    assert abs(dlat) < 1e-9          # no north/south drift for a northbound route
    assert abs(dlon) > 0.1           # pure lateral (east/west) offset


def test_perpendicular_offset_eastbound_gives_north_south_only():
    rng = random.Random(0)
    dlat, dlon = perpendicular_offset_deg(d_lat=0.0, d_lon=1.0, lat=10.0,
                                          offset_deg=0.5, rng=rng)
    assert abs(dlon) < 1e-9 or abs(dlon) < abs(dlat) * 1e-6 + 1e-9
    assert abs(dlat) > 0.1


def test_longitude_scaling_at_high_latitude():
    """Offset degrees must grow as cos(lat) shrinks (fixed 111km/deg bug).

    Uses a diagonal segment so the perpendicular has an east component —
    for an eastbound segment the perpendicular is pure-north (dlon=0)."""
    rng = random.Random(1)
    dlat_low, dlon_low = perpendicular_offset_deg(1.0, 1.0, lat=10.0, offset_deg=1.0, rng=rng)
    rng = random.Random(1)
    dlat_high, dlon_high = perpendicular_offset_deg(1.0, 1.0, lat=60.0, offset_deg=1.0, rng=rng)
    assert abs(dlon_high) > abs(dlon_low) * 1.5  # ~1/cos(60°)=2×
    _ = dlat_low, dlat_high


def test_sample_synthetic_location_provenance():
    rng = random.Random(42)
    corridors = load_corridors(CORRIDORS)
    loc = sample_synthetic_location(corridors, rng, offset_scale_km=50.0, retry_attempts=50)
    # Guide Step 4 required fields
    for key in ("synthetic_lat", "synthetic_lon", "source_corridor_id",
                "offset_km", "is_synthetic_location"):
        assert key in loc
    assert loc["is_synthetic_location"] is True
    assert loc["source_corridor_id"]
    assert -90 <= loc["synthetic_lat"] <= 90
    assert -180 <= loc["synthetic_lon"] <= 180


def test_sample_location_ocean_mostly():
    rng = random.Random(7)
    corridors = load_corridors(CORRIDORS)
    ocean = 0
    n = 20
    for _ in range(n):
        loc = sample_synthetic_location(corridors, rng, retry_attempts=80)
        if not loc.get("land_fallback"):
            ocean += 1
    assert ocean >= n * 0.8, f"only {ocean}/{n} open-water samples"


def test_synthetic_timestamp_in_range():
    rng = random.Random(3)
    start = datetime(2016, 1, 1)
    end = datetime(2025, 12, 31)
    for _ in range(20):
        ts = synthetic_timestamp(rng, start, end)
        assert "2016" <= ts[:4] <= "2025"
        assert ts.endswith("Z")


def test_synthetic_window_corners_scale_with_pixels():
    """~10 m/px: 256 px ≈ 2.56 km ≈ 0.023° lat (correctly cos(lat)-scaled lon too)."""
    g = synthetic_scene_georef(20.0, 60.0, 512, 512, pixel_size_m=10.0)
    corners = synthetic_window_corners(g, 0, 0, 256, 256)
    lat_tl, lat_bl = corners[0][1], corners[3][1]
    height_deg = lat_tl - lat_bl
    assert 0.02 < height_deg < 0.026  # ≈256×10m/111320 = 0.023°


def test_format_corners_spec_shape():
    s = format_corners([(10.0, 20.0), (10.1, 20.0), (10.1, 19.9), (10.0, 19.9)])
    assert s.startswith("[(") and s.endswith(")]")
    assert s.count("(") == 4
