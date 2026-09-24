#!/usr/bin/env python3
"""Task 1.8 — AIS historical tracks (mock generator, corridor-based).

Generates plausible vessel tracks along the shared shipping corridors for the
demo region (default: Arabian Sea / Gulf of Kutch approach — reuse corridor
``gulf_of_aden_mumbai`` / ``strait_of_hormuz`` context) with realistic cargo
speeds (10-20 kn).

Schema (matches the Backend team's agreed table)::

    mmsi, timestamp_utc, lat, lon, speed_knots, heading

Deliverable::

    data/ais/ais_tracks.csv

Optionally loads into PostGIS::

    python scripts/generate_ais.py --postgres-url postgresql://user:pass/db
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from proced.geo import load_corridors

COLUMNS = ["mmsi", "timestamp_utc", "lat", "lon", "speed_knots", "heading"]


def _corridor_point_with_offset(waypoints, s_km: float, lateral_km: float):
    """Interpolate (lat, lon) at arc-length ``s_km`` with lateral offset."""
    total = 0.0
    segs = []
    for i in range(len(waypoints) - 1):
        (la1, lo1), (la2, lo2) = waypoints[i], waypoints[i + 1]
        # rough km using mean latitude
        lat0 = math.radians((la1 + la2) / 2)
        dx = (lo2 - lo1) * 111.32 * math.cos(lat0)
        dy = (la2 - la1) * 110.57
        seg_len = math.hypot(dx, dy)
        segs.append((i, seg_len, la1, lo1, la2, lo2, dx, dy))
        total += seg_len
    if total <= 0:
        return waypoints[0]
    s = s_km % total
    acc = 0.0
    for i, seg_len, la1, lo1, la2, lo2, dx, dy in segs:
        if acc + seg_len >= s:
            t = (s - acc) / seg_len if seg_len else 0.0
            lat = la1 + t * (la2 - la1)
            lon = lo1 + t * (lo2 - lo1)
            # heading in degrees (0 = north)
            heading = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
            # perpendicular offset
            perp = math.radians(heading + 90.0)
            lat_off = (lateral_km / 110.57) * math.sin(perp)
            lon_off = (lateral_km / (111.32 * max(math.cos(math.radians(lat)), 1e-6))) * math.cos(perp)
            return lat + lat_off, lon + lon_off, heading
        acc += seg_len
    la1, lo1 = waypoints[-1]
    return la1, lo1, 0.0


def generate(
    corridors,
    corridor_ids,
    n_vessels: int,
    days: float,
    t0: datetime,
    step_min: int,
    rng: random.Random,
) -> pd.DataFrame:
    rows = []
    step = timedelta(minutes=step_min)
    n_steps = int(days * 24 * 60 / step_min)
    for v in range(n_vessels):
        cid = corridor_ids[v % len(corridor_ids)]
        corridor = next((c for c in corridors if c["id"] == cid), corridors[0])
        mmsi = 412_000_000 + rng.randrange(0, 8_999_999)
        speed_kn = rng.uniform(10.0, 20.0)          # cargo-ship range
        lateral = rng.gauss(0.0, 2.5)               # small lanekeeping offset
        s_km = rng.uniform(0.0, 200.0)              # start position along route
        for k in range(n_steps):
            # advance along corridor by speed
            s_km += speed_kn * 1.852 * (step_min / 60.0)
            jitter = rng.gauss(0.0, 0.05)
            speed = max(5.0, min(25.0, speed_kn + jitter))
            lat, lon, heading = _corridor_point_with_offset(
                corridor["waypoints"], s_km, lateral
            )
            ts = (t0 + k * step).strftime("%Y-%m-%dT%H:%M:%SZ")
            rows.append({
                "mmsi": mmsi,
                "timestamp_utc": ts,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "speed_knots": round(speed, 2),
                "heading": round(heading, 1),
            })
    return pd.DataFrame(rows, columns=COLUMNS)


def load_to_postgres(df: pd.DataFrame, url: str) -> None:
    try:
        import psycopg2
    except ImportError as exc:
        raise SystemExit("psycopg2-binary required for --postgres-url") from exc
    from io import StringIO
    conn = psycopg2.connect(url)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ais_tracks (
            mmsi BIGINT, timestamp_utc TIMESTAMPTZ, lat DOUBLE PRECISION,
            lon DOUBLE PRECISION, speed_knots DOUBLE PRECISION,
            heading DOUBLE PRECISION
        );
        CREATE INDEX IF NOT EXISTS ais_tracks_mmsi_time
            ON ais_tracks (mmsi, timestamp_utc);
    """)
    buf = StringIO()
    df.to_csv(buf, index=False, header=False)
    buf.seek(0)
    cur.copy_expert(
        "COPY ais_tracks FROM STDIN WITH (FORMAT CSV)", buf
    )
    conn.commit()
    cur.close()
    conn.close()
    print(f"Loaded {len(df)} AIS rows into ais_tracks")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "data" / "ais" / "ais_tracks.csv"))
    ap.add_argument("--corridors", default=str(ROOT / "data" / "reference" / "shipping_corridors.geojson"))
    ap.add_argument("--region-corridor", nargs="*", default=["gulf_of_aden_mumbai"],
                    help="Corridor ids to use (demo region: Arabian Sea)")
    ap.add_argument("--vessels", type=int, default=40)
    ap.add_argument("--days", type=float, default=5.0)
    ap.add_argument("--start", default="2023-06-01T00:00:00Z")
    ap.add_argument("--step-min", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--postgres-url", default=None)
    args = ap.parse_args(argv)

    corridors = load_corridors(Path(args.corridors))
    ids = [c["id"] for c in corridors]
    chosen = [i for i in args.region_corridor if i in ids] or ids
    t0 = datetime.fromisoformat(args.start.replace("Z", "+00:00")).replace(tzinfo=None)
    rng = random.Random(args.seed)

    df = generate(corridors, chosen, args.vessels, args.days, t0, args.step_min, rng)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Generated {len(df)} AIS positions for {df['mmsi'].nunique()} vessels → {out}")

    if args.postgres_url:
        load_to_postgres(df, args.postgres_url)
    else:
        print("Load later with: python scripts/generate_ais.py --postgres-url $DATABASE_URL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
