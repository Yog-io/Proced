#!/usr/bin/env python3
"""guide §2.2 — Assign synthetic ocean coordinates + timestamps.

Fills lat/lon/timestamp for every row in a metadata CSV where real geodata is
absent, using shipping-corridor sampling with half-normal perpendicular offset,
water-mask confirmation, and full provenance (guide Step 4).

Deliverable::

    data/features/synthetic_geo_assignments.csv

Usage::

    python scripts/assign_synthetic_geo.py \\
        --metadata data/master_dataset_processed/metadata.csv \\
        --out data/features/synthetic_geo_assignments.csv \\
        --update-metadata        # write lat/lon/timestamp back into metadata.csv

The main pipeline (sar_dataset_pipeline.py) already performs this during
Stage 3; this standalone script exists for guide compliance and for filling
older/foreign metadata tables.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from proced.config import PipelineConfig
from proced.geo import load_corridors, sample_synthetic_location, synthetic_timestamp
from proced.metadata import SYNTHETIC_GEO_COLUMNS, write_synthetic_assignments


def needs_synthesis(row: pd.Series) -> bool:
    """True only when the row lacks usable coordinates and isn't real-geo."""
    if str(row.get("has_real_geo", "")).upper() in ("Y", "TRUE", "1"):
        return False
    lat, lon = row.get("lat", ""), row.get("lon", "")
    try:
        if pd.notna(float(lat)) and pd.notna(float(lon)):
            return False  # already has coordinates (real or previously synthetic)
    except (TypeError, ValueError):
        pass
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", required=True, help="Path to metadata.csv")
    ap.add_argument("--out", default=str(ROOT / "data" / "features" / "synthetic_geo_assignments.csv"))
    ap.add_argument("--corridors", default=str(ROOT / "data" / "reference" / "shipping_corridors.geojson"))
    ap.add_argument("--update-metadata", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    cfg = PipelineConfig(seed=args.seed, corridors_path=Path(args.corridors))
    df = pd.read_csv(args.metadata).fillna("")
    # bool-parsed columns must become object before string assignment
    for col in ("is_synthetic_location", "has_real_geo"):
        if col in df.columns and df[col].dtype != object:
            df[col] = df[col].astype(object)
    corridors = load_corridors(cfg.corridors_path)
    rng = random.Random(cfg.seed)

    prov_rows = []
    for idx, row in df.iterrows():
        if not needs_synthesis(row):
            continue
        loc = sample_synthetic_location(
            corridors, rng,
            offset_scale_km=cfg.offset_scale_km,
            retry_attempts=cfg.land_retry_attempts,
        )
        ts = str(row.get("timestamp_utc", "")) or synthetic_timestamp(
            rng, cfg.ts_start_dt(), cfg.ts_end_dt()
        )
        image_id = str(row.get("crop_id") or row.get("image_id") or f"row{idx}")
        prov_rows.append({
            "image_id": image_id,
            "synthetic_lat": loc["synthetic_lat"],
            "synthetic_lon": loc["synthetic_lon"],
            "synthetic_timestamp_utc": ts,
            "source_corridor_id": loc["source_corridor_id"],
            "offset_km": loc["offset_km"],
            "is_synthetic_location": True,
        })
        if args.update_metadata:
            df.at[idx, "lat"] = loc["synthetic_lat"]
            df.at[idx, "lon"] = loc["synthetic_lon"]
            df.at[idx, "timestamp_utc"] = ts
            df.at[idx, "is_synthetic_location"] = "true"
            df.at[idx, "has_real_geo"] = "N"
            df.at[idx, "source_corridor_id"] = loc["source_corridor_id"]
            df.at[idx, "offset_km"] = loc["offset_km"]

    out = Path(args.out)
    # Write with the exact guide Step-4 schema
    pdf = pd.DataFrame(prov_rows, columns=SYNTHETIC_GEO_COLUMNS)
    pdf["is_synthetic_location"] = pdf["is_synthetic_location"].map(
        lambda v: "true" if v else "false"
    )
    write_synthetic_assignments(out, prov_rows)
    print(f"Assigned synthetic coordinates to {len(prov_rows)} row(s) → {out}")

    if args.update_metadata:
        df.to_csv(args.metadata, index=False)
        print(f"Updated metadata in place → {args.metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
