#!/usr/bin/env python3
"""Task 1.7 — Wind-speed join → the file the Model Training team needs.

Merges full-extent shape/radiometric features (C.0) with ERA5 10 m wind speed
at each row's (lat, lon, timestamp_utc) — real or synthetic coordinates, the
call is identical (guide Task 1.7).

Output::

    data/features/lookalike_training_data.csv
      = shape features + wind_speed_ms + label

Rows excluded from the classifier table (refinement A.5/C.1 + Part B):
  * truncated_by_scene_edge == Y  (unreliable full-extent shape)
  * is_calibrated != Y            (uncalibrated pixels harm radiometric features)
  * label == background           (classifier is oil vs look-alike)

``--source mock`` (default) produces a clearly-flagged mock column so the
handoff file can exist before CDS credentials are available; switch to
``--source era5`` once ``~/.cdsapirc`` is configured.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def mock_wind_speed(feature_id: str) -> float:
    """Deterministic per-feature mock wind speed (Weibull-ish ocean values).

    Flagged via wind_source='mock' — NEVER present this as real ERA5 data.
    """
    h = int(hashlib.sha1(feature_id.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(h)
    # Rayleigh (Weibull k=2) with scale 6 m/s ≈ typical ocean wind distribution
    return round(float(np.clip(rng.rayleigh(6.0), 0.3, 25.0)), 3)


def era5_join(df: pd.DataFrame, cache_dir: Path) -> pd.Series:
    """Real ERA5 path: group rows by (day, 1° cell) → fetch_winds → read u10/v10."""
    sys.path.insert(0, str(ROOT))
    from lib.fetch_winds import fetch_winds

    try:
        import netCDF4  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "netCDF4 is required for --source era5 (pip install netCDF4)"
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    wind = pd.Series(np.nan, index=df.index, dtype=float)

    df = df.copy()
    df["_lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["_lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["_day"] = pd.to_datetime(df["timestamp_utc"], errors="coerce").dt.strftime("%Y-%m-%d")
    valid = df["_lat"].notna() & df["_lon"].notna() & df["_day"].notna()

    groups = df[valid].groupby(["_day", df.loc[valid, "_lat"].round(0),
                                df.loc[valid, "_lon"].round(0)])
    for (day, lat0, lon0), grp in groups:
        south, north = lat0 - 0.5, lat0 + 0.5
        west, east = lon0 - 0.5, lon0 + 0.5
        start = f"{day}T00:00:00"
        end = f"{day}T23:59:59"
        try:
            nc_path = fetch_winds((south, west, north, east), start, end,
                                  out_dir=str(cache_dir))
        except Exception as exc:
            print(f"WARN fetch failed for {day} cell: {exc}", file=sys.stderr)
            continue
        speeds = _read_wind_speeds(nc_path, grp["_lat"].to_numpy(),
                                   grp["_lon"].to_numpy())
        for i, (idx, speed) in enumerate(zip(grp.index, speeds)):
            if speed is not None and not math.isnan(speed):
                wind.at[idx] = round(speed, 3)
    return wind


def _read_wind_speeds(nc_path, lats, lons):
    import netCDF4
    out = []
    with netCDF4.Dataset(nc_path) as ds:
        u_var = "u10" if "u10" in ds.variables else "u10m"
        v_var = "v10" if "v10" in ds.variables else "v10m"
        u = ds.variables[u_var]
        v = ds.variables[v_var]
        # Flatten first time step if present
        u_arr = u[:]
        v_arr = v[:]
        if hasattr(u_arr, "masked"):
            u_arr = u_arr.filled(np.nan)
            v_arr = v_arr.filled(np.nan)
        u_arr = np.squeeze(np.asarray(u_arr, dtype=float))
        v_arr = np.squeeze(np.asarray(v_arr, dtype=float))
        try:
            lat_dim = ds.variables[u_var].dimensions[0]
            lats_grid = np.asarray(ds.variables[lat_dim][:], dtype=float)
            lon_dim = ds.variables[u_var].dimensions[1]
            lons_grid = np.asarray(ds.variables[lon_dim][:], dtype=float)
        except Exception:
            lats_grid = np.arange(u_arr.shape[0])
            lons_grid = np.arange(u_arr.shape[1])
        for la, lo in zip(lats, lons):
            iy = int(np.argmin(np.abs(lats_grid - la)))
            ix = int(np.argmin(np.abs(lons_grid - lo)))
            uu, vv = u_arr[iy, ix], v_arr[iy, ix]
            if np.isnan(uu) or np.isnan(vv):
                out.append(float("nan"))
            else:
                out.append(math.sqrt(uu * uu + vv * vv))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features",
                    default=str(ROOT / "data" / "features" / "full_extent_features.csv"))
    ap.add_argument("--out",
                    default=str(ROOT / "data" / "features" / "lookalike_training_data.csv"))
    ap.add_argument("--source", choices=["mock", "era5"], default="mock",
                    help="mock = flagged synthetic winds (no CDS account needed)")
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "cache" / "winds"))
    args = ap.parse_args(argv)

    path = Path(args.features)
    if not path.is_file():
        raise SystemExit(f"features file not found: {path} — run sar_dataset_pipeline.py first")
    df = pd.read_csv(path).fillna("")

    # --- Filter to classifier-eligible rows (A.5 / C.1 / Part B) ----------
    mask_eligible = (
        (df.get("eligible_for_classifier", "Y").astype(str).str.upper() != "N")
        & (df.get("truncated_by_scene_edge", "N").astype(str).str.upper() != "Y")
        & (df.get("is_calibrated", "Y").astype(str).str.upper() == "Y")
        & (df.get("label", "").isin(["oil", "lookalike"]))
    )
    out_df = df[mask_eligible].copy()
    excluded = len(df) - len(out_df)

    if args.source == "era5":
        out_df["wind_speed_ms"] = era5_join(out_df, Path(args.cache_dir))
        out_df["wind_source"] = np.where(out_df["wind_speed_ms"].notna(), "era5", "")
    else:
        out_df["wind_speed_ms"] = [
            mock_wind_speed(str(f)) for f in out_df.get("feature_id", out_df.index).astype(str)
        ]
        out_df["wind_source"] = "mock"

    cols = [c for c in df.columns if c in out_df.columns] + ["wind_speed_ms", "wind_source"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df[cols].to_csv(out_path, index=False)

    n_missing = int((out_df["wind_speed_ms"] == "").sum()) if "wind_speed_ms" in out_df else 0
    print(f"Classifier training table → {out_path}")
    print(f"  rows: {len(out_df)}  (excluded {excluded}: truncated/uncalibrated/background)")
    print(f"  wind_source: {args.source}"
          + ("  ⚠ MOCK — not real ERA5; regenerate with --source era5 for production"
             if args.source == "mock" else ""))
    if n_missing:
        print(f"  wind_speed_ms missing on {n_missing} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
