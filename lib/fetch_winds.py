"""fetch_winds — ERA5 10 m wind wrapper (Copernicus CDS / cdsapi).

Backend team imports this directly::

    from lib.fetch_winds import fetch_winds

    path = fetch_winds(
        bbox=(30.0, 50.0, 32.0, 52.0),          # (south, west, north, east)
        start_time="2021-03-01T00:00:00",
        end_time="2021-03-02T00:00:00",
    )
    # path -> ".../winds_<hash>.nc"

Requires: ``pip install cdsapi`` + ``~/.cdsapirc`` with your CDS UID/key
(register free at cds.climate.copernicus.eu).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Tuple


def fetch_winds(
    bbox: Tuple[float, float, float, float],
    start_time: str,
    end_time: str,
    variables=("u10", "v10"),
    out_dir: str = "data/cache/winds",
    dataset: str = "reanalysis-era5-single-levels",
) -> str:
    """Download ERA5 10 m wind components for a bbox/time window → NetCDF path.

    Parameters
    ----------
    bbox : (south, west, north, east) decimal degrees
    start_time, end_time : ISO8601 timestamps
    variables : ERA5 short names (u10/v10 = 10 m U/V wind components)
    out_dir : download cache (identical requests reuse the file)
    dataset : CDS dataset id (swap for a forecast product if needed)

    Returns
    -------
    str : absolute path of the downloaded file

    Notes
    -----
    wind_speed_ms = sqrt(u10**2 + v10**2)  — see scripts/join_wind.py
    Example::

        p = fetch_winds((25.5, 64.5, 26.5, 65.5),
                        "2021-03-01T00:00:00", "2021-03-01T23:00:00")
    """
    try:
        import cdsapi
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "cdsapi is required for fetch_winds — pip install cdsapi "
            "(and create ~/.cdsapirc)"
        ) from exc

    south, west, north, east = bbox
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(
        f"{dataset}|{bbox}|{start_time}|{end_time}|{variables}".encode()
    ).hexdigest()[:16]
    dest = out / f"winds_{key}.nc"
    if dest.is_file():
        return str(dest.resolve())

    def _iso_d(s: str):
        return s[:10]

    client = cdsapi.Client()
    client.retrieve(dataset, {
        "product_type": "reanalysis",
        "variable": list(variables),
        "year": start_time[:4],
        "month": start_time[5:7],
        "day": {d for d in _day_range(start_time, end_time)},
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [north, west, south, east],
        "format": "netcdf",
    }).download(str(dest))
    return str(dest.resolve())


def _day_range(start_time: str, end_time: str):
    """Inclusive calendar-day list between two ISO dates (strings YYYY-MM-DD)."""
    from datetime import date, timedelta
    d0 = date.fromisoformat(start_time[:10])
    d1 = date.fromisoformat(end_time[:10])
    days = []
    while d0 <= d1:
        days.append(d0.isoformat()[-2:])
        d0 += timedelta(days=1)
        if len(days) > 31:  # safety cap for huge requests
            break
    return days


if __name__ == "__main__":  # pragma: no cover
    print(fetch_winds((25.9, 64.9, 26.1, 65.1),
                      "2021-03-01T00:00:00", "2021-03-01T12:00:00"))
