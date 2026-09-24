"""fetch_currents — Copernicus Marine (CMEMS) current-field wrapper.

Backend team imports this directly::

    from lib.fetch_currents import fetch_currents

    nc_path = fetch_currents(
        bbox=(24.0, 60.0, 26.0, 66.0),          # (south, west, north, east)
        start_time="2021-03-01T00:00:00",
        end_time="2021-03-02T00:00:00",
        variables=["uo", "vo"],                  # eastward / northward current
        out_dir="data/cache/currents",
    )
    # nc_path -> "data/cache/currents/currents_24.0_60.0_26.0_66.0_....nc"

Requires: ``pip install copernicusmarine`` + free credentials
(register at data.marine.copernicus.eu; first-run login caches a token).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence, Tuple


def fetch_currents(
    bbox: Tuple[float, float, float, float],
    start_time: str,
    end_time: str,
    variables: Sequence[str] = ("uo", "vo"),
    out_dir: str = "data/cache/currents",
) -> str:
    """Subset CMEMS currents for a bbox/time window → local NetCDF path.

    Parameters
    ----------
    bbox : (south, west, north, east) in decimal degrees WGS84
    start_time, end_time : ISO8601 timestamps (inclusive)
    variables : CMEMS variable ids (default uo, vo = 3D current components;
                surface-only users should add ``depth=0`` via copernicusmarine
                options — see Notes)
    out_dir : cache directory; identical requests hit the cache and skip download

    Returns
    -------
    str : absolute path of the downloaded ``.nc`` file

    Notes
    -----
    Example with a 0.05° box for one day::

        path = fetch_currents((25.9, 64.9, 26.1, 65.1),
                              "2021-03-01T00:00:00",
                              "2021-03-01T12:00:00")
    """
    try:
        import copernicusmarine
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "copernicusmarine is required for fetch_currents — "
            "pip install copernicusmarine"
        ) from exc

    south, west, north, east = bbox
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(
        f"{bbox}|{start_time}|{end_time}|{variables}".encode()
    ).hexdigest()[:16]
    dest = out / f"currents_{key}.nc"
    if dest.is_file():
        return str(dest.resolve())

    copernicusmarine.subset(
        dataset_id="global-analysis-forecast-phy-001-024",
        variables=list(variables),
        minimum_longitude=west,
        maximum_longitude=east,
        minimum_latitude=south,
        maximum_latitude=north,
        start_datetime=start_time,
        end_datetime=end_time,
        outputfilename=str(dest),
    )
    return str(dest.resolve())


if __name__ == "__main__":  # pragma: no cover
    p = fetch_currents((25.9, 64.9, 26.1, 65.1),
                       "2021-03-01T00:00:00", "2021-03-01T06:00:00")
    print(p)
