"""Raster I/O: windowed reads with edge padding, 16/8-bit PNG writes.

Uses rasterio (a first-class GDAL binding) per refinement A.4 —
``rasterio.windows`` is exactly the windowed-read primitive the refinement
endorses. Falls back to Pillow for PNG writes if the GDAL PNG driver refuses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window


def open_any(path) -> rasterio.DatasetReader:
    return rasterio.open(str(path))


def read_window(
    ds: rasterio.DatasetReader,
    xoff: int,
    yoff: int,
    width: int,
    height: int,
    band_index: int = 1,
    fill: float = 0,
) -> np.ndarray:
    """Windowed read that NEVER goes out of bounds.

    Windows extending past the scene edge are padded with ``fill`` — this keeps
    compact crops valid on scenes smaller than the tile size and jittered crops
    near borders (the archi.md script produced negative offsets / crashes here).
    """
    full_h, full_w = ds.height, ds.width
    out_dtype = ds.dtypes[band_index - 1]
    out = np.full((height, width), fill, dtype=np.dtype(out_dtype))

    src_x0, src_y0 = max(xoff, 0), max(yoff, 0)
    src_x1, src_y1 = min(xoff + width, full_w), min(yoff + height, full_h)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return out

    win = Window(src_x0, src_y0, src_x1 - src_x0, src_y1 - src_y0)
    data = ds.read(band_index, window=win)
    dst_x0 = src_x0 - xoff
    dst_y0 = src_y0 - yoff
    out[dst_y0:dst_y0 + data.shape[0], dst_x0:dst_x0 + data.shape[1]] = data
    return out


def read_full(ds: rasterio.DatasetReader, band_index: int = 1) -> np.ndarray:
    return ds.read(band_index)


def write_png(path, array: np.ndarray) -> None:
    """Write a single-band PNG: uint16 (calibrated) or uint8 (mask/native).

    Primary path: GDAL PNG driver via rasterio. Fallback: Pillow.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.ndim != 2:
        raise ValueError(f"write_png expects 2-D array, got {array.shape}")
    if array.dtype == np.uint16:
        dtype = "uint16"
    elif array.dtype == np.uint8:
        dtype = "uint8"
    else:
        raise ValueError(f"unsupported PNG dtype {array.dtype} for {path}")

    try:
        with rasterio.open(
            str(path), "w", driver="PNG",
            height=array.shape[0], width=array.shape[1],
            count=1, dtype=dtype,
        ) as dst:
            dst.write(array, 1)
        # GDAL PNG writes a sidecar .aux.xml for non-georeferenced data in some
        # builds — harmless, but keep the tree clean for validation.
        aux = Path(str(path) + ".aux.xml")
        if aux.exists():
            aux.unlink()
        return
    except Exception:
        from PIL import Image
        mode = "I;16" if dtype == "uint16" else "L"
        if dtype == "uint16":
            Image.fromarray(array, mode="I;16").save(str(path))
        else:
            Image.fromarray(array, mode="L").save(str(path))


def probe_profile(path) -> dict:
    """Cheap header probe: shape, dtype, crs, transform, count."""
    with rasterio.open(str(path)) as ds:
        return {
            "width": ds.width,
            "height": ds.height,
            "count": ds.count,
            "dtype": ds.dtypes[0],
            "crs": ds.crs,
            "transform": ds.transform,
            "has_transform": ds.transform is not None and not ds.transform.is_identity,
            "tags": ds.tags(),
        }


def has_real_georeference(ds: rasterio.DatasetReader) -> bool:
    """True only when BOTH a plausible transform and a CRS exist.

    The archi.md script only checked ``gt[0]==0 and gt[3]==0`` — a dataset can
    have origin (0,0) with a valid CRS, or an identity transform with no CRS
    (which it would have treated as real). Requiring crs + non-identity
    transform removes both failure modes.
    """
    if ds.crs is None:
        return False
    t = ds.transform
    if t is None or t.is_identity:
        return False
    return True
