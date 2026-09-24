"""Shared per-scene raster I/O used by the features / plan / convert stages."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from .catalog import SceneRecord
from .config import PipelineConfig
from .radiometry import detect_value_domain
from .raster_io import has_real_georeference, open_any


def load_scene_geometry(scene: SceneRecord):
    """Open every band file; return (datasets aligned to band_files, meta dict)."""
    datasets = []
    meta = None
    for path in scene.band_files:
        ds = open_any(path)
        datasets.append(ds)
        if meta is None:
            meta = {
                "width": ds.width,
                "height": ds.height,
                "transform": ds.transform,
                "crs": ds.crs,
                "has_geo": has_real_georeference(ds),
                "dtypes": [ds.dtypes[i] for i in range(ds.count)],
            }
    return datasets, meta


def band_slot(scene: SceneRecord, tag_index: int) -> Tuple[int, int]:
    """Map a pol_tags index → (dataset_index, band_index) for windowed reads.

    * split files (one file per pol): dataset i, band 1
    * stacked single file with multiple pol tags: dataset 0, band i+1
    """
    n_files = len(scene.band_files)
    stacked = n_files == 1 and len(scene.pol_tags) > 1
    if stacked:
        return 0, tag_index + 1
    ds_i = min(tag_index, max(0, n_files - 1))
    return ds_i, 1


def resolve_value_domains(scene: SceneRecord, cfg: PipelineConfig) -> Dict[str, str]:
    """Resolve each VV/VH tag's value domain (honours cfg.value_domain override)."""
    domains: Dict[str, str] = {}
    for i, tag in enumerate(scene.pol_tags):
        if tag not in ("VV", "VH"):
            continue
        if not scene.is_calibrated:
            domains[tag] = "native"
            continue
        if cfg.value_domain and cfg.value_domain != "auto":
            domains[tag] = cfg.value_domain
            continue
        ds_i, band_i = band_slot(scene, i)
        path = scene.band_files[ds_i]
        with open_any(path) as ds:
            h, w = min(256, ds.height), min(256, ds.width)
            x0 = max(0, (ds.width - w) // 2)
            y0 = max(0, (ds.height - h) // 2)
            sample = ds.read(band_i, window=((y0, y0 + h), (x0, x0 + w)))
        domains[tag] = detect_value_domain(sample, "auto")
    return domains


def read_mask(scene: SceneRecord, width: int, height: int) -> Tuple[Optional[np.ndarray], bool]:
    """Read the paired mask; size mismatch ⇒ (None, False) — never silently resize."""
    if scene.mask_path is None:
        return None, False
    try:
        with open_any(scene.mask_path) as mds:
            if mds.width == width and mds.height == height:
                arr = mds.read(1)
            else:
                return None, False
        if arr.dtype != np.uint8:
            arr = (arr > 0).astype(np.uint8) * 255
        return arr, True
    except Exception:
        return None, False


def scene_timestamp_from_tags(ds) -> Optional[str]:
    try:
        tags = ds.tags()
    except Exception:
        return None
    for key in ("TIFFTAG_DATETIME", "TIFFTAG_TIMESTAMP", "TIME", "ACQUISITION_TIME"):
        raw = tags.get(key)
        if not raw:
            continue
        raw = str(raw).strip()
        try:
            if ":" in raw[:4] and raw[4] == ":":
                dt = datetime.strptime(raw[:19], "%Y:%m:%d %H:%M:%S")
            else:
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            continue
    return None


def dataset_has_timestamp_tag(ds) -> bool:
    try:
        tags = ds.tags()
    except Exception:
        return False
    for key in ("TIFFTAG_DATETIME", "TIFFTAG_TIMESTAMP", "TIME", "ACQUISITION_TIME"):
        if tags.get(key):
            return True
    return False
