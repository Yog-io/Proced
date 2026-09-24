"""Open-and-verify every discovered raster independently (architecture §8)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .discover_tree import DiscoveredFile, TreeDiscovery


@dataclass
class IntegrityResult:
    opened: int = 0
    failed: List[dict] = field(default_factory=list)
    snapshots: List[dict] = field(default_factory=list)  # dtype/crs/domain per file
    resolution_outliers: List[dict] = field(default_factory=list)
    expected_size: tuple = (2000, 2000)  # ~2000×2000 Zenodo nominal
    size_tol_frac: float = 0.35  # flag if w or h outside ±35% of expected


def _coarse_domain(sample) -> str:
    """Independent value-domain guess (mirrors architecture §8 dtype snapshot)."""
    import numpy as np

    a = np.asarray(sample, dtype=np.float64)
    if a.size == 0:
        return "empty"
    if a.dtype == "uint8" or (a.max() <= 255 and a.min() >= 0 and np.issubdtype(np.asarray(sample).dtype, np.integer)):
        return "native"
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return "non-finite"
    if finite.min() < -1.0 and finite.max() <= 1.0:
        # float with negatives → likely dB
        return "db"
    p99 = float(np.percentile(np.abs(finite), 99))
    if p99 <= 1.5:
        return "power"
    return "amplitude"


def check_integrity(
    discovery: TreeDiscovery,
    *,
    sample_values: bool = True,
    max_sample_values: int = 50,
) -> IntegrityResult:
    import rasterio

    res = IntegrityResult()
    # Always open header for every file; optionally sample a center window for domain.
    for i, item in enumerate(discovery.rasters):
        try:
            with rasterio.open(str(item.path)) as ds:
                width, height = ds.width, ds.height
                count = ds.count
                dtypes = list(ds.dtypes)
                crs = str(ds.crs) if ds.crs else None
                res.opened += 1

                domain = None
                if sample_values and i < max_sample_values and count >= 1:
                    try:
                        h = min(64, height)
                        w = min(64, width)
                        x0 = max(0, (width - w) // 2)
                        y0 = max(0, (height - h) // 2)
                        arr = ds.read(1, window=((y0, y0 + h), (x0, x0 + w)))
                        domain = _coarse_domain(arr)
                    except Exception:
                        domain = None

                res.snapshots.append({
                    "rel": item.rel,
                    "width": width,
                    "height": height,
                    "count": count,
                    "dtypes": dtypes,
                    "crs": crs,
                    "domain_guess": domain,
                    "is_mask_by_name": item.is_mask_by_name,
                })

                ew, eh = res.expected_size
                if width and height:
                    if (abs(width - ew) / ew > res.size_tol_frac
                            or abs(height - eh) / eh > res.size_tol_frac):
                        # Masks from other sources / small exports can be smaller —
                        # only flag non-mask images as resolution outliers.
                        if not item.is_mask_by_name:
                            res.resolution_outliers.append({
                                "rel": item.rel,
                                "width": width,
                                "height": height,
                                "expected": [ew, eh],
                            })
        except Exception as exc:
            res.failed.append({
                "rel": item.rel,
                "error": f"{type(exc).__name__}: {exc}",
            })
    return res
