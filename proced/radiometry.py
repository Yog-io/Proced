"""Radiometric calibration: linear/dB domains → clipped [-30, 0] dB → uint16.

Fixes vs the embedded archi.md script:
  * correct dB formulas: power (σ⁰) uses 10·log10, AMPLITUDE uses 20·log10
    (the original applied 10·log10 to both);
  * robust value-domain detection: the original used ``max(arr) > 1.0`` which
    misclassified linear σ⁰ power (almost always ≤ 1) as already-dB;
  * uncalibrated (Kaggle-style uint8) data is passed through untouched per
    refinement C.2 — never dB-converted;
  * formula version tagged honestly per row.

Inverse for Block 2 (calibrated rows):
    db_value = (pixel_uint16 / 65535) * 30 - 30
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import DB_FORMULA_VERSION, NATIVE_FORMULA_VERSION

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    _HAS_TORCH = False


def gpu_available() -> bool:
    if not _HAS_TORCH:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover
        return False


def detect_value_domain(arr: np.ndarray, hint: str = "auto") -> str:
    """Classify array as 'db' | 'power' | 'amplitude' | 'native'.

    * hint != 'auto' → honoured verbatim (except native detection below).
    * integer dtypes  → 'native' (visual export, no dB conversion; C.2).
    * any negative value → already dB.
    * float, p99 ≤ 1.5 → linear power σ⁰ (ocean backscatter lives in 1e-3..1).
    * otherwise → linear amplitude.
    """
    if hint and hint != "auto":
        return hint
    if np.issubdtype(arr.dtype, np.integer):
        return "native"
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return "native"
    if float(np.min(finite)) < -0.5:
        return "db"
    p99 = float(np.percentile(finite, 99))
    return "power" if p99 <= 1.5 else "amplitude"


def to_db(arr: np.ndarray, domain: str, eps: float = 1e-7) -> np.ndarray:
    """Convert a band window to decibels (float32)."""
    a = arr.astype(np.float32, copy=False)
    if domain == "db":
        return a
    if domain == "power":
        return 10.0 * np.log10(np.clip(a, eps, None))
    if domain == "amplitude":
        return 20.0 * np.log10(np.clip(a, eps, None))
    raise ValueError(f"unknown value domain: {domain}")


def db_to_uint16(
    db: np.ndarray,
    lo: float = -30.0,
    hi: float = 0.0,
    use_gpu: bool = True,
) -> np.ndarray:
    """clip(db, lo, hi) → round(((db - lo) / (hi - lo)) * 65535) → uint16.

    Vectorised NumPy by default; CUDA tensors when available (archi §2).
    """
    if use_gpu and gpu_available():  # pragma: no cover - no CUDA in CI
        t = torch.as_tensor(np.ascontiguousarray(db), dtype=torch.float32, device="cuda")
        t = torch.clamp(t, lo, hi)
        scaled = torch.round(((t - lo) / (hi - lo)) * 65535.0)
        return scaled.to("cpu").numpy().astype(np.uint16)
    a = np.asarray(db, dtype=np.float32)
    a = np.clip(a, lo, hi)
    scaled = np.round(((a - lo) / (hi - lo)) * 65535.0)
    return np.clip(scaled, 0, 65535).astype(np.uint16)


def uint16_to_db(pix: np.ndarray, lo: float = -30.0, hi: float = 0.0) -> np.ndarray:
    """Exact inverse of the v1 mapping — Block 2 must use this."""
    p = np.asarray(pix, dtype=np.float32)
    return (p / 65535.0) * (hi - lo) + lo


def calibrate_window(
    band: np.ndarray,
    value_domain: str,
    lo: float = -30.0,
    hi: float = 0.0,
    eps: float = 1e-7,
    use_gpu: bool = True,
) -> Tuple[np.ndarray, str]:
    """Band window → (output_array, formula_version).

    * 'native' → returned unchanged (uint8/uint16 passthrough, C.2).
    * 'db' | 'power' | 'amplitude' → dB → clipped → uint16 (v1 formula).
    """
    if value_domain == "native":
        if np.issubdtype(band.dtype, np.floating):  # defensive: force integer out
            return np.clip(np.round(band), 0, 65535).astype(np.uint16), NATIVE_FORMULA_VERSION
        if band.dtype == np.uint8:
            return band, NATIVE_FORMULA_VERSION
        if band.dtype == np.uint16:
            return band, NATIVE_FORMULA_VERSION
        return np.clip(band, 0, 65535).astype(np.uint16), NATIVE_FORMULA_VERSION
    db = to_db(band, value_domain, eps=eps)
    return db_to_uint16(db, lo=lo, hi=hi, use_gpu=use_gpu), DB_FORMULA_VERSION


def calibrate_full_scene_db(
    band: np.ndarray,
    value_domain: str,
    eps: float = 1e-7,
) -> Optional[np.ndarray]:
    """Full-scene dB array for radiometric feature extraction (Stage 2/C.0).

    Returns None for native (uncalibrated) data — radiometric features that
    depend on calibrated σ⁰ are not meaningful there.
    """
    if value_domain == "native":
        return None
    return to_db(band, value_domain, eps=eps)
