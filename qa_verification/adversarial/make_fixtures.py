"""Build deliberately malformed fixtures under malformed_input_fixtures/.

Called by run_all_verification --adversarial and by the adversarial tests.
Does NOT import proced/ for fixture construction (writes raw rasters only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

HERE = Path(__file__).resolve().parent
FIXTURE_ROOT = HERE / "malformed_input_fixtures"


def write_fixture_tree(root: Path) -> Dict[str, Path]:
    """Create one small tree per failure mode. Returns {case_name: dir}."""
    root = Path(root)
    rng = np.random.default_rng(0)
    cases: Dict[str, Path] = {}

    def _tiff(path: Path, arr, crs="EPSG:4326"):
        import rasterio
        from rasterio.transform import from_origin
        path.parent.mkdir(parents=True, exist_ok=True)
        kwargs = dict(
            driver="GTiff", height=arr.shape[0], width=arr.shape[1],
            count=1, dtype=arr.dtype,
        )
        if crs is not None:
            kwargs["crs"] = crs
            kwargs["transform"] = from_origin(60.0, 25.0, 0.0001, 0.0001)
        else:
            kwargs["transform"] = from_origin(0, 0, 1, 1)
        with rasterio.open(str(path), "w", **kwargs) as ds:
            ds.write(arr.astype(arr.dtype), 1)

    def _png(path: Path, arr):
        from PIL import Image
        path.parent.mkdir(parents=True, exist_ok=True)
        if arr.dtype == np.uint16:
            Image.fromarray(arr, mode="I;16").save(str(path))
        else:
            Image.fromarray(arr.astype(np.uint8), mode="L").save(str(path))

    # 1) corrupted TIFF (truncated garbage)
    d = root / "corrupted_tiff"
    bad = d / "scene_bad.tif"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"II*\x00" + b"\x00" * 32)  # TIFF magic + junk, no IFD
    _png(d / "scene_bad_mask.png", np.zeros((64, 64), np.uint8))
    cases["corrupted_tiff"] = d

    # 2) mask dimensions != image dimensions
    d = root / "mask_size_mismatch"
    _tiff(d / "scene.tif", rng.uniform(0.01, 0.3, (128, 128)).astype(np.float32))
    _png(d / "scene_mask.png", np.zeros((64, 64), np.uint8))
    cases["mask_size_mismatch"] = d

    # 3) all-zero mask (verified-clean path — must not invent positives)
    d = root / "all_zero_mask"
    _tiff(d / "scene.tif", rng.uniform(0.01, 0.3, (300, 300)).astype(np.float32))
    _png(d / "scene_mask.png", np.zeros((300, 300), np.uint8))
    cases["all_zero_mask"] = d

    # 4) mask touching all four scene borders at once
    d = root / "mask_all_borders"
    m = np.zeros((300, 300), np.uint8)
    m[0, :] = 1
    m[-1, :] = 1
    m[:, 0] = 1
    m[:, -1] = 1
    m[100:140, 100:140] = 1  # interior blob so there is a real instance
    _tiff(d / "scene.tif", rng.uniform(0.01, 0.3, (300, 300)).astype(np.float32))
    _png(d / "scene_mask.png", m)
    cases["mask_all_borders"] = d

    # 5) scene smaller than 256×256
    d = root / "tiny_scene"
    _tiff(d / "scene.tif", rng.uniform(0.01, 0.3, (64, 64)).astype(np.float32))
    m = np.zeros((64, 64), np.uint8)
    m[10:30, 10:40] = 1
    _png(d / "scene_mask.png", m)
    cases["tiny_scene"] = d

    return cases


def build_default_fixtures() -> Dict[str, Path]:
    return write_fixture_tree(FIXTURE_ROOT)


if __name__ == "__main__":
    built = build_default_fixtures()
    for name, p in built.items():
        print(f"{name}: {p}")
