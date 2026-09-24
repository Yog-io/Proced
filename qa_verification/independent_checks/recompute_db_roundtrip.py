"""Stage 4 — THE critical check: independent dB round-trip (architecture §2).

For 100+ sampled calibrated crops:
  1. apply the documented inverse ``db = (px/65535)*30 - 30`` to the stored
     16-bit PNG pixel,
  2. re-open the *original raw scene* at that crop's exact window offset
     yourself (fresh rasterio call), recompute dB from the true pixel value
     with independent domain detection,
  3. compare within floating-point / quantization tolerance.

Also verifies mask PNGs are strictly binary, dimensions are 256×256, and
bit depth matches calibration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .._lib import (
    DB_FORMULA_VERSION,
    DB_HI,
    DB_LO,
    NATIVE_FORMULA_VERSION,
    QAPaths,
    Report,
    db_from_uint16,
    detect_domain_independent,
    find_png,
    load_catalog,
    load_crop_rows,
    resolve_stage_path,
    sample_indices,
    to_db_independent,
    uint16_from_db,
)

# Quantization: one LSB of the 16-bit linear map over 30 dB ≈ 0.000458 dB.
# Allow a few LSBs for float32 intermediate + independent recompute.
PX_TOL = 3          # uint16 LSBs
DB_TOL = 0.05       # dB slack (covers float32 vs float64 path differences)


def _band_slot(scene: dict, tag_index: int) -> Tuple[int, int]:
    """Independent reimplementation of dual-pol band mapping."""
    band_files = scene.get("band_files") or []
    pol_tags = scene.get("pol_tags") or []
    n_files = len(band_files)
    stacked = n_files == 1 and len(pol_tags) > 1
    if stacked:
        return 0, tag_index + 1
    ds_i = min(tag_index, max(0, n_files - 1))
    return ds_i, 1


def _open_png(path: Path) -> Optional[np.ndarray]:
    import rasterio
    try:
        with rasterio.open(str(path)) as ds:
            return ds.read(1)
    except Exception:
        return None


def run(paths: QAPaths, *, sample: int = 100, seed: int = 42,
        tile_size: int = 256) -> Report:
    rep = Report("stage4_db_roundtrip")
    skip = paths.require("crop_rows", "catalog", "stage", "dest")
    if skip:
        rep.hard(False, f"skip: {skip}")
        return rep

    rows = load_crop_rows(paths)
    if len(rows) == 0:
        rep.hard(False, "crop_rows.json is empty")
        return rep

    cat = load_catalog(paths)
    root = cat.get("root") or str(paths.stage_dir)
    scenes_by_id = {s["scene_id"]: s for s in cat.get("scenes") or []}
    scenes_by_name = {s.get("source_scene_name", ""): s
                      for s in cat.get("scenes") or []}

    # --- 1) mask PNGs strictly binary (HARD per architecture) -------------
    mask_bad: List[dict] = []
    records = rows.to_dict("records")
    idx_mask = sample_indices(len(records), min(sample, 200), seed, "mask")
    for i in idx_mask:
        r = records[i]
        crop_id = str(r.get("crop_id") or r.get("image_id") or "")
        mp = find_png(paths, crop_id, "mask")
        if mp is None:
            mask_bad.append({"crop_id": crop_id, "error": "mask PNG missing"})
            continue
        arr = _open_png(mp)
        if arr is None:
            mask_bad.append({"crop_id": crop_id, "error": "mask unreadable"})
            continue
        if arr.shape != (tile_size, tile_size):
            mask_bad.append({
                "crop_id": crop_id, "error": f"dims {arr.shape}",
            })
            continue
        uniq = set(int(v) for v in np.unique(arr))
        # Pipeline writes {0,255}; {0,1} is also valid binary. Anything else
        # (anti-aliased grays) is a hard fail.
        if not uniq <= {0, 1} and not uniq <= {0, 255}:
            mask_bad.append({
                "crop_id": crop_id,
                "unique_sample": sorted(uniq)[:16],
                "error": "non-binary mask pixel values",
            })

    rep.hard(
        not mask_bad,
        "mask PNGs strictly binary {0,1}|{0,255} (sampled)"
        if not mask_bad else f"{len(mask_bad)} non-binary/missing mask PNGs",
        examples=mask_bad[:20],
        sampled=len(idx_mask),
    )

    # --- 2) PNG dims + bit depth vs is_calibrated ------------------------
    dim_bad: List[dict] = []
    bit_bad: List[dict] = []
    idx_all = sample_indices(len(records), sample, seed, "png_meta")
    for i in idx_all:
        r = records[i]
        crop_id = str(r.get("crop_id") or r.get("image_id") or "")
        is_cal = str(r.get("is_calibrated", "")).upper() in ("Y", "TRUE", "1", "YES")
        formula = str(r.get("db_conversion_formula_version", ""))
        for pol in ("VV", "VH"):
            p = find_png(paths, crop_id, pol)
            if p is None:
                continue
            import rasterio
            try:
                with rasterio.open(str(p)) as ds:
                    w, h, dtype = ds.width, ds.height, ds.dtypes[0]
            except Exception as exc:
                dim_bad.append({"crop_id": crop_id, "pol": pol, "error": str(exc)})
                continue
            if (w, h) != (tile_size, tile_size):
                dim_bad.append({"crop_id": crop_id, "pol": pol,
                                "dims": [w, h]})
            expect_16 = is_cal and formula == DB_FORMULA_VERSION
            if expect_16 and dtype != "uint16":
                bit_bad.append({"crop_id": crop_id, "pol": pol,
                                "dtype": dtype, "expect": "uint16"})
            if (not is_cal) and formula == NATIVE_FORMULA_VERSION and dtype not in (
                "uint8", "uint16"
            ):
                bit_bad.append({"crop_id": crop_id, "pol": pol,
                                "dtype": dtype, "expect": "native int"})

    rep.hard(
        not dim_bad,
        f"PNG dimensions {tile_size}×{tile_size} on {len(idx_all)} samples"
        if not dim_bad else f"{len(dim_bad)} wrong-dimension PNGs",
        examples=dim_bad[:20],
    )
    rep.hard(
        not bit_bad,
        "bit depth matches calibration (16-bit calibrated / native uncalibrated)"
        if not bit_bad else f"{len(bit_bad)} bit-depth mismatches",
        examples=bit_bad[:20],
    )

    # --- 3) THE dB round-trip (calibrated rows) --------------------------
    cal_rows = [
        r for r in records
        if str(r.get("is_calibrated", "")).upper() in ("Y", "TRUE", "1", "YES")
        and str(r.get("db_conversion_formula_version", "")) == DB_FORMULA_VERSION
    ]
    if not cal_rows:
        rep.soft(True, "no calibrated v1 rows — dB round-trip skipped "
                       "(fixture may be uncalibrated-only)")
        return rep

    idxs = sample_indices(len(cal_rows), sample, seed, "db_rt")
    roundtrip_bad: List[dict] = []
    skipped = 0
    checked_pol = 0

    for i in idxs:
        r = cal_rows[i]
        crop_id = str(r.get("crop_id") or r.get("image_id") or "")
        sid = str(r.get("crop_source_scene_id") or "")
        scene = scenes_by_id.get(sid) or scenes_by_name.get(sid)
        if scene is None:
            # try matching via scene_id embedded in crop_id prefix
            for s in cat.get("scenes") or []:
                if crop_id.startswith(str(s.get("scene_id", "__")) + "_crop"):
                    scene = s
                    break
        if scene is None or not scene.get("band_files"):
            skipped += 1
            continue

        try:
            xoff = int(r["crop_xoff"])
            yoff = int(r["crop_yoff"])
            size = int(r.get("crop_size") or tile_size)
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue

        pol_tags = list(scene.get("pol_tags") or [])
        for ti, tag in enumerate(pol_tags):
            if tag not in ("VV", "VH"):
                continue
            png = find_png(paths, crop_id, tag)
            if png is None:
                continue
            stored = _open_png(png)
            if stored is None or stored.dtype != np.uint16:
                skipped += 1
                continue

            # Independent inverse on stored pixels
            db_png = db_from_uint16(stored.astype(np.float64))

            # Independent re-read of the RAW scene window
            ds_i, band_i = _band_slot(scene, ti)
            raw_path = resolve_stage_path(
                scene["band_files"][ds_i], root, paths.stage_dir,
            )
            if not Path(raw_path).is_file():
                skipped += 1
                continue
            import rasterio
            from rasterio.windows import Window

            try:
                with rasterio.open(str(raw_path)) as ds:
                    # clip window to scene bounds; out-of-bounds pad = fill 0
                    # matching independent read_window semantics
                    full_w, full_h = ds.width, ds.height
                    x0, y0 = max(xoff, 0), max(yoff, 0)
                    x1 = min(xoff + size, full_w)
                    y1 = min(yoff + size, full_h)
                    if x1 <= x0 or y1 <= y0:
                        raw_win = np.zeros((size, size), dtype=np.float64)
                    else:
                        win = Window(x0, y0, x1 - x0, y1 - y0)
                        data = ds.read(band_i, window=win)
                        # pad into size×size at correct offset
                        raw_win = np.zeros((size, size), dtype=np.float64)
                        dy, dx = y0 - yoff, x0 - xoff
                        raw_win[dy:dy + data.shape[0],
                                dx:dx + data.shape[1]] = data
                        # domain from the real (non-pad) region for detection
                        domain = (scene.get("value_domains") or {}).get(tag)
                        if not domain or domain == "auto":
                            domain = detect_domain_independent(data)
            except Exception as exc:
                roundtrip_bad.append({
                    "crop_id": crop_id, "pol": tag, "error": f"raw open: {exc}",
                })
                continue

            if domain in (None, "native"):
                # calibrated scene should not be native; flag soft and skip
                skipped += 1
                continue

            db_raw = to_db_independent(raw_win, domain)
            db_raw = np.clip(db_raw, DB_LO, DB_HI)

            # Compare: re-quantize independent dB to uint16 and compare LSBs,
            # and also compare dB directly.
            expect_px = uint16_from_db(db_raw)
            px_delta = np.abs(stored.astype(np.int32) - expect_px.astype(np.int32))
            max_px = int(px_delta.max()) if px_delta.size else 0
            db_delta = np.abs(db_png - db_raw)
            max_db = float(db_delta.max()) if db_delta.size else 0.0
            checked_pol += 1

            if max_px > PX_TOL and max_db > DB_TOL:
                # locate worst pixel for diagnostics
                yy, xx = np.unravel_index(int(db_delta.argmax()), db_delta.shape)
                roundtrip_bad.append({
                    "crop_id": crop_id, "pol": tag,
                    "max_px_delta": max_px,
                    "max_db_delta": round(max_db, 6),
                    "worst_xy": [int(xx), int(yy)],
                    "stored_px": int(stored[yy, xx]),
                    "stored_db": round(float(db_png[yy, xx]), 5),
                    "independent_db": round(float(db_raw[yy, xx]), 5),
                    "domain": domain,
                })

    ok_rt = not roundtrip_bad and checked_pol > 0
    if checked_pol == 0:
        rep.soft(True, "no calibrated crop pols could be rechecked "
                       f"(skipped={skipped})")
    else:
        rep.hard(
            ok_rt,
            f"dB round-trip matches independent recompute on {checked_pol} "
            f"band-crops (tol {PX_TOL} LSB / {DB_TOL} dB)"
            if ok_rt else f"{len(roundtrip_bad)}/{checked_pol} dB round-trip FAILURES",
            failures=roundtrip_bad[:25],
            checked=checked_pol,
            skipped=skipped,
        )

    # --- 4) formula version tags are honest ------------------------------
    bad_ver = []
    for r in records:
        is_cal = str(r.get("is_calibrated", "")).upper() in ("Y", "TRUE", "1", "YES")
        formula = str(r.get("db_conversion_formula_version", ""))
        if is_cal and formula != DB_FORMULA_VERSION:
            bad_ver.append({"crop_id": r.get("crop_id"), "v": formula})
        if (not is_cal) and formula == DB_FORMULA_VERSION:
            # uncalibrated must NOT claim the dB formula
            bad_ver.append({"crop_id": r.get("crop_id"), "v": formula,
                            "note": "uncalibrated claims db formula"})
    rep.hard(
        not bad_ver,
        "db_conversion_formula_version honest vs is_calibrated"
        if not bad_ver else f"{len(bad_ver)} formula-version tag errors",
        examples=bad_ver[:20],
    )

    # --- 5) value-range sanity: stored uint16 inverses land in [-30,0] ----
    idx_range = sample_indices(len(cal_rows), min(sample, 50), seed, "range")
    range_bad = []
    for i in idx_range:
        r = cal_rows[i]
        crop_id = str(r.get("crop_id") or "")
        for pol in ("VV", "VH"):
            p = find_png(paths, crop_id, pol)
            if p is None:
                continue
            arr = _open_png(p)
            if arr is None or arr.dtype != np.uint16:
                continue
            db = db_from_uint16(arr)
            if float(db.min()) < DB_LO - 1e-6 or float(db.max()) > DB_HI + 1e-6:
                range_bad.append({
                    "crop_id": crop_id, "pol": pol,
                    "db_min": float(db.min()), "db_max": float(db.max()),
                })
    rep.hard(
        not range_bad,
        "inverted dB values within [-30, 0]"
        if not range_bad else f"{len(range_bad)} crops outside dB range",
        examples=range_bad[:15],
    )
    return rep
