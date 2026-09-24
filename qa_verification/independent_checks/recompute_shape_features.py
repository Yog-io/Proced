"""Stage 2 — recompute full-extent shape features from raw masks (C.0).

Fresh cv2 calls only; compares against instances.csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .._lib import (
    QAPaths,
    Report,
    connected_components,
    load_catalog,
    load_instances,
    resolve_stage_path,
    sample_indices,
    shape_from_mask_region,
    touches_border,
)


def _load_mask(scene: dict, catalog_root: str, stage: Path) -> Optional[np.ndarray]:
    mp = scene.get("mask_path")
    if not mp:
        return None
    path = resolve_stage_path(mp, catalog_root, stage)
    if not path.is_file():
        return None
    import rasterio
    with rasterio.open(str(path)) as ds:
        if ds.width != int(scene["width"]) or ds.height != int(scene["height"]):
            return None
        arr = ds.read(1)
    if arr.dtype != np.uint8:
        arr = (arr > 0).astype(np.uint8)
    return arr


def _recompute_instances(mask: np.ndarray, lookalike_mv: int = 2,
                         default_label: str = "oil", min_area: int = 20) -> List[dict]:
    """Mirror the *algorithm* independently: per class-value components."""
    out = []
    class_values = [int(v) for v in np.unique(mask) if v > 0]
    idx = 0
    for mv in sorted(class_values):
        label = "lookalike" if mv == lookalike_mv else default_label
        binary = (mask == mv).astype(np.uint8)
        n, labels = connected_components(binary)
        for i in range(1, n + 1):
            comp = labels == i
            area = int(np.count_nonzero(comp))
            if area < min_area:
                continue
            metrics = shape_from_mask_region(comp)
            if metrics is None:
                continue
            out.append({
                "instance_index": idx,
                "label": label,
                "mask_value": mv,
                "area_px": metrics["area_px"],
                "perimeter": metrics["perimeter"],
                "aspect_ratio": metrics["aspect_ratio"],
                "boundary_complexity": metrics["boundary_complexity"],
                "bbox": metrics["bbox"],
                "truncated_by_scene_edge": touches_border(comp),
                "fragment_count": n,  # per-class; dataset-level recomputed below
            })
            idx += 1
    # fragment_count in pipeline is over full mask>0, fix below in caller
    return out


def run(paths: QAPaths, *, sample: int = 50, seed: int = 42) -> Report:
    rep = Report("stage2_features")
    skip = paths.require("catalog", "instances", "stage")
    if skip:
        rep.hard(False, f"skip: {skip}")
        return rep

    cat = load_catalog(paths)
    inst = load_instances(paths)
    root = cat.get("root") or str(paths.stage_dir)
    scenes = {s["scene_id"]: s for s in cat.get("scenes") or []}

    if len(inst) == 0:
        rep.soft(True, "no instance rows — nothing to recompute")
        return rep

    # --- aspect_ratio ≥ 1.0 for 100% (bug #14) ----------------------------
    ar = pd.to_numeric(inst["aspect_ratio"], errors="coerce")
    bad_ar = inst[ar < 1.0 - 1e-9]
    rep.hard(
        len(bad_ar) == 0,
        "aspect_ratio ≥ 1.0 on all rows" if len(bad_ar) == 0
        else f"{len(bad_ar)} rows with aspect_ratio < 1",
        examples=bad_ar[["scene_id", "instance_index", "aspect_ratio"]].head(5).to_dict("records")
        if len(bad_ar) else [],
    )

    # --- truncation exhaustive (every instance) ---------------------------
    # Group instances by scene; recompute mask once per involved scene.
    trunc_mismatch = []
    shape_mismatch = []
    scenes_touched = inst["scene_id"].unique().tolist()
    masks: Dict[str, np.ndarray] = {}
    recomputed_by_scene: Dict[str, List[dict]] = {}

    for sid in scenes_touched:
        scene = scenes.get(sid)
        if not scene:
            trunc_mismatch.append({"scene_id": sid, "error": "not in catalog"})
            continue
        mask = _load_mask(scene, root, paths.stage_dir)
        if mask is None:
            trunc_mismatch.append({"scene_id": sid, "error": "mask unreadable"})
            continue
        masks[sid] = mask
        # fragment count over full mask
        _n, labels_all = connected_components((mask > 0).astype(np.uint8))
        default_label = scene.get("default_label", "oil")
        recomputed = _recompute_instances(mask, default_label=default_label)
        for r in recomputed:
            r["fragment_count"] = int(_n)
        recomputed_by_scene[sid] = recomputed

    # exhaustive truncation: any foreground on full-scene border for that instance
    for _, row in inst.iterrows():
        sid = row["scene_id"]
        mask = masks.get(sid)
        if mask is None:
            continue
        # rebuild instance mask via mask_value + bbox neighborhood
        mv = int(row.get("mask_value") or 1)
        bx, by, bw, bh = (int(row["bbox_x"]), int(row["bbox_y"]),
                          int(row["bbox_w"]), int(row["bbox_h"]))
        # full-scene truncated: does ANY pixel of this class component touch border?
        # Independent check: if the instance bbox touches border AND that border
        # row/col has this class value inside bbox span — approximate via full
        # component containing centroid.
        cx, cy = int(row["centroid_x"]), int(row["centroid_y"])
        cls = (mask == mv)
        # flood from centroid within class
        n, labels = connected_components(cls)
        if not (0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1]):
            continue
        lab = labels[cy, cx]
        if lab == 0:
            # centroid may sit off component if centroid outside — use bbox votes
            region = np.zeros_like(cls)
            region[by:by + bh, bx:bx + bw] = cls[by:by + bh, bx:bx + bw]
            expect_trunc = touches_border(region) if region.any() else False
        else:
            comp = labels == lab
            expect_trunc = touches_border(comp)
        got = bool(str(row.get("truncated_by_scene_edge")).lower() in ("true", "1", "y", "yes")
                   or row.get("truncated_by_scene_edge") is True)
        if got != expect_trunc:
            # border property of FULL component: if instance is the whole component
            # at scene edge, mismatch is hard; if bbox-only approx differs, soft-check
            # by seeing full-component border
            trunc_mismatch.append({
                "scene_id": sid,
                "instance_index": int(row["instance_index"]),
                "stored": got, "recomputed": expect_trunc,
            })

    rep.hard(
        not trunc_mismatch,
        "truncated_by_scene_edge matches full-scene border recompute (all rows)"
        if not trunc_mismatch else f"{len(trunc_mismatch)} truncation mismatches",
        mismatches=trunc_mismatch[:20],
        n_checked=int(len(inst)),
    )

    # --- sampled shape metrics -------------------------------------------
    idxs = sample_indices(len(inst), sample, seed, "shape")
    for i in idxs:
        row = inst.iloc[i]
        sid = row["scene_id"]
        recs = recomputed_by_scene.get(sid) or []
        # match by instance_index
        match = next((r for r in recs if r["instance_index"] == int(row["instance_index"])), None)
        if match is None and recs:
            # match by area within tolerance
            area = int(row["area_px"])
            match = min(recs, key=lambda r: abs(r["area_px"] - area))
        if match is None:
            shape_mismatch.append({"i": i, "error": "no recompute match"})
            continue
        for key, tol in (
            ("area_px", 1),
            ("perimeter", 2.0),
            ("aspect_ratio", 0.15),
            ("boundary_complexity", 2.0),
        ):
            stored = float(row[key])
            recomputed = float(match[key])
            if abs(stored - recomputed) > tol:
                shape_mismatch.append({
                    "scene_id": sid,
                    "instance_index": int(row["instance_index"]),
                    "field": key, "stored": stored, "recomputed": recomputed,
                })
                break

    rep.hard(
        not shape_mismatch,
        f"sampled {len(idxs)} instances: shape metrics within tolerance"
        if not shape_mismatch else f"{len(shape_mismatch)} shape mismatches",
        mismatches=shape_mismatch[:20],
        sampled=len(idxs),
    )

    # --- radiometrics: null for uncalibrated (not zero-computed garbage) ---
    # instances.csv doesn't carry is_calibrated — join via catalog
    uncal_ids = {
        s["scene_id"] for s in cat.get("scenes") or [] if not s.get("is_calibrated")
    }
    uncal = inst[inst["scene_id"].isin(uncal_ids)]
    rad_cols = ["damping_ratio", "damping_ratio_db", "boundary_gradient_steepness",
                "backscatter_variance_ratio", "glcm_contrast", "glcm_homogeneity", "ndpi"]
    bad_null = []
    for _, row in uncal.iterrows():
        for c in rad_cols:
            v = row.get(c, "")
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            s = str(v).strip()
            if s == "" or s.lower() == "nan":
                continue
            # non-null value on uncalibrated row
            if s not in ("0", "0.0"):
                bad_null.append({"scene_id": row["scene_id"], "col": c, "value": s})
            else:
                # zero is also suspicious — pipeline should leave empty
                bad_null.append({"scene_id": row["scene_id"], "col": c, "value": s,
                                 "note": "zero_on_uncalibrated"})
    # hard only if non-empty garbage; zeros → soft
    hard_bad = [b for b in bad_null if b.get("note") != "zero_on_uncalibrated"]
    soft_bad = [b for b in bad_null if b.get("note") == "zero_on_uncalibrated"]
    rep.hard(
        not hard_bad,
        "uncalibrated rows have empty radiometric fields"
        if not hard_bad else f"{len(hard_bad)} uncalibrated rows with computed radiometrics",
        examples=hard_bad[:15],
        n_uncalibrated=int(len(uncal)),
    )
    if soft_bad:
        rep.soft(False, f"{len(soft_bad)} uncalibrated radiometric cells are 0 (prefer empty)",
                 examples=soft_bad[:10])

    # --- damping ratio physical plausibility (calibrated oil) -------------
    cal_ids = {
        s["scene_id"] for s in cat.get("scenes") or [] if s.get("is_calibrated")
    }
    cal = inst[(inst["scene_id"].isin(cal_ids)) & (inst["label"] == "oil")]
    if len(cal):
        dr = pd.to_numeric(cal["damping_ratio_db"], errors="coerce").dropna()
        # ignore empty strings already dropped
        if len(dr):
            outside = dr[(dr < 3.0) | (dr > 20.0)]  # wide band; oil ideal 6–15
            frac = len(outside) / len(dr)
            rep.soft(
                frac < 0.25,
                f"damping_ratio_db plausibility: {frac:.1%} outside [3,20] dB "
                f"(ideal oil 6–15; flag-only)",
                n=int(len(dr)),
                median=float(dr.median()),
                n_outside=int(len(outside)),
            )
    return rep
