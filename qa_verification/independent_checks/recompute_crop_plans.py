"""Stage 3b — independent crop-plan invariants (bugs #5, #6, neg ratio)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .._lib import (
    QAPaths,
    Report,
    load_catalog,
    load_crop_plans,
    load_instances,
    resolve_stage_path,
)


def _load_mask(scene: dict, root: str, stage: Path) -> Optional[np.ndarray]:
    mp = scene.get("mask_path")
    if not mp:
        return None
    path = resolve_stage_path(mp, root, stage)
    if not path.is_file():
        return None
    import rasterio
    with rasterio.open(str(path)) as ds:
        if ds.width != int(scene["width"]) or ds.height != int(scene["height"]):
            return None
        arr = ds.read(1)
    return (arr > 0).astype(np.uint8) if arr.dtype != np.uint8 else (arr > 0).astype(np.uint8)


def _configured(paths: QAPaths) -> Dict[str, float]:
    """Pull neg_buffer_px / neg_ratio from the run's recorded knobs.

    Sources (in order): state/run_config.json (written by every stage),
    dest/pipeline_summary.json (written by full `run`). Defaults only if
    neither exists.
    """
    out = {"neg_buffer_px": 300.0, "neg_ratio": 0.4}
    import json
    for cand in (
        paths.state_dir / "run_config.json",
        paths.dest_dir / "pipeline_summary.json",
    ):
        if not cand.is_file():
            continue
        try:
            payload = json.loads(cand.read_text()) or {}
            cfg = payload.get("config") if isinstance(payload, dict) else None
            if not isinstance(cfg, dict):
                # run_config.json is the config itself
                cfg = payload if isinstance(payload, dict) else {}
            if cfg.get("neg_buffer_px") is not None:
                out["neg_buffer_px"] = float(cfg["neg_buffer_px"])
            if cfg.get("neg_ratio") is not None:
                out["neg_ratio"] = float(cfg["neg_ratio"])
            if "neg_buffer_px" in cfg or "neg_ratio" in cfg:
                break
        except Exception:
            continue
    return out


def run(
    paths: QAPaths,
    *,
    neg_buffer_px: Optional[int] = None,
    neg_ratio: Optional[float] = None,
) -> Report:
    cfgk = _configured(paths)
    if neg_buffer_px is None:
        neg_buffer_px = int(cfgk["neg_buffer_px"])
    if neg_ratio is None:
        neg_ratio = float(cfgk["neg_ratio"])
    rep = Report("stage3b_plan")
    skip = paths.require("catalog", "crop_plans", "stage")
    if skip:
        rep.hard(False, f"skip: {skip}")
        return rep

    cat = load_catalog(paths)
    plans = load_crop_plans(paths)
    root = cat.get("root") or str(paths.stage_dir)
    scenes = {s["scene_id"]: s for s in cat.get("scenes") or []}

    # reload instances for compact vs overflow classification
    inst_path = paths.state_dir / "instances.csv"
    inst_by_scene: Dict[str, list] = {}
    if inst_path.is_file():
        inst = load_instances(paths)
        for _, row in inst.iterrows():
            inst_by_scene.setdefault(row["scene_id"], []).append(row)

    compact_bad = []
    overflow_bad = []
    empty_tile_bad = []
    masks: Dict[str, np.ndarray] = {}

    for sid, plist in plans.items():
        if not plist:
            continue
        scene = scenes.get(sid)
        if scene is None:
            continue

        # group plans by instance
        by_inst = defaultdict(list)
        negs = []
        for p in plist:
            if p.get("instance_index") is None:
                negs.append(p)
            else:
                by_inst[int(p["instance_index"])].append(p)

        for iidx, group in by_inst.items():
            inst_rows = inst_by_scene.get(sid) or []
            irow = next((r for r in inst_rows if int(r["instance_index"]) == iidx), None)
            is_overflow = bool(group[0].get("is_partial"))
            if irow is not None:
                w, h = int(irow["bbox_w"]), int(irow["bbox_h"])
                fits = w <= 256 and h <= 256
            else:
                fits = not is_overflow

            if fits and is_overflow:
                compact_bad.append({"scene_id": sid, "i": iidx, "note": "compact marked partial"})
            if not fits and len(group) == 1 and not is_overflow:
                overflow_bad.append({
                    "scene_id": sid, "i": iidx,
                    "note": "overflow produced single non-partial tile",
                })
            if is_overflow:
                if len(group) < 2:
                    overflow_bad.append({
                        "scene_id": sid, "i": iidx, "n": len(group),
                        "note": "overflow needs ≥2 tiles",
                    })
                gids = {p.get("parent_group_id") for p in group}
                if len(gids) != 1:
                    overflow_bad.append({
                        "scene_id": sid, "i": iidx, "groups": list(gids),
                        "note": "overflow tiles must share one parent_group_id",
                    })

            # every kept tile must contain ≥1 object pixel (bug #5) — reload mask
            if sid not in masks and scene is not None:
                masks[sid] = _load_mask(scene, root, paths.stage_dir)  # may be None
            mask = masks.get(sid)
            if mask is None:
                continue
            for p in group:
                x, y, s = int(p["xoff"]), int(p["yoff"]), 256
                tile = mask[y:y + s, x:x + s]
                if tile.size == 0 or not tile.any():
                    empty_tile_bad.append({
                        "scene_id": sid, "xoff": x, "yoff": y,
                        "instance_index": iidx,
                    })

        # negatives: full-window ≥ buffer (bug #6)
        if negs and sid in masks and masks[sid] is not None:
            mask = masks[sid]
            pos = (mask > 0).astype(np.uint8)
            if pos.any():
                import cv2
                dist = cv2.distanceTransform(
                    np.where(pos > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 3
                )
                for p in negs:
                    x, y, s = int(p["xoff"]), int(p["yoff"]), 256
                    window = dist[y:y + s, x:x + s]
                    if window.size == 0:
                        continue
                    # sample grid + edges for speed, but check corners/centre thoroughly
                    ys = np.unique(np.linspace(0, window.shape[0] - 1, 17).astype(int))
                    xs = np.unique(np.linspace(0, window.shape[1] - 1, 17).astype(int))
                    sub = window[np.ix_(ys, xs)]
                    if float(sub.min()) < neg_buffer_px - 1:
                        empty_tile_bad.append({
                            "scene_id": sid, "xoff": x, "yoff": y,
                            "min_clear_px": float(sub.min()),
                            "note": "negative buffer violated",
                        })

    rep.hard(
        not empty_tile_bad,
        "no empty overflow tiles / negative buffers respected"
        if not empty_tile_bad else f"{len(empty_tile_bad)} plan pixel violations (bugs #5/#6)",
        examples=empty_tile_bad[:20],
    )
    rep.hard(
        not overflow_bad,
        "overflow tiles ≥2, shared group, partial flag"
        if not overflow_bad else f"{len(overflow_bad)} overflow invariant failures",
        examples=overflow_bad[:20],
    )
    if compact_bad:
        rep.soft(False, f"{len(compact_bad)} compact/overflow flag oddities",
                 examples=compact_bad[:10])

    # --- neg:pos ratio across dataset -------------------------------------
    n_neg = 0
    n_pos = 0
    for plist in plans.values():
        for p in plist:
            if p.get("instance_index") is None:
                n_neg += 1
            else:
                n_pos += 1
    if n_pos > 0:
        ratio = n_neg / n_pos
        # architecture: roughly matches configured neg_ratio (default 0.4)
        ok = 0.05 <= ratio <= max(1.5, neg_ratio * 3)
        rep.soft(
            ok,
            f"negative:positive ratio = {ratio:.3f} (configured ≈ {neg_ratio})",
            n_neg=n_neg, n_pos=n_pos, ratio=round(ratio, 4),
        )
    else:
        rep.info("no positive crops — neg ratio skipped")
    return rep
