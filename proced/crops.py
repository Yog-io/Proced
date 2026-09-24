"""Stage 3 — crop geometry plan (guide A.2 / refinement C.1).

Decides WHERE every 256×256 window goes, before any pixel is read:

* compact objects    → centroid-centered crop + ±jitter (clamped in-bounds)
* overflowing objects → 50%-stride sliding windows covering the full bbox,
                        shared parent_group_id, is_partial_object=Y, and
                        (fix vs archi) empty tiles are dropped so no
                        background window is ever mislabelled as oil
* scene-edge truncation is a property of the OBJECT (mask pixels on the
  border), not of the crop window (fix vs archi)
* background patches  → ≥300 px from every positive pixel, computed with a
  distance transform + integral image (the whole window, not just its
  top-left corner, is guaranteed ≥300 px clear)
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import PipelineConfig
from .features import Instance


@dataclass
class CropPlan:
    xoff: int
    yoff: int
    label: str                       # oil | lookalike | background
    instance_index: Optional[int]    # None for background patches
    is_partial: bool
    parent_group_id: str
    truncated_by_scene_edge: bool
    scene_id: str
    seq: int = 0

    @property
    def crop_id(self) -> str:
        return f"{self.scene_id}_crop{self.seq:03d}"


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))


def axis_positions(start: int, length: int, limit: int, size: int, stride: int,
                   max_positions: int) -> List[int]:
    """Window origins along one axis covering [start, start+length).

    50%-stride sliding windows; the final window always reaches the far edge;
    duplicates removed; count capped (adaptive widening) by ``max_positions``.
    """
    if length <= size:
        return [_clamp(start + length // 2 - size // 2, 0, max(0, limit - size))]
    last = start + length - size
    first = start
    n = 1 + max(0, math.ceil((length - size) / stride))
    n = min(n, max_positions)
    if n == 1:
        return [first]
    step = (last - first) / (n - 1)
    pos = sorted({_clamp(int(round(first + i * step)), 0, max(0, limit - size)) for i in range(n)})
    return pos


def plan_crops(
    scene_w: int,
    scene_h: int,
    instances: List[Instance],
    mask: np.ndarray,
    cfg: PipelineConfig,
    scene_id: str,
    rng: np.random.Generator,
    has_mask: bool,
) -> List[CropPlan]:
    size = cfg.tile_size
    stride = cfg.effective_stride
    plans: List[CropPlan] = []
    pos_binary = (mask > 0).astype(np.uint8) if mask is not None else np.zeros((scene_h, scene_w), np.uint8)

    for inst in instances:
        x, y, w, h = inst.bbox
        group = uuid.uuid4().hex[:12]
        fits = w <= size and h <= size
        if fits:
            # Compact: centroid-centered + jitter, clamped to scene bounds
            jx = int(rng.integers(-cfg.jitter_px, cfg.jitter_px + 1))
            jy = int(rng.integers(-cfg.jitter_px, cfg.jitter_px + 1))
            xoff = _clamp(inst.centroid[0] - size // 2 + jx, 0, max(0, scene_w - size))
            yoff = _clamp(inst.centroid[1] - size // 2 + jy, 0, max(0, scene_h - size))
            plans.append(CropPlan(
                xoff=xoff, yoff=yoff, label=inst.label,
                instance_index=inst.instance_index,
                is_partial=False, parent_group_id=group,
                truncated_by_scene_edge=inst.truncated_by_scene_edge,
                scene_id=scene_id,
            ))
        else:
            # Overflowing: cover the bbox with overlapping windows (C.1)
            max_x_pos = max(1, cfg.max_tiles_per_object)
            max_y_pos = max(1, cfg.max_tiles_per_object)
            # Bias the tile budget toward the longer axis
            xs = axis_positions(x, w, scene_w, size, stride, max_x_pos)
            ys = axis_positions(y, h, scene_h, size, stride, max_y_pos)
            while len(xs) * len(ys) > cfg.max_tiles_per_object and (len(xs) > 1 or len(ys) > 1):
                if w >= h and len(xs) > 1:
                    xs = xs[::2] or xs[:1]
                elif len(ys) > 1:
                    ys = ys[::2] or ys[:1]
                else:
                    break
            made = 0
            for yoff in ys:
                for xoff in xs:
                    if made >= cfg.max_tiles_per_object:
                        break
                    tile = mask[yoff:yoff + size, xoff:xoff + size]
                    if tile.shape[0] == 0 or tile.shape[1] == 0:
                        continue
                    # FIX vs archi: only keep windows that actually contain object
                    # pixels — prevents background windows labelled "oil".
                    if not np.any(tile > 0):
                        continue
                    plans.append(CropPlan(
                        xoff=int(xoff), yoff=int(yoff), label=inst.label,
                        instance_index=inst.instance_index,
                        is_partial=True, parent_group_id=group,
                        truncated_by_scene_edge=inst.truncated_by_scene_edge,
                        scene_id=scene_id,
                    ))
                    made += 1

    # ---- Background / negative patches (C.3) -------------------------------
    want_neg = 0
    if has_mask:
        if instances:
            # At least one clean negative whenever candidates exist (C.3):
            # ratio·N rounds to 0 for small N otherwise.
            want_neg = max(1, int(round(len(instances) * cfg.neg_ratio)))
        else:
            want_neg = cfg.neg_per_empty_scene
    elif cfg.negatives_from_unlabeled:
        want_neg = cfg.neg_per_empty_scene
    want_neg = min(want_neg, cfg.max_neg_per_scene)

    if want_neg > 0 and scene_w >= size and scene_h >= size:
        neg_group = uuid.uuid4().hex[:12]
        candidates = _negative_candidates(pos_binary, size, cfg.neg_buffer_px)
        if len(candidates) > 0:
            take = min(want_neg, len(candidates))
            pick = rng.choice(len(candidates), size=take, replace=False)
            for i in np.sort(pick):
                yoff, xoff = candidates[int(i)]
                plans.append(CropPlan(
                    xoff=int(xoff), yoff=int(yoff), label="background",
                    instance_index=None,
                    is_partial=False, parent_group_id=neg_group,
                    truncated_by_scene_edge=False,
                    scene_id=scene_id,
                ))
        # else: scene too crowded to place a buffered negative — acceptable (logged by caller)

    # Deterministic sequence: positives first (instance order), then negatives
    positives = [p for p in plans if p.instance_index is not None]
    negatives = [p for p in plans if p.instance_index is None]
    ordered = positives + sorted(negatives, key=lambda p: (p.yoff, p.xoff))
    for seq, plan in enumerate(ordered):
        plan.seq = seq
    return ordered


def _negative_candidates(pos_binary: np.ndarray, size: int, buffer_px: int) -> np.ndarray:
    """Top-left corners (y, x) whose FULL size×size window lies ≥ buffer_px from
    every positive pixel.

    distanceTransform(src) = distance to nearest zero pixel → feed
    zeros exactly at positives. Then an integral image checks that the whole
    window contains no pixel closer than the buffer (a top-left-only check
    would let windows overlap the buffer zone).
    """
    h, w = pos_binary.shape
    if h < size or w < size:
        return np.zeros((0, 2), dtype=np.int64)
    dist = cv2.distanceTransform(np.where(pos_binary > 0, 0, 255).astype(np.uint8),
                                 cv2.DIST_L2, 3)
    ok = (dist >= float(buffer_px)).astype(np.uint8)  # 1 where a pixel itself is clear
    # Window valid ⇔ sum of (!ok) over the window == 0
    # integral has shape (h+1, w+1)
    not_ok_integral = cv2.integral(1 - ok)
    max_y, max_x = h - size, w - size
    # Vectorised window sums for all top-lefts in range
    sub = (not_ok_integral[size:, size:]
           - not_ok_integral[:-size, size:]
           - not_ok_integral[size:, :-size]
           + not_ok_integral[:-size, :-size])
    # sub has shape (h-size+1, w-size+1)? integral slicing: rows 0..h
    # not_ok_integral is (h+1, w+1); sub[r, c] = window at (r, c) for r in [0, h-size]
    valid = np.argwhere(sub[: max_y + 1, : max_x + 1] == 0)
    if valid.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return valid  # (y, x) pairs
