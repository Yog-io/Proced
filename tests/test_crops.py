"""Crop planning: compact / overflow / truncated / negatives (A.2, C.1, C.3)."""

import numpy as np
import pytest

from proced.config import PipelineConfig
from proced.crops import axis_positions, plan_crops
from proced.features import Instance, extract_full_extent_features


def _instance(mask, mv=1):
    feats = extract_full_extent_features(mask, PipelineConfig(),
                                         default_label="oil")
    assert feats.instances, "expected at least one instance"
    return feats.instances


def _blob(shape, xywh, value=1):
    m = np.zeros(shape, np.uint8)
    x, y, w, h = xywh
    m[y:y + h, x:x + w] = value
    return m


def test_compact_single_crop_with_jitter_in_bounds():
    cfg = PipelineConfig(tile_size=256, jitter_px=25)
    mask = _blob((512, 512), (200, 200, 80, 60))
    inst = _instance(mask)
    rng = np.random.default_rng(0)
    plans = plan_crops(512, 512, inst, mask, cfg, "s1", rng, has_mask=True)
    pos = [p for p in plans if p.instance_index is not None]
    assert len(pos) == 1
    p = pos[0]
    assert p.is_partial is False
    assert 0 <= p.xoff <= 512 - 256
    assert 0 <= p.yoff <= 512 - 256
    # jitter ±25 around centroid (240, 230) → window origin near 112±25
    assert 112 - 30 <= p.xoff <= 112 + 30
    assert p.parent_group_id


def test_overflow_tiles_share_group_and_are_partial():
    cfg = PipelineConfig(tile_size=256, stride=128, max_tiles_per_object=48)
    mask = _blob((600, 800), (50, 250, 600, 40))  # 600×40 slick → horizontal overflow
    inst = _instance(mask)
    rng = np.random.default_rng(0)
    plans = plan_crops(800, 600, inst, mask, cfg, "s2", rng, has_mask=True)
    pos = [p for p in plans if p.instance_index is not None]
    assert len(pos) >= 4, "oversized slick must be multi-tiled"
    assert all(p.is_partial for p in pos)
    assert len({p.parent_group_id for p in pos}) == 1  # shared parent_group_id
    # FIX vs archi: every kept tile must contain object pixels (no phantom oil)
    for p in pos:
        tile = mask[p.yoff:p.yoff + 256, p.xoff:p.xoff + 256]
        assert tile.size and tile.any(), "empty tile would be labelled oil"


def test_truncated_flag_is_object_property_not_crop_property():
    cfg = PipelineConfig()
    mask = _blob((400, 400), (0, 100, 60, 60))  # touches left border
    inst = _instance(mask)
    assert inst[0].truncated_by_scene_edge is True
    rng = np.random.default_rng(0)
    plans = plan_crops(400, 400, inst, mask, cfg, "s3", rng, has_mask=True)
    pos = [p for p in plans if p.instance_index is not None]
    assert all(p.truncated_by_scene_edge for p in pos)


def test_background_crop_far_from_positives():
    cfg = PipelineConfig(tile_size=256, neg_buffer_px=100, neg_ratio=1.0,
                         max_neg_per_scene=8)
    mask = _blob((700, 700), (0, 0, 60, 60))  # positives squeezed into corner
    inst = _instance(mask)
    rng = np.random.default_rng(0)
    plans = plan_crops(700, 700, inst, mask, cfg, "s4", rng, has_mask=True)
    negs = [p for p in plans if p.instance_index is None]
    assert negs, "expected background patches"
    pos_px = np.argwhere(mask > 0)
    for p in negs:
        # every pixel of the window must be ≥ buffer from every positive pixel
        ys = np.arange(p.yoff, p.yoff + 256, 8)
        xs = np.arange(p.xoff, p.xoff + 256, 8)
        grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
        # check window corners + centre (sufficient in test geometry)
        for gy, gx in [(grid_y[0, 0], grid_x[0, 0]),
                       (grid_y[-1, -1], grid_x[-1, -1]),
                       (grid_y[0, -1], grid_x[0, -1])]:
            d = np.min((pos_px[:, 0] - gy) ** 2 + (pos_px[:, 1] - gx) ** 2)
            # top-left of window is ≥buffer; full-window guarantee verified below
        dist = np.sqrt(((pos_px[None, :, 0] - (p.yoff + np.arange(0, 256, 4))[:, None]) ** 2
                        + (pos_px[None, :, 1] - (p.xoff + np.arange(0, 256, 4))[:, None]) ** 2).min())
        # Simpler: min distance from window pixels (sampled) to nearest positive
        wy = p.yoff + np.arange(0, 256, 2)
        wx = p.xoff + np.arange(0, 256, 2)
        GY, GX = np.meshgrid(wy, wx, indexing="ij")
        dd = np.sqrt((pos_px[None, :, 0] - GY[..., None]) ** 2 +
                     (pos_px[None, :, 1] - GX[..., None]) ** 2)
        min_d = dd.min()
        assert min_d >= cfg.neg_buffer_px - 1, f"negative window only {min_d:.0f}px clear"
        assert p.label == "background"


def test_no_negatives_from_unlabeled_by_default():
    cfg = PipelineConfig(tile_size=256, neg_per_empty_scene=3)
    mask = np.zeros((512, 512), np.uint8)
    rng = np.random.default_rng(0)
    plans = plan_crops(512, 512, [], mask, cfg, "s5", rng, has_mask=False)
    assert not plans  # unlabeled scene yields nothing by default
    plans2 = plan_crops(512, 512, [], mask, cfg, "s5", rng, has_mask=True)
    # verified-clean empty mask → gets empty-scene negatives
    assert len(plans2) == cfg.neg_per_empty_scene


def test_axis_positions_cover_length():
    pos = axis_positions(start=10, length=700, limit=800, size=256, stride=128,
                         max_positions=48)
    assert pos[0] == 10
    assert pos[-1] + 256 >= 10 + 700 - 1  # last window reaches the far edge
    assert all(0 <= p <= 800 - 256 for p in pos)
    assert len(pos) == len(set(pos))  # no duplicate windows
