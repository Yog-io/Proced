"""Full-extent feature extraction (C.0): shape, radiometry, eligibility."""

import numpy as np
import pytest

from proced.config import PipelineConfig
from proced.features import extract_full_extent_features, prepare_band_fields


def test_shape_features_on_disk():
    cfg = PipelineConfig(min_instance_area_px=10)
    mask = np.zeros((300, 300), np.uint8)
    yy, xx = np.mgrid[:300, :300]
    mask[(yy - 150) ** 2 + (xx - 150) ** 2 <= 40 ** 2] = 1  # r=40 disk
    feats = extract_full_extent_features(mask, cfg, default_label="oil")
    assert len(feats.instances) == 1
    inst = feats.instances[0]
    assert inst.area_px == pytest.approx(np.pi * 40 ** 2, rel=0.1)
    # circle: P²/A = 4π ≈ 12.57 (contour-based perimeter → some tolerance)
    assert 8.0 < inst.boundary_complexity < 18.0
    assert 0.8 < inst.aspect_ratio < 1.3
    assert inst.truncated_by_scene_edge is False
    assert inst.eligible_for_classifier is True
    assert feats.fragment_count == 1


def test_border_object_excluded_from_classifier_but_flagged():
    cfg = PipelineConfig(min_instance_area_px=10)
    mask = np.zeros((300, 300), np.uint8)
    mask[0:50, 100:150] = 1  # abuts top border
    feats = extract_full_extent_features(mask, cfg, default_label="oil")
    inst = feats.instances[0]
    assert inst.truncated_by_scene_edge is True
    assert inst.eligible_for_classifier is False
    assert feats.eligible_count == 0


def test_lookalike_mask_value_2_gets_label():
    cfg = PipelineConfig(min_instance_area_px=10, lookalike_mask_value=2)
    mask = np.zeros((300, 300), np.uint8)
    mask[50:100, 50:100] = 1
    mask[150:200, 150:200] = 2
    feats = extract_full_extent_features(mask, cfg, default_label="oil")
    labels = sorted(i.label for i in feats.instances)
    assert labels == ["lookalike", "oil"]


def test_radiometric_features_darker_object_has_positive_damping():
    cfg = PipelineConfig(min_instance_area_px=10, ring_px=15)
    H = W = 300
    mask = np.zeros((H, W), np.uint8)
    mask[120:180, 120:180] = 1
    # Object dark in dB (-18), background bright (-8)
    db = np.full((H, W), -8.0, dtype=np.float32)
    db[120:180, 120:180] = -18.0
    feats = extract_full_extent_features(
        mask, cfg, band_db={"VV": db}, default_label="oil",
    )
    inst = feats.instances[0]
    assert inst.damping_ratio_db == pytest.approx(10.0, abs=0.5)
    assert inst.damping_ratio > 5.0  # ≈10^(10/10)
    assert inst.glcm_contrast is not None
    assert inst.glcm_homogeneity is not None
    assert inst.boundary_gradient_steepness is not None


def test_ndpi_requires_dual_pol_linear_fields():
    cfg = PipelineConfig(min_instance_area_px=10, ring_px=10)
    H = W = 200
    mask = np.zeros((H, W), np.uint8)
    mask[80:120, 80:120] = 1
    db_vv = np.full((H, W), -10.0, dtype=np.float32)
    db_vh = np.full((H, W), -16.0, dtype=np.float32)
    from proced.features import linear_from_db
    lin = {
        "VV": linear_from_db(db_vv, "power"),
        "VH": linear_from_db(db_vh, "power"),
    }
    feats = extract_full_extent_features(
        mask, cfg, band_db={"VV": db_vv, "VH": db_vh},
        band_linear=lin, default_label="oil", has_dual_pol=True,
    )
    inst = feats.instances[0]
    assert inst.ndpi is not None
    # VV/VH = 10^(6/10) ≈ 3.98 → NDPI ≈ (3.98-1)/(3.98+1) ≈ 0.60
    assert 0.5 < inst.ndpi < 0.7


def test_prepare_band_fields_native_returns_none():
    arr = np.zeros((50, 50), np.uint8)
    db, lin, notes = prepare_band_fields({"VV": arr}, {"VV": "native"}, PipelineConfig())
    assert db is None and lin is None
    assert notes


def test_empty_mask_zero_instances():
    cfg = PipelineConfig()
    feats = extract_full_extent_features(np.zeros((100, 100), np.uint8), cfg)
    assert feats.instances == []
    assert feats.fragment_count == 0
