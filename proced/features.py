"""Stage 2 — full-extent (pre-crop) feature extraction, refinement C.0 / A.5.

Shape features AND radiometric features are computed from the FULL-SCENE mask
(+ local radiometric regions around each instance) BEFORE any tiling decision.
Nothing downstream may recompute these from cropped tiles.

Instances that touch the scene border are flagged ``truncated_by_scene_edge``
and marked ineligible for the look-alike classifier (their full-extent shape
features are unreliable by construction) — but their crops still feed Model 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import PipelineConfig
from .radiometry import calibrate_full_scene_db


@dataclass
class Instance:
    instance_index: int
    label: str                    # oil | lookalike
    mask_value: int               # source mask value (1 / 2 / ...)
    area_px: int                  # true pixel count over full extent
    perimeter: float
    aspect_ratio: float
    boundary_complexity: float
    bbox: Tuple[int, int, int, int]      # x, y, w, h (full scene)
    centroid: Tuple[int, int]
    truncated_by_scene_edge: bool
    # Contour/component mask exist only during extraction; they are dropped
    # when instances are serialised to instances.csv (plan/convert don't need them).
    contour: Optional[np.ndarray] = None
    comp_mask: Optional[np.ndarray] = None
    # Radiometric (None when uncalibrated / undetermined)
    damping_ratio: Optional[float] = None
    damping_ratio_db: Optional[float] = None
    boundary_gradient_steepness: Optional[float] = None
    backscatter_variance_ratio: Optional[float] = None
    glcm_contrast: Optional[float] = None
    glcm_homogeneity: Optional[float] = None
    ndpi: Optional[float] = None

    @property
    def eligible_for_classifier(self) -> bool:
        """A.5/C.1: scene-edge-truncated instances are excluded from the
        look-alike classifier's training rows."""
        return not self.truncated_by_scene_edge


@dataclass
class SceneFeatures:
    instances: List[Instance]
    fragment_count: int
    eligible_count: int
    notes: List[str] = field(default_factory=list)


def _instance_mask(labels: np.ndarray, idx: int) -> np.ndarray:
    return labels == idx


def _touches_border(m: np.ndarray) -> bool:
    return bool(
        m[0, :].any() or m[-1, :].any() or m[:, 0].any() or m[:, -1].any()
    )


def _shape_from_contour(cnt: np.ndarray, comp: np.ndarray) -> Tuple[int, float, float, Tuple]:
    area_px = int(np.count_nonzero(comp))
    perimeter = float(cv2.arcLength(cnt, True))
    complexity = (perimeter ** 2) / area_px if area_px > 0 else 0.0
    if len(cnt) >= 5:
        (_, _), (w, h), _ = cv2.fitEllipse(cnt)
        major, minor = max(w, h), min(w, h)
        aspect = float(major / minor) if minor > 1e-6 else 1.0
    else:
        rect = cv2.minAreaRect(cnt)
        (rw, rh) = rect[1]
        major, minor = max(rw, rh), min(rw, rh)
        aspect = float(major / minor) if minor > 1e-6 else 1.0
    return area_px, perimeter, complexity, aspect


def _ring_mask(comp: np.ndarray, scene_mask: np.ndarray, ring_px: int) -> np.ndarray:
    """Background ring: dilate(comp) ∩ (scene background). Never overlaps oil."""
    k = 2 * ring_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil = cv2.dilate(comp.astype(np.uint8), kernel) > 0
    ring = dil & (scene_mask == 0)
    return ring


def _boundary_band(contour: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    band = np.zeros(shape_hw, dtype=np.uint8)
    cv2.drawContours(band, [contour.astype(np.int32)], -1, 1, thickness=3)
    return band > 0


def _glcm_features(db_patch: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    from skimage.feature import graycomatrix, graycoprops

    if db_patch.size < 64 or min(db_patch.shape) < 4:
        return None, None
    q = np.clip((db_patch + 30.0) / 30.0 * 255.0, 0, 255).astype(np.uint8)
    try:
        glcm = graycomatrix(
            q, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=256, symmetric=True, normed=True,
        )
        contrast = float(np.mean(graycoprops(glcm, "contrast")))
        homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))
        return contrast, homogeneity
    except Exception:
        return None, None


def _region_window(bbox, shape_hw, pad):
    x, y, w, h = bbox
    H, W = shape_hw
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    return x0, y0, x1, y1


def extract_full_extent_features(
    mask: np.ndarray,
    cfg: PipelineConfig,
    band_db: Optional[Dict[str, np.ndarray]] = None,
    band_linear: Optional[Dict[str, np.ndarray]] = None,
    default_label: str = "oil",
    has_dual_pol: bool = False,
) -> SceneFeatures:
    """Compute C.0 features for every labeled instance on the FULL scene.

    Parameters
    ----------
    mask : full-scene integer mask (0 = background; 1+ = labels;
           ``cfg.lookalike_mask_value`` (default 2) marks look-alikes)
    band_db : optional per-pol full-scene dB arrays (None ⇒ radiometrics skipped)
    band_linear : optional per-pol linear arrays for NDPI (dual-pol only)
    """
    notes: List[str] = []
    mask_u8 = (mask > 0).astype(np.uint8)
    fragment_count = int(cv2.connectedComponents(mask_u8, connectivity=8)[0] - 1)

    instances: List[Instance] = []
    if fragment_count == 0:
        return SceneFeatures(instances=[], fragment_count=0, eligible_count=0,
                             notes=["no positive instances"])

    # Separate classes before component labelling so oil/lookalike labels survive
    class_values = [int(v) for v in np.unique(mask) if v > 0]
    idx_global = 0
    for mv in sorted(class_values):
        if mv == cfg.lookalike_mask_value:
            label = "lookalike"
        else:
            label = default_label  # binary mask → dataset default (oil|lookalike)
        binary = (mask == mv).astype(np.uint8)
        n, labels = cv2.connectedComponents(binary, connectivity=8)
        for i in range(1, n):
            comp = _instance_mask(labels, i)
            area = int(np.count_nonzero(comp))
            if area < cfg.min_instance_area_px:
                continue
            cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            area_px, perimeter, complexity, aspect = _shape_from_contour(cnt, comp)
            x, y, w, h = cv2.boundingRect(cnt)
            m = cv2.moments(cnt)
            cx = int(m["m10"] / m["m00"]) if m["m00"] else x + w // 2
            cy = int(m["m01"] / m["m00"]) if m["m00"] else y + h // 2
            inst = Instance(
                instance_index=idx_global,
                label=label,
                mask_value=mv,
                area_px=area_px,
                perimeter=perimeter,
                aspect_ratio=aspect,
                boundary_complexity=complexity,
                bbox=(int(x), int(y), int(w), int(h)),
                centroid=(cx, cy),
                truncated_by_scene_edge=_touches_border(comp),
                contour=cnt,
                comp_mask=comp,
            )
            idx_global += 1

            # ---- Radiometric features (calibrated data only) ------------------
            if band_db and "VV" in band_db:
                x0, y0, x1, y1 = _region_window(inst.bbox, mask.shape, cfg.ring_px + 2)
                sub_db = {p: arr[y0:y1, x0:x1] for p, arr in band_db.items()}
                sub_comp = inst.comp_mask[y0:y1, x0:x1]
                sub_bg = (mask[y0:y1, x0:x1] == 0)
                vv = sub_db.get("VV")
                obj_vals = vv[sub_comp] if vv is not None else np.array([])
                # Background ring inside the region window (dilate(comp) ∩ background)
                ring_local_bg = _ring_mask(sub_comp, mask[y0:y1, x0:x1], cfg.ring_px)
                ring_vals = vv[ring_local_bg] if vv is not None else np.array([])
                if obj_vals.size and ring_vals.size:
                    inst.damping_ratio_db = float(np.mean(ring_vals) - np.mean(obj_vals))
                    mean_obj_lin = float(np.mean(np.power(10.0, obj_vals / 10.0)))
                    mean_bg_lin = float(np.mean(np.power(10.0, ring_vals / 10.0)))
                    inst.damping_ratio = mean_bg_lin / max(mean_obj_lin, 1e-12)
                    var_obj = float(np.var(obj_vals))
                    var_bg = float(np.var(ring_vals))
                    inst.backscatter_variance_ratio = var_obj / max(var_bg, 1e-6)

                    # Boundary gradient steepness (|∇ dB| along the contour band)
                    band = _boundary_band(cnt, mask.shape)
                    band_sub = band[y0:y1, x0:x1]
                    gx = cv2.Sobel(vv, cv2.CV_32F, 1, 0, ksize=3)
                    gy = cv2.Sobel(vv, cv2.CV_32F, 0, 1, ksize=3)
                    grad = np.hypot(gx, gy)
                    if band_sub.any():
                        inst.boundary_gradient_steepness = float(np.mean(grad[band_sub]))

                    # GLCM contrast/homogeneity over the instance's local patch
                    cx0, cy0, cx1, cy1 = _region_window(inst.bbox, mask.shape, 0)
                    inst.glcm_contrast, inst.glcm_homogeneity = _glcm_features(
                        band_db["VV"][cy0:cy1, cx0:cx1]
                    )

                # NDPI from dual-pol linear fields
                if has_dual_pol and band_linear and "VV" in band_linear and "VH" in band_linear:
                    lv = band_linear["VV"][y0:y1, x0:x1][sub_comp]
                    lh = band_linear["VH"][y0:y1, x0:x1][sub_comp]
                    if lv.size:
                        denom = lv + lh
                        valid = np.abs(denom) > 1e-12
                        if valid.any():
                            inst.ndpi = float(np.mean((lv[valid] - lh[valid]) / denom[valid]))
            if inst.truncated_by_scene_edge:
                notes.append(
                    f"instance {inst.instance_index} truncated by scene edge — "
                    "excluded from look-alike classifier rows"
                )
            instances.append(inst)

    eligible = sum(1 for i in instances if i.eligible_for_classifier)
    return SceneFeatures(instances=instances, fragment_count=fragment_count,
                         eligible_count=eligible, notes=notes)


def linear_from_db(db: np.ndarray, domain: str) -> np.ndarray:
    """dB back to linear (power or amplitude) for NDPI/damping math."""
    if domain == "amplitude":
        return np.power(10.0, db / 20.0)
    return np.power(10.0, db / 10.0)


def prepare_band_fields(
    band_arrays: Dict[str, np.ndarray],
    domains: Dict[str, str],
    cfg: PipelineConfig,
) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, np.ndarray]], List[str]]:
    """Full-scene dB + linear dicts for radiometric features (or None if native)."""
    notes: List[str] = []
    db: Dict[str, np.ndarray] = {}
    lin: Dict[str, np.ndarray] = {}
    any_calibrated = False
    for pol, arr in band_arrays.items():
        domain = domains.get(pol, "auto")
        if domain == "auto" or domain is None:
            from .radiometry import detect_value_domain
            domain = detect_value_domain(arr, "auto")
        if domain == "native":
            continue
        any_calibrated = True
        d = calibrate_full_scene_db(arr, domain, eps=cfg.epsilon)
        if d is None:
            continue
        db[pol] = d
        lin[pol] = linear_from_db(d, domain)
    if not any_calibrated:
        notes.append("uncalibrated/native bands — radiometric features skipped (C.0 partial)")
        return None, None, notes
    return (db or None), (lin or None), notes
