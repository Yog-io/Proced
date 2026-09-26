"""Independent QA helpers — NEVER imports ``proced/``.

Every check recomputes from raw files (rasterio / cv2 / shapely / pandas) and
compares against pipeline artifacts. Constants below are intentional local
copies of the published schema so this suite stays decoupled from pipeline code.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# --- Local copies of published schema (do NOT import from proced/) ----------
GEODATA_COLUMNS = [
    "crop_id",
    "crop_source_scene_id",
    "is_calibrated",
    "has_dual_pol",
    "is_synthetic_location",
    "crop_bbox_corners",
    "is_partial_object",
    "parent_group_id",
    "truncated_by_scene_edge",
    "db_conversion_formula_version",
    "label",
    "full_extent_area_px",
    "full_extent_perimeter",
    "full_extent_boundary_complexity",
    "full_extent_aspect_ratio",
]
MASTER_EXTRA_COLUMNS = [
    "image_id",
    "dataset_source",
    "width",
    "height",
    "has_real_geo",
    "lat",
    "lon",
    "timestamp_utc",
    "source_corridor_id",
    "offset_km",
    "crop_xoff",
    "crop_yoff",
    "crop_size",
    "source_scene_relpath",
    "is_synthetic_timestamp",
]
MASTER_COLUMNS = GEODATA_COLUMNS + [c for c in MASTER_EXTRA_COLUMNS if c not in GEODATA_COLUMNS]
REQUIRED_NULL_FREE = (
    "crop_bbox_corners", "is_partial_object", "truncated_by_scene_edge",
    "crop_id", "label", "parent_group_id", "db_conversion_formula_version",
)
DB_LO, DB_HI = -30.0, 0.0
DB_FORMULA_VERSION = "v1_linear_neg30_to_0"
NATIVE_FORMULA_VERSION = "native_8bit_no_db"
# Spec C.2 inverse (Block 2): db = (px / 65535) * 30 - 30
DB_SCALE = 65535.0

# Zenodo label expectations (soft) — architecture §2 Stage 5
ZENODO_EXPECTED_LABELS = {"oil": 1400, "lookalike": 700}
LABEL_COUNT_TOLERANCE = 0.5  # ±50% soft band for fixture / partial runs


# --------------------------------------------------------------------- findings
@dataclass
class Finding:
    check: str
    severity: str  # "hard" | "soft" | "info"
    ok: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class Report:
    """Accumulates findings for one verification module."""

    def __init__(self, name: str):
        self.name = name
        self.findings: List[Finding] = []

    def hard(self, ok: bool, message: str, **details: Any) -> Finding:
        return self._add("hard", ok, message, details)

    def soft(self, ok: bool, message: str, **details: Any) -> Finding:
        return self._add("soft", ok, message, details)

    def info(self, message: str, **details: Any) -> Finding:
        return self._add("info", True, message, details)

    def _add(self, severity: str, ok: bool, message: str, details: dict) -> Finding:
        f = Finding(check=self.name, severity=severity, ok=bool(ok),
                    message=message, details=details)
        self.findings.append(f)
        return f

    @property
    def hard_failures(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "hard" and not f.ok]

    @property
    def soft_failures(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "soft" and not f.ok]

    @property
    def ok(self) -> bool:
        return not self.hard_failures

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "ok": self.ok,
            "n_hard_fail": len(self.hard_failures),
            "n_soft_fail": len(self.soft_failures),
            "findings": [f.to_dict() for f in self.findings],
        }


# ----------------------------------------------------------------------- paths
@dataclass
class QAPaths:
    stage_dir: Path
    dest_dir: Path
    state_dir: Path
    features_dir: Path
    splits_dir: Path
    corridors_path: Path
    output_archive: Optional[Path] = None
    source_archive: Optional[Path] = None  # raw input .7z for extract checks

    @classmethod
    def default(cls) -> "QAPaths":
        d = ROOT / "data"
        return cls(
            stage_dir=d / "scratch_unpacked",
            dest_dir=d / "master_dataset_processed",
            state_dir=d / "state",
            features_dir=d / "features",
            splits_dir=d / "splits",
            corridors_path=d / "reference" / "shipping_corridors.geojson",
            output_archive=d / "master_dataset_v1.7z",
        )

    @classmethod
    def from_env(cls) -> "QAPaths":
        base = cls.default()
        return cls(
            stage_dir=Path(os.environ.get("QA_STAGE_DIR", base.stage_dir)),
            dest_dir=Path(os.environ.get("QA_DEST_DIR", base.dest_dir)),
            state_dir=Path(os.environ.get("QA_STATE_DIR", base.state_dir)),
            features_dir=Path(os.environ.get("QA_FEATURES_DIR", base.features_dir)),
            splits_dir=Path(os.environ.get("QA_SPLITS_DIR", base.splits_dir)),
            corridors_path=Path(os.environ.get("QA_CORRIDORS", base.corridors_path)),
            output_archive=Path(os.environ["QA_OUTPUT_ARCHIVE"])
            if os.environ.get("QA_OUTPUT_ARCHIVE") else base.output_archive,
            source_archive=Path(os.environ["QA_SOURCE_ARCHIVE"])
            if os.environ.get("QA_SOURCE_ARCHIVE") else None,
        )

    def require(self, *names: str) -> Optional[str]:
        """Return a skip reason if required artifacts are missing."""
        missing = []
        mapping = {
            "catalog": self.state_dir / "catalog.json",
            "scene_geo": self.state_dir / "scene_geo.json",
            "instances": self.state_dir / "instances.csv",
            "crop_plans": self.state_dir / "crop_plans.json",
            "crop_rows": self.state_dir / "crop_rows.json",
            "metadata": self.dest_dir / "metadata.csv",
            "stage": self.stage_dir,
            "dest": self.dest_dir,
            "splits": self.splits_dir,
            "features": self.features_dir,
            "corridors": self.corridors_path,
        }
        for n in names:
            p = mapping[n]
            ok = p.is_dir() if n in ("stage", "dest", "splits", "features") else p.is_file()
            if not ok:
                missing.append(str(p))
        if missing:
            return "missing: " + ", ".join(missing)
        return None


# -------------------------------------------------------------------- loaders
def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_catalog(paths: QAPaths) -> dict:
    return load_json(paths.state_dir / "catalog.json")


def load_scene_geo(paths: QAPaths) -> dict:
    return load_json(paths.state_dir / "scene_geo.json")


def load_instances(paths: QAPaths) -> pd.DataFrame:
    return pd.read_csv(paths.state_dir / "instances.csv").fillna("")


def load_crop_plans(paths: QAPaths) -> dict:
    return load_json(paths.state_dir / "crop_plans.json")


def load_crop_rows(paths: QAPaths) -> pd.DataFrame:
    return pd.DataFrame(load_json(paths.state_dir / "crop_rows.json")).fillna("")


def load_metadata(paths: QAPaths) -> pd.DataFrame:
    return pd.read_csv(paths.dest_dir / "metadata.csv").fillna("")


def iter_geodata(paths: QAPaths) -> Iterable[Tuple[Path, pd.DataFrame]]:
    for dirpath, _d, files in os.walk(paths.dest_dir):
        if "GEODATA.csv" in files:
            p = Path(dirpath) / "GEODATA.csv"
            try:
                yield p, pd.read_csv(p).fillna("")
            except Exception:
                continue


def find_png(paths: QAPaths, crop_id: str, pol: str) -> Optional[Path]:
    matches = list(paths.dest_dir.rglob(f"{crop_id}_{pol}.png"))
    return matches[0] if matches else None


def scene_by_id(catalog: dict) -> Dict[str, dict]:
    return {s["scene_id"]: s for s in catalog.get("scenes") or []}


def resolve_stage_path(rel_or_abs: str, catalog_root: str, stage_dir: Path) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    # catalog stores paths relative to stage root (or absolute)
    cand = stage_dir / p
    if cand.exists():
        return cand
    cand2 = Path(catalog_root) / p
    return cand2


# ------------------------------------------------------------------ math utils
def db_from_uint16(px: np.ndarray, lo: float = DB_LO, hi: float = DB_HI) -> np.ndarray:
    """Documented inverse — independent of pipeline code."""
    p = np.asarray(px, dtype=np.float64)
    return (p / DB_SCALE) * (hi - lo) + lo


def uint16_from_db(db: np.ndarray, lo: float = DB_LO, hi: float = DB_HI) -> np.ndarray:
    a = np.clip(np.asarray(db, dtype=np.float64), lo, hi)
    scaled = np.round(((a - lo) / (hi - lo)) * DB_SCALE)
    return np.clip(scaled, 0, DB_SCALE).astype(np.uint16)


def to_db_independent(arr: np.ndarray, domain: str, eps: float = 1e-7) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if domain == "db":
        return a
    if domain == "power":
        return 10.0 * np.log10(np.clip(a, eps, None))
    if domain == "amplitude":
        return 20.0 * np.log10(np.clip(a, eps, None))
    raise ValueError(domain)


def detect_domain_independent(arr: np.ndarray) -> str:
    if np.issubdtype(arr.dtype, np.integer):
        return "native"
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return "native"
    if float(np.min(finite)) < -0.5:
        return "db"
    p99 = float(np.percentile(finite, 99))
    return "power" if p99 <= 1.5 else "amplitude"


def parse_corners(s: str) -> List[Tuple[float, float]]:
    """Parse GEODATA crop_bbox_corners string → [(lon, lat), ...].

    format_corners() renders ``[(lon, lat), (lon, lat)]`` (separator ``), ``);
    tolerate any spacing and plain-number/scientific-notation coords.
    """
    pairs = re.findall(r"\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)", str(s))
    if not pairs:
        raise ValueError(f"bad corners: {str(s)[:40]}")
    return [(float(lon), float(lat)) for lon, lat in pairs]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def point_segment_distance_km(
    lat: float, lon: float,
    lat1: float, lon1: float, lat2: float, lon2: float,
) -> Tuple[float, Tuple[float, float]]:
    """Great-circle-ish local metric distance from point to segment (km).

    Works in a local ENU frame around the point (adequate for corridor offsets).
    Returns (distance_km, closest_point_latlon).
    """
    # Local projection: km per degree
    lat0 = math.radians(lat)
    m_lat = 111.32
    m_lon = 111.32 * max(math.cos(lat0), 1e-6)

    def to_xy(la: float, lo: float) -> Tuple[float, float]:
        return ((lo - lon) * m_lon, (la - lat) * m_lat)

    ax, ay = to_xy(lat1, lon1)
    bx, by = to_xy(lat2, lon2)
    px, py = 0.0, 0.0
    dx, dy = bx - ax, by - ay
    seg_len2 = dx * dx + dy * dy
    if seg_len2 < 1e-18:
        closest_xy = (ax, ay)
    else:
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len2))
        closest_xy = (ax + t * dx, ay + t * dy)
    dist = math.hypot(*closest_xy)
    c_lat = lat + closest_xy[1] / m_lat
    c_lon = lon + closest_xy[0] / m_lon
    return dist, (c_lat, c_lon)


def min_distance_to_polyline_km(
    lat: float, lon: float, waypoints: Sequence[Tuple[float, float]]
) -> Tuple[float, Tuple[float, float]]:
    """waypoints are (lat, lon). Returns (km, closest_latlon)."""
    best = (float("inf"), (lat, lon))
    for i in range(len(waypoints) - 1):
        la1, lo1 = waypoints[i]
        la2, lo2 = waypoints[i + 1]
        d, c = point_segment_distance_km(lat, lon, la1, lo1, la2, lo2)
        if d < best[0]:
            best = (d, c)
    return best


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from point 1 → point 2 (degrees, 0=N)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def angular_diff_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


def load_corridors_independent(path: Path) -> List[dict]:
    """Load LineString corridors from GeoJSON — (lat, lon) waypoints."""
    gj = load_json(path)
    out = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry") or {}
        if geom.get("type") != "LineString":
            continue
        props = feat.get("properties") or {}
        wps = [(float(c[1]), float(c[0])) for c in geom.get("coordinates", [])]
        if len(wps) >= 2:
            out.append({
                "id": str(props.get("id", f"c{len(out)}")),
                "weight": float(props.get("weight", 1.0)),
                "waypoints": wps,
            })
    return out


def stable_rng(seed: int, *parts: str) -> random.Random:
    h = seed
    for p in parts:
        for ch in str(p):
            h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return random.Random(h)


def sample_indices(n: int, k: int, seed: int, *parts: str) -> List[int]:
    if n <= 0:
        return []
    rng = stable_rng(seed, *parts)
    k = min(k, n)
    idx = list(range(n))
    rng.shuffle(idx)
    return sorted(idx[:k])


def is_truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in ("true", "1", "y", "yes")


def is_yn_yes(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().upper() in ("Y", "TRUE", "1", "YES")


def syn_true(v: Any) -> bool:
    return str(v).strip().lower() in ("true", "1", "y", "yes")


# -------------------------------------------------------------- independent CV
def connected_components(binary: np.ndarray) -> Tuple[int, np.ndarray]:
    """8-connected components of a binary mask (fresh cv2 call)."""
    import cv2
    n, labels = cv2.connectedComponents((binary > 0).astype(np.uint8), connectivity=8)
    return max(0, n - 1), labels


def shape_from_mask_region(region: np.ndarray) -> Optional[dict]:
    """Fresh contour metrics for a boolean/0-1 region."""
    import cv2
    m = (region > 0).astype(np.uint8)
    if not m.any():
        return None
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    area = int(np.count_nonzero(m))
    perim = float(cv2.arcLength(cnt, True))
    complexity = (perim ** 2) / area if area > 0 else 0.0
    # Match the published C.0 algorithm independently: fitEllipse when the
    # contour has enough points (pipeline bug #14 fix: sort major/minor),
    # else minAreaRect. Using the same definition here is still independent —
    # fresh cv2 calls on the raw mask, no proced/ import.
    if len(cnt) >= 5:
        try:
            (_, _), (ew, eh), _ = cv2.fitEllipse(cnt)
            major, minor = max(ew, eh), min(ew, eh)
            aspect = float(major / minor) if minor > 1e-6 else 1.0
        except cv2.error:
            rect = cv2.minAreaRect(cnt)
            (rw, rh) = rect[1]
            major, minor = max(rw, rh), min(rw, rh)
            aspect = float(major / minor) if minor > 1e-6 else 1.0
    else:
        rect = cv2.minAreaRect(cnt)
        (rw, rh) = rect[1]
        major, minor = max(rw, rh), min(rw, rh)
        aspect = float(major / minor) if minor > 1e-6 else 1.0
    x, y, w, h = cv2.boundingRect(cnt)
    return {
        "area_px": area,
        "perimeter": perim,
        "boundary_complexity": complexity,
        "aspect_ratio": aspect,
        "bbox": (int(x), int(y), int(w), int(h)),
        "touches_border": bool(
            m[0, :].any() or m[-1, :].any() or m[:, 0].any() or m[:, -1].any()
        ),
    }


def touches_border(mask: np.ndarray) -> bool:
    return bool(
        mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any()
    )


def apply_affine(transform6: Sequence[float], col: float, row: float) -> Tuple[float, float]:
    """x = c + a*col + b*row ; y = f + d*col + e*row  (GDAL geotransform order
    stored as [a,b,c,d,e,f])."""
    a, b, c, d, e, f = transform6
    return (c + a * col + b * row, f + d * col + e * row)


def parse_transform_from_rasterio(ds) -> List[float]:
    t = ds.transform
    # rasterio Affine: a b c / d e f in matrix form x = a*col + b*row + c
    return [float(t.a), float(t.b), float(t.c), float(t.d), float(t.e), float(t.f)]


def ensure_dir(p: Path) -> Path:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p


def write_report(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
