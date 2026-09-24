"""Intermediate pipeline state under ``data/state/``.

Each stage reads its upstream artifact and writes its own, so every stage is
independently runnable:

    catalog.json   ← scan
    scene_geo.json ← geo
    instances.csv  ← features
    crop_plans.json← plan
    crop_rows.json ← convert   (feeds metadata + split + resume)
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .catalog import Catalog, SceneRecord
from .config import PipelineConfig

# Column order/types for instances.csv (loader coerces by name).
INSTANCE_COLUMNS = [
    "scene_id", "instance_index", "label", "mask_value",
    "area_px", "perimeter", "aspect_ratio", "boundary_complexity",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "centroid_x", "centroid_y",
    "truncated_by_scene_edge", "fragment_count",
    "crop_source_scene_id", "dataset_source",
    "damping_ratio", "damping_ratio_db",
    "boundary_gradient_steepness", "backscatter_variance_ratio",
    "glcm_contrast", "glcm_homogeneity", "ndpi",
]

_INT_FIELDS = {
    "instance_index", "mask_value", "area_px",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "centroid_x", "centroid_y", "fragment_count",
}
_FLOAT_FIELDS = {
    "perimeter", "aspect_ratio", "boundary_complexity",
    "damping_ratio", "damping_ratio_db",
    "boundary_gradient_steepness", "backscatter_variance_ratio",
    "glcm_contrast", "glcm_homogeneity", "ndpi",
}
_BOOL_FIELDS = {"truncated_by_scene_edge"}


@dataclass
class StatePaths:
    root: Path
    catalog: Path
    instances: Path
    scene_geo: Path
    crop_plans: Path
    crop_rows: Path

    @classmethod
    def from_config(cls, cfg: PipelineConfig) -> "StatePaths":
        root = Path(cfg.state_dir)
        return cls(
            root=root,
            catalog=root / "catalog.json",
            instances=root / "instances.csv",
            scene_geo=root / "scene_geo.json",
            crop_plans=root / "crop_plans.json",
            crop_rows=root / "crop_rows.json",
        )

    def ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------- paths
def _rel_or_abs(p: Optional[Union[str, Path]], root: Path) -> Optional[str]:
    if p is None:
        return None
    path = Path(p)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _abs_path(s: Optional[str], root: Path) -> Optional[Path]:
    if s is None:
        return None
    path = Path(s)
    return path if path.is_absolute() else root / path


# ------------------------------------------------------------------ catalog
def scene_to_dict(s: SceneRecord, root: Path) -> dict:
    return {
        "scene_id": s.scene_id,
        "source_scene_name": s.source_scene_name,
        "rel_folder": s.rel_folder,
        "band_files": [_rel_or_abs(p, root) for p in s.band_files],
        "pol_tags": list(s.pol_tags),
        "mask_path": _rel_or_abs(s.mask_path, root),
        "mask_present": s.mask_present,
        "is_calibrated": s.is_calibrated,
        "value_domain_hint": s.value_domain_hint,
        "has_dual_pol": s.has_dual_pol,
        "width": s.width,
        "height": s.height,
        "default_label": s.default_label,
        "source_dataset": s.source_dataset,
        "rel_primary": s.rel_primary,
        "value_domains": dict(s.value_domains),
    }


def scene_from_dict(d: dict, root: Path) -> SceneRecord:
    return SceneRecord(
        scene_id=d["scene_id"],
        source_scene_name=d["source_scene_name"],
        rel_folder=d["rel_folder"],
        band_files=[_abs_path(p, root) for p in d["band_files"]],
        pol_tags=list(d["pol_tags"]),
        mask_path=_abs_path(d.get("mask_path"), root),
        mask_present=bool(d.get("mask_present")),
        is_calibrated=bool(d.get("is_calibrated")),
        value_domain_hint=d.get("value_domain_hint", "auto"),
        has_dual_pol=bool(d.get("has_dual_pol")),
        width=int(d["width"]),
        height=int(d["height"]),
        default_label=d.get("default_label", "oil"),
        source_dataset=d.get("source_dataset", ""),
        rel_primary=d.get("rel_primary", ""),
        value_domains=dict(d.get("value_domains") or {}),
    )


def save_catalog(cat: Catalog, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "root": str(cat.root),
        "all_dirs": list(cat.all_dirs),
        "mask_only_folders": list(cat.mask_only_folders),
        "errors": list(cat.errors),
        "scenes": [scene_to_dict(s, cat.root) for s in cat.scenes],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_catalog(path: Path) -> Catalog:
    path = Path(path)
    if not path.is_file():
        raise SystemExit(
            f"state missing: {path}\nRun the 'scan' stage first: "
            f"python sar_dataset_pipeline.py scan --stage_dir <extracted_tree>"
        )
    payload = json.loads(path.read_text())
    root = Path(payload["root"])
    return Catalog(
        root=root,
        all_dirs=list(payload.get("all_dirs") or []),
        scenes=[scene_from_dict(d, root) for d in payload.get("scenes") or []],
        mask_only_folders=list(payload.get("mask_only_folders") or []),
        errors=list(payload.get("errors") or []),
    )


# ---------------------------------------------------------------- instances
def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def save_instances(rows: List[dict], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=INSTANCE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k)) for k in INSTANCE_COLUMNS})
    return path


def _coerce_instance(row: dict) -> dict:
    out = dict(row)
    for key in _INT_FIELDS:
        raw = out.get(key, "")
        if raw is None or raw == "":
            out[key] = 0
        else:
            out[key] = int(float(raw))
    for key in _FLOAT_FIELDS:
        raw = out.get(key, "")
        if raw is None or raw == "":
            out[key] = None
        else:
            try:
                out[key] = float(raw)
            except ValueError:
                out[key] = None
    for key in _BOOL_FIELDS:
        raw = str(out.get(key, "")).strip().lower()
        out[key] = raw in ("true", "1", "y", "yes")
    return out


def load_instances(path: Path) -> List[dict]:
    path = Path(path)
    if not path.is_file():
        raise SystemExit(
            f"state missing: {path}\nRun the 'features' stage first: "
            f"python sar_dataset_pipeline.py features"
        )
    with open(path, newline="") as fh:
        return [_coerce_instance(row) for row in csv.DictReader(fh)]


# --------------------------------------------------------------- scene geo
def save_scene_geo(geo: Dict[str, dict], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(geo, indent=2, default=str))
    return path


def load_scene_geo(path: Path) -> Dict[str, dict]:
    path = Path(path)
    if not path.is_file():
        raise SystemExit(
            f"state missing: {path}\nRun the 'geo' stage first: "
            f"python sar_dataset_pipeline.py geo"
        )
    return json.loads(path.read_text())


# ------------------------------------------------------------- crop plans
def save_crop_plans(plans: Dict[str, List[dict]], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plans, indent=2))
    return path


def load_crop_plans(path: Path) -> Dict[str, List[dict]]:
    path = Path(path)
    if not path.is_file():
        raise SystemExit(
            f"state missing: {path}\nRun the 'plan' stage first: "
            f"python sar_dataset_pipeline.py plan"
        )
    return json.loads(path.read_text())


# -------------------------------------------------------------- crop rows
def save_crop_rows(rows: List[dict], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, default=str))
    return path


def load_crop_rows(path: Path) -> List[dict]:
    path = Path(path)
    if not path.is_file():
        raise SystemExit(
            f"state missing: {path}\nRun the 'convert' stage first: "
            f"python sar_dataset_pipeline.py convert"
        )
    return json.loads(path.read_text())


# ----------------------------------------------------------- convenience
def require(paths: StatePaths, *names: str) -> None:
    """Raise SystemExit listing missing upstream artifacts."""
    missing = []
    label = {
        "catalog": "scan",
        "scene_geo": "geo",
        "instances": "features",
        "crop_plans": "plan",
        "crop_rows": "convert",
    }
    for name in names:
        p = getattr(paths, name)
        if not p.is_file():
            missing.append(f"{p} (run '{label[name]}' first)")
    if missing:
        raise SystemExit("missing pipeline state:\n  " + "\n  ".join(missing))
