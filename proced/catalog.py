"""Stage 1 — tree parsing & dependency discovery.

Walks the extracted tree, pairs polarization files (VV/VH split across files or
stacked as bands), pairs scene↔mask files with several naming conventions,
detects calibration/dual-pol, resolves label defaults per dataset, mirrors the
full directory tree (including empty / mask-only folders — required by the
archi.md directory-equivalence check), and disambiguates globally-colliding
scene stems so crop_ids stay unique.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import MASK_NAME_TOKENS, PipelineConfig, RASTER_EXTENSIONS
from .raster_io import open_any, probe_profile

_POL_RE = re.compile(r"(?:^|[_\-.])(vv|vh|v v|v h)(?:$|[_\-.])", re.IGNORECASE)


@dataclass
class SceneRecord:
    scene_id: str                 # globally-unique id used in crop_id
    source_scene_name: str        # original file stem (spec: crop_source_scene_id)
    rel_folder: str               # folder relative to stage root
    band_files: List[Path]        # ordered: one entry per stacked band OR per pol file
    pol_tags: List[str]           # e.g. ["VV"] / ["VV", "VH"]
    mask_path: Optional[Path]
    mask_present: bool
    is_calibrated: bool
    value_domain_hint: str        # "auto" unless config forces
    has_dual_pol: bool
    width: int
    height: int
    default_label: str            # oil | lookalike (for binary masks)
    source_dataset: str           # top-level folder under stage root
    rel_primary: str              # rel path of first band file (provenance)
    # Resolved during scan (center-sample domain detection for calibrated bands)
    value_domains: Dict[str, str] = field(default_factory=dict)


@dataclass
class Catalog:
    root: Path
    all_dirs: List[str]           # every relative dir under root (tree mirror)
    scenes: List[SceneRecord] = field(default_factory=list)
    mask_only_folders: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def folder_to_scenes(self) -> Dict[str, List[SceneRecord]]:
        out: Dict[str, List[SceneRecord]] = {}
        for s in self.scenes:
            out.setdefault(s.rel_folder, []).append(s)
        return out


def _is_raster(p: Path) -> bool:
    return p.suffix.lower() in RASTER_EXTENSIONS and not p.name.startswith(".")


def _is_mask_name(stem: str) -> bool:
    toks = re.split(r"[_\-.]+", stem.lower())
    return any(t in MASK_NAME_TOKENS for t in toks)


def _strip_pol_suffix(stem: str) -> Tuple[str, Optional[str]]:
    """('scene_VV', 'scene', 'VV') from pol-suffixed stems; pol = None otherwise."""
    m = re.search(r"(?:^|[_\-.])(vv|vh)$", stem, re.IGNORECASE)
    if m:
        base = stem[: m.start()] if m.start() > 0 else stem[: m.end()]
        base = base.rstrip("_-. ")
        return base, m.group(1).upper()
    return stem, None


def _dataset_default_label(rel_path: str, cfg: PipelineConfig) -> str:
    low = rel_path.lower()
    for key, label in (cfg.label_overrides or {}).items():
        if key.lower() in low:
            return label
    if "lookalike" in low or "look-alike" in low or "look_alike" in low:
        return "lookalike"
    return "oil"


def _calibration_for(rel_path: str, dtype: str, cfg: PipelineConfig) -> bool:
    low = rel_path.lower()
    for pat in cfg.uncalibrated_match:
        if pat.lower() in low:
            return False
    for pat in cfg.calibrated_match:
        if pat.lower() in low:
            return True
    # Heuristic: calibrated σ⁰ ships as float GeoTIFF; visual exports are int.
    return dtype.startswith("float")


def _find_mask(scene_base: str, folder: Path, mask_files: List[Path]) -> Optional[Path]:
    """Several real-world naming conventions for mask files."""
    base_l = scene_base.lower()
    candidates: List[Path] = []
    for mf in mask_files:
        stem = mf.stem.lower()
        stem_nopol, _ = _strip_pol_suffix(stem)
        # exact base pairs: scene↔scene_mask, mask_scene, scene_label, labels/scene
        if stem == base_l or stem_nopol == base_l:
            candidates.append(mf)
        elif stem.replace("_mask", "").replace("_label", "").replace("_gt", "") == base_l:
            candidates.append(mf)
        elif stem.endswith("_mask") and stem[: -len("_mask")] == base_l:
            candidates.append(mf)
        elif stem.startswith("mask_") and stem[len("mask_"):] == base_l:
            candidates.append(mf)
        elif stem == f"{base_l}_mask" or stem == f"mask_{base_l}":
            candidates.append(mf)
        elif stem in (f"{base_l}_label", f"label_{base_l}", f"{base_l}_gt", f"{base_l}_ann"):
            candidates.append(mf)
    # Prefer same-folder, then masks/labels/gt sibling dirs
    if candidates:
        candidates.sort(key=lambda p: (p.parent != folder, len(p.parts)))
        return candidates[0]
    # Sibling conventional subfolders
    for sub in ("masks", "mask", "labels", "label", "gt", "annotations"):
        d = folder / sub
        if d.is_dir():
            for mf in sorted(d.iterdir()):
                if _is_raster(mf) and _strip_pol_suffix(mf.stem)[0].lower() == base_l:
                    return mf
    return None


def _stacked_band_info(path: Path, pol_tags: List[str]) -> Tuple[List[Path], List[str], dict]:
    prof = probe_profile(path)
    tags = pol_tags
    if len(tags) == 0:
        if prof["count"] >= 2:
            tags = ["VV", "VH"] + [f"B{i+1}" for i in range(2, prof["count"])]
        else:
            tags = ["VV"]
    return [path], tags, prof


def scan_tree(cfg: PipelineConfig) -> Catalog:
    root = Path(cfg.stage_dir)
    cat = Catalog(root=root, all_dirs=[])
    if not root.is_dir():
        cat.errors.append(f"stage_dir does not exist: {root}")
        return cat

    import os
    all_dirs: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        rel = "" if rel == "." else rel
        all_dirs.append(rel)
    cat.all_dirs = sorted(all_dirs)

    # Collect rasters per folder
    folder_scenes_raw: List[Tuple[str, Path, List[Path]]] = []
    for rel in cat.all_dirs:
        folder = root / rel if rel else root
        rasters = sorted(p for p in folder.iterdir() if p.is_file() and _is_raster(p)) if folder.is_dir() else []
        if rasters:
            folder_scenes_raw.append((rel, folder, rasters))

    # First pass: build per-folder (base → band files), masks
    prelim: List[dict] = []
    for rel, folder, rasters in folder_scenes_raw:
        masks = [p for p in rasters if _is_mask_name(p.stem)]
        non_masks = [p for p in rasters if p not in masks]
        # Group pol-split files by base stem
        groups: Dict[str, Dict[str, Path]] = {}
        stacked: List[Path] = []
        for p in non_masks:
            base, pol = _strip_pol_suffix(p.stem)
            if pol is not None:
                groups.setdefault(base, {})[pol] = p
            else:
                stacked.append(p)
        # A stacked file whose stem also has pol-splitted siblings? keep both tracks
        bases_pol = set(groups.keys())
        scene_entries: List[Tuple[str, Optional[List[Path]], Optional[List[str]], Optional[Path]]] = []
        for base, polmap in groups.items():
            tags = sorted(polmap.keys(), key=lambda t: (t != "VV", t))  # VV first
            scene_entries.append((base, [polmap[t] for t in tags], tags, None))
        for p in stacked:
            base, _ = _strip_pol_suffix(p.stem)
            # If a pol-split group with same base exists, skip the stacked one (dup)
            if base in bases_pol:
                continue
            scene_entries.append((base, None, None, p))
        prelim.append({"rel": rel, "folder": folder, "masks": masks, "entries": scene_entries,
                       "all_rasters": rasters})

    # Global stem uniqueness for crop_id disambiguation
    stem_counter: Counter = Counter()
    for item in prelim:
        for base, _, _, _ in item["entries"]:
            stem_counter[base] += 1

    def _folder_slug(rel: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "-", rel).strip("-").replace("/", "-") if rel else "root"

    def _unique_id(rel: str, base: str) -> str:
        if stem_counter[base] == 1:
            return base
        return f"{_folder_slug(rel)}__{base}"

    # Second pass: probe profiles, build SceneRecords
    for item in prelim:
        rel, folder, masks = item["rel"], item["folder"], item["masks"]
        source_dataset = rel.split("/")[0] if rel else root.name
        default_label = _dataset_default_label(rel or source_dataset, cfg)
        for base, band_paths, pol_tags, stacked_path in item["entries"]:
            try:
                if band_paths is not None:
                    # Probe first band file for geometry; count total bands
                    prof = probe_profile(band_paths[0])
                    tags = [t for t in pol_tags if t in ("VV", "VH")] or pol_tags
                    files = band_paths
                    count = len(band_paths)
                else:
                    files, tags, prof = _stacked_band_info(stacked_path, [])
                    count = prof["count"]
                has_dual = len([t for t in tags if t in ("VV", "VH")]) >= 2 or count >= 2
                rel_primary = str((Path(rel) / files[0].name) if rel else files[0].name)
                is_calibrated = _calibration_for(rel_primary, prof["dtype"], cfg)
                mask_path = _find_mask(base, folder, masks)
                # Mask-only folders: rasters all masks → no scene entries — handled above
                cat.scenes.append(SceneRecord(
                    scene_id=_unique_id(rel, base),
                    source_scene_name=base,
                    rel_folder=rel,
                    band_files=files,
                    pol_tags=tags if tags else ["VV"],
                    mask_path=mask_path,
                    mask_present=mask_path is not None,
                    is_calibrated=is_calibrated,
                    value_domain_hint="auto",
                    has_dual_pol=bool(has_dual),
                    width=int(prof["width"]),
                    height=int(prof["height"]),
                    default_label=default_label,
                    source_dataset=source_dataset,
                    rel_primary=rel_primary,
                ))
            except Exception as exc:  # corrupt header etc.
                cat.errors.append(f"{rel}/{base}: {exc}")

    # Folders that contain only masks (e.g. reference_masks/) — record for reporting
    for item in prelim:
        if not item["entries"] and item["masks"]:
            cat.mask_only_folders.append(item["rel"])

    return cat
