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


def _dir_tokens(name: str) -> set:
    return {t for t in re.split(r"[_\-\s.]+", name.lower()) if t}


def _is_mask_dir_name(name: str) -> bool:
    """Folder name marks a mask tree (``*_mask``, ``masks``, ``labels``, …)."""
    toks = _dir_tokens(name)
    if toks & set(MASK_NAME_TOKENS):
        return True
    low = name.lower()
    return (
        low.endswith(("_mask", "_masks", "_label", "_labels", "_gt"))
        or low.startswith(("mask_", "masks_", "label_", "labels_"))
        or low in ("mask", "masks", "gt", "labels")
    )


def _is_image_dir_name(name: str) -> bool:
    toks = _dir_tokens(name)
    return bool(toks & {"images", "image", "imgs", "img", "scenes", "scene", "sar"})


def _rel_parent(rel_path: str) -> str:
    if not rel_path:
        return ""
    p = str(Path(rel_path).parent)
    return "" if p == "." else p.replace("\\", "/")


def _is_mask_rel(rel_path: str, stem: str) -> bool:
    """Mask if stem has mask tokens OR any ancestor folder is a mask tree."""
    if _is_mask_name(stem):
        return True
    parent = _rel_parent(rel_path)
    if not parent:
        return False
    return any(_is_mask_dir_name(part) for part in parent.split("/"))


def _strip_mask_tokens(stem: str) -> str:
    low = stem
    for suf in ("_mask", "_masks", "_label", "_labels", "_gt", "_lbl", "_ann"):
        if low.lower().endswith(suf):
            low = low[: -len(suf)]
            break
    for pre in ("mask_", "masks_", "label_", "labels_", "gt_", "lbl_", "ann_"):
        if low.lower().startswith(pre):
            low = low[len(pre):]
            break
    return low


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
    if "oil_spill" in low or "oil-spill" in low or "oilspill" in low:
        return "oil"
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


def _stem_key(stem: str) -> str:
    """Normalized pairing key: no pol suffix, no mask tokens, lowercased."""
    base, _ = _strip_pol_suffix(stem)
    base = _strip_mask_tokens(base)
    return base.lower()


def _parallel_mask_rel(rel_folder: str) -> Optional[str]:
    """Map an image-tree folder to its parallel mask-tree folder.

    ``01_…_images/sub`` → ``01_…_mask/sub`` (any depth; first matching
    image-like component is swapped). Returns None if nothing to swap.
    """
    if not rel_folder:
        return None
    parts = rel_folder.split("/")
    out: List[str] = []
    swapped = False
    img_to_mask = (
        ("_images_and_ground_truth", "_mask"),
        ("_images", "_mask"),
        ("_image", "_mask"),
        ("_imgs", "_mask"),
        ("_img", "_mask"),
    )
    for part in parts:
        low = part.lower()
        new = part
        for suf_img, suf_mask in img_to_mask:
            if low.endswith(suf_img):
                new = part[: -len(suf_img)] + suf_mask
                break
        else:
            if low in ("images", "image", "imgs", "img"):
                new = "mask" if part.islower() else ("Mask" if part[:1].isupper() else "mask")
        if new != part:
            swapped = True
        out.append(new)
    return "/".join(out) if swapped else None


def _mask_search_dirs(rel_folder: str) -> List[str]:
    """Candidate relative dirs that may hold the mask for this image folder."""
    dirs: List[str] = [rel_folder] if rel_folder else [""]
    seen = set(dirs)

    def _add(d: Optional[str]) -> None:
        if d is not None and d not in seen:
            seen.add(d)
            dirs.append(d)

    _add(_parallel_mask_rel(rel_folder))
    # conventional sibling names at the same parent
    parent = _rel_parent(rel_folder) if rel_folder else ""
    leaf = rel_folder.split("/")[-1] if rel_folder else ""
    for name in ("masks", "mask", "labels", "label", "gt", "annotations"):
        _add(f"{parent}/{name}" if parent else name)
        # swap image leaf → mask leaf under same parent
        if leaf and _is_image_dir_name(leaf):
            base = re.sub(
                r"(_images|_image|_imgs|_img|images|image|imgs|img)$",
                "",
                leaf,
                flags=re.IGNORECASE,
            )
            _add(f"{parent}/{base}_{name}" if parent else f"{base}_{name}")
            _add(f"{parent}/{name}_{base}" if parent else f"{name}_{base}")
    # parent itself if it is a mask tree (image nested under mask root — rare)
    if rel_folder and any(_is_mask_dir_name(p) for p in rel_folder.split("/")):
        _add(rel_folder)
    return dirs


def _find_mask(
    scene_base: str,
    folder: Path,
    mask_files: List[Path],
    *,
    root: Optional[Path] = None,
    rel_folder: str = "",
    mask_index: Optional[Dict[str, List[Path]]] = None,
) -> Optional[Path]:
    """Several real-world naming conventions for mask files."""
    base_l = scene_base.lower()
    base_key = _stem_key(scene_base)
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
        elif _stem_key(mf.stem) == base_key:
            candidates.append(mf)
    # Prefer same-folder, then parallel mask tree, then conventional subdirs
    if candidates:
        def _rank(p: Path) -> tuple:
            try:
                rel = str(p.relative_to(root)) if root else str(p)
            except ValueError:
                rel = str(p)
            par = _rel_parent(rel.replace("\\", "/"))
            same = 0 if (root and folder in p.parents) or p.parent == folder else 1
            parallel = 0 if _parallel_mask_rel(rel_folder) == par else 1
            return (same, parallel, len(p.parts))
        candidates.sort(key=_rank)
        return candidates[0]

    # Sibling / conventional directories (with or without prebuilt index)
    wanted_names = {
        f"{base_l}.tif", f"{base_l}.tiff", f"{base_l}.png", f"{base_l}.jp2",
        f"{base_l}_mask.tif", f"{base_l}_mask.tiff", f"{base_l}_mask.png",
        f"mask_{base_l}.tif", f"mask_{base_l}.png",
        f"{base_l}_label.tif", f"{base_l}_label.png",
        f"{base_l}_gt.tif", f"{base_l}_gt.png",
    }
    if root is not None:
        for rel_dir in _mask_search_dirs(rel_folder):
            d = root / rel_dir if rel_dir else root
            if not d.is_dir():
                continue
            for mf in sorted(d.iterdir()):
                if not _is_raster(mf):
                    continue
                rel_cand = str(Path(rel_dir) / mf.name) if rel_dir else mf.name
                # Only consider files classified as masks (stem or mask-tree folder).
                # Never treat the image itself as its own mask.
                if not (_is_mask_rel(rel_cand, mf.stem) or _is_mask_name(mf.stem)):
                    continue
                if _stem_key(mf.stem) == base_key or mf.name.lower() in wanted_names:
                    return mf

    # Global stem index (built over entire tree) — prefer parallel mask dirs
    if mask_index:
        hits = mask_index.get(base_key) or mask_index.get(base_l) or []
        if hits:
            par = _parallel_mask_rel(rel_folder)
            if par:
                for h in hits:
                    try:
                        h_rel = str(h.relative_to(root)) if root else str(h)
                    except ValueError:
                        h_rel = str(h)
                    if _rel_parent(h_rel.replace("\\", "/")) == par or h_rel.replace("\\", "/").startswith(par + "/"):
                        return h
            # Prefer any path under a mask-named folder
            for h in hits:
                try:
                    h_rel = str(h.relative_to(root)) if root else str(h)
                except ValueError:
                    h_rel = str(h)
                if _is_mask_rel(h_rel.replace("\\", "/"), h.stem):
                    return h
            return hits[0]
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

    # Global mask index by normalized stem key (for cross-tree sibling pairing)
    mask_index: Dict[str, List[Path]] = {}
    for rel, _folder, rasters in folder_scenes_raw:
        for p in rasters:
            if _is_mask_rel(str(Path(rel) / p.name) if rel else p.name, p.stem):
                key = _stem_key(p.stem)
                mask_index.setdefault(key, []).append(p)

    # First pass: build per-folder (base → band files), masks
    prelim: List[dict] = []
    for rel, folder, rasters in folder_scenes_raw:
        masks = [
            p for p in rasters
            if _is_mask_rel(str(Path(rel) / p.name) if rel else p.name, p.stem)
        ]
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
                mask_path = _find_mask(
                    base, folder, masks,
                    root=root, rel_folder=rel, mask_index=mask_index,
                )
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