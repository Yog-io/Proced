"""Independent VV/VH/mask pairing re-check (architecture §8).

Second implementation — deliberately NOT a call into ``proced/catalog.py``.
Reports orphaned images (no matching mask) and orphaned masks (no matching
image), plus VV/VH split-file pairing gaps within a folder.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .discover_tree import (
    DiscoveredFile,
    TreeDiscovery,
    parallel_image_folder,
    parallel_mask_folder,
)

_POL_SUFFIX_RE = re.compile(r"(?:^|[_\-.])(vv|vh)$", re.IGNORECASE)


def strip_pol_suffix(stem: str) -> Tuple[str, Optional[str]]:
    m = _POL_SUFFIX_RE.search(stem)
    if m:
        base = stem[: m.start()] if m.start() > 0 else stem[: m.end()]
        base = base.rstrip("_-. ")
        return base, m.group(1).upper()
    return stem, None


def _mask_base(stem: str) -> str:
    """Reduce a mask stem to the scene base it pairs with."""
    low = stem.lower()
    for suf in ("_mask", "_label", "_labels", "_gt", "_lbl", "_ann"):
        if low.endswith(suf):
            stem = stem[: -len(suf)]
            break
    for pre in ("mask_", "label_", "labels_", "gt_", "lbl_", "ann_"):
        if stem.lower().startswith(pre):
            stem = stem[len(pre):]
            break
    stem_nopol, _ = strip_pol_suffix(stem)
    return stem_nopol.lower()


@dataclass
class PairingResult:
    orphan_images: List[dict] = field(default_factory=list)
    orphan_masks: List[dict] = field(default_factory=list)
    pol_pair_gaps: List[dict] = field(default_factory=list)  # VV without VH etc.
    matched_pairs: int = 0
    folders_checked: int = 0


def check_pairing(discovery: TreeDiscovery) -> PairingResult:
    """Pair images↔masks within a folder AND across parallel sibling trees.

    ``01_…_images/pass/scene.tif`` ↔ ``01_…_mask/pass/scene.tif`` (and the
    reverse) count as a match; mask-only trees are not orphan-flagged when a
    parallel image tree holds the matching base.
    """
    res = PairingResult()
    by_folder: Dict[str, List[DiscoveredFile]] = defaultdict(list)
    for item in discovery.rasters:
        by_folder[item.folder].append(item)

    # Global mask bases by folder (for cross-tree sibling lookup)
    mask_bases_by_folder: Dict[str, set] = defaultdict(set)
    image_bases_by_folder: Dict[str, Dict[str, List[DiscoveredFile]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for folder, items in by_folder.items():
        for i in items:
            if i.is_mask_by_name:
                mask_bases_by_folder[folder].add(_mask_base(i.stem))
            else:
                base, _pol = strip_pol_suffix(i.stem)
                image_bases_by_folder[folder][base.lower()].append(i)

    def _mask_folders_for(image_folder: str) -> List[str]:
        folders = [image_folder]
        par = parallel_mask_folder(image_folder)
        if par and par != image_folder:
            folders.append(par)
        # conventional sibling mask dirs under same parent
        parent = str(Path(image_folder).parent) if image_folder not in ("", ".") else ""
        leaf = Path(image_folder).name if image_folder not in ("", ".") else ""
        for name in ("masks", "mask", "labels", "label", "gt", "annotations"):
            cand = f"{parent}/{name}" if parent else name
            if cand not in folders:
                folders.append(cand)
        return folders

    def _image_folders_for(mask_folder: str) -> List[str]:
        folders = [mask_folder]
        par = parallel_image_folder(mask_folder)
        if par and par != mask_folder:
            folders.append(par)
        return folders

    for folder, items in sorted(by_folder.items()):
        res.folders_checked += 1
        masks = [i for i in items if i.is_mask_by_name]
        images = [i for i in items if not i.is_mask_by_name]

        mask_bases = mask_bases_by_folder.get(folder, set())
        image_bases = image_bases_by_folder.get(folder, {})

        # Union of mask bases reachable from this image folder (same + sibling)
        reachable_mask_bases: set = set()
        for mf in _mask_folders_for(folder):
            reachable_mask_bases |= mask_bases_by_folder.get(mf, set())

        for base, imgs in image_bases.items():
            if base in reachable_mask_bases:
                res.matched_pairs += 1
            else:
                for img in imgs:
                    res.orphan_images.append({
                        "rel": img.rel,
                        "folder": folder,
                        "base": base,
                        "note": "no mask with matching base in same or parallel mask folder",
                    })

        # Reverse: masks with no image base in same folder OR parallel image folder
        for m in masks:
            mb = _mask_base(m.stem)
            found = mb in image_bases or m.stem.lower() in image_bases
            if not found:
                for imf in _image_folders_for(folder):
                    ib = image_bases_by_folder.get(imf, {})
                    if mb in ib or m.stem.lower() in ib:
                        found = True
                        break
            if not found and images:
                # mask-only folder with a parallel image tree is fine;
                # only flag when this folder itself has images (mixed, unmatched)
                res.orphan_masks.append({
                    "rel": m.rel,
                    "folder": folder,
                    "base": mb,
                    "note": "mask present but no image with matching base in same or parallel image folder",
                })

        # VV/VH split pairing within folder
        pol_map: Dict[str, set] = defaultdict(set)
        for img in images:
            base, pol = strip_pol_suffix(img.stem)
            if pol:
                pol_map[base.lower()].add(pol)
        for base, pols in sorted(pol_map.items()):
            if len(pols) == 1 and ("VV" in pols or "VH" in pols):
                pass
            if "VV" in pols and "VH" not in pols:
                res.pol_pair_gaps.append({
                    "folder": folder, "base": base,
                    "have": sorted(pols), "missing": "VH",
                })
            if "VH" in pols and "VV" not in pols:
                res.pol_pair_gaps.append({
                    "folder": folder, "base": base,
                    "have": sorted(pols), "missing": "VV",
                })

    return res
