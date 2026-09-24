"""Independent VV/VH/mask pairing re-check (architecture §8).

Second implementation — deliberately NOT a call into ``proced/catalog.py``.
Reports orphaned images (no matching mask) and orphaned masks (no matching
image), plus VV/VH split-file pairing gaps within a folder.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .discover_tree import DiscoveredFile, TreeDiscovery

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
    res = PairingResult()
    by_folder: Dict[str, List[DiscoveredFile]] = defaultdict(list)
    for item in discovery.rasters:
        by_folder[item.folder].append(item)

    for folder, items in sorted(by_folder.items()):
        res.folders_checked += 1
        masks = [i for i in items if i.is_mask_by_name]
        images = [i for i in items if not i.is_mask_by_name]

        # image base → mask lookup (same folder only for this pre-check;
        # catalog also looks in masks/labels subdirs — we report same-folder
        # orphans and note mask-dir presence as a soft signal).
        mask_bases = {_mask_base(m.stem) for m in masks}
        image_bases: Dict[str, List[DiscoveredFile]] = defaultdict(list)
        for img in images:
            base, _pol = strip_pol_suffix(img.stem)
            image_bases[base.lower()].append(img)

        for base, imgs in image_bases.items():
            if base in mask_bases:
                res.matched_pairs += 1
            else:
                for img in imgs:
                    res.orphan_images.append({
                        "rel": img.rel,
                        "folder": folder,
                        "base": base,
                        "note": "no mask with matching base in same folder",
                    })

        # Reverse: masks with no image base in same folder.
        for m in masks:
            mb = _mask_base(m.stem)
            # also try full stem without pol (e.g. mask name == scene name)
            if mb not in image_bases and m.stem.lower() not in image_bases:
                # soft: mask-only folders are legal (tree equivalence) — flag
                # as orphan only if the folder has images at all.
                if images:
                    res.orphan_masks.append({
                        "rel": m.rel,
                        "folder": folder,
                        "base": mb,
                        "note": "mask present but no image with matching base",
                    })

        # VV/VH split pairing within folder
        pol_map: Dict[str, set] = defaultdict(set)
        for img in images:
            base, pol = strip_pol_suffix(img.stem)
            if pol:
                pol_map[base.lower()].add(pol)
        for base, pols in sorted(pol_map.items()):
            if len(pols) == 1 and ("VV" in pols or "VH" in pols):
                # single pol file is OK (single-pol scene); only flag if
                # the other pol exists elsewhere? For pre-audit, note as info.
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
