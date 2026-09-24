"""Recursive tree walk with no depth/naming assumptions (architecture §8)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set

# Local copies — do NOT import from proced/config.py.
RASTER_EXTENSIONS = {".tif", ".tiff", ".png", ".jp2", ".img", ".jpg", ".jpeg"}
MASK_NAME_TOKENS = ("mask", "label", "labels", "gt", "lbl", "ann")
# Images that are almost always masks under the naming conventions we care about.
MASK_SUFFIX_TOKENS = ("_mask", "_label", "_labels", "_gt", "_lbl", "_ann")


@dataclass
class DiscoveredFile:
    path: Path
    rel: str
    folder: str  # parent relative path ("" for root)
    suffix: str
    stem: str
    is_mask_by_name: bool
    size_bytes: int


@dataclass
class TreeDiscovery:
    root: Path
    files: List[DiscoveredFile] = field(default_factory=list)
    rasters: List[DiscoveredFile] = field(default_factory=list)
    masks: List[DiscoveredFile] = field(default_factory=list)
    images: List[DiscoveredFile] = field(default_factory=list)  # non-mask rasters
    all_dirs: List[str] = field(default_factory=list)
    file_count_by_folder: Dict[str, int] = field(default_factory=dict)
    total_bytes: int = 0
    errors: List[str] = field(default_factory=list)


def is_mask_stem(stem: str) -> bool:
    toks = [t for t in stem.replace("-", "_").replace(".", "_").split("_") if t]
    if any(t.lower() in MASK_NAME_TOKENS for t in toks):
        return True
    low = stem.lower()
    return any(low.endswith(suf) for suf in MASK_SUFFIX_TOKENS)


def discover_tree(root: Path) -> TreeDiscovery:
    root = Path(root)
    out = TreeDiscovery(root=root)
    if not root.is_dir():
        out.errors.append(f"root is not a directory: {root}")
        return out

    seen_dirs: Set[str] = set()
    for dirpath, dirnames, filenames in root.walk() if hasattr(root, "walk") else _os_walk(root):
        dirnames.sort()
        rel_dir = str(Path(dirpath).relative_to(root)) if Path(dirpath) != root else ""
        if rel_dir != "." and rel_dir not in seen_dirs:
            seen_dirs.add(rel_dir)
            out.all_dirs.append(rel_dir)

        for name in sorted(filenames):
            if name.startswith("."):
                continue
            p = Path(dirpath) / name
            try:
                size = p.stat().st_size
            except OSError as exc:
                out.errors.append(f"{p}: {exc}")
                continue
            suffix = p.suffix.lower()
            if suffix not in RASTER_EXTENSIONS:
                continue
            rel = str(p.relative_to(root))
            folder = rel_dir if rel_dir != "." else ""
            stem = p.stem
            item = DiscoveredFile(
                path=p, rel=rel, folder=folder, suffix=suffix, stem=stem,
                is_mask_by_name=is_mask_stem(stem), size_bytes=size,
            )
            out.files.append(item)
            out.rasters.append(item)
            out.total_bytes += size
            out.file_count_by_folder[folder] = out.file_count_by_folder.get(folder, 0) + 1
            if item.is_mask_by_name:
                out.masks.append(item)
            else:
                out.images.append(item)

    out.files.sort(key=lambda f: f.rel)
    out.rasters.sort(key=lambda f: f.rel)
    out.masks.sort(key=lambda f: f.rel)
    out.images.sort(key=lambda f: f.rel)
    return out


def _os_walk(root: Path):
    import os
    for dirpath, dirnames, filenames in os.walk(root):
        yield dirpath, dirnames, filenames
