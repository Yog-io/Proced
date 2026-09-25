"""Partial-file hashing to catch byte-identical duplicates (architecture §8)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List

from .._progress import bar as progress_bar
from .discover_tree import TreeDiscovery


@dataclass
class DuplicateResult:
    duplicate_groups: List[dict] = field(default_factory=list)
    hashed: int = 0
    errors: List[dict] = field(default_factory=list)


def _partial_hash(path, *, head: int = 65536, tail: int = 65536) -> str:
    import os

    size = os.path.getsize(path)
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(head))
        if size > head + tail:
            fh.seek(size - tail)
            h.update(fh.read(tail))
        elif size > head:
            h.update(fh.read())
    h.update(str(size).encode())
    return h.hexdigest()


def check_duplicates(discovery: TreeDiscovery) -> DuplicateResult:
    res = DuplicateResult()
    by_hash: Dict[str, List[str]] = {}
    pb = progress_bar(len(discovery.rasters), "raw audit hashing", unit="file")
    try:
        for item in discovery.rasters:
            pb.update()
            try:
                key = _partial_hash(item.path)
                by_hash.setdefault(key, []).append(item.rel)
                res.hashed += 1
            except OSError as exc:
                res.errors.append({"rel": item.rel, "error": str(exc)})
    finally:
        pb.close()
    for key, rels in by_hash.items():
        if len(rels) > 1:
            res.duplicate_groups.append({"sha256_partial": key, "files": sorted(rels)})
    res.duplicate_groups.sort(key=lambda g: g["files"][0])
    return res
