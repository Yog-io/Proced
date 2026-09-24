"""Group-aware train/val/test splitting (guide Task 1.5 / refinement C.5).

Crops from the SAME original scene must never land in different splits —
that leaks the physical spill event and inflates val/test scores. We group by
``crop_source_scene_id`` and stratify by the group's dominant label.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from .config import PipelineConfig

SPLIT_NAMES = ("train", "val", "test")


def _dominant_label(labels: Sequence[str]) -> str:
    positives = [l for l in labels if l in ("oil", "lookalike")]
    if positives:
        counts = Counter(positives)
        # oil beats lookalike on ties (deterministic)
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0] == "lookalike", kv[0]))[0][0]
    return "background"


def assign_splits(
    rows: Sequence[Dict],
    cfg: PipelineConfig,
) -> Tuple[Dict[str, List[str]], List[Dict]]:
    """Return {split: [image_id, ...]} plus a per-crop manifest.

    Guarantees: every crop assigned exactly once; no group spans two splits;
    per-label ratios approximately match the configured split ratios.
    """
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        gid = str(r.get("crop_source_scene_id") or r.get("image_id") or r.get("crop_id"))
        groups[gid].append(r)

    ratios = {"train": cfg.split_train, "val": cfg.split_val, "test": cfg.split_test}
    total = sum(ratios.values()) or 1.0
    ratios = {k: v / total for k, v in ratios.items()}

    rng = random.Random(cfg.seed)

    # Bucket groups by dominant label → stratify within each bucket
    buckets: Dict[str, List[str]] = defaultdict(list)
    for gid, grows in groups.items():
        buckets[_dominant_label([str(g.get("label", "background")) for g in grows])].append(gid)

    out: Dict[str, List[str]] = {s: [] for s in SPLIT_NAMES}
    manifest: List[Dict] = []

    for label in sorted(buckets.keys()):
        gids = buckets[label]
        rng.shuffle(gids)
        n = len(gids)
        n_train = int(round(n * ratios["train"]))
        n_val = int(round(n * ratios["val"]))
        # remainder → test
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        n_test = n - n_train - n_val
        # Guarantee non-empty val/test when there are ≥3 groups
        if n >= 3:
            if n_val == 0:
                n_val = 1
                n_train = max(1, n_train - 1) if n_train > 1 else n_train
            if n_test == 0:
                n_test = 1
                if n_train > 1:
                    n_train -= 1
                elif n_val > 1:
                    n_val -= 1
        assignment: Dict[str, str] = {}
        i = 0
        for gid in gids[i:i + n_train]:
            assignment[gid] = "train"
        i += n_train
        for gid in gids[i:i + n_val]:
            assignment[gid] = "val"
        i += n_val
        for gid in gids[i:i + n_test]:
            assignment[gid] = "test"

        for gid, split in assignment.items():
            for g in groups[gid]:
                image_id = str(g.get("crop_id") or g.get("image_id"))
                out[split].append(image_id)
                manifest.append({
                    "image_id": image_id,
                    "split": split,
                    "crop_source_scene_id": gid,
                    "label": g.get("label", ""),
                })

    _assert_no_leakage(groups, assignment=None, manifest=manifest)
    for s in out:
        out[s].sort()
    return out, manifest


def _assert_no_leakage(groups, assignment, manifest) -> None:
    seen: Dict[str, str] = {}
    for m in manifest:
        gid = m["crop_source_scene_id"]
        if gid in seen and seen[gid] != m["split"]:
            raise AssertionError(
                f"LEAKAGE: scene group {gid} spans splits {seen[gid]} and {m['split']}"
            )
        seen[gid] = m["split"]


def write_splits(splits_dir: Path, splits: Dict[str, List[str]], manifest: List[Dict]) -> Dict[str, Path]:
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for name in SPLIT_NAMES:
        p = splits_dir / f"{name}.csv"
        pd.DataFrame({"image_id": splits.get(name, [])}).to_csv(p, index=False)
        paths[name] = p
    man = splits_dir / "manifest.csv"
    pd.DataFrame(manifest, columns=["image_id", "split", "crop_source_scene_id", "label"]).to_csv(man, index=False)
    paths["manifest"] = man
    return paths


def split_from_metadata(metadata_csv: Path, cfg: PipelineConfig) -> Dict[str, Path]:
    """(Re)build splits from an existing master metadata.csv."""
    df = pd.read_csv(metadata_csv).fillna("")
    rows = df.to_dict("records")
    splits, manifest = assign_splits(rows, cfg)
    return write_splits(cfg.splits_dir, splits, manifest)
