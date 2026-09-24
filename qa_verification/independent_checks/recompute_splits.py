"""Stage 6 — independent train/val/test verification (architecture §2).

Recomputes group leakage and stratification sanity purely from the split
CSVs + metadata.csv with pandas — never trusts the pipeline's internal
assertion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

from .._lib import QAPaths, Report, load_metadata

SPLIT_NAMES = ("train", "val", "test")


def _load_split(paths: QAPaths, name: str) -> Optional[pd.DataFrame]:
    p = paths.splits_dir / f"{name}.csv"
    if not p.is_file():
        return None
    return pd.read_csv(p).fillna("")


def run(
    paths: QAPaths,
    *,
    ratios: Optional[Dict[str, float]] = None,
    ratio_tol: float = 0.20,
) -> Report:
    rep = Report("stage6_splits")
    skip = paths.require("splits")
    if skip:
        rep.hard(False, f"skip: {skip}")
        return rep

    frames = {n: _load_split(paths, n) for n in SPLIT_NAMES}
    missing = [n for n, f in frames.items() if f is None]
    if missing:
        rep.hard(False, f"missing split CSVs: {missing}")
        return rep

    # --- every image_id appears exactly once ------------------------------
    all_ids: List[str] = []
    per_split: Dict[str, Set[str]] = {}
    for n, f in frames.items():
        if "image_id" not in f.columns:
            rep.hard(False, f"{n}.csv missing image_id column")
            return rep
        ids = [str(x) for x in f["image_id"].tolist()]
        per_split[n] = set(ids)
        all_ids.extend(ids)

    dupes = sorted({i for i in all_ids if all_ids.count(i) > 1})
    # O(n) instead of count-per-item for large sets
    from collections import Counter
    counts = Counter(all_ids)
    dupes = sorted([k for k, c in counts.items() if c > 1])
    rep.hard(
        not dupes,
        "every image_id in exactly one split" if not dupes
        else f"{len(dupes)} image_ids appear in multiple/overlapping splits",
        examples=dupes[:20],
        n_total=len(all_ids),
        n_unique=len(set(all_ids)),
    )

    overlaps = []
    for a in SPLIT_NAMES:
        for b in SPLIT_NAMES:
            if a >= b:
                continue
            inter = per_split[a] & per_split[b]
            if inter:
                overlaps.append({"splits": [a, b], "n": len(inter),
                                 "examples": sorted(inter)[:10]})
    rep.hard(
        not overlaps,
        "no image_id overlap between split pairs" if not overlaps
        else f"{len(overlaps)} split-pair overlaps",
        overlaps=overlaps,
    )

    # --- group leakage via manifest (primary) or metadata join ------------
    man_p = paths.splits_dir / "manifest.csv"
    if man_p.is_file():
        man = pd.read_csv(man_p).fillna("")
        if "crop_source_scene_id" not in man.columns or "split" not in man.columns:
            rep.hard(False, "manifest.csv missing crop_source_scene_id/split")
        else:
            # independent recompute: each scene group → exactly one split value
            bad = []
            for gid, grp in man.groupby("crop_source_scene_id"):
                splits = sorted({str(s) for s in grp["split"]})
                if len(splits) != 1:
                    bad.append({"group": str(gid), "splits": splits,
                                "n": int(len(grp))})
            rep.hard(
                not bad,
                f"no group leakage across {man['crop_source_scene_id'].nunique()} "
                "scene groups (manifest recompute)"
                if not bad else f"{len(bad)} scene groups span multiple splits",
                examples=bad[:20],
            )
            # manifest coverage vs split files
            man_ids = {str(x) for x in man["image_id"]}
            split_ids = set().union(*per_split.values()) if per_split else set()
            only_man = man_ids - split_ids
            only_split = split_ids - man_ids
            rep.hard(
                not only_man and not only_split,
                "manifest image_ids match union of split CSVs"
                if not only_man and not only_split
                else "manifest ↔ split CSV id mismatch",
                only_manifest=sorted(only_man)[:10],
                only_splits=sorted(only_split)[:10],
            )
    else:
        # fallback: join metadata on image_id/crop_id via split membership
        if paths.dest_dir.joinpath("metadata.csv").is_file():
            md = load_metadata(paths)
            id_col = "crop_id" if "crop_id" in md.columns else "image_id"
            scene_col = "crop_source_scene_id"
            if scene_col in md.columns and id_col in md.columns:
                assign: Dict[str, str] = {}
                for n, s in per_split.items():
                    for i in s:
                        assign[i] = n
                md = md.copy()
                md["_split"] = md[id_col].astype(str).map(assign)
                md = md[md["_split"] != ""]
                bad = []
                for gid, grp in md.groupby(scene_col):
                    splits = sorted({str(s) for s in grp["_split"]})
                    if len(splits) != 1:
                        bad.append({"group": str(gid), "splits": splits})
                rep.hard(
                    not bad,
                    "no group leakage (metadata join recompute)" if not bad
                    else f"{len(bad)} groups leak via metadata join",
                    examples=bad[:20],
                )
            else:
                rep.soft(False, "metadata lacks id/scene columns for leakage fallback")
        else:
            rep.soft(False, "manifest.csv and metadata.csv both missing — "
                            "leakage check limited to id disjointness")

    # --- stratification sanity: label ratios ~ overall --------------------
    if paths.dest_dir.joinpath("metadata.csv").is_file():
        md = load_metadata(paths)
        if "label" in md.columns and "crop_id" in md.columns:
            overall = md["label"].value_counts(normalize=True).to_dict()
            problems = []
            for n, s in per_split.items():
                if not s:
                    continue
                sub = md[md["crop_id"].astype(str).isin(s) | md.get(
                    "image_id", pd.Series(dtype=str)
                ).astype(str).isin(s)]
                if len(sub) == 0 or "label" not in sub.columns:
                    continue
                ratios_n = sub["label"].value_counts(normalize=True).to_dict()
                for lab, p_overall in overall.items():
                    p_split = ratios_n.get(lab, 0.0)
                    if abs(p_split - p_overall) > ratio_tol:
                        problems.append({
                            "split": n, "label": lab,
                            "share_split": round(float(p_split), 4),
                            "share_overall": round(float(p_overall), 4),
                        })
            rep.soft(
                not problems,
                f"label stratification within ±{ratio_tol:.0%} of overall"
                if not problems else f"{len(problems)} stratification deviations",
                problems=problems[:20],
            )

    # --- non-empty val/test when dataset is large enough -----------------
    n_all = len(all_ids)
    for n in ("val", "test"):
        k = len(per_split[n])
        if n_all >= 30:
            rep.hard(k > 0, f"{n} split non-empty" if k else f"{n} split is empty")
        else:
            rep.info(f"{n} has {k} ids (small fixture)")

    rep.info(
        f"split sizes: train={len(per_split['train'])} "
        f"val={len(per_split['val'])} test={len(per_split['test'])}"
    )
    return rep
