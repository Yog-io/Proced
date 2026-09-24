"""Structural — train/val/test group leakage (architecture Stage 6 / hard-fail).

Independent recompute from split CSVs + metadata.csv using pure pandas —
never trusts the pipeline's own assertion inside proced/splits.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

from .._lib import QAPaths, Report, load_metadata

SPLIT_NAMES = ("train", "val", "test")


def _read_ids(path: Path) -> List[str]:
    if not path.is_file():
        return []
    df = pd.read_csv(path)
    col = "image_id" if "image_id" in df.columns else df.columns[0]
    return [str(x) for x in df[col].tolist()]


def run(paths: QAPaths, *, ratio_tol: float = 0.20) -> Report:
    rep = Report("stage6_split_leakage")
    skip = paths.require("splits")
    if skip:
        rep.hard(False, f"skip: {skip}")
        return rep

    split_ids: Dict[str, List[str]] = {}
    for name in SPLIT_NAMES:
        split_ids[name] = _read_ids(paths.splits_dir / f"{name}.csv")
        if not split_ids[name]:
            # val/test may be empty on tiny fixtures; train must exist if any split does
            if name == "train" and any(split_ids.values()):
                rep.hard(False, "train.csv empty while other splits exist")
            elif name in ("val", "test") and split_ids["train"]:
                rep.info(f"{name}.csv empty (small fixture?)")

    if not any(split_ids.values()):
        rep.hard(False, "all split CSVs empty or missing")
        return rep

    # --- disjointness of image_id sets ------------------------------------
    seen: Dict[str, str] = {}
    dups = []
    for name, ids in split_ids.items():
        for i in ids:
            if i in seen and seen[i] != name:
                dups.append({"image_id": i, "splits": sorted({seen[i], name})})
            seen[i] = name
    rep.hard(
        not dups,
        "no image_id appears in two splits" if not dups
        else f"{len(dups)} image_ids span multiple splits",
        examples=dups[:20],
        sizes={k: len(v) for k, v in split_ids.items()},
    )

    # --- group leakage via metadata join ----------------------------------
    md_path = paths.dest_dir / "metadata.csv"
    man_path = paths.splits_dir / "manifest.csv"
    assign: Dict[str, str] = {}
    for name, ids in split_ids.items():
        for i in ids:
            assign[i] = name

    source_df: Optional[pd.DataFrame] = None
    id_col = "image_id"
    scene_col = "crop_source_scene_id"

    if man_path.is_file():
        man = pd.read_csv(man_path).fillna("")
        if scene_col in man.columns and "split" in man.columns:
            # direct manifest recompute
            leaks = []
            for gid, grp in man.groupby(scene_col):
                splits = sorted({str(s) for s in grp["split"]})
                if len(splits) > 1:
                    leaks.append({"group": str(gid), "splits": splits,
                                  "n": int(len(grp))})
            rep.hard(
                not leaks,
                f"manifest: no scene-group leakage across "
                f"{man[scene_col].nunique()} groups"
                if not leaks else f"{len(leaks)} groups leak across splits (manifest)",
                examples=leaks[:20],
            )
            # coverage: every split id should be in the manifest
            man_ids = set(man["image_id"].astype(str)) if "image_id" in man.columns else set()
            missing = sorted(set(assign) - man_ids)
            rep.hard(
                not missing,
                f"manifest covers all split image_ids ({len(assign)})"
                if not missing else f"{len(missing)} split ids missing from manifest",
                examples=missing[:20],
            )
            source_df = man
            id_col = "image_id" if "image_id" in man.columns else id_col
        else:
            rep.soft(False, "manifest.csv missing split/scene columns — falling back to metadata")
    if source_df is None:
        if md_path.is_file():
            source_df = load_metadata(paths)
            id_col = "crop_id" if "crop_id" in source_df.columns else "image_id"
        else:
            rep.hard(False, "neither manifest.csv nor metadata.csv available for leakage recompute")
            return rep

    if scene_col not in source_df.columns:
        rep.hard(False, f"source table missing {scene_col}")
        return rep

    source_df = source_df.copy()
    source_df["_id"] = source_df[id_col].astype(str)
    source_df["_split"] = source_df["_id"].map(assign)
    matched = source_df[source_df["_split"].notna()]

    leaks = []
    for gid, grp in matched.groupby(scene_col):
        splits = sorted({str(s) for s in grp["_split"]})
        if len(splits) > 1:
            leaks.append({"group": str(gid), "splits": splits, "n": int(len(grp))})
    rep.hard(
        not leaks,
        f"no scene-group leakage across {matched[scene_col].nunique()} groups "
        f"({len(matched)} matched rows)"
        if not leaks else f"{len(leaks)} scene groups leak across splits",
        examples=leaks[:20],
    )

    # --- unmatched split ids (in splits but not in source table) -----------
    unmatched = sorted(set(assign) - set(source_df["_id"]))
    if unmatched:
        rep.soft(
            len(unmatched) <= max(5, int(0.05 * len(assign))),
            f"{len(unmatched)} split image_ids not found in source table",
            examples=unmatched[:20],
        )

    # --- stratification sanity --------------------------------------------
    if "label" in matched.columns and len(matched):
        overall = matched["label"].astype(str).value_counts(normalize=True)
        deviations = []
        for name in SPLIT_NAMES:
            sub = matched[matched["_split"] == name]
            if len(sub) < 2:
                continue
            share = sub["label"].astype(str).value_counts(normalize=True)
            for lab, p_all in overall.items():
                p_s = float(share.get(lab, 0.0))
                if abs(p_s - p_all) > ratio_tol:
                    deviations.append({
                        "split": name, "label": lab,
                        "overall": round(float(p_all), 4),
                        "split_share": round(p_s, 4),
                    })
        rep.soft(
            not deviations,
            f"label stratification within ±{ratio_tol:.0%} of overall"
            if not deviations else f"{len(deviations)} stratification deviations",
            deviations=deviations[:20],
        )

    # --- coverage: total unique ids vs metadata ---------------------------
    if md_path.is_file():
        md = load_metadata(paths)
        mid = "crop_id" if "crop_id" in md.columns else "image_id"
        md_ids = set(md[mid].astype(str))
        split_set = set(assign)
        only_md = md_ids - split_set
        only_split = split_set - md_ids
        # every metadata crop should land in a split (C.5)
        rep.hard(
            not only_md,
            f"every metadata crop_id assigned to a split ({len(md_ids)})"
            if not only_md else f"{len(only_md)} metadata crops in no split",
            examples=sorted(only_md)[:25],
        )
        if only_split:
            rep.soft(
                False,
                f"{len(only_split)} split ids absent from metadata",
                examples=sorted(only_split)[:15],
            )

    return rep
