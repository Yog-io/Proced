"""Structural — full null-sweep on required schema columns (Stage 5 / bug #16).

Zero tolerance: every required column in master metadata.csv and every
per-folder GEODATA.csv must be null-free. Also enforces boolean vocabulary
(true/false or Y/N) on flag columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import pandas as pd

from .._lib import (
    GEODATA_COLUMNS,
    MASTER_COLUMNS,
    QAPaths,
    REQUIRED_NULL_FREE,
    Report,
    iter_geodata,
    load_metadata,
)


def _nullish_mask(s: pd.Series) -> pd.Series:
    as_str = s.astype(str).str.strip().str.lower()
    # pandas reads empty as NaN → already "" after fillna(""), but catch None/nan
    return as_str.isin(["", "nan", "none", "null"])


def _nulls_in(df: pd.DataFrame, cols: Sequence[str], origin: str) -> List[dict]:
    bad = []
    for c in cols:
        if c not in df.columns:
            bad.append({"origin": origin, "column": c, "error": "missing column"})
            continue
        mask = _nullish_mask(df[c])
        n = int(mask.sum())
        if n:
            examples = df.loc[mask, c].head(5).tolist()
            bad.append({
                "origin": origin, "column": c, "n_null": n,
                "examples": [str(x) for x in examples],
            })
    return bad


def run(paths: QAPaths) -> Report:
    rep = Report("stage5_schema_nulls")
    skip = paths.require("dest")
    if skip:
        rep.hard(False, f"skip: {skip}")
        return rep

    # --- master metadata.csv ----------------------------------------------
    md_path = paths.dest_dir / "metadata.csv"
    if not md_path.is_file():
        rep.hard(False, f"metadata.csv missing: {md_path}")
        return rep

    md = load_metadata(paths)
    if len(md) == 0:
        rep.hard(False, "metadata.csv has zero rows")
        return rep

    # REQUIRED_NULL_FREE: absolute zero tolerance (bug #16)
    bad_crit = _nulls_in(md, REQUIRED_NULL_FREE, "metadata.csv")
    rep.hard(
        not bad_crit,
        f"required-null-free columns empty on 0/{len(md)} rows (bug #16)"
        if not bad_crit else f"{len(bad_crit)} critical columns contain nulls",
        failures=bad_crit,
        n_rows=int(len(md)),
    )

    # full GEODATA + MASTER columns must exist; non-critical empties soft
    all_cols = list(dict.fromkeys(GEODATA_COLUMNS + MASTER_COLUMNS))
    missing_cols = [c for c in all_cols if c not in md.columns]
    rep.hard(
        not missing_cols,
        f"all {len(all_cols)} schema columns present in metadata.csv"
        if not missing_cols else f"missing columns: {missing_cols}",
        missing=missing_cols,
    )
    bad_all = _nulls_in(md, all_cols, "metadata.csv")
    # separate hard (required) from soft (others already covered above)
    soft_bad = [b for b in bad_all if b["column"] not in REQUIRED_NULL_FREE
                and "missing column" not in str(b.get("error", ""))]
    if soft_bad:
        # full_extent_* must never be empty either (numeric 0 is fine; empty is not)
        hard_soft = [b for b in soft_bad if b["column"].startswith("full_extent_")]
        rep.hard(
            not hard_soft,
            "full_extent_* columns non-empty"
            if not hard_soft else f"{len(hard_soft)} full_extent columns have nulls",
            failures=hard_soft,
        )
        other = [b for b in soft_bad if b not in hard_soft]
        if other:
            rep.soft(False, f"{len(other)} non-critical columns have nulls",
                     failures=other[:20])

    # --- boolean / Y-N vocabulary -----------------------------------------
    vocab_bad = []
    yn_cols = ["is_calibrated", "has_dual_pol", "is_partial_object",
               "truncated_by_scene_edge", "has_real_geo"]
    bool_cols = ["is_synthetic_location", "is_synthetic_timestamp"]
    for c in yn_cols:
        if c not in md.columns:
            continue
        vals = {str(v).strip().upper() for v in md[c].dropna().unique()}
        bad = vals - {"Y", "N"}
        if bad:
            vocab_bad.append({"column": c, "bad_values": sorted(bad)[:10]})
    for c in bool_cols:
        if c not in md.columns:
            continue
        vals = {str(v).strip().lower() for v in md[c].dropna().unique()}
        bad = vals - {"true", "false", "1", "0"}
        if bad:
            vocab_bad.append({"column": c, "bad_values": sorted(bad)[:10]})
    rep.hard(
        not vocab_bad,
        "boolean/Y-N vocabularies valid on all flag columns"
        if not vocab_bad else f"{len(vocab_bad)} columns with illegal flag values",
        failures=vocab_bad,
    )

    # --- crop_id uniqueness -----------------------------------------------
    if "crop_id" in md.columns:
        dup = md["crop_id"].astype(str).duplicated()
        n_dup = int(dup.sum())
        rep.hard(
            n_dup == 0,
            f"crop_id unique across {len(md)} rows"
            if n_dup == 0 else f"{n_dup} duplicate crop_id values",
            examples=md.loc[dup, "crop_id"].astype(str).head(10).tolist(),
        )

    # --- every per-folder GEODATA.csv null-free on required cols ----------
    gd_bad = []
    n_files = 0
    for p, gdf in iter_geodata(paths):
        n_files += 1
        rel = str(p)
        miss = [c for c in GEODATA_COLUMNS if c not in gdf.columns]
        if miss:
            gd_bad.append({"file": rel, "missing": miss})
            continue
        gd_bad.extend(_nulls_in(gdf, GEODATA_COLUMNS, rel))
    if n_files == 0:
        rep.soft(False, "no per-folder GEODATA.csv found under dest_dir")
    else:
        rep.hard(
            not gd_bad,
            f"{n_files} GEODATA.csv files null-free on required columns"
            if not gd_bad else f"{len(gd_bad)} GEODATA null/schema failures",
            failures=gd_bad[:30],
            n_files=n_files,
        )

    # --- label domain ------------------------------------------------------
    if "label" in md.columns:
        labels = {str(v).strip().lower() for v in md["label"].dropna().unique()}
        allowed = {"oil", "lookalike", "background"}
        bad_l = sorted(labels - allowed)
        rep.hard(
            not bad_l,
            f"label domain ⊆ {sorted(allowed)}" if not bad_l
            else f"illegal labels: {bad_l}",
            seen=sorted(labels),
        )

    # --- Zenodo label-count soft band -------------------------------------
    if "label" in md.columns:
        from .._lib import LABEL_COUNT_TOLERANCE, ZENODO_EXPECTED_LABELS
        vc = md["label"].astype(str).str.lower().value_counts().to_dict()
        soft_bad = []
        for lab, expected in ZENODO_EXPECTED_LABELS.items():
            got = int(vc.get(lab, 0))
            if got == 0:
                soft_bad.append({"label": lab, "got": 0, "expected": expected,
                                 "note": "absent (fixture/partial?)"})
            elif abs(got - expected) > expected * LABEL_COUNT_TOLERANCE:
                soft_bad.append({"label": lab, "got": got, "expected": expected})
        if soft_bad:
            rep.soft(False, "label counts outside Zenodo ±50% band",
                     counts={str(k): int(v) for k, v in vc.items()},
                     problems=soft_bad)
        else:
            rep.info(f"label counts: {vc}")

    # --- lookalike_training_data filters (if join_wind ran) ---------------
    look_p = paths.features_dir / "lookalike_training_data.csv"
    if look_p.is_file():
        try:
            ldf = pd.read_csv(look_p).fillna("")
        except Exception as exc:
            rep.soft(False, f"lookalike_training_data unreadable: {exc}")
            ldf = None
        if ldf is not None and len(ldf) >= 0:
            problems = []
            if "truncated_by_scene_edge" in ldf.columns:
                n = int(ldf["truncated_by_scene_edge"].astype(str).str.upper()
                        .isin(("Y", "TRUE", "1")).sum())
                if n:
                    problems.append({"filter": "truncated_by_scene_edge==Y", "n": n})
            if "is_calibrated" in ldf.columns:
                n = int((~ldf["is_calibrated"].astype(str).str.upper()
                         .isin(("Y", "TRUE", "1"))).sum())
                if n:
                    problems.append({"filter": "is_calibrated==N", "n": n})
            if "label" in ldf.columns:
                n = int((ldf["label"].astype(str).str.lower() == "background").sum())
                if n:
                    problems.append({"filter": "label==background", "n": n})
            rep.hard(
                not problems,
                "lookalike_training_data filters held"
                if not problems else f"{len(problems)} classifier-filter violations",
                failures=problems,
                n_rows=int(len(ldf)),
            )
            # mock-wind rows must be visibly flagged in the file itself
            if "wind_source" in ldf.columns:
                srcs = {str(s).strip().lower() for s in ldf["wind_source"].unique()}
                n_mock = int(ldf["wind_source"].astype(str).str.lower()
                              .eq("mock").sum())
                if n_mock:
                    rep.soft(
                        n_mock == len(ldf) or n_mock > 0,
                        f"{n_mock} rows flagged wind_source=mock "
                        "(must not be presented as ERA5)",
                        sources=sorted(srcs),
                    )
                else:
                    rep.info(f"wind_source values: {sorted(srcs)}")
            elif "wind_speed_ms" in ldf.columns:
                rep.soft(
                    False,
                    "wind_speed_ms present but wind_source column missing — "
                    "mock winds not visibly flagged",
                )
    else:
        rep.info("lookalike_training_data.csv absent — skip classifier filters "
                 "(run scripts/join_wind.py after the pipeline)")

    return rep
