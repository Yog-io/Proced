"""Stage 1 — independent scan/catalog verification (architecture §2)."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np

from .._lib import (
    QAPaths,
    Report,
    detect_domain_independent,
    load_catalog,
    resolve_stage_path,
    sample_indices,
)

_POL_SPLIT = re.compile(r"(?:^|[_\-.])(vv|vh)$", re.IGNORECASE)


def run(paths: QAPaths, *, sample: int = 30, seed: int = 42) -> Report:
    rep = Report("stage1_scan")
    skip = paths.require("catalog", "stage")
    if skip:
        rep.hard(False, f"skip: {skip}")
        return rep

    cat = load_catalog(paths)
    scenes = cat.get("scenes") or []
    rep.hard(len(scenes) > 0, "catalog has scenes", n=len(scenes))
    root = cat.get("root") or str(paths.stage_dir)

    # --- sample: dtype / bands / CRS / value range vs catalog -------------
    idxs = sample_indices(len(scenes), sample, seed, "scan")
    mismatches = []
    for i in idxs:
        s = scenes[i]
        try:
            path = resolve_stage_path(s["band_files"][0], root, paths.stage_dir)
            import rasterio
            with rasterio.open(str(path)) as ds:
                dtype = ds.dtypes[0]
                count = ds.count
                crs = str(ds.crs) if ds.crs else None
                # centre sample for domain
                h, w = min(256, ds.height), min(256, ds.width)
                x0 = max(0, (ds.width - w) // 2)
                y0 = max(0, (ds.height - h) // 2)
                win = ds.read(1, window=((y0, y0 + h), (x0, x0 + w)))
                domain = detect_domain_independent(win)

            # calibration heuristic re-derived: float ⇒ calibrated (unless overrides)
            expect_cal = dtype.startswith("float")
            # overrides not re-applied here — flag only clear contradictions
            if s.get("is_calibrated") and dtype.startswith("uint") and "match" not in str(
                s.get("value_domain_hint", "")
            ):
                # catalog may force calibrated via path overrides — soft
                rep.soft(
                    True,
                    f"sampled {s['scene_id']}: uint but is_calibrated=Y (path override?)",
                    dtype=dtype,
                )
            if bool(s.get("is_calibrated")) != expect_cal and not s.get("value_domains"):
                mismatches.append({
                    "scene_id": s["scene_id"], "dtype": dtype,
                    "catalog_cal": s.get("is_calibrated"), "heuristic": expect_cal,
                })

            # width/height match
            if int(s.get("width", -1)) != ds.width or int(s.get("height", -1)) != ds.height:
                mismatches.append({
                    "scene_id": s["scene_id"],
                    "size": (ds.width, ds.height),
                    "catalog": (s.get("width"), s.get("height")),
                })

            # value domain for calibrated non-native
            vd = (s.get("value_domains") or {}).get("VV")
            if vd and vd != "native" and domain != vd:
                # centre sample can differ from pipeline's sample window slightly
                # only hard-fail on native vs non-native contradiction
                if (vd == "native") != (domain == "native"):
                    mismatches.append({
                        "scene_id": s["scene_id"], "domain_catalog": vd,
                        "domain_independent": domain,
                    })
        except Exception as exc:
            mismatches.append({"scene_id": s.get("scene_id"), "error": str(exc)})

    rep.hard(
        not mismatches,
        "sampled scenes match independent raster probe"
        if not mismatches else f"{len(mismatches)} catalog mismatches",
        mismatches=mismatches[:20],
        sampled=len(idxs),
    )

    # --- regression bug #4: no VV/VH split double-count --------------------
    # Two entries must not be base + base_VV / base_VH of the same folder.
    by_folder = defaultdict(list)
    for s in scenes:
        by_folder[s.get("rel_folder", "")].append(s)

    dup_pairs = []
    for rel, group in by_folder.items():
        bases = []
        for s in group:
            name = s.get("source_scene_name", "")
            m = _POL_SPLIT.search(name)
            bases.append(m.group(1).upper() if m else name)
        # if both scene "foo" and scene "foo_VV" exist as separate records with
        # same stripped base in same folder → pairing failed
        stripped = []
        for s in group:
            name = s.get("source_scene_name", "")
            m = _POL_SPLIT.search(name)
            if m:
                stripped.append(name[: m.start()].rstrip("_-. ") or name)
            else:
                stripped.append(name)
        # count collisions where one entry is the pol-split of another
        names = {s.get("source_scene_name", "") for s in group}
        for s in group:
            name = s.get("source_scene_name", "")
            m = _POL_SPLIT.search(name)
            if m:
                base = name[: m.start()].rstrip("_-. ")
                if base in names:
                    dup_pairs.append(f"{rel}: {base} and {name} both present")

    rep.hard(
        not dup_pairs,
        "no VV/VH split double-processing in catalog"
        if not dup_pairs else f"dual-pol pairing regression: {dup_pairs[:5]}",
        dup_pairs=dup_pairs[:10],
    )
    return rep
