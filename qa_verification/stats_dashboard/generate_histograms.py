"""Layer 2 — statistical histograms (architecture §4).

* full_extent_area_px / aspect_ratio / damping_ratio_db by label
* output crop file-size distribution (22GB/full-scene regression)

Writes PNGs + a small index JSON under --out-dir. Soft-skips missing files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .._lib import QAPaths, Report, load_metadata

LABEL_COLORS = {
    "oil": "#e74c3c",
    "lookalike": "#9b59b6",
    "background": "#7f8c8d",
}


def _hist_by_label(ax, series: pd.Series, labels: pd.Series, bins: int = 40,
                   title: str = "") -> None:
    for lab in sorted(set(labels.astype(str))):
        vals = pd.to_numeric(series[labels.astype(str) == lab], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=bins, alpha=0.55, label=str(lab),
                color=LABEL_COLORS.get(str(lab), None),
                edgecolor="none")
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)


def run(
    paths: QAPaths,
    *,
    out_dir: Optional[Path] = None,
    sample: int = 400,
    seed: int = 42,
) -> Report:
    rep = Report("stats_histograms")
    skip = paths.require("dest")
    if skip:
        rep.hard(False, f"skip: {skip}")
        return rep

    md_path = paths.dest_dir / "metadata.csv"
    if not md_path.is_file():
        rep.hard(False, "metadata.csv missing")
        return rep

    md = load_metadata(paths)
    if len(md) == 0:
        rep.hard(False, "metadata.csv empty")
        return rep

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        rep.soft(False, f"matplotlib unavailable — histograms skipped: {exc}")
        return rep

    out = Path(out_dir) if out_dir else (paths.features_dir / "qa_stats")
    out.mkdir(parents=True, exist_ok=True)

    labels = md["label"] if "label" in md.columns else pd.Series(["?"] * len(md))
    written: List[str] = []

    # --- shape / radiometric histograms -----------------------------------
    metrics = [
        ("full_extent_area_px", "area px"),
        ("full_extent_aspect_ratio", "aspect ratio"),
        ("full_extent_boundary_complexity", "boundary complexity"),
        ("full_extent_perimeter", "perimeter"),
    ]
    # damping lives in features CSV, not always in metadata
    features_p = paths.features_dir / "full_extent_features.csv"
    damping = None
    if features_p.is_file():
        try:
            fdf = pd.read_csv(features_p).fillna("")
            if "damping_ratio_db" in fdf.columns and "label" in fdf.columns:
                damping = fdf
        except Exception:
            damping = None

    present = [(c, t) for c, t in metrics if c in md.columns]
    if present:
        n = len(present) + (1 if damping is not None else 0)
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.4))
        if n == 1:
            axes = [axes]
        for ax, (col, title) in zip(axes, present):
            _hist_by_label(ax, md[col], labels, title=f"{col} ({title})")
        if damping is not None:
            ax = axes[-1]
            _hist_by_label(
                ax,
                pd.to_numeric(damping["damping_ratio_db"], errors="coerce"),
                damping["label"],
                title="damping_ratio_db by label",
            )
        fig.tight_layout()
        p1 = out / "hist_shape_and_damping.png"
        fig.savefig(p1, dpi=120)
        plt.close(fig)
        written.append(str(p1))

        # damping comparison note: oil median vs lookalike median
        if damping is not None:
            meds = {}
            for lab in ("oil", "lookalike"):
                vals = pd.to_numeric(
                    damping.loc[damping["label"].astype(str) == lab, "damping_ratio_db"],
                    errors="coerce",
                ).dropna()
                if len(vals):
                    meds[lab] = float(vals.median())
            if "oil" in meds and "lookalike" in meds:
                ok = meds["oil"] >= meds["lookalike"] - 1e-6
                rep.soft(
                    ok,
                    f"oil median damping {meds.get('oil'):.2f} dB vs lookalike "
                    f"{meds.get('lookalike'):.2f} dB "
                    f"({'oil higher as expected' if ok else 'COMPLETE overlap? oil ≤ lookalike'})",
                    medians=meds,
                )
            elif meds:
                rep.info(f"damping medians (partial): {meds}")

    # --- crop file-size histogram (22GB regression) ------------------------
    sizes: List[int] = []
    for p in paths.dest_dir.rglob("*.png"):
        try:
            sizes.append(p.stat().st_size)
        except OSError:
            continue
    if sizes:
        arr = np.asarray(sizes, dtype=np.int64)
        fig, ax = plt.subplots(figsize=(6, 3.4))
        # log-ish bins for wide dynamic range
        ax.hist(arr / 1024.0, bins=50, color="#3498db", edgecolor="none")
        ax.set_xlabel("PNG size (KB)")
        ax.set_ylabel("count")
        ax.set_title(f"output crop file sizes (n={len(arr)})")
        fig.tight_layout()
        p2 = out / "hist_crop_file_sizes.png"
        fig.savefig(p2, dpi=120)
        plt.close(fig)
        written.append(str(p2))

        median_kb = float(np.median(arr) / 1024.0)
        max_mb = float(arr.max() / (1024.0 * 1024.0))
        # "a few hundred KB" clustering; hard cap well below full-scene scale
        p95_mb = float(np.percentile(arr, 95) / (1024.0 * 1024.0))
        ok = p95_mb < 20.0  # 256×256 uint16 ≈ 128KB raw; compressed smaller
        rep.soft(
            ok,
            f"crop sizes: median={median_kb:.1f} KB p95={p95_mb:.2f} MB "
            f"max={max_mb:.2f} MB (expect tight cluster, not full scenes)",
            n=len(sizes),
            median_kb=round(median_kb, 1),
            p95_mb=round(p95_mb, 3),
            max_mb=round(max_mb, 3),
        )
    else:
        rep.info("no PNGs found for size histogram")

    # --- area / aspect by-label medians (soft sanity) ----------------------
    if "full_extent_area_px" in md.columns and "label" in md.columns:
        med = md.groupby("label")["full_extent_area_px"].median().to_dict()
        rep.info(f"median area_px by label: { {k: round(float(v), 1) for k, v in med.items()} }")

    index = {"written": written, "n_rows": int(len(md)), "n_pngs": len(sizes)}
    (out / "histogram_index.json").write_text(json.dumps(index, indent=2))
    rep.info(f"histograms → {out}", out=str(out), files=written)
    return rep
