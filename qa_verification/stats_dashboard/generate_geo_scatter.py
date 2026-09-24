"""Layer 2 — world-map scatter of real + synthetic coordinates (architecture §4).

Real points should fall in Zenodo coverage; synthetic points should cluster
along the 8 corridors, not sit on land/at (0,0).

Prefers folium (interactive HTML). Soft-skips if folium is unavailable and
falls back to a matplotlib scatter + corridor overlay.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .._lib import (
    QAPaths,
    Report,
    load_corridors_independent,
    load_metadata,
    min_distance_to_polyline_km,
    sample_indices,
)

# Rough Zenodo / global SAR ocean coverage sanity box (soft)
LAT_OK = (-75.0, 85.0)
LON_OK = (-180.0, 180.0)


def _collect_points(md: pd.DataFrame) -> List[dict]:
    out = []
    if "lat" not in md.columns or "lon" not in md.columns:
        return out
    for _, row in md.iterrows():
        try:
            la = float(row["lat"])
            lo = float(row["lon"])
        except (TypeError, ValueError):
            continue
        syn = str(row.get("is_synthetic_location", "")).lower() in (
            "true", "1", "y", "yes",
        )
        out.append({
            "lat": la, "lon": lo,
            "label": str(row.get("label", "")),
            "synthetic": syn,
            "crop_id": str(row.get("crop_id") or row.get("image_id") or ""),
            "corridor": str(row.get("source_corridor_id") or ""),
            "offset_km": row.get("offset_km", ""),
        })
    return out


def _folium_map(points: List[dict], corridors_path: Path, out_html: Path) -> bool:
    try:
        import folium
    except Exception:
        return False
    if not points:
        return False

    lats = [p["lat"] for p in points]
    lons = [p["lon"] for p in points]
    m = folium.Map(location=[float(np.median(lats)), float(np.median(lons))],
                   zoom_start=2, tiles="OpenStreetMap")

    # corridors first (under points)
    if corridors_path.is_file():
        try:
            corridors = load_corridors_independent(corridors_path)
            for c in corridors:
                wps = c["waypoints"]
                folium.PolyLine(
                    [(la, lo) for la, lo in wps],
                    color="#f1c40f", weight=2, opacity=0.7,
                    tooltip=c["id"],
                ).add_to(m)
        except Exception:
            pass

    # cluster-ish: sample if huge
    show = points if len(points) <= 5000 else [
        points[i] for i in sample_indices(len(points), 5000, 42, "map")
    ]
    for p in show:
        color = "#e67e22" if p["synthetic"] else (
            "#e74c3c" if p["label"] == "oil" else (
                "#9b59b6" if p["label"] == "lookalike" else "#7f8c8d"
            )
        )
        folium.CircleMarker(
            [p["lat"], p["lon"]], radius=3, color=color,
            fill=True, fillOpacity=0.55, weight=0,
            popup=f"{p['crop_id']} {p['label']}",
        ).add_to(m)

    m.save(str(out_html))
    return True


def _matplotlib_scatter(points: List[dict], corridors_path: Path,
                        out_png: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    if not points:
        return False
    fig, ax = plt.subplots(figsize=(11, 5.5))
    if corridors_path.is_file():
        try:
            for c in load_corridors_independent(corridors_path):
                la = [w[0] for w in c["waypoints"]]
                lo = [w[1] for w in c["waypoints"]]
                ax.plot(lo, la, color="#f1c40f", lw=1.5, alpha=0.8, label="_c")
        except Exception:
            pass
    real = [p for p in points if not p["synthetic"]]
    syn = [p for p in points if p["synthetic"]]
    if real:
        ax.scatter([p["lon"] for p in real], [p["lat"] for p in real],
                   s=6, c="#e74c3c", alpha=0.5, label=f"real (n={len(real)})")
    if syn:
        ax.scatter([p["lon"] for p in syn], [p["lat"] for p in syn],
                   s=6, c="#e67e22", alpha=0.5, label=f"synthetic (n={len(syn)})")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    ax.set_title("crop coordinates (yellow = shipping corridors)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return True


def run(
    paths: QAPaths,
    *,
    out_dir: Optional[Path] = None,
    sample: int = 400,
    seed: int = 42,
) -> Report:
    rep = Report("stats_geo_scatter")
    skip = paths.require("dest")
    if skip:
        rep.hard(False, f"skip: {skip}")
        return rep

    md_path = paths.dest_dir / "metadata.csv"
    if not md_path.is_file():
        rep.hard(False, "metadata.csv missing")
        return rep

    md = load_metadata(paths)
    points = _collect_points(md)
    if not points:
        rep.soft(False, "no lat/lon columns populated in metadata — map skipped")
        return rep

    out = Path(out_dir) if out_dir else (paths.features_dir / "qa_stats")
    out.mkdir(parents=True, exist_ok=True)

    # --- (0,0) / out-of-range soft checks ---------------------------------
    origin = [p for p in points if p["lat"] == 0 and p["lon"] == 0]
    oob = [
        p for p in points
        if not (LAT_OK[0] <= p["lat"] <= LAT_OK[1])
        or not (LON_OK[0] <= p["lon"] <= LON_OK[1])
    ]
    rep.hard(
        not oob,
        f"all {len(points)} coordinates within lat/lon bounds "
        f"(no (0,0) or transposed-axis garbage)"
        if not oob else f"{len(oob)} coordinates out of bounds / invalid",
        examples=[{"crop_id": p["crop_id"], "lat": p["lat"], "lon": p["lon"]}
                  for p in oob[:15]],
        n_origin=len(origin),
    )

    # --- synthetic points near corridors (sampled) ------------------------
    corridors = []
    if paths.corridors_path.is_file():
        corridors = load_corridors_independent(paths.corridors_path)
    syn = [p for p in points if p["synthetic"]]
    if syn and corridors:
        idxs = sample_indices(len(syn), min(sample, len(syn)), seed, "geo_scatter")
        far = []
        for i in idxs:
            p = syn[i]
            best = min(
                min_distance_to_polyline_km(p["lat"], p["lon"], c["waypoints"])[0]
                for c in corridors
            )
            try:
                allow = float(p["offset_km"]) + 20.0 if p["offset_km"] not in ("", None) else 100.0
            except (TypeError, ValueError):
                allow = 100.0
            if best > max(allow, 25.0):
                far.append({
                    "crop_id": p["crop_id"],
                    "dist_km": round(best, 2),
                    "offset_km": p["offset_km"],
                })
        rep.hard(
            not far,
            f"synthetic points cluster near corridors ({len(idxs)} sampled)"
            if not far else f"{len(far)}/{len(idxs)} synthetic points far from corridors",
            examples=far[:15],
        )
    elif syn:
        rep.soft(False, f"{len(syn)} synthetic points but corridors GeoJSON missing")

    # --- render ------------------------------------------------------------
    html_path = out / "geo_scatter.html"
    png_path = out / "geo_scatter.png"
    used_folium = False
    if corridors:
        used_folium = _folium_map(points, paths.corridors_path, html_path)
    else:
        # still render points without corridors
        used_folium = _folium_map(points, Path("/nonexistent"), html_path)

    fell_back = False
    if not used_folium:
        # synthesize empty corridors path for pure points plot
        fell_back = True
        if not _matplotlib_scatter(
            points,
            paths.corridors_path if paths.corridors_path.is_file() else Path("/nonexistent"),
            png_path,
        ):
            rep.soft(False, "neither folium nor matplotlib available for geo scatter")
            return rep

    n_real = sum(1 for p in points if not p["synthetic"])
    n_syn = sum(1 for p in points if p["synthetic"])
    rep.info(
        f"geo scatter: {n_real} real + {n_syn} synthetic → "
        + (str(html_path) if used_folium else str(png_path))
        + (" (matplotlib fallback; folium not installed)" if fell_back else ""),
        n_points=len(points),
        folium=used_folium,
    )
    return rep
