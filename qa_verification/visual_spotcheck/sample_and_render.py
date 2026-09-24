"""Layer 3 — visual spot-check: sample crops → one static HTML gallery.

Oversamples high-risk categories (overflow/tiled, synthetic-location,
lookalike) per architecture §3. Renders side-by-side raw (inverse-dB +
contrast stretch for viewing), mask overlay, and a metadata panel.
No server required — open the HTML in any browser.
"""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._lib import (
    QAPaths,
    Report,
    db_from_uint16,
    find_png,
    load_metadata,
    sample_indices,
)

# fraction of rows to sample per label (architecture: ~2% overall, but
# oversample risk categories — here we use max(min_n, frac*n) capped).
DEFAULT_FRAC = 0.02
DEFAULT_MIN_PER_CLASS = 4
RISK_CAP = 25  # max samples drawn from each risk bucket


def _b64_png(arr: np.ndarray) -> str:
    from PIL import Image
    if arr.dtype == np.uint16:
        # contrast-stretch for human viewing via inverse dB → 8-bit display
        db = db_from_uint16(arr)
        lo, hi = float(db.min()), float(db.max())
        if hi - lo < 1e-9:
            disp = np.zeros(db.shape, np.uint8)
        else:
            disp = np.clip(((db - lo) / (hi - lo)) * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(disp, mode="L")
    elif arr.dtype == np.uint8:
        img = Image.fromarray(arr, mode="L")
    else:
        a = arr.astype(np.float64)
        lo, hi = float(a.min()), float(a.max())
        disp = np.clip(((a - lo) / (hi - lo + 1e-12)) * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(disp, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _overlay_b64(img_arr: np.ndarray, mask_arr: Optional[np.ndarray]) -> str:
    from PIL import Image
    if img_arr.dtype == np.uint16:
        db = db_from_uint16(img_arr)
        lo, hi = float(db.min()), float(db.max())
        base = np.zeros(db.shape, np.float64) if hi <= lo else (
            (db - lo) / (hi - lo) * 255.0
        )
        base8 = np.clip(base, 0, 255).astype(np.uint8)
    else:
        base8 = img_arr.astype(np.uint8) if img_arr.max() > 1 else (
            img_arr.astype(np.uint8) * 255
        )
    rgb = np.stack([base8] * 3, axis=-1).astype(np.float64)
    if mask_arr is not None:
        m = mask_arr > 0
        if m.shape == base8.shape:
            # semi-transparent red where mask
            rgb[m] = rgb[m] * 0.45 + np.array([220, 40, 40]) * 0.55
    out = np.clip(rgb, 0, 255).astype(np.uint8)
    img = Image.fromarray(out, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _pick_samples(md: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Stratified sample with risk oversampling."""
    picked: Dict[str, pd.Series] = {}

    def _bucket(name: str, mask: pd.Series, k: int) -> None:
        sub = md[mask]
        if len(sub) == 0:
            return
        idxs = sample_indices(len(sub), min(k, len(sub)), seed, name)
        for i in idxs:
            row = sub.iloc[i]
            cid = str(row.get("crop_id") or row.get("image_id"))
            picked[cid] = row

    n = len(md)
    overall_k = max(int(round(n * frac)), DEFAULT_MIN_PER_CLASS)

    # per-label baseline
    for lab in ("oil", "lookalike", "background"):
        mask = md["label"].astype(str).str.lower() == lab if "label" in md.columns \
            else pd.Series(False, index=md.index)
        k = max(DEFAULT_MIN_PER_CLASS, int(round(mask.sum() * frac)))
        _bucket(f"label_{lab}", mask, min(k, overall_k * 3))

    # risk buckets (oversample)
    if "is_partial_object" in md.columns:
        risk = md["is_partial_object"].astype(str).str.upper().isin(("Y", "TRUE", "1"))
        _bucket("risk_partial", risk, RISK_CAP)
    if "is_synthetic_location" in md.columns:
        risk = md["is_synthetic_location"].astype(str).str.lower().isin(
            ("true", "1", "y", "yes")
        )
        _bucket("risk_synth", risk, RISK_CAP)
    if "label" in md.columns:
        risk = md["label"].astype(str).str.lower() == "lookalike"
        _bucket("risk_lookalike", risk, RISK_CAP)
    if "truncated_by_scene_edge" in md.columns:
        risk = md["truncated_by_scene_edge"].astype(str).str.upper().isin(
            ("Y", "TRUE", "1")
        )
        _bucket("risk_truncated", risk, RISK_CAP)

    if not picked:
        idxs = sample_indices(n, min(overall_k, n), seed, "fallback")
        return md.iloc[idxs]
    ids = list(picked.keys())
    id_col = "crop_id" if "crop_id" in md.columns else "image_id"
    return md[md[id_col].astype(str).isin(ids)].copy()


def _panel_html(row: pd.Series) -> str:
    keys = [
        "crop_id", "label", "is_partial_object", "truncated_by_scene_edge",
        "is_synthetic_location", "is_calibrated", "parent_group_id",
        "crop_source_scene_id", "offset_km", "source_corridor_id",
        "full_extent_area_px", "full_extent_aspect_ratio",
        "db_conversion_formula_version",
    ]
    items = []
    for k in keys:
        if k in row.index:
            v = row[k]
            items.append(
                f"<tr><th>{html.escape(str(k))}</th>"
                f"<td>{html.escape(str(v))}</td></tr>"
            )
    return "<table class='panel'>" + "".join(items) + "</table>"


_CSS = """
body{font-family:system-ui,Segoe UI,sans-serif;margin:16px;background:#111;color:#ddd}
h1{font-size:1.3rem} h2{font-size:1.1rem;margin-top:2rem}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px}
.card{background:#1b1b1b;border:1px solid #333;border-radius:8px;padding:10px}
.card .imgs{display:flex;gap:6px}
.card img{width:48%;image-rendering:pixelated;background:#000;border:1px solid #444}
.card h3{margin:0 0 6px 0;font-size:.95rem;color:#8cf}
.panel{width:100%;border-collapse:collapse;font-size:.72rem}
.panel th{text-align:left;color:#999;padding:1px 6px 1px 0;font-weight:500}
.panel td{padding:1px 0;color:#eee;word-break:break-all}
.risk{display:inline-block;background:#a33;color:#fff;border-radius:3px;
      font-size:.7rem;padding:1px 5px;margin-right:4px}
.meta{color:#888;font-size:.85rem;margin-bottom:1rem}
"""


def run(
    paths: QAPaths,
    *,
    sample: int = 60,
    seed: int = 42,
    frac: float = DEFAULT_FRAC,
    out_html: Optional[Path] = None,
) -> Report:
    rep = Report("visual_spotcheck")
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

    samples = _pick_samples(md, frac, seed)
    if sample and len(samples) > sample:
        idxs = sample_indices(len(samples), sample, seed, "gallery_cap")
        samples = samples.iloc[idxs]

    cards: List[str] = []
    rendered = 0
    risks_hit = {"partial": 0, "synth": 0, "lookalike": 0, "trunc": 0}

    for _, row in samples.iterrows():
        crop_id = str(row.get("crop_id") or row.get("image_id") or "")
        vv = find_png(paths, crop_id, "VV")
        if vv is None:
            vv = find_png(paths, crop_id, "VH")
        mask_p = find_png(paths, crop_id, "mask")
        if vv is None:
            continue

        import rasterio
        try:
            with rasterio.open(str(vv)) as ds:
                img = ds.read(1)
        except Exception:
            continue
        marr = None
        if mask_p is not None:
            try:
                with rasterio.open(str(mask_p)) as ds:
                    marr = ds.read(1)
            except Exception:
                marr = None

        risks = []
        if str(row.get("is_partial_object", "")).upper() in ("Y", "TRUE", "1"):
            risks.append("overflow/tiled")
            risks_hit["partial"] += 1
        if str(row.get("is_synthetic_location", "")).lower() in ("true", "1", "y"):
            risks.append("synthetic-geo")
            risks_hit["synth"] += 1
        if str(row.get("label", "")).lower() == "lookalike":
            risks.append("lookalike")
            risks_hit["lookalike"] += 1
        if str(row.get("truncated_by_scene_edge", "")).upper() in ("Y", "TRUE", "1"):
            risks.append("edge-truncated")
            risks_hit["trunc"] += 1

        risk_html = "".join(f"<span class='risk'>{html.escape(r)}</span>"
                            for r in risks)
        b64_img = _b64_png(img)
        b64_ov = _overlay_b64(img, marr)
        b64_mask = _b64_png(marr) if marr is not None else ""
        mask_cell = (
            f"<img alt='mask' src='data:image/png;base64,{b64_mask}'/>"
            if b64_mask else "<div style='width:48%'></div>"
        )
        cards.append(f"""
<div class='card'>
  <h3>{html.escape(crop_id)} {risk_html}</h3>
  <div class='imgs'>
    <img alt='raw' src='data:image/png;base64,{b64_img}'/>
    <img alt='overlay' src='data:image/png;base64,{b64_ov}'/>
  </div>
  <div class='imgs' style='margin-top:6px'>
    <div style='width:48%'></div>
    {mask_cell}
  </div>
  {_panel_html(row)}
</div>
""")
        rendered += 1

    # Optional Leaflet map for coordinate review (only if folium available)
    map_html = ""
    try:
        map_html = _try_leaflet(md)
    except Exception:
        map_html = ""

    out = out_html or (paths.dest_dir / "qa_visual_spotcheck.html")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = f"""<!DOCTYPE html>
<html lang='en'><head><meta charset='utf-8'/>
<title>QA visual spot-check</title>
<style>{_CSS}</style></head>
<body>
<h1>SAR dataset — visual spot-check (Layer 3)</h1>
<div class='meta'>samples rendered: {rendered} · seed={seed} ·
risk hits: partial={risks_hit['partial']} synth={risks_hit['synth']}
lookalike={risks_hit['lookalike']} truncated={risks_hit['trunc']}
<br/>Checklist: mask aligns with a dark patch? oil/lookalike mask empty?
Black borders / banding / NaN holes?</div>
{map_html}
<div class='grid'>
{''.join(cards)}
</div>
</body></html>
"""
    out.write_text(doc, encoding="utf-8")

    rep.info(
        f"wrote static gallery with {rendered} crops → {out}",
        out=str(out), rendered=rendered, risks=risks_hit,
    )
    # Soft review flag: gallery exists is not pass/fail — human marks issues.
    rep.soft(
        True,
        f"review queue ready ({rendered} samples) — human pass/fail required "
        "for Layer 3",
        out=str(out),
    )
    if rendered == 0:
        rep.hard(False, "no samples could be rendered (PNGs missing?)")
    return rep


def _try_leaflet(md: pd.DataFrame) -> str:
    """Build an offline-CDN Leaflet map snippet; soft-skip if no coords."""
    if "lat" not in md.columns or "lon" not in md.columns:
        return ""
    pts = []
    for _, row in md.iterrows():
        try:
            la, lo = float(row["lat"]), float(row["lon"])
        except (TypeError, ValueError):
            continue
        if la == 0 and lo == 0:
            continue
        lab = str(row.get("label", ""))
        syn = str(row.get("is_synthetic_location", "")).lower() in ("true", "1", "y")
        pts.append((la, lo, lab, syn))
    if not pts:
        return ""
    # Cap for HTML size
    pts = pts[:2000]
    payload = json.dumps(pts)
    return f"""
<h2>Coordinate scatter (Leaflet via CDN)</h2>
<div id='map' style='height:420px;background:#222'></div>
<link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'/>
<script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'></script>
<script>
(function(){{
  var pts = {payload};
  var map = L.map('map').setView([20, 60], 2);
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
    {{attribution:'© OSM'}}).addTo(map);
  pts.forEach(function(p){{
    var color = p[3] ? '#e67e22' : (p[2]==='lookalike' ? '#9b59b6' :
                 (p[2]==='oil' ? '#e74c3c' : '#7f8c8d'));
    L.circleMarker([p[0], p[1]], {{radius:4, color:color, fillOpacity:0.6}}).addTo(map);
  }});
}})();
</script>
"""
