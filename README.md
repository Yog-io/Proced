# Proced — SAR Oil-Spill Dataset Engineering System (Block 1)

Builds the **fuel** for the project: a high-performance pipeline that turns raw
`.7z` SAR archives into a standardised, validated master dataset — plus every
other Block 1 deliverable (splits, feature tables, AIS, synthetic drift pairs,
fetch wrappers).

The three spec documents in this folder are **final and unchanged**:

| Document | Role |
|---|---|
| `BLOCK1_DATA_TEAM_GUIDE.md` | Task list 1.1–1.10, folder structure, definition of done |
| `DATASET_FORMAT_REFINEMENT_REPORT.md` | Refinements C.0–C.6 — supersede/amend guide tasks |
| `archi.md` | Stage 0–6 architecture, concurrency plan, GEODATA spec, embedded reference script |

This repo **implements** those specs and fixes the issues found in
`archi.md`'s embedded script (analysis below).

---

## Quick start

```bash
pip install -r requirements.txt

# Full gated Zenodo build (brief Steps 0–4) — cross-platform entry point;
# single-dash args work on Windows (PowerShell/cmd) as well as macOS/Linux:
python run_zenodo_build.py -path /path/to/extracted/zenodo -seed 42 -workers 8
python run_zenodo_build.py -path D:\Zenodo-dataset -seed 42 -workers 8
#   (./run_zenodo_build.sh is a thin POSIX wrapper around the same script;
#    exit 0 = all gates passed, 2 = QA/validate fail, 3 = Step 0 STOP, 1 = harness error)

# 1) Unpack is a SEPARATE subcommand — the pipeline never auto-extracts.
#    (Or point --stage_dir at a tree you already extracted yourself.)
python sar_dataset_pipeline.py extract \
    --archive data/raw_sar_inputs.7z \
    --stage_dir data/scratch_unpacked

# 2) Full run scan→validate (no --archive here; exits 2 if validation fails)
#    Zenodo-style asymmetric tree (sibling *_images / *_mask folders) is supported;
#    the output tree mirrors the input layout under --dest_dir.
python sar_dataset_pipeline.py run \
    --stage_dir data/scratch_unpacked \
    --dest_dir data/master_dataset_processed \
    --output_archive data/master_dataset_v1.7z
#    (--workers N is optional — it defaults to every CPU core)

# Example for the Zenodo PC layout (D:/Zenodo-dataset/ … → Zenodo-Dataset_final/):
# python sar_dataset_pipeline.py run \
#     --stage_dir /path/to/Zenodo-dataset \
#     --dest_dir /path/to/Zenodo-Dataset_final \
#     --workers 8 --seed 42

# Or run stages one at a time (intermediate artifacts under --state_dir):
python sar_dataset_pipeline.py scan     --stage_dir data/scratch_unpacked
python sar_dataset_pipeline.py geo      --stage_dir data/scratch_unpacked
python sar_dataset_pipeline.py features --stage_dir data/scratch_unpacked
python sar_dataset_pipeline.py plan     --stage_dir data/scratch_unpacked
python sar_dataset_pipeline.py convert  --stage_dir data/scratch_unpacked
python sar_dataset_pipeline.py metadata --stage_dir data/scratch_unpacked
python sar_dataset_pipeline.py split
python sar_dataset_pipeline.py archive
python sar_dataset_pipeline.py validate

# Validate an existing output tree only
python scripts/validate_dataset.py

# Then the handoff chain:
python scripts/join_wind.py            # → data/features/lookalike_training_data.csv  (Model Training)
python scripts/generate_ais.py         # → data/ais/ais_tracks.csv                    (Backend)
python scripts/generate_model2_pairs.py --pairs 2500   # → data/synthetic/model2_pairs/ (Model Training)
pytest tests/ -q                       # 127 tests incl. end-to-end + standalone stages
```

---

## Pipeline stages (archi.md §1)

Every stage is a **separate subcommand**; intermediate artifacts live under
`--state_dir` (default `data/state/`) so any stage can be re-run alone.

| Stage | Subcommand | What | Module |
|---|---|---|---|
| 0 | `extract` | `.7z` decompression only (7z CLI → py7zr fallback) — **never auto-run** | `proced/archive.py`, `proced/cli.py` |
| 1 | `scan` | Tree parsing, VV/VH↔mask pairing, calibration/label detection, value-domain resolution, **full tree mirror** → `catalog.json` | `proced/catalog.py`, `proced/stage_scan.py` |
| 2 | `features` | **Pre-crop** full-extent shape + radiometric features (C.0) → `instances.csv` | `proced/features.py`, `proced/stage_features.py` |
| 3a | `geo` | Real/synthetic GPS + timestamps → `scene_geo.json` | `proced/geo.py`, `proced/stage_geo.py` |
| 3b | `plan` | Crop geometry (compact/overflow/truncated/background) → `crop_plans.json` | `proced/crops.py`, `proced/stage_plan.py` |
| 4 | `convert` | Parallel windowed tiling → dB calibration → 16-bit uint16 PNG + GEODATA → `crop_rows.json` | `proced/stage_convert.py`, `proced/radiometry.py` |
| 5 | `metadata` | Master `metadata.csv` + feature/provenance CSVs | `proced/metadata.py`, `proced/stage_metadata.py` |
| 6 | `split` / `archive` / `validate` | Group-safe splits (C.5) + `.7z` pack + validation checklist | `proced/splits.py`, `proced/validate.py`, `proced/stage_*.py` |
| — | `run` | Executes `scan → geo → features → plan → convert → metadata → split → archive → validate` in order; **never extracts** | `proced/cli.py` |

Shared infrastructure: `proced/state.py` (artifact I/O), `proced/pool.py`
(spawn ProcessPoolExecutor + GDAL worker tuning), `proced/scene_io.py`
(per-scene raster helpers).

Concurrency per archi §2: `ProcessPoolExecutor` (explicit **spawn** context —
safe with GDAL), per-worker `GDAL_CACHEMAX`/`GDAL_NUM_THREADS`, windowed reads
only (never full-scene reloads per crop), executor recycled every
`--pool-chunk` folders to bound worker RSS, optional CUDA path (NumPy
fallback), deterministic per-scene RNG seeds (`--seed`).

---

## Issues found in `archi.md` and how they're tackled

### Correctness bugs

| # | Issue in the reference script | Fix here |
|---|---|---|
| 1 | **Perpendicular offset wasn't perpendicular.** Offset used a fixed `±π/2` in raw lat/lon space, so every sample drifted *purely east–west* regardless of corridor heading; longitude degrees treated as constant 111 km (wrong off-equator). | `geo.perpendicular_offset_deg`: heading computed in a local east/north metric frame, rotated 90°, longitude scaled by `cos(lat)`. Regression-tested in `tests/test_geo.py`. |
| 2 | **dB heuristic `max(arr) > 1.0` misclassified linear σ⁰ power** (ocean backscatter is almost always ≤ 1 → treated as already-dB) and applied `10·log10` to *amplitude* (needs `20·log10`). | Explicit domain detection (`db`/`power`/`amplitude`/`native`) + correct formulas in `radiometry.py`; overridable via `--value-domain`. Formula version tagged honestly per row. |
| 3 | **`truncated_by_scene_edge` keyed off the crop window** abutting the scene border — flagged background crops near edges, missed objects whose crop sits inside the scene. Spec (A.5/C.1) is an *object* property. | Flag computed from **mask pixels on the scene border** in Stage 2 (`features._touches_border`), propagated to that object's crops only; background always `N`. |
| 4 | **Dual-pol split files double-processed.** `scene_VV.tif` and `scene_VH.tif` were two "scenes": wrong band tags, jittered windows written twice, VV/VH crops misaligned. | Catalog pairs pol-suffixed files into one logical scene (`catalog._strip_pol_suffix`), one crop plan, aligned windows for both bands. |
| 5 | **Overflow tiling produced phantom-oil tiles** (thin slicks got a 2-row grid whose second row contained no object pixels → background labelled `oil`) and duplicate clamped windows. | Axis positions computed from true coverage, de-duplicated, and **every kept tile must contain ≥1 object pixel** (`crops.plan_crops`). |
| 6 | **Negative sampling could overlap the buffer zone** (checked only the top-left corner) and crashed/misbehaved on scenes ≤ 256 px; fixed cap of 10 per scene regardless of positive count. | Distance transform + **integral image over the whole window** (all pixels ≥ 300 px clear), in-bounds candidate grid, ratio-based count with `max(1, ⌈ratio·N⌉)` floor (C.3), configurable via `--neg-ratio/--neg-buffer-px`. |
| 7 | **Negative samples taken from mask-less scenes** — unlabeled oil would become false "clean sea". | Disabled by default (`--negatives-from-unlabeled` to opt in); verified-clean empty masks still get `neg_per_empty_scene` patches. |
| 8 | **Calibration hardcoded `Y` + always dB-converted** — Kaggle 8-bit exports would be destroyed (C.2 says keep native). | Detection heuristics (float GeoTIFF ⇒ calibrated) + `--calibrated-match/--uncalibrated-match` overrides; uncalibrated rows pass through as native 8-bit with `db_conversion_formula_version=native_8bit_no_db`. |
| 9 | **Label hardcoded `"oil"`** — no look-alike class anywhere. | Mask value `2` ⇒ `lookalike`; binary masks inherit a per-folder default (path heuristic / `--label-override`); guide's `oil/lookalike/background` trichotomy respected end-to-end (incl. classifier filtering). |
| 10 | **Synthetic-location provenance incomplete** — no `source_corridor_id`, `offset_km`, or `synthetic_timestamp_utc` (guide §2.2 Steps 3–4), no corridor GeoJSON (Step 1). | Corridors live in `data/reference/shipping_corridors.geojson` (8 lanes incl. Suez & South China Sea); every synthetic row carries full provenance + uniform 2016–2025 timestamp; written to `synthetic_geo_assignments.csv`. |
| 11 | **Synthetic bbox corners were a fixed `0.02°` box** regardless of pixel size/latitude. | Scene-level synthetic georef at ~10 m/px with `cos(lat)` longitude scaling (`geo.synthetic_scene_georef`). |
| 12 | **Georeference test `gt[0]==0 and gt[3]==0`** false-negatives (valid origin at 0,0) and false-positives (identity transform without CRS → bogus "real" coordinates). | `raster_io.has_real_georeference`: requires CRS **and** non-identity transform. |
| 13 | **Mask pairing** only handled `stem.replace("_mask","") in scene_name`; size mismatches would crash or silently misalign. | Convention matcher (same stem, `mask_*`, `*_mask/_label/_gt`, sibling `masks/` dirs) **plus asymmetric sibling trees** (`*_images` → `*_mask` at any depth, folder-based mask detection, global stem index); dimension mismatch ⇒ scene treated as unlabeled (never resize labels silently). |
| 14 | **Aspect ratio from `cv2.fitEllipse` without sorting axes** → could return < 1. | Major/minor sorted; `minAreaRect` fallback for tiny contours. |
| 15 | **`parent_group_id` 8-hex chars** (collision-prone), `"none"` for compact crops. | Full `uuid4` hex-12 per object group; negatives get their own group id (never empty → null-free CSVs). |
| 16 | **`bool(nan) is True` class of bugs** in metadata coercion (NaN → `"Y"`). | All normalisers check NaN/None first (`metadata._is_nan`); unit-tested. |

### Missing capabilities (required by the fixed specs)

| Spec requirement | Status |
|---|---|
| C.0 radiometric features (damping ratio, boundary gradient, backscatter variance ratio, GLCM contrast/homogeneity, NDPI) — *script only did shape* | Implemented in `features.py` from **pre-crop** full-extent geometry + local radiometric regions; eligible/ineligible rows flagged |
| Master `metadata.csv` (Task 1.3/C.4 superset) — *script only wrote per-folder GEODATA* | Written at dest root; GEODATA keeps the archi column spec first |
| Look-alike classifier table (Task 1.7) | `scripts/join_wind.py` → `lookalike_training_data.csv` (filters truncated + uncalibrated + background per A.5/Part B; mock winds flagged until CDS creds exist) |
| Train/val/test split grouped by scene (Task 1.5/C.5) | `proced/splits.py` — leakage asserted in both unit tests and validation checklist |
| Task 1.1 audit / 1.3 unify / 1.8 AIS / 1.9 fetch wrappers / 1.10 OpenDrift pairs | `scripts/audit_datasets.py`, `scripts/unify_datasets.py`, `scripts/generate_ais.py`, `lib/fetch_*.py`, `scripts/generate_model2_pairs.py` |
| Validation checklist (archi §5) | `proced/validate.py` runs automatically at end of every pipeline run (exit code 2 on failure) + standalone script |

### Optimisations vs the reference script

* **Windowed everything** — radiometric features read instance bbox+ring only; crops stream 256×256 windows (archi §2 memory guardrails honoured).
* **Executor recycling** (`--pool-chunk`) — bounds worker RSS on long runs without needing Python 3.11 `max_tasks_per_child`.
* **`rasterio` as the GDAL binding** — identical GDAL semantics, pip-installable wheels, and the exact `rasterio.windows` primitive the refinement (A.4) endorses; `gdalinfo` still used for bit-depth checks when present (falls back to rasterio).
* **Per-worker GDAL cache** defaults to 256 MB (configurable) so N workers can't silently claim `N × 512 MB`.
* **All cores by default** — `--workers` defaults to `os.cpu_count()` (and `run_zenodo_build.sh` picks the same), with BLAS/OpenMP (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`) pinned to 1 thread per worker so N processes don't fight over the same cores; `--gdal-threads` caps GDAL's own threads per worker (default `2`).
* **Progress bars for every step and stage** — `run` prints `▶ [3/9] features` / `✓ features 12.4s` and a master `pipeline 3/9` bar; each heavy loop (pool folders, scenes, value-domain probes, validation folders, archive entries, raw-audit files, QA modules) has its own bar. Interactive terminals get a live bar; `tee`/log output gets throttled `[progress] …` lines; `PROGRESS=0` disables everything.
* **Immediate memory release** — full-scene arrays are dropped the moment derived data exists (raw bands after dB conversion, scene masks after planning/conversion), pool results are cleared as they are consumed, and the parent runs `gc.collect()` between pool chunks, so RAM never stacks up across scenes/folders on a long Zenodo run.
* **Deterministic seeding** per `(scene folder, scene id)` — parallel scheduling no longer changes jitter/negative choices; reruns reproduce byte-identical plans.
* **Resume** — folders with an existing `GEODATA.csv` are skipped unless `--force`.
* **GPU only when real** — CUDA used if actually available; MPS/CPU never take a fake GPU path.
* **7z without p7zip** — py7zr fallbacks for extract *and* pack (the reference script crashed outright), preserving empty directories for the tree-equivalence check.

### Honest limitations (flag upward — guide §7)

* Synthetic coordinates are a **known, deliberate approximation** — never show them as evidentiary spill locations (`is_synthetic_location` is load-bearing; preserved on every row and in `synthetic_geo_assignments.csv`).
* `join_wind.py --source mock` (default) emits **flagged mock winds** so the handoff file can exist before CDS credentials; switch to `--source era5` for production.
* `generate_model2_pairs.py` defaults to the **`simple` physics backend** (uniform fields, windage 3 %, random-walk diffusion — every pair's `wind_field_ref`/`current_field_ref` says `synthetic_*`). Use `--backend opendrift --currents-nc … --winds-nc …` once forcing data is available.
* Scene-edge-truncated instances are **excluded from the classifier table** but kept for Model 1 crops (A.5/C.1).

---

## Deliverables map (guide §4)

```
data/
├── raw/DATASET_AUDIT.md                     ← scripts/audit_datasets.py (Task 1.1)
├── processed/                               ← scripts/unify_datasets.py (Task 1.3, supplementary sets)
│   ├── unified_masks/*.png  unified_images/*.png  metadata.csv
├── state/                                   ← intermediate stage artifacts (--state_dir)
│   ├── catalog.json  scene_geo.json  instances.csv  crop_plans.json  crop_rows.json
├── splits/{train,val,test,manifest}.csv     ← Stage 6 (grouped by scene, C.5)
├── features/
│   ├── shape_features.csv                   ← Stage 5 (Task 1.6 / C.0)
│   ├── full_extent_features.csv             ← Stage 5 (C.0 shape + radiometric)
│   ├── synthetic_geo_assignments.csv        ← Stage 5 / scripts/assign_synthetic_geo.py (§2.2)
│   └── lookalike_training_data.csv          ← scripts/join_wind.py (Task 1.7) ★ hand to Model Training
├── ais/ais_tracks.csv                       ← scripts/generate_ais.py (Task 1.8)
├── synthetic/model2_pairs/*.geojson         ← scripts/generate_model2_pairs.py (Task 1.10) ★ hand to Model Training
└── reference/shipping_corridors.geojson     ← shared by geo assignment, AIS, Model 2

lib/fetch_currents.py, lib/fetch_winds.py    ← Task 1.9 ★ hand to Backend

data/master_dataset_processed/               ← pipeline output
├── metadata.csv                             ← master table (guide Task 1.3 + C.4 columns)
├── pipeline_summary.json                    ← run stats + validation report
└── <region>/<subpass>/… crops … + GEODATA.csv   (tree mirrors the raw archive)
```

**Block 2 needs this for inverting the pixel mapping** (C.2):

```python
db_value = (pixel_uint16 / 65535) * 30 - 30    # v1_linear_neg30_to_0
```

---

## GEODATA.csv columns (archi §4)

`crop_id, crop_source_scene_id, is_calibrated, has_dual_pol,
is_synthetic_location, crop_bbox_corners, is_partial_object, parent_group_id,
truncated_by_scene_edge, db_conversion_formula_version, label,
full_extent_area_px, full_extent_perimeter, full_extent_boundary_complexity,
full_extent_aspect_ratio`
— plus master-only provenance (`lat, lon, timestamp_utc, dataset_source,
has_real_geo, source_corridor_id, offset_km, crop_xoff/yoff/size, …`).

Booleans render as spec `true`/`false` or `Y`/`N` per column; required
columns are never null (validated every run).

---

## Independent QA suite (`qa_verification/`)

Second layer of verification per `DATASET_VERIFICATION_ARCHITECTURE.md`.
**Hard boundary: `qa_verification/` never imports `proced/`** — every check
recomputes from raw files with fresh `rasterio`/`cv2`/`shapely`/`pandas` and
compares against pipeline artifacts. Schema constants are duplicated locally
in `qa_verification/_lib.py`.

```
qa_verification/
├── independent_checks/     ← stage-wise recomputes (dB round-trip, shape, plan, …)
├── structural_checks/      ← schema nulls, cross-file joins, split leakage
├── visual_spotcheck/       ← risk-sampled HTML gallery (Layer 3)
├── stats_dashboard/        ← histograms + geo scatter (Layer 2)
├── adversarial/            ← malformed fixtures, determinism, graceful failure
├── raw_audit/              ← standalone pre-pipeline raw-folder audit (§8)
└── run_all_verification.py ← aggregates → verification_report.json
```

```bash
# full suite (exit 0 pass / 2 hard-fail / 1 harness error)
python qa_verification/run_all_verification.py

# knobs
python qa_verification/run_all_verification.py \
    --sample-size 200 --seed 7 \
    --only stage4_db_roundtrip,stage5_schema_nulls \
    --skip-dashboards --skip-visual --skip-adversarial \
    --out /tmp/verification_report.json

# path overrides (else defaults under data/)
QA_STAGE_DIR=… QA_DEST_DIR=… QA_STATE_DIR=… \
QA_FEATURES_DIR=… QA_SPLITS_DIR=… QA_CORRIDORS=… \
python qa_verification/run_all_verification.py
```

Report is deliberately **separate from** `pipeline_summary.json` so it's clear
which findings came from independent recomputation vs. the pipeline grading its
own homework. Hard-fail blocks handoff (nulls, orphans, dB mismatch, split
leakage, non-binary mask, offset bearing); soft findings are flagged for review.

### Standalone raw-folder pre-pipeline audit (`qa_verification/raw_audit/`)

Runs **before any pipeline stage** (architecture §8): point it at a freshly
extracted `.7z` tree and get a fast "is this raw data intact" verdict —
corrupted opens, multi-part extraction gaps, orphaned VV/VH/mask files,
resolution outliers, and byte-identical duplicates. No `--stage_dir` /
`--dest_dir` / pipeline state required; same hard rule — never imports `proced/`.

```bash
python qa_verification/raw_audit/run_raw_folder_audit.py \
    --root path/to/extracted/dataset \
    --out raw_folder_audit_report.json   # optional; defaults under --root
```

Console summary prints pass/warn/fail counts + `ready for pipeline: yes/no`
(exit 0 ready / 2 not ready). Full detail lands in `raw_folder_audit_report.json`.

---

## Tests

```bash
pytest tests/ -q          # 127 tests (86 pipeline + 16 runner + 25 QA)
pytest tests/ -q -m "not slow"   # skip the end-to-end / standalone-stage runs
```

Covers: radiometry round-trip & domain detection · perpendicular-offset
regression · corridor provenance · compact/overflow/truncated/negative crop
invariants · C.0 features (incl. damping/NDPI/GLCM) · group-split leakage ·
catalog pairing/calibration/labels · **asymmetric sibling image/mask trees**
(Zenodo `*_images` / `*_mask` layout, nested subfolders, folder-based mask
detection, tree mirror) · metadata null-freeness · 7z round-trip ·
validation checklist · **state serialisation round-trips** · **full Stage 0–6
end-to-end** (extract → run, bit depth, tree equivalence, provenance, splits,
archive) · **every stage runnable standalone** · `run` rejects `--archive`
with a hint to use `extract` · **independent QA suite** (boundary rule, dB
round-trip, shape/plan recompute, full `run_all_verification` on a fixture) ·
**raw-folder pre-pipeline audit** (discovery, integrity, pairing, duplicates,
CLI exit codes, sibling-tree pairing) · **gated build runner**
(`run_zenodo_build.py`: single-dash `-path/-seed/-workers` CLI for Windows,
§4.1 stop gates, `BUILD_STATUS_*.md` writer, streamed step logs).
