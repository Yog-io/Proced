# Dataset Pipeline Verification & QA Architecture
### For the Block 1 team to build: an independent verification suite for the `Proced` pipeline — current scope: Zenodo dataset only

---

## 0. Purpose, Scope & Ownership

**Scope:** this plan currently covers only the **Zenodo dataset** (the multi-part `.7z` archives from the org), since that's the dataset being processed right now. The team is tackling datasets one at a time, so checks specific to other sources — e.g. the RGB multi-class mask palette used by OSD/M4D-style datasets, which doesn't apply to Zenodo's mask convention — are deliberately out of scope here. When Kaggle/OSD processing starts later, extend this document with whatever source-specific checks that dataset needs.

**Ownership:** this is built and run by the **Block 1 team itself**, as a second, independent layer alongside their existing `pytest tests/` suite — not handed to a separate QA team. The distinction that matters is architectural, not organizational: the pipeline's own 68 tests verify the **code behaves as designed**, often against fixtures the same team constructed. This suite's rule is different: **every check here recomputes the answer independently, from raw first principles (fresh `rasterio`/`shapely`/`cv2` calls on the original files, never importing `proced/`'s own functions), and compares against what the pipeline produced.** A check that calls into `proced/` to verify `proced/` is a duplicate unit test, not an independent one — keep the two clearly separated in the codebase (Section 7), even though the same team writes both.

---

## 1. Three Verification Layers

1. **Structural/Schema** — do the right files exist, in the right format, with no forbidden nulls or orphaned rows?
2. **Statistical/Distributional** — do value ranges, class balances, and counts make physical/statistical sense?
3. **Semantic/Visual** — does a human, looking at a sample of actual images, agree the output is correct?

All three run every time; Layer 3 needs a human in the loop and produces a review queue rather than a pass/fail.

---

## 2. Stage-by-Stage Independent Checks

### Stage 0 — Extract
- File count and total size of the unpacked tree matches the source manifest (no partial extraction).
- **Multi-part archive completeness (Zenodo-specific):** since the org distributes this dataset as multiple `.7z` parts, explicitly verify every part was present and successfully joined/extracted before the pipeline ran — confirm the pipeline actually errors out (rather than silently proceeding on a partial dataset) if a part is missing or fails its own archive checksum. This is worth checking deliberately: a silently-incomplete multi-part extraction is exactly the kind of failure that produces a merely-smaller-than-expected dataset rather than an obvious crash, and could easily go unnoticed until much later.

### Stage 1 — Scan (`catalog.json`)
- For 30 randomly sampled scenes: open the raw file yourself with `rasterio`, independently read dtype, band count, CRS, and value range. Compare against what `catalog.json` recorded for calibration/label/value-domain detection.
- Regression check for bug #4: assert no two catalog scene entries reference the same underlying scene via a stripped VV/VH suffix (i.e., confirm dual-pol pairing actually happened, not two independent "scenes").

### Stage 2 — Features (`instances.csv`, C.0 full-extent)
- For 50 sampled instances: recompute area, perimeter, aspect ratio, and fragment count **from the raw mask polygon yourself** (fresh `cv2.findContours` + `shapely`), compare to `instances.csv` within a small numerical tolerance.
- Assert `aspect_ratio ≥ 1.0` for 100% of rows (regression for bug #14 — sorted major/minor axes).
- Recompute `truncated_by_scene_edge` independently: does the mask actually have foreground pixels touching row 0, row max, col 0, or col max of the *original scene*? Compare to the stored flag for every instance, not a sample — this one's cheap enough to check exhaustively.
- For calibrated rows only: recompute damping ratio independently (mean background ring dB − mean slick dB, using your own ring definition) and confirm it's in a physically plausible range (oil: roughly 6–15 dB per the earlier research; flag, don't hard-fail, anything wildly outside that as worth a look).
- Confirm radiometric feature columns are genuinely null/ineligible (not zero, not a silently-computed garbage value) for every row where `is_calibrated == N`.

### Stage 3a — Geo (`scene_geo.json`)
- **Real-geo scenes:** for a sample, independently reproject the scene's corner pixels using the stored CRS/transform and confirm the resulting lat/lon falls within the Zenodo dataset's known geographic coverage area (won't catch every error, but catches gross ones like a transposed axis or wrong EPSG).
- **Synthetic-geo scenes:** confirm every row has `is_synthetic_location = True`; confirm the assigned coordinate actually lies on (or within `offset_km` of) one of the 8 corridors in `shipping_corridors.geojson` — recompute the point-to-line distance yourself, don't trust the stored `offset_km`; confirm `synthetic_timestamp_utc` falls in 2016–2025.
- **Regression check for bug #1 (perpendicular offset):** for 20 synthetic assignments, recompute the corridor's true heading at that point and the true bearing from the corridor line to the assigned point in a proper local metric frame (not raw lat/lon degrees). Confirm the two bearings differ by ~90° ± a few degrees. This is the single most important geo check — it's the exact bug that was found and fixed, so it's the one most worth re-verifying rather than assuming fixed.

### Stage 3b — Plan (`crop_plans.json`)
- Every compact object → exactly 1 crop. Every overflow object → ≥2 tiles, all sharing one `parent_group_id`, and **every tile independently confirmed to contain at least 1 object-mask pixel** (regression for bug #5 — recheck by loading each tile's mask window fresh, don't trust the plan's own bookkeeping).
- For every background/negative crop: independently recompute the minimum distance from every pixel in that crop's window to the nearest positive polygon (fresh distance computation) and confirm it's ≥ the configured buffer everywhere in the window, not just at the top-left corner (regression for bug #6).
- Confirm negative-count-to-positive-count ratio roughly matches the configured `--neg-ratio` across the dataset as a whole.

### Stage 4 — Convert (crops, PNGs, `crop_rows.json`)
- **The single most important check in this whole document — the dB round-trip:** for 100+ sampled calibrated crops, take the stored 16-bit pixel value, apply the documented inverse formula (`db = (px/65535)*30 - 30`), and compare against the dB value you compute **completely independently** by re-opening the *original raw scene* at that crop's exact window offset and reading the true pixel value yourself. These must match within floating-point rounding tolerance. This is the check that actually proves the conversion pipeline is correct end-to-end, not just internally self-consistent.
- Every PNG opens without a decode error; correct bit depth (16-bit calibrated / native 8-bit uncalibrated); correct dimensions (256×256).
- Every mask crop PNG contains strictly `{0, 1}` values after loading — no stray anti-aliased gray values (a common silent PNG-encoding bug).
- Recompute each crop's `crop_bbox_corners` independently from `crop_xoff/yoff/size` + the parent scene's transform (the A.4 math), compare to what's stored.

### Stage 5 — Metadata (`metadata.csv`, feature CSVs)
- Full null-sweep on every column the spec marks as required — zero tolerance, not sampled (regression for bug #16).
- **Join integrity, both directions:** every `crop_id` in `metadata.csv` has a corresponding PNG on disk and a corresponding row in the per-folder `GEODATA.csv`; every PNG on disk has a corresponding metadata row (catches orphaned files either direction).
- Label distribution (`value_counts()` on `label`) roughly matches the expected ~1,400 oil / ~700 look-alike from Zenodo plus whatever negative-patch volume was targeted — flag large unexplained deviations.
- In `lookalike_training_data.csv` specifically: assert zero rows where `truncated_by_scene_edge == Y` or `is_calibrated == N` — these are supposed to be filtered out per spec; verify the filter actually ran, don't assume it did because the README says it does.
- Confirm every mock-wind row is visibly flagged as such in the file itself, not just in the script's default parameter.

### Stage 6 — Split / Archive / Validate
- **Group leakage, independently reverified:** for every `crop_source_scene_id`, confirm it appears in exactly one of train/val/test — recompute this yourself from the split CSVs rather than trusting the pipeline's own internal assertion.
- Unpack the output `.7z` fresh into a clean scratch directory; diff file list and checksums against the pre-archive tree to confirm no corruption occurred during packing.
- Confirm train/val/test roughly preserve the overall oil/look-alike/background ratio (stratification sanity, not exact match).

---

## 3. Visual Spot-Check Protocol (Layer 3 — human in the loop)

- Randomly sample ~2% of crops per label class, **oversampling the highest-risk categories for Zenodo's pipeline**: overflow/tiled objects (phantom-oil risk from bug #5), synthetic-location rows (perpendicular-offset risk from bug #1), and label==lookalike rows generally (since mask-value-2 detection is a newer code path than plain oil/background and has had less real-world exposure so far).
- For each sampled crop, render one side-by-side view: the raw image (converted back from 16-bit via the inverse dB formula + a contrast stretch purely for human viewing), the mask overlay, and a small text panel showing `label`, `is_partial_object`, `truncated_by_scene_edge`, and `is_synthetic_location`.
- Reviewer checklist per sample: does the mask actually align with a visible dark patch in the image? Is the mask suspiciously empty for a row labeled `oil`/`lookalike`? Any visible artifacts (black borders, banding, NaN holes) from a windowed read gone wrong?
- Output: a static HTML gallery (one file, grid layout, no server needed) so this can be reviewed by anyone on the team without touching the pipeline code.

---

## 4. Statistical Sanity Dashboards

- Histograms of `full_extent_area_px`, `aspect_ratio`, and `damping_ratio_db`, split by label — oil should show a visibly higher median damping ratio than look-alike (some overlap is expected and fine; a *complete* overlap would suggest something's wrong upstream).
- Histogram of output crop file sizes — should cluster tightly around a few hundred KB. Any file anomalously large is a regression check for the original 22GB/full-scene problem resurfacing.
- A world-map scatter of every real + synthetic coordinate — real points should fall within Zenodo's known coverage region; synthetic points should visibly cluster along the 8 defined corridors, not scattered randomly or sitting on land/at (0,0).

---

## 5. Adversarial & Determinism Checks

- Feed the actual pipeline binary (not a mocked unit test) a handful of deliberately malformed inputs: a corrupted TIFF, a mask with mismatched dimensions, an all-zero mask, a mask touching all four scene borders at once, a scene smaller than 256×256. Confirm each fails gracefully (flagged/skipped/logged) rather than crashing or silently producing a bad row.
- Run the full pipeline twice on the same input subset with the same `--seed`; assert byte-identical output. This directly tests the determinism claim in the README rather than accepting it on faith.

---

## 6. Pass/Fail Criteria

**Hard-fail (blocks handoff to Block 2/Backend):** any null in a required column · any orphaned crop/metadata row in either direction · any dB round-trip mismatch beyond tolerance · any confirmed train/val/test group leakage · any non-binary mask pixel · any confirmed perpendicular-offset bearing error.

**Soft-fail (flagged for review, doesn't block):** label class imbalance beyond an agreed range · a visual spot-check reviewer flags a sample · mock-wind rows present without their flag being obviously visible downstream.

Aggregate everything into one `verification_report.json`, separate from the pipeline's own `pipeline_summary.json`, so it's clear which findings came from independent recomputation vs. the pipeline grading its own homework.

---

## 7. Suggested Folder Structure

```
qa_verification/
├── independent_checks/
│   ├── recompute_shape_features.py       # fresh cv2/shapely, no proced/ imports
│   ├── recompute_geo_transform.py
│   ├── recompute_db_roundtrip.py         # the single most important script here
│   └── recompute_perpendicular_offset.py
├── structural_checks/
│   ├── test_schema_nulls.py
│   ├── test_cross_file_joins.py
│   └── test_split_leakage.py
├── visual_spotcheck/
│   └── sample_and_render.py              # outputs one static HTML gallery
├── stats_dashboard/
│   ├── generate_histograms.py
│   └── generate_geo_scatter.py
├── adversarial/
│   ├── malformed_input_fixtures/
│   ├── test_graceful_failure.py
│   └── test_determinism.py
└── run_all_verification.py               # aggregates into verification_report.json
```

**Tooling:** `rasterio`/`shapely`/`cv2` for independent recomputation, `pandas` for join/statistical checks, `matplotlib` + `folium` for the histograms and geo-scatter, plain HTML/CSS for the visual gallery (no server needed). Keep `qa_verification/` as its own top-level folder alongside `proced/` and `tests/` in the same repo — same team, same repo, but no imports crossing from `qa_verification/` into `proced/`. That's the boundary that keeps the checks independent rather than circular.

---

## 8. New Addition: Standalone Raw-Folder Pre-Pipeline Audit

### Why this is different from everything above
Sections 2–7 all verify **pipeline output** — they run *after* `scan`/`convert`/`metadata` have already processed the data. This section covers a different moment: right after `extract`, before you've committed to a full pipeline run at all. The goal is a fast, standalone tool you can point at any freshly-extracted `.7z` tree and get an immediate "is this raw data actually intact" answer — catching a corrupted or incomplete extraction in seconds/minutes, rather than discovering it hours into a `run`.

### What it does
- **Input:** one root folder path — no assumption about depth or naming convention. Recursively discovers every raster (`.tif`/`.tiff`) and mask (`.png`/`.tif`) file at any nesting level under it, exactly mirroring however the multi-part `.7z` archives happened to extract.
- **File-level integrity:** independently attempt to open every discovered file with `rasterio`; log any that fail (corrupted extraction, truncated file, wrong format) rather than letting the full pipeline fail on it mid-run.
- **Multi-part-extraction gap detection:** flag any subfolder whose file count is suspiciously low relative to sibling subfolders — the classic signature of one `.7z` part having failed to extract or merge in cleanly.
- **Pairing completeness (freshly re-derived, not imported from `catalog.py`):** re-implement the same VV/VH/mask naming logic independently in the QA library, and report every orphaned image (no matching mask) or orphaned mask (no matching image), in both directions.
- **Resolution sanity:** flag any scene whose dimensions deviate meaningfully from the expected ~2000×2000 — could indicate an accidentally-mixed-in file from an unrelated source.
- **Dtype/CRS/domain snapshot:** record dtype, band count, CRS presence, and a coarse value-domain guess per file, independently. Cross-reference this later against `scan`'s own `catalog.json` conclusions once the pipeline actually runs, to catch any drift between the fast pre-check and the pipeline's own detection logic.
- **Duplicate detection:** partial-file hashing to catch byte-identical files accidentally duplicated across merged archive parts.
- **Output:** `raw_folder_audit_report.json` + a console summary (pass/warn/fail counts, the specific problem files, and a plain "ready for pipeline: yes/no" verdict).

### Usage
```bash
python qa_verification/raw_audit/run_raw_folder_audit.py --root <path/to/extracted/dataset>
```
No `--stage_dir`/`--dest_dir`/pipeline state required — this is fully standalone, runs before any pipeline stage, and doesn't care whether `scan` has ever been run on this folder.

### Folder addition
```
qa_verification/
└── raw_audit/
    ├── discover_tree.py       # recursive walk, no depth assumption
    ├── check_integrity.py     # open-and-verify every raster independently
    ├── check_pairing.py       # independent VV/VH/mask pairing re-check
    ├── check_duplicates.py    # partial-hash duplicate detection
    └── run_raw_folder_audit.py
```

Same hard boundary as the rest of the suite applies here too: no imports from `proced/` — the pairing/domain-detection logic in `check_pairing.py` is a second, independent implementation, deliberately not a call into `catalog.py`. If the two ever disagree once `scan` runs, that disagreement is itself a useful signal worth investigating, not a bug to silently reconcile.

**Extending this later:** when the team starts processing the next dataset (Kaggle/OSD), revisit every stage above for that source's specific quirks — most notably, add a mask-palette verification step if that source turns out to use an RGB multi-class convention rather than Zenodo's integer mask values.