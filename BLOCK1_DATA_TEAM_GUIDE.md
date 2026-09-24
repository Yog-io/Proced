# BLOCK 1: Data & Dataset Engineering — Team Working Guide
### Read this end-to-end before starting. Everything you need is here.

---

## 0. Your Mission

You are building the **fuel** for the whole project. Two other teams (Model Training, Backend) are entirely blocked on your output. Your job has three parts:

1. Collect and clean **real SAR oil-spill imagery/masks** → feeds Model 1 + the look-alike classifier.
2. Solve the **"we don't know where these images actually are" problem** → without this, nobody can join wind/current data to anything.
3. Generate the **synthetic OpenDrift drift-pair dataset** → feeds Model 2.

You do NOT need to know anything about neural networks. You need to know: file handling, basic geospatial Python, and care about data integrity (correct labels, no leakage between train/test).

---

## 1. Dataset Inventory — What You Found, Analyzed

Your team found 4 Kaggle datasets. Here's the read on each, and **what to do before trusting any of them**:

| Dataset | Link | What it likely is | Verification needed |
|---|---|---|---|
| SAR Oil Spill Segmentation Dataset (SOS) | [bitsandlayers/sar-oil-spill-segmentation-dataset-sos](https://www.kaggle.com/datasets/bitsandlayers/sar-oil-spill-segmentation-dataset-sos/data) | Combines PALSAR + Sentinel-1 imagery for segmentation — sounds like it includes actual masks, not just raw images | **Task 1.1** below |
| Sentinel-1 SAR Oil Spill Train Images | [rohithsheregar/sentinel1-sar-oil-spill-train](https://www.kaggle.com/datasets/rohithsheregar/sentinel1-sar-oil-spill-train) | Likely a raw-image collection (possibly no masks — name doesn't say "segmentation") | **Task 1.1** below |
| Sentinel-1 SAR Oil Spill Test | [rohithsheregar/sentinel1-sar-oil-spill-test](https://www.kaggle.com/datasets/rohithsheregar/sentinel1-sar-oil-spill-test) | Test-split counterpart to the above | **Task 1.1** below |
| Deep-SAR Oil Spill Segmentation (Refined) | [bakhtiyar2222/deep-sar-oil-spill-segmentation-refined](https://www.kaggle.com/datasets/bakhtiyar2222/deep-sar-oil-spill-segmentation-refined) | A "refined" (likely cleaned-up masks) segmentation set | **Task 1.1** below |

**Important honesty note for your team and for judges:** I could not verify the exact class lists (does "look-alike" exist as a separate label, or is it just binary oil/no-oil?), image counts, or file formats inside these from the Kaggle page alone — Kaggle's page content isn't fetchable without actually downloading. **Do not assume anything about these datasets until Task 1.1 is done.**

None of these four are the same as the three datasets we identified earlier from published research (OSD Dataset, the Zenodo Sentinel-1 Lookalike Set, the ESSD 2025 Mediterranean Set) — those are peer-reviewed/expert-annotated with a confirmed separate "look-alike" class. **Treat the Kaggle sets as a supplementary/bonus pool, and the three research datasets as your primary source for the look-alike classifier specifically.** The Kaggle sets may still be very useful for Model 1 (pure oil-vs-background segmentation) even if they turn out to lack a look-alike class.

### Task 1.1 — Verify actual dataset contents (DO THIS FIRST, before anything else in this doc)
1. Download all 4 Kaggle datasets (`kaggle datasets download -d <dataset-slug>`, requires a free Kaggle API token — see kaggle.com/settings for how to generate one).
2. For each, open the folder and record in a shared sheet:
   - Total image count
   - Are there mask files? What format (PNG/TIFF/NumPy array)?
   - Is there a class label file, or just a folder-per-class structure?
   - Do filenames or any metadata.csv/json contain a date, location, or sensor pass ID?
   - Image dimensions and channel count (1-band grayscale vs. 2-band VV/VH)
3. Write this into `data/raw/DATASET_AUDIT.md` — a short table, one row per dataset, columns: `name, image_count, has_masks(Y/N), has_lookalike_class(Y/N), has_any_geo_or_time_metadata(Y/N), format, notes`.
4. **Do not proceed to Task 1.2 until this audit is done and shared with the Model Training team** — it directly changes what they can train on.

---

## 2. THE CORE PROBLEM: No Dataset Has Real GPS Coordinates

None of the public SAR oil-spill datasets we've found — Kaggle or the research ones — reliably ship with the exact lat/lon of each spill. This matters because two downstream things need a coordinate for every image:
- The **look-alike classifier** needs "wind speed at that location/time" as a feature.
- The **Model 2 synthetic drift generator** needs a real ocean location to pull real historical current/wind data from (you can't run OpenDrift without picking somewhere on Earth's actual ocean).

### 2.1 Our Approach: Assigned Synthetic Coordinates

Since we can't know the real location, we **assign each image a synthetically generated but plausible ocean coordinate**, biased toward realistic shipping-lane geography (because real spills disproportionately happen near vessel traffic, not in the middle of nowhere). This is a **known, deliberate approximation** — document it as such everywhere, and never present these coordinates as real evidence in the final dashboard/dossier for demo images.

**Why this is defensible for a hackathon:** the exact coordinate doesn't need to be "true" for training purposes — what matters statistically is that the *wind speed* and *current* values used in training are realistic ocean values, not that they match one specific real historical event. As long as we're honest about this in the README/pitch, it's a reasonable, explainable stopgap.

### 2.2 The Algorithm (build this as `scripts/assign_synthetic_geo.py`)

**Step 1 — Define shipping corridors.**
Hand-define 6–10 major shipping lane corridors as simple polylines (a list of a few lat/lon waypoints each). Use publicly known busy sea routes as reference, e.g.:
- Strait of Malacca corridor
- Gulf of Aden → Arabian Sea → Mumbai corridor
- Persian Gulf / Strait of Hormuz corridor
- Suez Canal approach → Eastern Mediterranean corridor
- English Channel / North Sea corridor
- Gulf of Mexico corridor

Store these as `data/reference/shipping_corridors.geojson` — each corridor is a `LineString` feature with an `id`.

**Step 2 — For each image needing a coordinate, sample a point:**
```
1. Randomly pick one corridor (weighted equally, or weight busier corridors higher if you want realism)
2. Pick a random point t ∈ [0,1] along that corridor's polyline → interpolate to get (lat_c, lon_c)
3. Sample a perpendicular offset distance `d` from a half-normal distribution
   (scale ≈ 50 km) — this means MOST points land close to the route,
   a few land farther away, matching how real spills cluster near traffic
   but aren't always exactly on the lane centerline
4. Pick a random perpendicular direction (left or right of route heading)
5. Compute the offset point (lat_final, lon_final) using the direction + distance
6. Check the point is actually in water:
   use the `global-land-mask` Python package (`from global_land_mask import globe;
   globe.is_ocean(lat, lon)`) — if it's on land, go back to step 2 and resample
7. Assign (lat_final, lon_final) to the image
```

**Step 3 — Assign a synthetic timestamp.**
Randomly sample a timestamp uniformly within a range where ERA5 (from 1940 onward, but use 2015–present for data-recency reasons) and CMEMS reanalysis both have full coverage — e.g., uniformly random between `2016-01-01` and `2025-12-31`.

**Step 4 — Record provenance, always.**
Every synthetic-location row must carry:
```json
{
  "image_id": "string",
  "synthetic_lat": "float",
  "synthetic_lon": "float",
  "synthetic_timestamp_utc": "ISO8601",
  "source_corridor_id": "string",
  "offset_km": "float",
  "is_synthetic_location": true
}
```
That `is_synthetic_location: true` flag must **never be dropped** as this data moves downstream — it's how the Backend/Dashboard team knows never to display this coordinate as if it were a real, evidentiary spill location.

**Deliverable:** `data/features/synthetic_geo_assignments.csv` with the schema above, one row per image across all datasets you're using for the look-alike classifier / Model 1 training.

### 2.3 What this does NOT affect
This synthetic-location step is **only for training data**. The live operational pipeline (Stage 0 onward in the main README) always uses the **real** GPS bounding box that comes attached to the actual live satellite tile at inference time — real deployments never have this problem, only our historical training images do.

---

## 3. Full Task List (in order)

### Task 1.1 — Dataset audit *(see Section 1 above — do this first)*

### Task 1.2 — Acquire the 3 research datasets
- Download: OSD Dataset, the Zenodo Sentinel-1 Lookalike Set, ESSD 2025 Mediterranean Set (search each by name — they're published academic datasets, usually on Zenodo or linked from their paper).
- Same audit process as Task 1.1 for each: record image count, mask format, confirm the look-alike class actually exists and how it's labeled.
- **Deliverable:** append to `data/raw/DATASET_AUDIT.md`.

### Task 1.3 — Unify all datasets into one consistent format
- Convert every mask to binary PNG (0 = background, 255 = foreground), every image to GeoTIFF-like array (even if you have to fake a dummy CRS for datasets with no georeferencing — just needs to be a consistent array shape).
- Resize/crop all images to a consistent tile size (recommend 256×256) — pad or tile larger images, don't just squash-resize (distorts shape features you'll need later).
- Write one `metadata.csv` for ALL datasets combined: `image_id, dataset_source, label(oil/lookalike/background), width, height, has_real_geo(Y/N)`.
- **Deliverable:** `data/processed/unified_masks/*.png` + `data/processed/unified_images/*.tif` + `data/processed/metadata.csv`.

### Task 1.4 — Attach coordinates
- For any image where `has_real_geo == Y` (rare, but check for it in your audit), keep the real coordinate.
- For everything else, run the Section 2.2 algorithm.
- Merge into `metadata.csv` as new columns (`lat, lon, timestamp_utc, is_synthetic_location`).

### Task 1.5 — Train/val/test split
- Stratify by `label` (oil / look-alike / background) so class balance is roughly equal across splits.
- Group by `dataset_source` + rough spatial/temporal proximity if you can (avoid putting near-duplicate images in both train and test — check for this manually if the dataset seems to have repeated/near-identical tiles, which some Kaggle sets do).
- **Deliverable:** `data/splits/train.csv`, `val.csv`, `test.csv` (each just a list of `image_id`).

### Task 1.6 — Shape feature extraction (for the look-alike classifier)
- For every mask: compute `area`, `perimeter`, `aspect_ratio` (via `cv2.fitEllipse` major/minor axis ratio), `boundary_complexity` (perimeter²/area), `fragment_count` (`cv2.connectedComponents`).
- Use library: `opencv-python` or `scikit-image` (`skimage.measure.regionprops`).
- **Deliverable:** `data/features/shape_features.csv` — `image_id, area, perimeter, aspect_ratio, boundary_complexity, fragment_count`.

### Task 1.7 — Wind-speed join
- For every row in `shape_features.csv`, using its `lat, lon, timestamp_utc` (real or synthetic — doesn't matter which, same call either way), fetch ERA5 10m wind components.
- Library: `cdsapi` (Copernicus Climate Data Store Python client — requires free CDS account + API key from cds.climate.copernicus.eu).
- Compute `wind_speed_ms = sqrt(u10² + v10²)`.
- **Deliverable:** final training table `data/features/lookalike_training_data.csv` = shape features + `wind_speed_ms` + `label`. **This is the file the Model Training team needs.**

### Task 1.8 — AIS historical data pipeline
- Pick ONE region + time window to keep scope small for the demo (recommend: Arabian Sea / Gulf of Kutch, since that's a real busy corridor and gives you a concrete story for the pitch).
- Source real historical AIS if you have institutional/academic access; otherwise, generate a **plausible mock AIS dataset** — a few dozen vessels with realistic tracks along your defined shipping corridors (Section 2.2's corridors are reusable here!), moving at realistic speeds (10–20 knots for cargo ships), over a multi-day window.
- Clean/generate into schema: `mmsi, timestamp_utc, lat, lon, speed_knots, heading`.
- **Deliverable:** `data/ais/ais_tracks.csv` + load script into PostGIS table `ais_tracks` (schema handed to you by Backend team — coordinate with them).

### Task 1.9 — Live API wrapper scripts
- Build two reusable, documented Python functions (Backend will import these):
  - `fetch_currents(bbox, start_time, end_time) -> "path/to/currents.nc"` — wraps the `copernicusmarine` Python package.
  - `fetch_winds(bbox, start_time, end_time) -> "path/to/winds.nc"` — wraps `cdsapi` (ERA5) for historical, or NOAA GFS API for forecast data.
- **Deliverable:** `lib/fetch_currents.py`, `lib/fetch_winds.py` — each with a docstring showing exact input/output types and an example call.

### Task 1.10 — OpenDrift synthetic dataset for Model 2
- Pick a handful of real historical time windows within your chosen region (Task 1.8's region is fine to reuse) where you have real CMEMS + ERA5 data available.
- Using the `opendrift` Python package (`from opendrift.models.openoil import OpenOil`):
  - Generate **point-source releases**: seed a single random-sized blob at t=0, run forward 24h, snapshot every 30 min.
  - Generate **line-source releases**: simulate a moving-vessel discharge by seeding particles progressively along a short synthetic track (reuse your shipping corridor logic — pick a corridor segment, "sail" along it while continuously releasing particles for 1–4 hours), then let the whole thing drift forward.
  - Convert each particle cloud into a polygon using `shapely.geometry.MultiPoint(...).convex_hull` (simpler) or an alpha-shape library for a tighter, more realistic non-convex outline.
- Generate at minimum 2,000–5,000 of these paired sequences for the MVP.
- **Deliverable:** `data/synthetic/model2_pairs/*.geojson`, each containing `{initial_shape, wind_field_ref, current_field_ref, elapsed_hours, resulting_shape, release_type: "point"|"line"}`.

---

## 4. Final Folder Structure You're Responsible For

```
data/
├── raw/
│   ├── <each kaggle/research dataset>/
│   └── DATASET_AUDIT.md
├── processed/
│   ├── unified_masks/*.png
│   ├── unified_images/*.tif
│   └── metadata.csv
├── splits/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── features/
│   ├── shape_features.csv
│   ├── synthetic_geo_assignments.csv
│   └── lookalike_training_data.csv     <- hand this to Model Training team
├── ais/
│   └── ais_tracks.csv
├── synthetic/
│   └── model2_pairs/*.geojson           <- hand this to Model Training team
└── reference/
    └── shipping_corridors.geojson

lib/
├── fetch_currents.py                    <- hand this to Backend team
└── fetch_winds.py                       <- hand this to Backend team
```

---

## 5. Tools & Libraries You'll Need (install list)

```
pip install opencv-python scikit-image pandas geopandas shapely
pip install global-land-mask cdsapi copernicusmarine opendrift
pip install kaggle
```
- Kaggle API token: kaggle.com → Account → Create New API Token → save `kaggle.json` to `~/.kaggle/`
- CDS API key (for `cdsapi`/ERA5): cds.climate.copernicus.eu → register → get UID+key → save to `~/.cdsapirc`
- Copernicus Marine credentials (for `copernicusmarine`): register at data.marine.copernicus.eu

---

## 6. Definition of Done (checklist)

- [ ] `DATASET_AUDIT.md` completed for all 7 candidate datasets (4 Kaggle + 3 research)
- [ ] All usable datasets unified into one `metadata.csv` with consistent image/mask format
- [ ] Every image has a coordinate + timestamp (real where available, synthetic otherwise, always flagged)
- [ ] Train/val/test split done with no leakage, roughly class-balanced
- [ ] `lookalike_training_data.csv` complete and handed off to Model Training team
- [ ] AIS mock/real dataset ready and loaded into agreed DB schema
- [ ] `fetch_currents.py` / `fetch_winds.py` tested and handed off to Backend team
- [ ] At least 2,000 Model 2 synthetic drift pairs generated and handed off to Model Training team

---

## 7. Things to Flag Upward (don't solve these yourselves — tell the team)

- If the Task 1.1 audit shows the Kaggle datasets have **no look-alike class at all**, say so immediately — the look-alike classifier may need to rely entirely on the 3 research datasets, which changes how much training data Block 2 actually has.
- The synthetic-GPS approach is a **known limitation to state explicitly in the final pitch** — don't let it get presented as real data by accident in the demo.
