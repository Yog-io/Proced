# Dataset Preparation: Research Findings, Block 2 Approach & Block 1 Refinement Guide
### Prepared for: cross-team alignment on SAR image storage, cropping, and multi-source training strategy

---

## PART A: Research Findings — Answering the Three Open Questions

### A.1 Why crop instead of resize to 256×256?

**The core issue: resizing a 2000×2000 scene to 256×256 is a ~7.8x downsample per axis (~61x fewer total pixels).** This isn't a minor quality loss — it breaks the task in two specific, provable ways:

1. **Oil slicks become sub-pixel or disappear entirely.** Sentinel-1 GRD imagery has roughly 10m ground pixel spacing, so a 2000×2000 tile covers ~20km × 20km. Confirmed spills already occupy only 1–3% of a tile's pixel area (this exact number is why we chose Focal Tversky loss for Model 1). Downsampling by 61x shrinks that already-small object further — a 50-pixel-wide slick at native resolution becomes ~6 pixels wide after resize. Thin filamentous slicks (the shape that's most diagnostically "oil-like," per the boundary-sharpness features in Block 2's classifier) can shrink below one pixel and vanish from the resized image entirely.
2. **Resizing corrupts the exact features the look-alike classifier depends on.** Aspect ratio, boundary complexity, and boundary gradient steepness all depend on true edge geometry. Interpolation used during resize smooths and rounds edges — which specifically erases the "oil has sharp edges, look-alikes have gradual ones" signal that's the classifier's main job to detect.

**Conclusion: crop to native resolution, never resize the full scene.** This preserves the true physical scale (a 500m-wide slick stays ~50 pixels wide, regardless of which part of the scene it's in), which both models depend on.

### A.2 How do we decide *where* to crop?

Not arbitrarily — the crop location is driven by the existing label geometry, using a standard strategy from satellite object-detection literature (the same general approach used in large-scene detection datasets like DOTA/xView):

1. **Positive (labeled) crops:** for every labeled polygon (oil or look-alike) in a scene, compute its centroid and bounding box. Center a 256×256 window on that centroid, with a small random jitter offset (±20–30px) applied per sample — this both avoids the model learning "the object is always dead-center" and gives you free data augmentation.
2. **Oversized objects:** if a polygon's bounding box is larger than 256×256 (a long linear slick spanning kilometers), don't just crop-and-lose the tails — tile multiple overlapping 256×256 windows along the polygon's length (50% stride overlap), each carrying whichever portion of the mask falls inside that window. This ensures the full object gets represented across several training tiles rather than truncated into one.
3. **Negative (background) crops — this is a new requirement, see Part C.3:** randomly sample additional 256×256 windows from open water areas that are *not* near any labeled polygon (minimum buffer distance, e.g. 300px, from any positive polygon's edge). Without true negative examples of clean sea, Model 1 has no signal for what "no spill" looks like at inference time on a full scene — it would only ever have seen spill-centered crops during training.

### A.3 Does Kaggle-pretrain → Zenodo-fine-tune actually cause a format mismatch?

**Short answer: not a *format* mismatch once the preprocessing pipeline is designed correctly — but a real *domain/precision* gap, which is expected and already the same category of gap we accepted when choosing an ImageNet-pretrained encoder for SAR data in the first place.**

- **What "format" means at the point the model actually sees data:** regardless of whether the source file was 8-bit Kaggle PNG or 16-bit calibrated Zenodo TIFF/PNG, the dataloader's job is to convert both into the *same* final tensor shape and value range (float32, normalized to `[0,1]` or `[-1,1]`) before it reaches the network. If this conversion step is built consistently, the model never "sees" a bit-depth difference — both phases feed it identically-shaped, identically-ranged tensors. This is a preprocessing design requirement Block 1 needs to implement consistently (see Part C.1), not something Block 2 works around at training time.
- **What genuinely differs, and can't be fully eliminated:** the *information content*. Kaggle's 8-bit visual exports have already had their dynamic range clipped/stretched by whatever tool generated them — that's a real, permanent information loss versus the calibrated Zenodo data, no amount of careful tensor formatting recovers it. This is why Kaggle data is scoped strictly to warm-up pretraining (learning general blob/edge/texture detection) and the calibrated Zenodo set does the fine-tuning (learning the true task).
### A.4 Does cropping destroy geospatial correlation?

**No — not if it's done via a proper windowed read, which is standard GIS practice, not something exotic.** A GeoTIFF doesn't store coordinates per-pixel; it stores one affine transform (pixel size + origin corner) applied across the whole array. Cropping a window at pixel offset `(col_off, row_off)` just requires recomputing the new origin:
```
new_origin_x = old_origin_x + col_off * pixel_width
new_origin_y = old_origin_y + row_off * pixel_height
```
Pixel size stays identical. This is exactly what `rasterio.windows.transform()` (or `gdal_translate -srcwin`) does automatically. **The geolocation problem only occurs if the crop is done by slicing the numpy array and discarding the transform** — that's a bookkeeping bug, not an inherent limitation of cropping. Section C.4 below is updated to carry the full recomputed bounding box forward per crop, not just a centroid, so this is never lost.

**Separately — "resize with more bits" doesn't solve the actual problem.** Bit depth controls intensity precision; it's unrelated to spatial resolution. A thin slick filament that shrinks to 6 pixels wide after a 2000→256 resize is still 6 pixels wide at 16-bit — more bits per pixel doesn't recover detail that was never spatially represented in the first place. Resizing isn't a substitute for cropping regardless of bit depth.

### A.5 Handling slicks that overflow a fixed crop window (the edge-to-edge case)

This is the real problem your inspection surfaced, and it needs a different fix than "make the crop bigger" — there's no fixed crop size that always contains an arbitrarily long slick. The fix is to **stop treating shape-feature extraction and CNN-training-image cropping as the same operation**, since they actually have different requirements:

- **Shape/radiometric features (feeds the look-alike classifier)** — compute these **once, directly from the full-scene polygon/mask, before any cropping decision is made.** This is exactly what your `output.geojson` output already represents. Aspect ratio, area, boundary complexity, damping ratio, etc. are only meaningful when measured against the object's true full extent — if computed from a cropped tile that happens to truncate the object, they'd be wrong. So these features should never be derived from a cropped tile at all; they come from the full-resolution scene/polygon, and travel into `metadata.csv`/`lookalike_training_data.csv` independent of whatever tiling strategy gets applied for Model 1's training images.
- **Image crops (feeds Model 1's segmentation training only)** — Model 1 is a pixel-wise segmentation model; it doesn't need to see a "whole" object to learn "is this pixel oil or not," so a partial/truncated view is legitimate training signal — this is standard practice for training segmentation on any large scene. So:
  - **Compact objects** (bounding box fits within 256×256): single centroid-centered crop with jitter, as already planned.
  - **Extended/overflowing objects** (bounding box exceeds 256×256, or the mask touches the tile's edge): tile with 50%-overlap 256×256 windows along the object's extent (as already specified in A.2), tagging all tiles from one object with a shared `parent_group_id` — but critically, don't re-derive shape features from any of these individual tiles.
  - **Swath-edge-truncated objects** (the object's true extent is cut off by the satellite scene boundary itself, not just by a crop choice): even the full-scene mask underestimates the real size here. Flag these explicitly (`truncated_by_scene_edge: Y`) and exclude them from the look-alike classifier's training data — their shape features are unreliable by construction — but they're still fine to include in Model 1's segmentation training images.

---

## PART B: How Block 2 Is Handling This (for your awareness — no action needed from Block 1 on this part)

- **Two-phase training schedule for Model 1:** Phase 1 — pretrain on Kaggle's larger, lower-fidelity set (single-channel duplicated to 3 channels) purely to learn general segmentation ability. Phase 2 — fine-tune on Zenodo's calibrated, cropped set (true VV/VH/NDPI channels) with a reduced overall learning rate and an even smaller learning rate specifically on the first convolutional layer.
- **Dataset routing is now finalized by model, not shared indiscriminately:**
  - Model 1 (segmentation) → Kaggle (pretrain) + Zenodo (fine-tune)
  - Look-alike classifier → Zenodo calibrated set only (~2,100 rows is sufficient; Kaggle's uncalibrated pixels would actively harm the radiometric features)
- **Unified tensor preprocessing** (channel normalization, value range, resolution) is being implemented once in a shared dataloader utility, so both Kaggle and Zenodo sources always produce identically-shaped, identically-ranged model inputs regardless of their original file format or bit depth.

---

## PART C: Refinement Guide for Block 1 — Updated Tasks & Deliverables

**This section supersedes/amends the relevant tasks in Block 1's original guide. Please update your pipeline and deliverables to match.**

### C.0 New required step, done BEFORE any cropping: extract full-extent shape & radiometric features
For every labeled polygon, on the full ~2000×2000 scene: compute the shape features (area, perimeter, aspect ratio, boundary complexity, fragment count) and radiometric features (damping ratio, boundary gradient steepness, backscatter variance ratio, GLCM contrast/homogeneity) directly from the full mask/polygon and its surrounding background ring — **before** any tiling/cropping decision is made. Store these in `lookalike_training_data.csv` exactly as before. This is what your `output.geojson` polygon output already supports — it just needs to happen first, and independently of whatever cropping strategy is applied next. These full-extent features are what feeds the look-alike classifier; nothing downstream should recompute them from a cropped tile.

### C.1 Crop, don't resize — and store the crop, not the full scene (for Model 1's training images ONLY)
- Stop delivering full ~2000×2000 scenes as Model 1's training input. This step is entirely separate from C.0 above — cropping affects only the pixel-image data used for Model 1's segmentation training, never the shape/radiometric features (which come from the full scene per C.0).
- **Compact objects** (bounding box fits within 256×256): one crop centered on the polygon's centroid, with small random jitter (Part A.2).
- **Extended/overflowing objects** (bounding box bigger than 256×256, or mask touches the scene edge — this is the case you found in your inspection): tile with 50% overlapping 256×256 windows along the object's extent. Tag every tile from the same object with a shared `parent_group_id`. Each tile only needs to carry whichever fragment of the mask falls inside it — do not attempt to recompute shape features from these fragments (C.0 already covers that from the full extent).
- **Swath-edge-truncated objects** (the object's true extent is cut off by the satellite scene boundary itself, not just by your crop choice): flag these (`truncated_by_scene_edge: Y`) and exclude them from the look-alike classifier's training rows — their full-extent features from C.0 are unreliable by construction — but keep their image crops for Model 1's segmentation training, since pixel-level learning doesn't need the "whole" object.
- Recompute a valid geotransform for every crop (new origin = old origin + offset × pixel size, per Part A.4) — a proper windowed read (`rasterio.windows`) does this automatically; don't crop the raw array and discard the transform.
- **This alone should resolve the 22GB/698-file size problem** — a 256×256 crop at 16-bit precision is on the order of a few hundred KB, roughly 60x smaller than a full uncompressed scene.

### C.2 Storage format: 16-bit PNG per band, not 8-bit PNG/JPG, not full-scene TIFF
- Per-band single-channel 16-bit PNG (lossless): `<image_id>_VV.png`, `<image_id>_VH.png`. Keep bands as separate files — multi-band 16-bit PNG support is inconsistent across libraries, and separate files avoid that entirely.
- Never JPG for any imagery fed into radiometric/texture features — JPG's lossy compression introduces artifacts that directly corrupt the GLCM texture features Block 2 depends on.
- Never plain 8-bit PNG for the calibrated (Zenodo) imagery — quantizes away the precision the damping ratio / NDPI calculations need.
- Conversion formula (document this exactly, once, in your pipeline script): clip calibrated value to `[-30, 0]` dB, then `pixel_uint16 = round(((db_value + 30) / 30) * 65535)`. Store this formula in your handoff doc so Block 2 can invert it exactly: `db_value = (pixel_uint16 / 65535) * 30 - 30`.
- Kaggle sets keep whatever native 8-bit format they already have — no conversion needed there, they're already scoped to pretraining-only use.

### C.3 New deliverable: negative/background patches
- Per Part A.2, point 3: sample additional 256×256 crops of open water with no labeled polygon nearby (minimum 300px buffer from any positive polygon edge), across a range of scenes/conditions (calm sea, rougher sea, if available near-coastline vs open-ocean) for diversity.
- Target roughly the same order of magnitude as your positive counts (a few hundred to ~1,000 negative patches from the Zenodo scenes) so Model 1 gets a real "here's what clean sea looks like" training signal, not just spill-centered crops.
- Label these `background` in `metadata.csv` (this label value was already anticipated in the schema — just needs data behind it now).

### C.4 Updated `metadata.csv` schema — add these columns
On top of the existing columns, add:
| New column | Values | Purpose |
|---|---|---|
| `is_calibrated` | Y/N | Flags whether pixel values are true calibrated σ⁰ (Zenodo) or an uncalibrated visual export (Kaggle) — Block 2 needs this to decide which features/phase to use this row for |
| `has_dual_pol` | Y/N | Flags whether real separate VV+VH bands exist, or the source is single-channel — determines whether NDPI channel construction is possible for this row |
| `crop_source_scene_id` | string | Which original full scene this crop came from — needed to avoid train/val/test leakage (don't let two crops from the same original scene end up in different splits) |
| `crop_bbox_corners` | 4 × (lat,lon) | Full crop bounding box in real-world coordinates (all four corners, not just centroid) — recomputed via the windowed-read transform (Part A.4), sufficient to map any pixel in the crop back to a real coordinate |
| `is_partial_object` | Y/N | Y if this crop is one tile of a larger, multi-tile object (per C.1) — tells Block 2 not to derive shape features from this specific crop |
| `parent_group_id` | string | Links all tiles belonging to the same original overflowing object together |
| `truncated_by_scene_edge` | Y/N | Y if the object's true extent is cut off by the satellite scene boundary — excludes this row from the look-alike classifier's training data (see C.0/C.1), but not from Model 1's image set |
| `db_conversion_formula_version` | string/int | In case the normalization formula changes later, tag which version was used per file |

### C.5 Re-check train/val/test splitting after this change
Since scenes are now split into multiple crops (potentially several per original scene, including tiled overlaps), re-verify your stratified split (`data/splits/train.csv`/`val.csv`/`test.csv`) groups by `crop_source_scene_id`, not by individual crop — otherwise crops from the same physical spill event could leak across splits, inflating validation/test scores artificially.

### C.6 Summary of what changes vs. your original guide
| Original plan | Updated plan |
|---|---|
| Shape/radiometric features computed however/whenever | Compute once from the **full scene**, before any cropping (C.0) — never recompute from a cropped tile |
| Deliver full-scene GeoTIFFs | Deliver 256×256 crops only, centered on labels or tiled for oversized objects |
| One crop per object, regardless of size | Compact objects → single crop; overflowing objects → multi-tile with `parent_group_id`; scene-edge-truncated objects → flagged and excluded from classifier data |
| Format unspecified / raw calibrated TIFF | 16-bit lossless PNG, one file per polarization band |
| Only labeled (oil/lookalike) patches | Add background/negative patches, buffered from any positive label |
| `metadata.csv` base schema | Add `is_calibrated`, `has_dual_pol`, `crop_source_scene_id`, `crop_centroid_lat/lon`, `db_conversion_formula_version` |
| Split by whatever grouping was used | Must group split by `crop_source_scene_id` to prevent leakage |
