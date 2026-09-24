System Architecture & High-Performance Plan

1. Architectural Overview & Workflow Lifecycle

[ .7z Raw Archives ] │ ▼ (Stage 0: Fast Multi-Core Decompression via py7zr / 7z CLI) [ Staged Extracted Tree ] (Root: raw_extracted/) │ ▼ (Stage 1: Tree Parsing & Dependency Discovery) Catalog subfolders, pair SAR scenes with corresponding mask TIFFs │ ▼ (Stage 2: Pre-Crop Full-Scene Feature Extraction - Section C.0) Calculate shape & radiometric features on uncropped ~2000×2000 arrays │ ▼ (Stage 3: Crop Geometry & Coordinate Resolution) Identify compact, overflowing, and negative crops; assign real or synthetic GPS │ ▼ (Stage 4: Parallel Windowed Tiling & 16-bit Conversion) Multi-Process / GPU execution: Windowed GDAL read -> dB calibration -> 16-bit uint16 PNG │ ▼ (Stage 5: Local Metadata Synthesis) Emit compliant `GEODATA.csv` inside each corresponding output subfolder │ ▼ (Stage 6: Target Archival) [ Master .7z Compressed Dataset ] 

2. Concurrency & Hardware Acceleration Strategy

To process large uncompressed SAR TIFF files (~40MB to 100MB+ each, 2000×2000+ pixels) without memory leaks or race conditions:

Process-Level Isolation (ProcessPoolExecutor): Python's Global Interpreter Lock (GIL) and GDAL's internal C-bindings do not allow true multi-core parallel array manipulation within standard threads. Processing occurs via worker processes mapped across physical CPU cores (os.cpu_count()).

GDAL Native Multi-Threading: Configure GDAL engine flags within each worker process:

gdal.SetCacheMax(1024 * 1024 * 512) (512MB RAM cache per worker process).

gdal.SetConfigOption('GDAL_NUM_THREADS', '2') (multithreaded I/O decompression per worker).

GPU-Accelerated Calibration (Optional Fallback to Vectorized NumPy): When an NVIDIA CUDA device is present, batch radiometric dB scaling and normalization (clip -> scale -> uint16) via CuPy or PyTorch tensor operations. If unavailable, use optimized SIMD-vectorized NumPy routines (np.clip, in-place arithmetic).

Memory Guardrails: Raw TIFFs are never loaded entirely into memory multiple times. Extraction of sub-windows uses GDAL's native windowed raster read (ds.ReadAsArray(xoff, yoff, xsize, ysize)), streaming only the required 256×256 pixels directly into RAM.

3. Core Processing Specifications

A. Pre-Crop Analysis (Full Extent)

Features that feed the look-alike classifier (area, perimeter, boundary complexity, aspect ratio, damping ratio, GLCM contrast) must be computed on the complete scene mask before tiling. Tiling alters geometry and breaks edge-to-edge boundary metrics.

B. Dynamic Crop Generation

Compact Objects (BBox \le 256\times 256): Centroid-centered crop with random jitter (\pm 20\text{--}30\text{ px}).

Overflowing Objects (BBox > 256\times 256): Multi-tile sliding window with 50% stride overlap. Link tiles via parent_group_id and set is_partial_object: Y.

Swath-Edge Truncated Objects: Flagged with truncated_by_scene_edge: Y if any mask pixel abuts the 0th or maximum scene coordinate.

Background/Negative Patches: Sample 256×256 patches positioned at least 300px away from any positive spill polygon.

C. Coordinate Geometry & Missing Geodata Fallback

Native Geodata Present: Read the 6-parameter affine transform via ds.GetGeoTransform(). Compute the cropped origin corner: \text{Origin}_X' = \text{Origin}_X + (\text{col\_off} \times \text{pixel\_width}) \text{Origin}_Y' = \text{Origin}_Y + (\text{row\_off} \times \text{pixel\_height}) Project all 4 crop corners to WGS84 (EPSG:4326) using osgeo.osr.CoordinateTransformation.

Missing Geodata (Masks / Unreferenced Imagery): Sample coordinates along high-density maritime shipping corridors with a half-normal perpendicular offset, confirmed as open water via global_land_mask, setting is_synthetic_location: true.

D. Radiometric Calibration to 16-Bit Single-Band PNG

Input: Calibrated backscatter \sigma^0 or raw linear amplitudes converted to decibels: \text{dB} = 10 \cdot \log_{10}(\text{amplitude} + \epsilon) 

Scale decibels from [-30, 0]\text{ dB} to [0, 65535]: \text{pixel\_uint16} = \text{round}\left(\frac{\text{clip}(\text{dB}, -30, 0) + 30}{30} \times 65535\right) 

Output: Separate single-channel files: <image_id>_VV.png and <image_id>_VH.png written via the GDAL PNG 16-bit driver.

4. Target Directory Structure & GEODATA.csv Specification

The target folder tree mirrors the original dataset structure. Each terminal subfolder containing crops has an independent GEODATA.csv:

master_dataset_processed/ ├── region_atlantic/ │ ├── subpass_01/ │ │ ├── scene01_crop001_VV.png │ │ ├── scene01_crop001_VH.png │ │ ├── scene01_crop001_mask.png │ │ └── GEODATA.csv │ └── subpass_02/ │ ├── scene02_crop001_VV.png │ ├── scene02_crop001_mask.png │ └── GEODATA.csv └── reference_masks/ └── GEODATA.csv 

Required GEODATA.csv Columns:

crop_id: Target crop filename base.

crop_source_scene_id: Original scene file name.

is_calibrated: Y (Zenodo calibrated) or N (Kaggle/uncalibrated).

has_dual_pol: Y (VV+VH available) or N (single channel).

is_synthetic_location: true if synthetic coordinates were assigned; false otherwise.

crop_bbox_corners: Coordinate string [(lon_tl, lat_tl), (lon_tr, lat_tr), (lon_br, lat_br), (lon_bl, lat_bl)].

is_partial_object: Y if part of a multi-tile oversized slick; else N.

parent_group_id: Shared UUID linking partial tiles of the same spill.

truncated_by_scene_edge: Y if intersecting the physical boundary of the parent scene; else N.

db_conversion_formula_version: Set to "v1_linear_neg30_to_0".

label: oil, lookalike, or background.

full_extent_area_px: Uncropped pixel area (or 0 for background).

full_extent_perimeter: Uncropped perimeter length.

full_extent_boundary_complexity: Computed \frac{\text{Perimeter}^2}{\text{Area}}.

full_extent_aspect_ratio: Major axis / minor axis ratio.

Production Processing Script

Save this script as sar_dataset_pipeline.py. It runs stand-alone or via CLI, incorporating GDAL, multi-processing, optional GPU acceleration (Torch/CuPy), synthetic geodata fallback, and automated .7z reconstruction.

#!/usr/bin/env python3 """ SAR Master Dataset Converter & Preprocessor High-performance GDAL-based pipeline for SAR oil spill dataset standardization. """ import os import sys import glob import math import uuid import shutil import random import logging import subprocess from pathlib import Path from concurrent.futures import ProcessPoolExecutor, as_completed import cv2 import numpy as np import pandas as pd from osgeo import gdal, osr # Enable GDAL exceptions and configure global cache gdal.UseExceptions() gdal.SetCacheMax(1024 * 1024 * 512) # 512 MB per process logging.basicConfig( level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(processName)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)], ) # Optional GPU support HAS_GPU = False try: import torch if torch.cuda.is_available(): HAS_GPU = True logging.info("CUDA GPU detected. Accelerated tensor math active.") except ImportError: pass # Check for global_land_mask try: from global_land_mask import globe HAS_LAND_MASK = True except ImportError: HAS_LAND_MASK = False logging.warning("global_land_mask not installed. Falling back to oceanic bounding heuristic.") # Reference Major Shipping Corridors for Synthetic Coordinates SHIPPING_CORRIDORS = [ {"id": "strait_of_malacca", "waypoints": [(1.2, 103.8), (2.5, 101.5), (5.5, 97.5)]}, {"id": "gulf_of_aden", "waypoints": [(12.5, 43.5), (13.0, 48.0), (14.5, 53.0)]}, {"id": "hormuz", "waypoints": [(26.0, 56.5), (25.0, 55.0), (27.0, 51.5)]}, {"id": "english_channel", "waypoints": [(50.0, -1.0), (50.5, 0.0), (51.2, 1.8)]}, {"id": "gulf_of_mexico", "waypoints": [(27.5, -90.0), (28.0, -89.0), (26.0, -86.0)]}, ] def generate_synthetic_coordinate(): """Generates realistic offshore ocean coordinates along major shipping lanes.""" for _ in range(50): corridor = random.choice(SHIPPING_CORRIDORS) wp = corridor["waypoints"] idx = random.randint(0, len(wp) - 2) p1, p2 = wp[idx], wp[idx + 1] t = random.random() base_lat = p1[0] + t * (p2[0] - p1[0]) base_lon = p1[1] + t * (p2[1] - p1[1]) # Perpendicular offset (half-normal distribution ~50km, 1 deg ~ 111km) offset_km = abs(random.gauss(0, 50)) offset_deg = offset_km / 111.0 angle = random.choice([math.pi / 2, -math.pi / 2]) d_lat = offset_deg * math.cos(angle) d_lon = offset_deg * math.sin(angle) syn_lat = base_lat + d_lat syn_lon = base_lon + d_lon if HAS_LAND_MASK: if globe.is_ocean(syn_lat, syn_lon): return syn_lat, syn_lon else: return syn_lat, syn_lon return 15.0, 65.0 # Fallback to Arabian Sea def extract_full_scene_features(mask_array): """ Computes shape features on the uncropped native scene before tiling. Prevents truncation artifacts from skewing downstream classifiers. """ contours, _ = cv2.findContours(mask_array.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) if not contours: return [] features_list = [] for cnt in contours: area = float(cv2.contourArea(cnt)) if area < 20: # Skip single-pixel sensor noise continue perimeter = float(cv2.arcLength(cnt, True)) complexity = (perimeter ** 2) / area if area > 0 else 0.0 if len(cnt) >= 5: (_, _), (MA, ma), _ = cv2.fitEllipse(cnt) aspect_ratio = float(MA / ma) if ma > 0 else 1.0 else: aspect_ratio = 1.0 x, y, w, h = cv2.boundingRect(cnt) moments = cv2.moments(cnt) cx = int(moments["m10"] / moments["m00"]) if moments["m00"] != 0 else x + w // 2 cy = int(moments["m01"] / moments["m00"]) if moments["m00"] != 0 else y + h // 2 features_list.append({ "contour": cnt, "area": area, "perimeter": perimeter, "boundary_complexity": complexity, "aspect_ratio": aspect_ratio, "bbox": (x, y, w, h), "centroid": (cx, cy), }) return features_list def convert_radiometric_to_uint16(band_array): """ Calibrates SAR data to decibels and maps [-30, 0] dB to uint16 [0, 65535]. Utilizes GPU acceleration if available, falling back to vectorized NumPy. """ if HAS_GPU: tensor = torch.from_numpy(band_array).cuda().float() eps = 1e-7 # Convert amplitude/power to dB if values are linear if torch.max(tensor) > 1.0: db = 10.0 * torch.log10(torch.clamp(tensor, min=eps)) else: db = tensor db_clipped = torch.clamp(db, -30.0, 0.0) scaled = torch.round(((db_clipped + 30.0) / 30.0) * 65535.0) return scaled.cpu().numpy().astype(np.uint16) else: eps = 1e-7 arr = band_array.astype(np.float32) if np.max(arr) > 1.0: db = 10.0 * np.log10(np.clip(arr, eps, None)) else: db = arr db_clipped = np.clip(db, -30.0, 0.0) scaled = np.round(((db_clipped + 30.0) / 30.0) * 65535.0) return scaled.astype(np.uint16) def get_crop_geotransform_and_corners(orig_gt, orig_srs, xoff, yoff, xsize, ysize): """Recomputes the affine transform and corner GPS coordinates for windowed crops.""" if not orig_gt or (orig_gt[0] == 0.0 and orig_gt[3] == 0.0): # Image does not possess valid spatial references lat, lon = generate_synthetic_coordinate() delta = 0.02 return None, f"[({lon:.4f},{lat+delta:.4f}),({lon+delta:.4f},{lat+delta:.4f}),({lon+delta:.4f},{lat:.4f}),({lon:.4f},{lat:.4f})]", True # Recalculate affine transform new_origin_x = orig_gt[0] + xoff * orig_gt[1] + yoff * orig_gt[2] new_origin_y = orig_gt[3] + xoff * orig_gt[4] + yoff * orig_gt[5] new_gt = (new_origin_x, orig_gt[1], orig_gt[2], new_origin_y, orig_gt[4], orig_gt[5]) # Transform corners to EPSG:4326 (WGS84) wgs84 = osr.SpatialReference() wgs84.ImportFromEPSG(4326) tx = osr.CoordinateTransformation(orig_srs, wgs84) def pixel_to_latlon(px, py): x = new_origin_x + px * orig_gt[1] + py * orig_gt[2] y = new_origin_y + px * orig_gt[4] + py * orig_gt[5] point = tx.TransformPoint(x, y) return round(point[0], 6), round(point[1], 6) # lon, lat c_tl = pixel_to_latlon(0, 0) c_tr = pixel_to_latlon(xsize, 0) c_br = pixel_to_latlon(xsize, ysize) c_bl = pixel_to_latlon(0, ysize) corners_str = f"[{c_tl}, {c_tr}, {c_br}, {c_bl}]" return new_gt, corners_str, False def save_16bit_png_gdal(output_path, array_uint16): """Saves uint16 raster arrays strictly using GDAL's native PNG driver.""" driver = gdal.GetDriverByName("PNG") ysize, xsize = array_uint16.shape out_ds = driver.Create(str(output_path), xsize, ysize, 1, gdal.GDT_UInt16) out_band = out_ds.GetRasterBand(1) out_band.WriteArray(array_uint16) out_band.FlushCache() out_ds = None def process_subfolder(subfolder_tuple): """ Subfolder worker execution unit. Converts and crops SAR images, writes 16-bit PNGs, and generates GEODATA.csv. """ src_subfolder, dest_subfolder = subfolder_tuple dest_path = Path(dest_subfolder) dest_path.mkdir(parents=True, exist_ok=True) csv_records = [] tif_files = glob.glob(os.path.join(src_subfolder, "*.tif")) if not tif_files: return # Categorize into scenes and masks mask_files = [f for f in tif_files if "mask" in Path(f).stem.lower()] scene_files = [f for f in tif_files if f not in mask_files] for scene_file in scene_files: scene_name = Path(scene_file).stem corresponding_mask = None for mf in mask_files: if Path(mf).stem.replace("_mask", "") in scene_name: corresponding_mask = mf break try: ds = gdal.Open(scene_file, gdal.GA_ReadOnly) width = ds.RasterXSize height = ds.RasterYSize bands_count = ds.RasterCount gt = ds.GetGeoTransform() srs = osr.SpatialReference() if ds.GetProjectionRef(): srs.ImportFromWkt(ds.GetProjectionRef()) else: srs = None mask_arr = np.zeros((height, width), dtype=np.uint8) if corresponding_mask: mask_ds = gdal.Open(corresponding_mask, gdal.GA_ReadOnly) mask_arr = mask_ds.GetRasterBand(1).ReadAsArray(0, 0, width, height) mask_ds = None # Stage 2: Feature extraction on full scene before cropping features = extract_full_scene_features(mask_arr) crop_plan = [] occupied_mask = np.zeros((height, width), dtype=bool) for feat in features: x, y, w, h = feat["bbox"] cx, cy = feat["centroid"] occupied_mask[max(0, y - 300):min(height, y + h + 300), max(0, x - 300):min(width, x + w + 300)] = True # Overflowing or extended object check if w > 256 or h > 256: group_id = str(uuid.uuid4())[:8] stride = 128 # 50% overlap x_steps = math.ceil(max(1, (w - 256) / stride)) + 1 y_steps = math.ceil(max(1, (h - 256) / stride)) + 1 for xi in range(x_steps): for yi in range(y_steps): cx_off = min(x + xi * stride, width - 256) cy_off = min(y + yi * stride, height - 256) crop_plan.append({ "xoff": cx_off, "yoff": cy_off, "is_partial": "Y", "parent_id": group_id, "label": "oil", "feat": feat }) else: # Compact object: centroid centered + random jitter jitter_x = random.randint(-25, 25) jitter_y = random.randint(-25, 25) xoff = min(max(0, cx - 128 + jitter_x), width - 256) yoff = min(max(0, cy - 128 + jitter_y), height - 256) crop_plan.append({ "xoff": xoff, "yoff": yoff, "is_partial": "N", "parent_id": "none", "label": "oil", "feat": feat }) # Negative/Background Patches (minimum 300px from any slick) unoccupied_y, unoccupied_x = np.where(~occupied_mask[0:height - 256, 0:width - 256]) if len(unoccupied_y) > 0: num_backgrounds = min(max(2, len(features)), 10) indices = np.random.choice(len(unoccupied_y), size=num_backgrounds, replace=False) for idx in indices: crop_plan.append({ "xoff": int(unoccupied_x[idx]), "yoff": int(unoccupied_y[idx]), "is_partial": "N", "parent_id": "none", "label": "background", "feat": None }) # Stage 4: Windowed reading and conversion for c_idx, plan in enumerate(crop_plan): xoff, yoff = plan["xoff"], plan["yoff"] crop_base = f"{scene_name}_crop{c_idx:03d}" # Swath boundary check edge_truncated = "Y" if (xoff == 0 or yoff == 0 or xoff + 256 >= width or yoff + 256 >= height) else "N" # Spatial geometry extraction _, bbox_corners, is_synthetic = get_crop_geotransform_and_corners(gt, srs, xoff, yoff, 256, 256) # Process polarization bands band_tags = ["VV", "VH"] if bands_count >= 2 else ["VV"] for b_i, b_tag in enumerate(band_tags): band = ds.GetRasterBand(b_i + 1) raw_window = band.ReadAsArray(xoff, yoff, 256, 256) uint16_crop = convert_radiometric_to_uint16(raw_window) save_16bit_png_gdal(dest_path / f"{crop_base}_{b_tag}.png", uint16_crop) # Process matching mask window mask_window = mask_arr[yoff:yoff + 256, xoff:xoff + 256] mask_bin = (mask_window > 0).astype(np.uint8) * 255 cv2.imwrite(str(dest_path / f"{crop_base}_mask.png"), mask_bin) # Populate metadata record feat = plan["feat"] csv_records.append({ "crop_id": crop_base, "crop_source_scene_id": scene_name, "is_calibrated": "Y", "has_dual_pol": "Y" if bands_count >= 2 else "N", "is_synthetic_location": is_synthetic, "crop_bbox_corners": bbox_corners, "is_partial_object": plan["is_partial"], "parent_group_id": plan["parent_id"], "truncated_by_scene_edge": edge_truncated, "db_conversion_formula_version": "v1_linear_neg30_to_0", "label": plan["label"], "full_extent_area_px": feat["area"] if feat else 0, "full_extent_perimeter": feat["perimeter"] if feat else 0.0, "full_extent_boundary_complexity": feat["boundary_complexity"] if feat else 0.0, "full_extent_aspect_ratio": feat["aspect_ratio"] if feat else 0.0, }) ds = None except Exception as e: logging.error(f"Error processing scene {scene_file}: {e}") # Emit local GEODATA.csv for this subfolder if csv_records: df = pd.DataFrame(csv_records) df.to_csv(dest_path / "GEODATA.csv", index=False) logging.info(f"Generated GEODATA.csv with {len(df)} records in {dest_path}") def unpack_7z(archive_path, extract_dir): """Extracts .7z archive using 7z command line with multi-threading.""" logging.info(f"Decompressing {archive_path} into {extract_dir}...") cmd = ["7z", "x", f"-o{extract_dir}", str(archive_path), "-y", "-mmt=on"] subprocess.run(cmd, check=True) def pack_7z(source_dir, output_archive): """Packs target directory into final master .7z file.""" logging.info(f"Compressing {source_dir} into {output_archive}...") cmd = ["7z", "a", "-t7z", "-m0=lzma2", "-mx=7", "-mmt=on", str(output_archive), f"{source_dir}/*"] subprocess.run(cmd, check=True) def main(): import argparse parser = argparse.ArgumentParser(description="SAR Master Dataset Processing & Compression Pipeline") parser.add_argument("--archive", type=str, required=True, help="Path to raw source .7z archive") parser.add_argument("--stage_dir", type=str, default="data/staging_raw", help="Scratch directory for extraction") parser.add_argument("--dest_dir", type=str, default="data/master_processed", help="Directory for cropped master output") parser.add_argument("--output_archive", type=str, default="data/master_processed_dataset.7z", help="Final output .7z archive") parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Parallel worker count") args = parser.parse_args() stage_path = Path(args.stage_dir) dest_path = Path(args.dest_dir) # 1. Extraction unpack_7z(args.archive, stage_path) # 2. Directory Tree Mapping subfolder_jobs = [] for root, dirs, files in os.walk(stage_path): if any(f.endswith(".tif") for f in files): rel_path = os.path.relpath(root, stage_path) target_subfolder = dest_path / rel_path subfolder_jobs.append((root, str(target_subfolder))) logging.info(f"Found {len(subfolder_jobs)} subfolders containing SAR scenes. Dispatching across {args.workers} workers.") # 3. Parallel Execution across CPU/GPU with ProcessPoolExecutor(max_workers=args.workers) as executor: futures = [executor.submit(process_subfolder, job) for job in subfolder_jobs] for future in as_completed(futures): future.result() # 4. Final Archival pack_7z(dest_path, args.output_archive) logging.info(f"Pipeline complete. Master archive created: {args.output_archive}") if __name__ == "__main__": main() 

Step-by-Step Implementation & Run Guide

1. Environment & Dependency Setup

Verify that GDAL and system-level p7zip utilities are installed:

# Ubuntu / Debian OS packages sudo apt-get update && sudo apt-get install -y p7zip-full gdal-bin libgdal-dev # Python virtual environment pip install numpy pandas opencv-python global-land-mask torch torchvision pip install GDAL=="$(gdal-config --version).*" 

2. Execution Command

To process the dataset end-to-end:

python sar_dataset_pipeline.py \ --archive "data/raw_sar_inputs.7z" \ --stage_dir "data/scratch_unpacked" \ --dest_dir "data/master_dataset_processed" \ --output_archive "data/master_dataset_v1.7z" \ --workers 8 

3. Verification & Validation Checklist

Run these checks before passing outputs to downstream training:

Directory Equivalence: Verify that running find data/scratch_unpacked -type d and find data/master_dataset_processed -type d return identical subfolder trees.

Metadata Integrity: Check that every terminal subfolder contains a valid GEODATA.csv with no null values in crop_bbox_corners, is_partial_object, or truncated_by_scene_edge.

Bit-Depth Accuracy: Run gdalinfo data/master_dataset_processed/.../*_VV.png and verify the band reports Type=UInt16, confirming conversion to 16-bit single-band PNG.

