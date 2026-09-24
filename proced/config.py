"""Pipeline configuration (single source of truth for every tunable)."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]

# Spec-mandated dB -> uint16 mapping version (DATASET_FORMAT_REFINEMENT_REPORT C.2).
# Inverse for Block 2: db = (pixel_uint16 / 65535) * 30 - 30
DB_FORMULA_VERSION = "v1_linear_neg30_to_0"
NATIVE_FORMULA_VERSION = "native_8bit_no_db"

RASTER_EXTENSIONS = (".tif", ".tiff", ".png", ".jp2", ".img", ".jpg", ".jpeg")
MASK_NAME_TOKENS = ("mask", "label", "labels", "gt", "lbl", "ann")


@dataclass
class PipelineConfig:
    # Paths (CLI overrides these)
    archive: Optional[Path] = None
    stage_dir: Path = ROOT / "data" / "scratch_unpacked"
    dest_dir: Path = ROOT / "data" / "master_dataset_processed"
    output_archive: Path = ROOT / "data" / "master_dataset_v1.7z"
    features_dir: Path = ROOT / "data" / "features"
    splits_dir: Path = ROOT / "data" / "splits"
    corridors_path: Path = ROOT / "data" / "reference" / "shipping_corridors.geojson"
    # Intermediate stage artifacts (catalog/instances/geo/plans/rows) so every
    # stage is independently runnable. Never auto-extracted from an archive —
    # use the `extract` subcommand (or your own unpack) to populate stage_dir.
    state_dir: Path = ROOT / "data" / "state"

    # Concurrency / memory guardrails (archi.md §2)
    workers: int = 4
    gdal_cache_mb_per_worker: int = 256
    gdal_num_threads: str = "2"
    pool_chunk: int = 16  # folders per executor before recycling (bounds worker memory growth)

    # Crop geometry (refinement A.2 / C.1)
    tile_size: int = 256
    jitter_px: int = 25  # +/- 20-30 px per spec
    stride: int = 0  # 0 => tile_size // 2 (50% overlap)
    max_tiles_per_object: int = 48

    # Background / negative patches (refinement C.3)
    neg_buffer_px: int = 300
    neg_ratio: float = 0.4  # negatives ≈ ratio × positives per scene
    neg_per_empty_scene: int = 4  # negatives from scenes whose mask is verified clean
    max_neg_per_scene: int = 64
    negatives_from_unlabeled: bool = False  # scenes WITHOUT a mask: no negatives (unlabeled oil risk)

    # Radiometry (refinement C.2)
    db_min: float = -30.0
    db_max: float = 0.0
    value_domain: str = "auto"  # auto | db | power | amplitude
    epsilon: float = 1e-7
    gpu: bool = True  # used only when CUDA is actually available

    # Full-extent feature extraction (refinement C.0)
    min_instance_area_px: int = 20
    ring_px: int = 20
    lookalike_mask_value: int = 2

    # Geolocation (guide §2.2)
    offset_scale_km: float = 50.0
    land_retry_attempts: int = 50
    synthetic_pixel_size_m: float = 10.0  # Sentinel-1 GRD ~10 m ground spacing
    timestamp_start: str = "2016-01-01T00:00:00"
    timestamp_end: str = "2025-12-31T23:59:59"

    # Splits (guide Task 1.5 / refinement C.5)
    split_train: float = 0.70
    split_val: float = 0.15
    split_test: float = 0.15
    seed: int = 42

    # Dataset routing overrides (substring match against relative path)
    calibrated_match: tuple = ()
    uncalibrated_match: tuple = ()
    label_overrides: dict = field(default_factory=dict)  # {"kaggle": "oil", ...}

    # Behaviour flags
    no_pack: bool = False
    no_validate: bool = False
    force: bool = False  # reprocess folders that already have GEODATA.csv

    def __post_init__(self) -> None:
        for name in (
            "stage_dir", "dest_dir", "output_archive", "features_dir",
            "splits_dir", "corridors_path", "archive", "state_dir",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Path):
                setattr(self, name, Path(value))
        if self.stride <= 0:
            self.stride = self.tile_size // 2

    # ------------------------------------------------------------------ utils
    @property
    def effective_stride(self) -> int:
        return self.stride

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, Path):
                d[k] = str(v)
        return d

    def ts_start_dt(self) -> datetime:
        return datetime.fromisoformat(self.timestamp_start)

    def ts_end_dt(self) -> datetime:
        return datetime.fromisoformat(self.timestamp_end)


def default_config() -> PipelineConfig:
    return PipelineConfig()
