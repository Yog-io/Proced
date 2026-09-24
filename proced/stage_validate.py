"""Stage 6c — validation checklist wrapper (archi §5)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from .config import PipelineConfig
from .validate import validate_dataset

log = logging.getLogger("proced.validate_stage")


def run_validate(cfg: PipelineConfig) -> Dict[str, Any]:
    log.info("validate: running checklist …")
    return validate_dataset(
        Path(cfg.stage_dir), Path(cfg.dest_dir),
        cfg=cfg, splits_dir=Path(cfg.splits_dir),
    )
