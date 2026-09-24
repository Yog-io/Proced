"""Stage 6b — pack the processed tree into the output .7z archive."""

from __future__ import annotations

import logging
from pathlib import Path

from .archive import pack_7z
from .config import PipelineConfig

log = logging.getLogger("proced.archive_stage")


def run_archive(cfg: PipelineConfig) -> Path:
    dest = Path(cfg.dest_dir)
    out = Path(cfg.output_archive)
    if not dest.is_dir():
        raise SystemExit(f"archive: dest_dir not found: {dest}")
    log.info("archive: packing %s → %s", dest, out)
    return pack_7z(dest, out)
