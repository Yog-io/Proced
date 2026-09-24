"""Stage 6a — group-safe stratified splits from crop_rows / metadata (C.5)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

from .config import PipelineConfig
from .splits import assign_splits, write_splits
from .state import StatePaths, load_crop_rows

log = logging.getLogger("proced.split")


def run_split(cfg: PipelineConfig) -> Dict[str, Path]:
    paths = StatePaths.from_config(cfg)
    rows = None
    if paths.crop_rows.is_file():
        rows = load_crop_rows(paths.crop_rows)
    else:
        md = Path(cfg.dest_dir) / "metadata.csv"
        if md.is_file():
            import pandas as pd
            rows = pd.read_csv(md).fillna("").to_dict("records")
            log.info("split: state crop_rows missing — using %s", md)
        else:
            raise SystemExit(
                "split: need crop_rows.json (run 'convert') or metadata.csv under dest_dir"
            )

    if not rows:
        log.warning("split: no rows — skipping split write")
        return {}

    splits, manifest = assign_splits(rows, cfg)
    paths_out = write_splits(cfg.splits_dir, splits, manifest)
    log.info("split: train=%d val=%d test=%d (grouped by scene, C.5)",
             len(splits["train"]), len(splits["val"]), len(splits["test"]))
    return paths_out
