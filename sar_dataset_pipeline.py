#!/usr/bin/env python3
"""SAR Master Dataset Converter & Preprocessor — Stage 0-6 pipeline (CLI).

Thin entry point over ``proced.cli`` subcommands. Implements the architecture
in ``archi.md`` under the constraints of ``BLOCK1_DATA_TEAM_GUIDE.md`` +
``DATASET_FORMAT_REFINEMENT_REPORT.md``.

Stages (each runnable standalone; intermediate state under ``--state_dir``)
----------------------------------------------------------------------------
0. extract   — .7z decompression ONLY (7z CLI / py7zr fallback). The pipeline
               never auto-extracts; point ``--stage_dir`` at your extracted tree.
1. scan      — tree parsing, VV/VH↔mask pairing, calibration detection, mirror
2. features  — pre-crop full-scene (C.0) shape + radiometric features
3a. geo      — real/synthetic GPS + timestamps → scene_geo.json
3b. plan     — crop geometry classification → crop_plans.json
4. convert   — parallel windowed tiling → dB → 16-bit uint16 PNG + GEODATA
5. metadata  — master metadata.csv + feature/provenance CSVs
6. split / archive / validate — group-safe splits, .7z pack, checklist

``run`` executes scan→validate in order and **never** extracts.

Example
-------
::

    # 1) unpack (or use your own extracted folder)
    python sar_dataset_pipeline.py extract \\
        --archive data/raw_sar_inputs.7z \\
        --stage_dir data/scratch_unpacked

    # 2) full pipeline (no --archive here)
    python sar_dataset_pipeline.py run \\
        --stage_dir data/scratch_unpacked \\
        --dest_dir data/master_dataset_processed \\
        --output_archive data/master_dataset_v1.7z \\
        --workers 8

    # Or run stages one at a time:
    python sar_dataset_pipeline.py scan --stage_dir data/scratch_unpacked
    python sar_dataset_pipeline.py geo  --stage_dir data/scratch_unpacked
    ...
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root importable when invoked as a script
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from proced.cli import build_parser, config_from_args, main, run_pipeline  # noqa: F401

__all__ = ["main", "run_pipeline", "build_parser", "config_from_args"]


if __name__ == "__main__":
    raise SystemExit(main())
