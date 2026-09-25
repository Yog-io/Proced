"""Stage 1 — tree scan → catalog.json + value-domain resolution + tree mirror."""

from __future__ import annotations

import logging
from pathlib import Path

from .catalog import Catalog, scan_tree
from .config import PipelineConfig
from .progress import bar as progress_bar
from .scene_io import resolve_value_domains
from .state import StatePaths, save_catalog

log = logging.getLogger("proced.scan")


def ensure_output_tree(cfg: PipelineConfig, cat: Catalog) -> None:
    """Mirror the full source tree under dest_dir (directory-equivalence check)."""
    dest = Path(cfg.dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for rel in cat.all_dirs:
        (dest / rel).mkdir(parents=True, exist_ok=True)


def run_scan(cfg: PipelineConfig) -> Catalog:
    log.info("scan: walking %s …", cfg.stage_dir)
    cat = scan_tree(cfg)
    for e in cat.errors:
        log.error("catalog: %s", e)

    n_resolved = 0
    pb = progress_bar(len(cat.scenes), "scan domains", unit="scene")
    try:
        for scene in cat.scenes:
            try:
                scene.value_domains = resolve_value_domains(scene, cfg)
                n_resolved += 1
            except Exception as exc:
                cat.errors.append(f"{scene.scene_id}: domain resolve failed: {exc}")
                scene.value_domains = {
                    t: ("native" if not scene.is_calibrated else cfg.value_domain)
                    for t in scene.pol_tags if t in ("VV", "VH")
                }
            finally:
                pb.update()
    finally:
        pb.close()

    ensure_output_tree(cfg, cat)

    paths = StatePaths.from_config(cfg)
    paths.ensure_root()
    save_catalog(cat, paths.catalog)

    log.info(
        "scan: %d scenes, %d folders, %d mask-only, %d domains resolved → %s",
        len(cat.scenes), len(cat.all_dirs), len(cat.mask_only_folders),
        n_resolved, paths.catalog,
    )
    return cat
