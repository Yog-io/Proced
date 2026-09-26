"""Determinism: run the real pipeline twice with the same seed → byte-identical.

Architecture §5. Marked slow — full mini-pipeline twice. Compares hashes of
every file under dest_dir + state artifact bytes (excluding volatile logs).
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

from .._lib import QAPaths, Report

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "sar_dataset_pipeline.py"


def _hash_tree(root: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not root.exists():
        return out
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root))
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out[rel] = h.hexdigest()
    return out


def _run_once(stage: Path, work: Path, seed: int) -> int:
    cmd = [
        sys.executable, str(PIPELINE), "run",
        "--stage_dir", str(stage),
        "--state_dir", str(work / "state"),
        "--dest_dir", str(work / "dest"),
        "--features_dir", str(work / "f"),
        "--splits_dir", str(work / "s"),
        "--output_archive", str(work / "out.7z"),
        "--workers", "1",
        "--neg-buffer-px", "120",
        "--neg-ratio", "0.5",
        "--seed", str(seed),
        "--tile-size", "256",
        "--no-pack",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=300, cwd=str(ROOT),
    )
    return proc.returncode


def run(
    paths: QAPaths,
    *,
    stage_dir: Optional[Path] = None,
    seed: int = 42,
) -> Report:
    rep = Report("adversarial_determinism")
    stage = Path(stage_dir) if stage_dir else paths.stage_dir
    if not stage.is_dir():
        rep.hard(False, f"stage_dir missing for determinism run: {stage}")
        return rep

    with tempfile.TemporaryDirectory(prefix="qa_det_") as td:
        w1 = Path(td) / "run1"
        w2 = Path(td) / "run2"
        w1.mkdir()
        w2.mkdir()

        rc1 = _run_once(stage, w1, seed)
        rc2 = _run_once(stage, w2, seed)

        if rc1 != 0 or rc2 != 0:
            rep.hard(
                False,
                f"determinism runs failed (rc1={rc1}, rc2={rc2}) — cannot compare",
                stderr_note="inspect pipeline logs",
            )
            return rep

        h1 = _hash_tree(w1 / "dest")
        h2 = _hash_tree(w2 / "dest")
        # state too (catalog/plans should be seed-stable)
        s1 = _hash_tree(w1 / "state")
        s2 = _hash_tree(w2 / "state")

        if not h1:
            rep.hard(False, "first determinism run produced no dest files")
            return rep

        only1 = sorted(set(h1) - set(h2))
        only2 = sorted(set(h2) - set(h1))
        differ = sorted(k for k in set(h1) & set(h2) if h1[k] != h2[k])
        s_differ = sorted(k for k in set(s1) & set(s2) if s1[k] != s2[k])

        # pipeline_summary.json is volatile: embeds elapsed_s + per-run paths.
        volatile = [k for k in differ if k.endswith("pipeline_summary.json")]
        real_diffs = [k for k in differ if k not in set(volatile)]

        rep.hard(
            not only1 and not only2 and not real_diffs,
            f"byte-identical dest trees across 2 seeded runs "
            f"({len(h1)} files)"
            if not only1 and not only2 and not real_diffs
            else f"determinism broken: +{len(only1)}/{len(only2)} files, "
                 f"{len(real_diffs)} content diffs",
            only_run1=only1[:20],
            only_run2=only2[:20],
            content_diffs=real_diffs[:20],
            volatile_diffs=volatile[:20],
            state_diffs=s_differ[:20],
        )
        if volatile and not real_diffs and not only1 and not only2:
            rep.soft(
                True,
                "only volatile pipeline_summary.json differs (elapsed_s / "
                "run paths) — crop bytes identical",
                volatile=volatile,
            )
    return rep
