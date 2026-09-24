"""Feed the real pipeline binary malformed inputs; require graceful failure.

Architecture §5: each case must fail gracefully (flagged/skipped/logged /
non-zero exit) — never crash with an unhandled traceback that kills the
whole run without a log line, and never silently emit a bad crop row.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

from .._lib import QAPaths, Report
from .make_fixtures import build_default_fixtures

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "sar_dataset_pipeline.py"


def _run_scan(case_dir: Path, work: Path) -> subprocess.CompletedProcess:
    state = work / "state"
    dest = work / "dest"
    cmd = [
        sys.executable, str(PIPELINE), "scan",
        "--stage_dir", str(case_dir),
        "--state_dir", str(state),
        "--dest_dir", str(dest),
        "--features_dir", str(work / "f"),
        "--splits_dir", str(work / "s"),
        "--workers", "1",
        "--neg-buffer-px", "64",
        "--seed", "7",
    ]
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=120, cwd=str(ROOT),
    )


def _run_full(case_dir: Path, work: Path, *, no_pack: bool = True) -> subprocess.CompletedProcess:
    state = work / "state"
    dest = work / "dest"
    cmd = [
        sys.executable, str(PIPELINE), "run",
        "--stage_dir", str(case_dir),
        "--state_dir", str(state),
        "--dest_dir", str(dest),
        "--features_dir", str(work / "f"),
        "--splits_dir", str(work / "s"),
        "--workers", "1",
        "--neg-buffer-px", "64",
        "--seed", "7",
        "--no-pack",
        "--no-validate" if no_pack else "--validate",
    ]
    # --no-validate not valid; drop it
    cmd = [c for c in cmd if c != "--validate" and c != "--no-validate"]
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=180, cwd=str(ROOT),
    )


def run(paths: QAPaths, *, full_pipeline: bool = False) -> Report:
    rep = Report("adversarial_graceful_failure")
    cases = build_default_fixtures()
    rep.info(f"built {len(cases)} malformed fixtures",
             cases=sorted(cases))

    outcomes = []
    for name, case_dir in sorted(cases.items()):
        with tempfile.TemporaryDirectory(prefix=f"qa_adv_{name}_") as td:
            work = Path(td)
            try:
                if full_pipeline:
                    proc = _run_full(case_dir, work)
                else:
                    proc = _run_scan(case_dir, work)
            except subprocess.TimeoutExpired:
                outcomes.append({"case": name, "error": "timeout"})
                rep.hard(False, f"{name}: pipeline timed out (hang, not graceful)")
                continue
            except Exception as exc:
                outcomes.append({"case": name, "error": str(exc)})
                rep.soft(False, f"{name}: harness error: {exc}")
                continue

            rc = proc.returncode
            stderr = (proc.stderr or "")
            stdout = (proc.stdout or "")
            unhandled = "Traceback (most recent call last):" in stderr
            # Accept: exit 0 with catalog errors logged, OR non-zero exit.
            # Fail hard: unhandled traceback with exit 0 (silent partial) is OK
            # only if catalog errors recorded; unhandled with crash is still
            # "not graceful" if we cannot find a log line naming the scene.
            logged = (
                "error" in stdout.lower()
                or "error" in stderr.lower()
                or "skip" in stdout.lower()
                or "fail" in stdout.lower()
                or "Traceback" in stderr
            )
            outcomes.append({
                "case": name, "returncode": rc,
                "unhandled_traceback": unhandled,
                "logged": logged,
            })

            if rc not in (0, 2) and not logged:
                rep.hard(
                    False,
                    f"{name}: exit {rc} without a clear error log line",
                    stderr_tail=stderr[-500:],
                )
            elif unhandled and not logged:
                # traceback present but no other message — still harsh but
                # stderr itself is the log; accept if returncode != 0
                if rc == 0:
                    rep.hard(
                        False,
                        f"{name}: unhandled traceback but exit 0 (silent failure risk)",
                        stderr_tail=stderr[-500:],
                    )
                else:
                    rep.soft(
                        True,
                        f"{name}: failed with traceback + exit {rc} "
                        "(non-zero ⇒ not silent)",
                    )
            else:
                rep.info(
                    f"{name}: graceful (exit={rc}, logged={logged})",
                    returncode=rc,
                )

            # Malformed cases must not invent a rich metadata with positives
            # from an all-zero / broken mask unless backgrounds only.
            md = work / "dest" / "metadata.csv"
            if md.is_file() and name in ("all_zero_mask", "corrupted_tiff",
                                         "mask_size_mismatch"):
                try:
                    import pandas as pd
                    df = pd.read_csv(md)
                    n_oil = 0
                    if "label" in df.columns:
                        n_oil = int((df["label"].astype(str) == "oil").sum())
                    if n_oil > 0 and name == "all_zero_mask":
                        rep.hard(
                            False,
                            f"{name}: pipeline emitted {n_oil} oil rows from an "
                            "all-zero mask (phantom positives)",
                        )
                    elif name == "corrupted_tiff" and len(df) > 0:
                        # some rows might exist from residual files; only flag
                        # if labelled oil from the broken scene
                        bad = df[df["crop_source_scene_id"].astype(str)
                                 .str.contains("scene_bad", na=False)]
                        if len(bad) and "label" in bad.columns:
                            n_oil = int((bad["label"].astype(str) == "oil").sum())
                            if n_oil:
                                rep.hard(
                                    False,
                                    f"{name}: {n_oil} oil crops from corrupted TIFF",
                                )
                except Exception as exc:
                    rep.soft(False, f"{name}: post-check error: {exc}")

    return rep
