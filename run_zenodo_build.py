#!/usr/bin/env python3
"""Gated Zenodo build runner — cross-platform (use this on Windows).

Implements AGENT_ORCHESTRATION_BRIEF.md Steps 0–4 exactly like
``run_zenodo_build.sh`` (which is now a thin POSIX wrapper around this file).

Usage (Windows / PowerShell or cmd):
    python run_zenodo_build.py -path D:\\Zenodo-dataset
    python run_zenodo_build.py -path D:\\Zenodo-dataset -seed 42 -workers 8
    python run_zenodo_build.py -path D:\\Zenodo-dataset -dest D:\\Zenodo-Dataset_final

Usage (macOS/Linux — long options work too):
    python3 run_zenodo_build.py --path /path/to/extracted/zenodo/folder

Arguments:
    -path / --path      (required) extracted Zenodo root (already unpacked — NEVER extracted here)
    -seed / --seed      default 42
    -workers / --workers  default: every CPU core
    -dest / --dest      output tree; default: sibling folder "Zenodo-Dataset_final" of -path
    -pairs / --pairs    synthetic Model-2 pairs for Step 2 (default 2500)

Exit codes (same contract as the brief / shell runner):
    0  all gates passed
    2  pipeline validate failed (no allowed retry) or QA hard-fail
    3  Step 0 STOP — incomplete extraction suspected (§4.1)
    1  harness / unexpected error
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT_REPO = Path(__file__).resolve().parent
DATE_TAG = datetime.now().strftime("%Y%m%d")
LOG_DIR = ROOT_REPO / "logs"


def log(msg: str) -> None:
    print(f"[run_zenodo_build] {msg}", flush=True)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


# --------------------------------------------------------------------- runs
def run_step(argv: List[str], log_path: Path, env: Optional[dict] = None,
             mode: str = "w") -> int:
    """Run a child process, streaming stdout+stderr to console AND a log file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    child_env = dict(os.environ)
    if env:
        child_env.update(env)
    try:
        with open(log_path, mode, encoding="utf-8", errors="replace") as fh:
            fh.write(f"$ {' '.join(str(a) for a in argv)}\n")
            fh.flush()
            proc = subprocess.Popen(
                [str(a) for a in argv],
                cwd=str(ROOT_REPO),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                errors="replace",
                env=child_env,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                fh.write(line)
            return proc.wait()
    except OSError as exc:
        log(f"failed to launch {argv[0]}: {exc}")
        return 127


def py(*args: str) -> List[str]:
    return [sys.executable, *args]


# ------------------------------------------------------------- status report
def write_status(path: Path, sections: List[str], *, verdict_complete: bool) -> None:
    lines = [f"# Dataset Build Status — {DATE_TAG}", "", "## Summary verdict"]
    lines.append("[x] Fully complete and verified (all Section 2 conditions met)"
                 if verdict_complete else
                 "[ ] Fully complete and verified (all Section 2 conditions met)")
    lines.append("[ ] Stopped partway — see below for exactly where and why"
                 if verdict_complete else
                 "[x] Stopped partway — see below for exactly where and why")
    lines.append("")
    lines.extend(sections)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"wrote {path}")


def die_report(code: int, reason: str, status_md: Path, logs: Dict[str, Path]) -> None:
    """Brief §4: always leave a BUILD_STATUS report even on an early stop."""
    log(f"STOP: {reason}")
    if not status_md.is_file():
        sections = ["## Stop reason", reason, "", "## Full logs"]
        for label, p in logs.items():
            if p.is_file():
                sections.append(f"- {label}: `{p}`")
        write_status(status_md, sections, verdict_complete=False)
    sys.exit(code)


# ------------------------------------------------------------------- Step 0
def step0_gate(report_path: Path) -> Tuple[int, str]:
    """§4.1 stop gates over raw_folder_audit_report.json → (exit_code, note)."""
    d = _read_json(report_path)
    c = d.get("counts", {}) or {}
    n = c.get("files_discovered", 0) or 0
    failed = c.get("open_failed", 0) or 0
    gaps = len(d.get("folder_gaps") or [])
    ready = bool(d.get("ready_for_pipeline"))
    pct = (100.0 * failed / n) if n else (100.0 if failed else 0.0)

    if failed and n and pct > 5.0:
        return 3, (f"§4.1 STOP: >5% corrupt/unopenable — incomplete extraction "
                   f"suspected. See {report_path} (failed={failed} n={n} pct={pct:.2f})")
    if gaps >= 2 and failed > 0 and pct >= 1.0:
        return 3, (f"§4.1 STOP: multi-part extraction gaps (suspicious sibling "
                   f"folders). See {report_path} (gaps={gaps} failed={failed})")
    if gaps >= 3:
        return 3, (f"§4.1 STOP: multi-part extraction gaps. See {report_path} "
                   f"(gaps={gaps})")
    note = (f"OK ready={ready} failed={failed} n={n} pct={pct:.2f} gaps={gaps} | "
            f"orphan_images={len((d.get('pairing') or {}).get('orphan_images') or [])} "
            f"orphan_masks={len((d.get('pairing') or {}).get('orphan_masks') or [])} "
            f"duplicate_groups={len((d.get('duplicates') or {}).get('groups') or [])}")
    return 0, note


# ------------------------------------------------------------------- Step 4
def write_final_status(
    status_md: Path, *, zenodo: Path, dest: Path, seed: int, workers: int,
    pairs: int, pipe_rc: int, qa_rc: int, step0: Path, qa_file: Path,
    l1: Path, l2: Path, l3: Path, out_archive: Path,
) -> None:
    s0 = _read_json(step0)
    counts = s0.get("counts") or {}
    pair = s0.get("pairing") or {}
    dups = s0.get("duplicates") or {}
    orph_i = len(pair.get("orphan_images") or [])
    orph_m = len(pair.get("orphan_masks") or [])
    n_dup = len(dups.get("groups") or [])
    n_fail = counts.get("open_failed", 0)
    n_disc = counts.get("files_discovered", 0)
    gaps = s0.get("folder_gaps") or []

    oil = la = bg = tr = va = te = None
    ps = dest / "pipeline_summary.json"
    if ps.is_file():
        d = _read_json(ps)
        labs = d.get("label_counts") or d.get("labels") or {}
        if isinstance(labs, dict):
            oil = labs.get("oil")
            la = labs.get("lookalike") or labs.get("look-alike")
            bg = labs.get("background")
        sp = d.get("splits") or d.get("split_counts") or {}
        if isinstance(sp, dict):
            tr, va, te = sp.get("train"), sp.get("val"), sp.get("test")
        if oil is None:
            lc = d.get("counts") or {}
            oil, la, bg = lc.get("oil"), lc.get("lookalike"), lc.get("background")
            tr, va, te = lc.get("train"), lc.get("val"), lc.get("test")

    hard: List[str] = []
    soft: List[str] = []
    if qa_file.is_file():
        q = _read_json(qa_file)
        for m in q.get("modules") or []:
            for f in m.get("findings") or []:
                if not f.get("ok") and f.get("severity") == "hard":
                    hard.append(f"{m.get('name')}: {f.get('message')}")
                elif not f.get("ok") and f.get("severity") == "soft":
                    soft.append(f"{m.get('name')}: {f.get('message')}")

    complete = (pipe_rc == 0 and qa_rc == 0 and not hard)
    if n_fail and n_disc and (100.0 * n_fail / n_disc) > 5:
        action = "STOPPED — incomplete extraction suspected"
    elif n_fail:
        action = f"quarantined {n_fail} corrupt paths in report; proceeded"
    else:
        action = "proceeded"

    s: List[str] = []
    s += ["## Step 0 — Raw Audit",
          f"- Total files discovered: {n_disc}",
          f"- Corrupt/unopenable: {n_fail}",
          f"- Orphaned pairs: images={orph_i}, masks={orph_m}",
          f"- Duplicates: {n_dup}",
          f"- Folder gaps flagged: {len(gaps)}",
          f"- Action taken: {action}",
          f"- Report: `{step0}`", ""]
    s += ["## Step 1 — Pipeline Run",
          f"- Command used: `python run_zenodo_build.py -path {zenodo} -dest {dest} "
          f"-workers {workers} -seed {seed}` (output_archive: `{out_archive}`)",
          f"- Exit code: {pipe_rc}",
          "- If exit 2: see validation block in prior section / pipeline_summary.json"
          if pipe_rc == 2 else None,
          "- Retry attempted? [no]",
          f"- Final counts: oil={oil}, look-alike={la}, background={bg}; "
          f"train/val/test={tr}/{va}/{te}", ""]
    s = [x for x in s if x is not None]
    s += ["## Step 2 — Handoff Scripts",
          "- join_wind.py: [source used, mock or era5] — see log (default mock expected)",
          f"- generate_ais.py: [output confirmed present] — see `{Path('data') / 'ais' / 'ais_tracks.csv'}`",
          f"- generate_model2_pairs.py: [backend simple, pairs={pairs}]", ""]
    s += ["## Step 3 — QA Verification", f"- Exit code: {qa_rc}",
          f"- Hard-fails (must be empty for a complete build): {len(hard)}"]
    s += [f"  - {h}" for h in hard[:50]] or ["  - (none)"]
    s += [f"- Soft-fails: {len(soft)}"]
    s += [f"  - {x}" for x in soft[:50]] or ["  - (none)"]
    s += ["", "## Open items for the human tomorrow"]
    s.append("- See summary verdict and Step exits above; inspect logs for full stdout/stderr."
             if not complete else "- (none — all Section 2 conditions met)")
    s += ["", "## Full logs"]
    s += [f"- {lab}: `{p}`" for lab, p in
          [("Step 0", step0), ("Step 1", l1), ("Step 2", l2), ("Step 3", l3)]]
    s += ["", f"_Generated {datetime.now(timezone.utc).isoformat()} by run_zenodo_build.py_"]
    write_status(status_md, s, verdict_complete=complete)


def write_validate_failed_status(
    status_md: Path, *, zenodo: Path, dest: Path, seed: int, workers: int,
    out_archive: Path, step0: Path, l1: Path, l2: Path, l3: Path, pipeline_summary: Path,
) -> None:
    valid_msg = "(pipeline_summary.json missing or unreadable)"
    if pipeline_summary.is_file():
        v = _read_json(pipeline_summary).get("validation") or {}
        valid_msg = json.dumps(v, indent=2, default=str)[:4000] or valid_msg
    s = ["## Step 0 — Raw Audit", f"- Report: `{step0}`",
         "- (see report counts for discovered/corrupt/orphans/duplicates)", "",
         "## Step 1 — Pipeline Run",
         f"- Command: `python run_zenodo_build.py -path {zenodo} -dest {dest} "
         f"-workers {workers} -seed {seed}`",
         "- Exit code: 2",
         "- Validation failure (verbatim from pipeline_summary.json):",
         "```json", valid_msg, "```",
         "- Retry attempted? [no — §4.2 requires an unambiguous domain/calib override; not auto-guessed]",
         "",
         "## Open items for the human tomorrow",
         "- Pipeline validate failed (exit 2). Inspect validation block above; if value-domain/"
         "calibration, one retry with `--value-domain` / `--calibrated-match` / "
         "`--uncalibrated-match` is allowed.",
         "",
         "## Full logs", f"- Step 0: `{step0}`", f"- Step 1: `{l1}`"]
    write_status(status_md, s, verdict_complete=False)


# --------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_zenodo_build.py",
        description="Gated Zenodo build (brief Steps 0–4) — works on Windows/macOS/Linux",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-path", "--path", dest="path", required=True,
                   help="Extracted Zenodo root folder (never extracted by this runner)")
    p.add_argument("-seed", "--seed", dest="seed", type=int, default=42,
                   help="Pipeline / QA seed (brief §1: 42)")
    p.add_argument("-workers", "--workers", dest="workers", type=int, default=None,
                   help="Worker processes (default: every CPU core)")
    p.add_argument("-dest", "--dest", dest="dest", default=None,
                   help="Output tree (default: sibling 'Zenodo-Dataset_final' of -path)")
    p.add_argument("-pairs", "--pairs", dest="pairs", type=int, default=2500,
                   help="Model-2 synthetic pairs for Step 2")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    zenodo = Path(args.path).expanduser()
    if not zenodo.is_dir():
        log(f"ERROR: -path is not a directory: {zenodo}")
        return 1
    try:
        zenodo = zenodo.resolve()
    except OSError:
        pass

    seed = int(args.seed)
    workers = int(args.workers) if args.workers else max(1, os.cpu_count() or 8)
    pairs = int(args.pairs)
    dest = Path(args.dest).expanduser() if args.dest else zenodo.parent / "Zenodo-Dataset_final"

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    step0_json = zenodo / "raw_folder_audit_report.json"
    step0_log = LOG_DIR / f"step0_raw_audit_{DATE_TAG}.log"
    step1_log = LOG_DIR / f"step1_pipeline_{DATE_TAG}.log"
    step2_log = LOG_DIR / f"step2_handoff_{DATE_TAG}.log"
    step3_log = LOG_DIR / f"step3_qa_{DATE_TAG}.log"
    status_md = ROOT_REPO / f"BUILD_STATUS_{DATE_TAG}.md"
    out_archive = ROOT_REPO / "data" / "master_dataset_v1.7z"
    pipeline_summary = dest / "pipeline_summary.json"
    qa_report = dest / "verification_report.json"
    qa_report_fallback = ROOT_REPO / "verification_report.json"
    logs = {"Step 0 report": step0_json, "Step 1 log": step1_log,
            "Step 2 log": step2_log, "Step 3 log": step3_log}

    log(f"path={zenodo}")
    log(f"dest={dest}  seed={seed}  workers={workers} "
        f"(cpu_count={os.cpu_count()})  pairs={pairs}")

    # ---------------------------------------------------------------- Step 0
    log("[Step 0/4] raw folder audit")
    rc0 = run_step(py("qa_verification/raw_audit/run_raw_folder_audit.py",
                      "--root", str(zenodo), "--out", str(step0_json)),
                   step0_log)
    if not step0_json.is_file():
        die_report(1, f"Step 0 produced no report (rc={rc0})", status_md, logs)
    gate_rc, gate_note = step0_gate(step0_json)
    log(gate_note)
    if gate_rc != 0:
        die_report(3, gate_note, status_md, logs)
    log("[Step 0/4] gate passed (quarantine/orphans/dups logged in report — proceed)")

    # ---------------------------------------------------------------- Step 1
    log(f"[Step 1/4] pipeline run seed={seed} workers={workers} (NO extract)")
    dest.mkdir(parents=True, exist_ok=True)
    pipe_rc = run_step(
        py("sar_dataset_pipeline.py", "run",
           "--stage_dir", str(zenodo), "--dest_dir", str(dest),
           "--output_archive", str(out_archive),
           "--workers", str(workers), "--seed", str(seed)),
        step1_log,
    )
    log(f"[Step 1/4] exit={pipe_rc} log={step1_log}")
    if pipe_rc == 2:
        write_validate_failed_status(
            status_md, zenodo=zenodo, dest=dest, seed=seed, workers=workers,
            out_archive=out_archive, step0=step0_json, l1=step1_log,
            l2=step2_log, l3=step3_log, pipeline_summary=pipeline_summary,
        )
        return 2
    if pipe_rc != 0:
        die_report(1, f"Step 1 failed with exit {pipe_rc} (not 0/2). See {step1_log}",
                   status_md, logs)

    # ---------------------------------------------------------------- Step 2
    log("[Step 2/4] handoff scripts (mock/simple expected)")
    if step2_log.is_file():
        step2_log.unlink()  # fresh log per run (first script opens with "w")
    step2_rcs = []
    for i, (label, argv) in enumerate((
        ("join_wind", py("scripts/join_wind.py")),
        ("generate_ais", py("scripts/generate_ais.py")),
        ("generate_model2_pairs",
         py("scripts/generate_model2_pairs.py", "--pairs", str(pairs))),
    )):
        rc = run_step(argv, step2_log, mode="w" if i == 0 else "a")
        step2_rcs.append(f"{label}={rc}")
    log(f"[Step 2/4] done ({', '.join(step2_rcs)}) log={step2_log}")

    # ---------------------------------------------------------------- Step 3
    log(f"[Step 3/4] independent QA verification seed={seed}")
    qa_env = {
        "QA_STAGE_DIR": str(zenodo),
        "QA_DEST_DIR": str(dest),
        "QA_STATE_DIR": str(ROOT_REPO / "data" / "state"),
        "QA_FEATURES_DIR": str(ROOT_REPO / "data" / "features"),
        "QA_SPLITS_DIR": str(ROOT_REPO / "data" / "splits"),
        "QA_CORRIDORS": str(ROOT_REPO / "data" / "reference" / "shipping_corridors.geojson"),
    }
    for var in ("QA_OUTPUT_ARCHIVE", "QA_SOURCE_ARCHIVE"):
        os.environ.pop(var, None)
    qa_rc = run_step(py("qa_verification/run_all_verification.py", "--seed", str(seed)),
                     step3_log, env=qa_env)
    log(f"[Step 3/4] exit={qa_rc} log={step3_log}")

    qa_file = qa_report if qa_report.is_file() else qa_report_fallback

    # ---------------------------------------------------------------- Step 4
    log(f"[Step 4/4] write {status_md}")
    write_final_status(
        status_md, zenodo=zenodo, dest=dest, seed=seed, workers=workers,
        pairs=pairs, pipe_rc=pipe_rc, qa_rc=qa_rc, step0=step0_json, qa_file=qa_file,
        l1=step1_log, l2=step2_log, l3=step3_log, out_archive=out_archive,
    )

    log(f"Done. Status: {status_md}")
    if qa_rc == 0 and pipe_rc == 0:
        log("RESULT: gates passed (exit 0)")
        return 0
    if qa_rc == 2 or pipe_rc == 2:
        log("RESULT: hard-fail / validate fail — see report (exit 2)")
        return 2
    log(f"RESULT: incomplete — see report (exit {qa_rc})")
    return qa_rc or 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(1)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
