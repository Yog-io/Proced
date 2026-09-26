"""run_zenodo_build.py — Windows-friendly gated runner contract.

Covers the single-dash CLI interface the dataset (Windows) PC uses
(``-path -seed -workers -dest -pairs``), the §4.1 Step 0 stop gates, the
BUILD_STATUS writer, and run_step() log streaming.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_zenodo_build", ROOT / "run_zenodo_build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner = _load_runner()


# ---------------------------------------------------------------- CLI args
def test_single_dash_args_accepted():
    args = runner.build_parser().parse_args(
        ["-path", "/data/zenodo", "-seed", "7", "-workers", "4",
         "-dest", "/data/out", "-pairs", "50"])
    assert args.path == "/data/zenodo"
    assert args.seed == 7
    assert args.workers == 4
    assert args.dest == "/data/out"
    assert args.pairs == 50


def test_long_form_args_accepted():
    args = runner.build_parser().parse_args(
        ["--path", "/data/zenodo", "--seed", "42", "--workers", "8"])
    assert args.path == "/data/zenodo"
    assert args.seed == 42
    assert args.workers == 8


def test_defaults_seed_42_and_pairs_2500():
    args = runner.build_parser().parse_args(["-path", "/data/zenodo"])
    assert args.seed == 42
    assert args.pairs == 2500
    assert args.workers is None          # resolved to cpu_count at run time
    assert args.dest is None             # sibling Zenodo-Dataset_final


def test_path_is_required():
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args([])


def test_help_mentions_single_dash_usage():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "run_zenodo_build.py"), "--help"],
        capture_output=True, text=True, cwd=str(ROOT))
    assert proc.returncode == 0
    assert "-path" in proc.stdout and "-workers" in proc.stdout


def test_main_rejects_missing_path_dir(tmp_path):
    rc = runner.main(["-path", str(tmp_path / "does_not_exist")])
    assert rc == 1


# ------------------------------------------------------------- Step 0 gate
def _report(tmp_path, *, n=100, failed=0, gaps=0, ready=True):
    p = tmp_path / "raw_folder_audit_report.json"
    p.write_text(json.dumps({
        "counts": {"files_discovered": n, "open_failed": failed},
        "folder_gaps": ["g"] * gaps,
        "ready_for_pipeline": ready,
        "pairing": {"orphan_images": [], "orphan_masks": []},
        "duplicates": {"groups": []},
    }), encoding="utf-8")
    return p


def test_gate_passes_clean_tree(tmp_path):
    rc, note = runner.step0_gate(_report(tmp_path))
    assert rc == 0
    assert "OK" in note and "gaps=0" in note


def test_gate_stops_over_5pct_corrupt(tmp_path):
    rc, note = runner.step0_gate(_report(tmp_path, n=100, failed=6))
    assert rc == 3
    assert "4.1" in note


def test_gate_stops_on_three_folder_gaps(tmp_path):
    rc, note = runner.step0_gate(_report(tmp_path, gaps=3))
    assert rc == 3
    assert "gaps" in note


def test_gate_stops_gaps_with_failures(tmp_path):
    rc, _ = runner.step0_gate(_report(tmp_path, n=200, failed=4, gaps=2))
    assert rc == 3


# ---------------------------------------------------------- status reports
def test_write_status_verdict_boxes(tmp_path):
    complete = tmp_path / "complete.md"
    runner.write_status(complete, ["## Step 1 — Pipeline Run", "- ok"],
                        verdict_complete=True)
    txt = complete.read_text(encoding="utf-8")
    assert "[x] Fully complete and verified" in txt
    assert "[ ] Stopped partway" in txt

    stopped = tmp_path / "stopped.md"
    runner.write_status(stopped, ["## Stop reason", "boom"],
                        verdict_complete=False)
    txt2 = stopped.read_text(encoding="utf-8")
    assert "[ ] Fully complete and verified" in txt2
    assert "[x] Stopped partway" in txt2


def test_die_report_writes_status_and_exits(tmp_path):
    status = tmp_path / "BUILD_STATUS.md"
    log = tmp_path / "step1.log"
    log.write_text("traceback\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        runner.die_report(3, "§4.1 STOP: gaps", status,
                          {"Step 1 log": log})
    assert exc.value.code == 3
    txt = status.read_text(encoding="utf-8")
    assert "§4.1 STOP: gaps" in txt
    assert "Stopped partway" in txt
    assert "step1.log" in txt


def test_write_final_status_complete_and_hard_fail(tmp_path):
    step0 = _report(tmp_path, n=50, failed=1)
    qa = tmp_path / "verification_report.json"
    summary = tmp_path / "pipeline_summary.json"
    summary.write_text(json.dumps({
        "label_counts": {"oil": 3, "lookalike": 4, "background": 2},
        "splits": {"train": 6, "val": 2, "test": 1},
    }), encoding="utf-8")

    def status_for(tag, hard):
        qa.write_text(json.dumps({"modules": [
            {"name": "stage1_scan",
             "findings": [{"severity": "hard", "ok": not hard,
                           "message": "boom" if hard else "ok"}]},
        ]}), encoding="utf-8")
        out = tmp_path / f"BUILD_STATUS_{tag}.md"
        runner.write_final_status(
            out, zenodo=tmp_path, dest=tmp_path, seed=42, workers=4,
            pairs=10, pipe_rc=0, qa_rc=0 if not hard else 2, step0=step0,
            qa_file=qa, l1=tmp_path / "l1.log", l2=tmp_path / "l2.log",
            l3=tmp_path / "l3.log", out_archive=tmp_path / "o.7z")
        return out.read_text(encoding="utf-8")

    ok_txt = status_for("ok", hard=False)
    assert "[x] Fully complete and verified" in ok_txt
    assert "oil=3" in ok_txt and "train/val/test=6/2/1" in ok_txt
    assert "quarantined 1 corrupt paths" in ok_txt

    bad_txt = status_for("hard", hard=True)
    assert "[ ] Fully complete and verified" in bad_txt
    assert "stage1_scan: boom" in bad_txt


# -------------------------------------------------------------- run_step()
def test_run_step_streams_to_log(tmp_path):
    log = tmp_path / "child.log"
    rc = runner.run_step(runner.py("-c", "print('hello-from-child')"), log)
    assert rc == 0
    txt = log.read_text(encoding="utf-8")
    assert "hello-from-child" in txt
    assert txt.splitlines()[0].startswith("$ ")


def test_run_step_append_mode(tmp_path):
    log = tmp_path / "multi.log"
    runner.run_step(runner.py("-c", "print('one')"), log, mode="w")
    runner.run_step(runner.py("-c", "print('two')"), log, mode="a")
    txt = log.read_text(encoding="utf-8")
    assert "one" in txt and "two" in txt


def test_wrapper_is_posix_thin_shell():
    sh = (ROOT / "run_zenodo_build.sh").read_text(encoding="utf-8")
    assert "run_zenodo_build.py" in sh
    proc = subprocess.run(["bash", "-n", str(ROOT / "run_zenodo_build.sh")],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
