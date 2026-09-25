"""Progress bars + CPU/memory guardrail tests (proced.progress & pool pinning)."""

import io
import os
import re
from pathlib import Path

from proced import progress
from qa_verification import _progress as qa_progress


class _TTY(io.StringIO):
    def isatty(self):
        return True


def test_progress_enabled_by_default():
    assert progress.progress_enabled() is True


def test_log_mode_emits_initial_and_final_lines():
    buf = io.StringIO()
    p = progress.bar(20, "folders", unit="folder", file=buf)
    assert "[progress] folders: 0/20" in buf.getvalue()
    for _ in range(20):
        p.update()
    p.close()
    out = buf.getvalue()
    assert "folders: 20/20" in out
    assert out.endswith("\n")
    assert "\r" not in out  # pipe/log friendly — no carriage returns


def test_log_mode_throttles_small_totals_to_at_most_one_line_per_item():
    buf = io.StringIO()
    p = progress.bar(4, "scenes", unit="scene", file=buf)
    for _ in range(4):
        p.update()
    p.close()
    lines = [ln for ln in buf.getvalue().splitlines() if ln]
    assert len(lines) <= 6  # initial + per-item + final, never unbounded


def test_tty_mode_uses_carriage_return_and_final_newline():
    buf = _TTY()
    p = progress.bar(4, "scenes", unit="scene", file=buf)
    for _ in range(4):
        p.update()
    p.close()
    out = buf.getvalue()
    assert "\r" in out
    assert out.endswith("\n")
    assert "4/4" in out


def test_disabled_by_env_is_silent(monkeypatch):
    monkeypatch.setenv("PROGRESS", "0")
    buf = io.StringIO()
    p = progress.bar(3, "x", file=buf)
    p.update(3)
    p.close()
    assert buf.getvalue() == ""


def test_context_manager_closes_bar():
    buf = io.StringIO()
    with progress.bar(2, "walk", unit="it", file=buf) as p:
        p.update(2)
    assert buf.getvalue().endswith("\n")


def test_stage_progress_banners_skips_and_master_bar():
    buf = _TTY()
    with progress.StageProgress(3, file=buf) as st:
        st.begin("scan")
        st.end("scan")
        st.begin("convert")
        st.end("convert")
        st.skip("archive")
    out = buf.getvalue()
    assert "▶ [1/3] scan" in out
    assert "✓ scan" in out
    assert "▶ [2/3] convert" in out
    assert "· archive skipped" in out
    assert "3/3" in out  # master bar completed
    assert out.endswith("\n")


def test_stage_progress_marks_interrupted_stage():
    buf = _TTY()
    st = progress.StageProgress(2, file=buf)
    st.begin("features")
    st.close()
    assert "✗ features interrupted" in buf.getvalue()


def test_serial_pool_shows_progress_and_returns_results():
    from proced.config import PipelineConfig
    from proced.pool import run_serial_or_pool

    cfg = PipelineConfig(workers=1)
    out = run_serial_or_pool(cfg, [1, 2, 3], lambda x: x * 2, label="jobs")
    assert out == [2, 4, 6]
    assert run_serial_or_pool(cfg, [], lambda x: x, label="jobs") == []


def test_config_default_workers_is_every_cpu_core():
    from proced.config import PipelineConfig

    assert PipelineConfig().workers == max(1, os.cpu_count() or 4)


def test_init_worker_pins_blas_threads_to_one():
    from proced.pool import BLAS_THREAD_VARS, init_worker

    saved = {k: os.environ.get(k) for k in BLAS_THREAD_VARS}
    try:
        for k in BLAS_THREAD_VARS:
            os.environ.pop(k, None)
        init_worker(64, "2")
        for k in BLAS_THREAD_VARS:
            assert os.environ.get(k) == "1"
        # user-provided values must win (setdefault semantics)
        os.environ["OMP_NUM_THREADS"] = "4"
        init_worker(64, "2")
        assert os.environ["OMP_NUM_THREADS"] == "4"
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_qa_progress_module_renders_and_respects_boundary():
    buf = io.StringIO()
    p = qa_progress.bar(5, "qa", unit="module", file=buf)
    for _ in range(5):
        p.update()
    p.close()
    assert "qa: 5/5" in buf.getvalue()

    # QA must never import proced (boundary rule)
    src = Path(qa_progress.__file__).read_text()
    assert not re.search(r"^\s*(?:from\s+proced[\s.]|import\s+proced\b)", src, re.M)
