"""Progress bars for every pipeline step and stage.

Self-contained (no tqdm dependency) so it behaves the same on the builder PC
and the dataset PC:

* TTY (interactive terminal) → live ``\\r`` bar, redrawn at most ~3x/second
* pipe / ``tee`` log file    → plain one-line updates, throttled to every 5 %
  of the work or 15 s, whichever comes first (keeps run logs readable)
* ``PROGRESS=0`` (or off/false/no) → completely silent

Every heavy loop drives one of these: pool jobs (features/plan/convert),
per-scene stages (scan domains, geo), validation walks, archive packing —
and :class:`StageProgress` wraps a whole ``run`` with per-stage banners plus
a master ``pipeline 3/9`` bar.
"""

from __future__ import annotations

import os
import sys
import time
from typing import IO, Optional

_TTY_INTERVAL = 0.35   # seconds between live redraws
_LOG_INTERVAL = 15.0   # seconds between log-mode lines while work continues
_LOG_STEP = 0.05       # log-mode line at least every 5% of total
_WIDTH = 24            # bar width in characters (TTY only)
_DISABLE_VALUES = {"0", "off", "false", "no"}


def progress_enabled() -> bool:
    """True unless the environment sets ``PROGRESS`` to a falsy value."""
    return os.environ.get("PROGRESS", "1").strip().lower() not in _DISABLE_VALUES


def _is_tty(stream) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _fmt_dur(seconds: float) -> str:
    s = max(0, int(seconds))
    if s >= 3600:
        return f"{s // 3600}h{s % 3600 // 60:02d}m"
    if s >= 60:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s}s"


def bar(total, desc: str = "", unit: str = "it", **kwargs) -> "ProgressBar":
    """Convenience constructor — ``with bar(100, "scenes") as p: p.update()``."""
    return ProgressBar(total, desc, unit, **kwargs)


class ProgressBar:
    """One progress bar over ``total`` items (``total=None`` → count-only)."""

    def __init__(
        self,
        total = None,
        desc: str = "",
        unit: str = "it",
        *,
        file: Optional[IO] = None,
        enabled: Optional[bool] = None,
        width: int = _WIDTH,
    ) -> None:
        self.total = int(total) if total else 0
        self.desc = desc
        self.unit = unit
        self.n = 0
        self.width = width
        self.file = file if file is not None else sys.stderr
        self.enabled = progress_enabled() if enabled is None else bool(enabled)
        self.is_tty = _is_tty(self.file)
        self.t0 = time.time()
        self.elapsed = 0.0
        self._closed = False
        self._last_render = 0.0
        self._last_len = 0
        self._last_log_n = -1
        self._last_log_t = self.t0
        if self.enabled:
            self._render(force=True)

    # ------------------------------------------------------------- public
    def update(self, n: int = 1) -> None:
        if not self.enabled or self._closed:
            return
        self.n += n
        self._render()

    def set_desc(self, desc: str) -> None:
        """Change the label (used when one bar spans differently-named units)."""
        self.desc = desc
        if self.enabled and not self._closed and self.is_tty:
            self._render(force=True)

    def clear(self) -> None:
        """Erase the current TTY line so another bar/banner can use it."""
        if not (self.enabled and self.is_tty and self._last_len and not self._closed):
            return
        try:
            self.file.write("\r" + " " * self._last_len + "\r")
            self.file.flush()
        except Exception:
            pass
        self._last_len = 0

    def close(self) -> None:
        if not self.enabled or self._closed:
            return
        try:
            if self.is_tty:
                self._render(force=True)
                self.file.write("\n")
            elif self._last_log_n != self.n:
                self._emit_log_line()
            self.file.flush()
        except Exception:
            pass
        self._closed = True

    # --------------------------------------------------------------- dunders
    def __enter__(self) -> "ProgressBar":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    # ------------------------------------------------------------ internals
    def _line(self) -> str:
        elapsed = time.time() - self.t0
        self.elapsed = elapsed
        head = f"{self.desc}: {self.n}/{self.total}" if self.total else f"{self.desc}: {self.n} {self.unit}"
        if not self.total:
            return f"{head} ({_fmt_dur(elapsed)})"
        pct = 100.0 * self.n / self.total
        filled = int(round(self.width * self.n / self.total))
        filled = max(0, min(self.width, filled))
        parts = [head, f"[{'#' * filled}{'-' * (self.width - filled)}]", f"{pct:5.1f}%"]
        if 0 < self.n < self.total and elapsed > 0:
            rate = self.n / elapsed
            parts.append(f"eta {_fmt_dur((self.total - self.n) / rate)}")
        return " ".join(parts)

    def _render(self, force: bool = False) -> None:
        if not self.enabled or self._closed:
            return
        now = time.time()
        finished = bool(self.total) and self.n >= self.total
        if self.is_tty:
            if not force and not finished and now - self._last_render < _TTY_INTERVAL:
                return
            self._last_render = now
            line = self._line()
            pad = max(0, self._last_len - len(line))
            try:
                self.file.write("\r" + line + (" " * pad if pad else ""))
                self.file.flush()
            except Exception:
                pass
            self._last_len = max(len(line), self._last_len) if pad else len(line)
            return
        # log mode: emit on step/time thresholds (never when idle)
        step = max(1, int(self.total * _LOG_STEP)) if self.total else 100
        due_n = (self.n - self._last_log_n) >= step
        due_t = (now - self._last_log_t) >= _LOG_INTERVAL and self.n != self._last_log_n
        if force or due_n or due_t or finished:
            self._emit_log_line()

    def _emit_log_line(self) -> None:
        try:
            self.file.write(f"[progress] {self._line()}\n")
            self.file.flush()
        except Exception:
            pass
        self._last_log_n = self.n
        self._last_log_t = time.time()


class StageProgress:
    """Master bar over the top-level stages of a ``run``.

    Prints ``▶ [3/9] features`` before each stage and ``✓ features 12.4s``
    after it, while a ``pipeline: 3/9 [...]`` bar tracks overall completion.
    The master bar yields the terminal line while a stage is running so the
    stage's own item bar can use it.
    """

    def __init__(
        self,
        total: int,
        desc: str = "pipeline",
        *,
        file: Optional[IO] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self.file = file if file is not None else sys.stderr
        self.enabled = progress_enabled() if enabled is None else bool(enabled)
        self.total = int(total)
        self._bar = ProgressBar(self.total, desc, unit="stage",
                                file=self.file, enabled=self.enabled)
        self._count = 0
        self._t0_stage: Optional[float] = None
        self._name: Optional[str] = None
        self._closed = False

    def begin(self, name: str) -> None:
        if self._closed:
            return
        self._name, self._t0_stage = name, time.time()
        if not self.enabled:
            return
        self._bar.clear()
        self._write(f"▶ [{self._count + 1}/{self.total}] {name}\n")

    def end(self, name: Optional[str] = None) -> None:
        if self._closed:
            return
        label = name or self._name or ""
        dt = time.time() - self._t0_stage if self._t0_stage else 0.0
        self._name, self._t0_stage = None, None
        self._count += 1
        if not self.enabled:
            return
        self._write(f"✓ {label} {_fmt_dur(dt)}\n")
        self._bar.update(1)

    def skip(self, name: str) -> None:
        if self._closed:
            return
        self._count += 1
        if self.enabled:
            self._bar.clear()
            self._write(f"· {name} skipped\n")
            self._bar.update(1)

    def _write(self, s: str) -> None:
        try:
            self.file.write(s)
            self.file.flush()
        except Exception:
            pass

    def close(self) -> None:
        if self._closed:
            return
        if self._t0_stage is not None and self._name and self.enabled:
            dt = time.time() - self._t0_stage
            self._write(f"✗ {self._name} interrupted after {_fmt_dur(dt)}\n")
        self._bar.close()
        self._closed = True

    def __enter__(self) -> "StageProgress":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False
