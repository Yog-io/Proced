"""Progress bars for the independent QA suite (Step 0 audit + Step 3 modules).

Deliberately self-contained: ``qa_verification`` must never import ``proced``
(boundary rule), so this mirrors ``proced.progress`` in trimmed form:

* TTY → live ``\\r`` bar; pipe/log → throttled ``[progress]`` lines
* ``PROGRESS=0`` → silent
"""

from __future__ import annotations

import os
import sys
import time
from typing import IO, Optional

_TTY_INTERVAL = 0.35
_LOG_INTERVAL = 15.0
_LOG_STEP = 0.05
_WIDTH = 24
_DISABLE_VALUES = {"0", "off", "false", "no"}


def progress_enabled() -> bool:
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
    return ProgressBar(total, desc, unit, **kwargs)


class ProgressBar:
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
        self._closed = False
        self._last_render = 0.0
        self._last_len = 0
        self._last_log_n = -1
        self._last_log_t = self.t0
        if self.enabled:
            self._render(force=True)

    def update(self, n: int = 1) -> None:
        if not self.enabled or self._closed:
            return
        self.n += n
        self._render()

    def set_desc(self, desc: str) -> None:
        self.desc = desc
        if self.enabled and not self._closed and self.is_tty:
            self._render(force=True)

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

    def __enter__(self) -> "ProgressBar":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def _line(self) -> str:
        elapsed = time.time() - self.t0
        head = (f"{self.desc}: {self.n}/{self.total}" if self.total
                else f"{self.desc}: {self.n} {self.unit}")
        if not self.total:
            return f"{head} ({_fmt_dur(elapsed)})"
        pct = 100.0 * self.n / self.total
        filled = max(0, min(self.width, int(round(self.width * self.n / self.total))))
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
