"""Platform-independent human-readable run transcript (the ``run.log`` tee).

This mirrors everything printed during a run to both the interactive console and
``<run-dir>/run.log`` so an operator can follow a live run and diagnose
warnings, stalls, gates, and crashes without shell-specific redirection
(``Tee-Object``/``tee``/``nohup``).  It is the human transcript only; the
structured ``training_log.jsonl`` and the manifest are written independently and
are unaffected by disabling it.

Scope and threading: the tee is installed in the **orchestrator (parent)
process** and serializes its own threads with a lock so worker progress cannot
interleave bytes.  It deliberately does not attempt any cross-process file
locking -- self-play/gate workers run in separate spawned processes, return
their results, and never write ``run.log`` directly (a parent-held lock could
not serialize them anyway).

Failure policy: logging must never swallow a training failure.  On any exception
the tee records a termination block with the traceback and then re-raises,
preserving the original traceback and exit code.  If the log file cannot be
opened it warns once to the console and continues with console-only output.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import threading
import traceback
from typing import Any, TextIO

_RULE = "=" * 60


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class _TeeStream:
    """A text stream that writes to the console and (optionally) a file."""

    def __init__(self, console: TextIO, file: TextIO | None, lock: threading.Lock):
        self._console = console
        self._file = file
        self._lock = lock

    def write(self, data: str) -> int:
        with self._lock:
            self._console.write(data)
            self._console.flush()
            if self._file is not None:
                self._file.write(data)
                # Flush every write so a crash loses at most an incomplete line.
                self._file.flush()
        return len(data)

    def flush(self) -> None:
        with self._lock:
            self._console.flush()
            if self._file is not None:
                self._file.flush()

    def isatty(self) -> bool:
        return getattr(self._console, "isatty", lambda: False)()

    def writable(self) -> bool:
        return True


class RunLog:
    """Context manager that tees stdout/stderr to a run transcript file.

    Parameters
    ----------
    path:
        Destination transcript path, or ``None`` to disable the file entirely.
    enabled:
        When ``False`` the transcript file is not written (console only); the
        structured JSONL/manifest persistence is unaffected.
    header:
        Ordered key/value fields written under the startup header.
    """

    def __init__(
        self,
        path: str | Path | None,
        *,
        enabled: bool = True,
        header: dict[str, Any] | None = None,
    ):
        self.path = Path(path) if path is not None else None
        self.enabled = bool(enabled) and self.path is not None
        self.header_fields = dict(header or {})
        self.completion_fields: dict[str, Any] = {}
        self._file: TextIO | None = None
        self._saved: tuple[TextIO, TextIO] | None = None

    def __enter__(self) -> "RunLog":
        import sys

        if not self.enabled:
            # A true no-op: leave the console untouched so embedding/tests see
            # only the ordinary console output.
            return self
        console_out, console_err = sys.stdout, sys.stderr
        self._file = None
        if self.enabled and self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                # newline="\n": never translate to CRLF, so the transcript is
                # byte-identical on Windows and Linux.
                self._file = open(self.path, "a", encoding="utf-8", newline="\n")
            except OSError as exc:
                console_err.write(
                    f"WARNING: could not open run log {self.path}: {exc}; "
                    "continuing with console output only\n"
                )
                console_err.flush()
                self._file = None
        lock = threading.Lock()
        self._saved = (console_out, console_err)
        sys.stdout = _TeeStream(console_out, self._file, lock)
        sys.stderr = _TeeStream(console_err, self._file, lock)
        self._emit_header()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        import sys

        if not self.enabled:
            return False
        try:
            self._emit_footer(exc_type, exc, tb)
        finally:
            if self._saved is not None:
                sys.stdout, sys.stderr = self._saved
            if self._file is not None:
                self._file.flush()
                self._file.close()
                self._file = None
        return False  # never suppress a training failure

    # -- narrative ----------------------------------------------------------

    @staticmethod
    def _emit(text: str = "") -> None:
        print(text)

    def _emit_header(self) -> None:
        self._emit(_RULE)
        self._emit(f"Run invocation started: {_utc_now()}")
        for key, value in self.header_fields.items():
            self._emit(f"{key}: {value}")
        self._emit(_RULE)

    def _emit_footer(self, exc_type, exc, tb) -> None:
        self._emit(_RULE)
        if exc_type is not None:
            label = (
                "Run interrupted"
                if issubclass(exc_type, KeyboardInterrupt)
                else "Run failed"
            )
            self._emit(f"{label} ({exc_type.__name__}): {_utc_now()}")
            self._emit(
                "".join(traceback.format_exception(exc_type, exc, tb)).rstrip()
            )
        else:
            self._emit(f"Run completed: {_utc_now()}")
            for key, value in self.completion_fields.items():
                self._emit(f"{key}: {value}")
        self._emit(_RULE)
