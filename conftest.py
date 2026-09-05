"""Repo-wide test setup: bounded CPU threads, and a log that survives a stall.

Both exist because of one incident. A full parallel run appeared to hang; it
was actually a test whose input is the BGA game log -- a corpus that grows every
time a game is played -- crawling through 317 exact solves while several other
Python processes fought it for the same cores. Nothing in the run said which
test was unfinished, and nothing bounded how many threads each worker took.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

# --- CPU thread budget -------------------------------------------------------
#
# These must be set before Torch, NumPy or MKL are imported, and a root conftest
# is imported before any test module, so this is the last honest place to do it.
# Without them every xdist worker builds its own intra-op pool sized for the
# WHOLE machine: 12 workers x 12 threads on a 12-core box is 144 runnable
# threads fighting for 12 cores, which Torch's own guidance calls out as a large
# slowdown rather than a small one.

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _thread_budget() -> int:
    """Cores this worker may use: the machine's, split across the workers."""

    cores = os.cpu_count() or 1
    workers = os.environ.get("PYTEST_XDIST_WORKER_COUNT")
    try:
        count = max(1, int(workers)) if workers else 1
    except ValueError:
        count = 1
    return max(1, cores // count)


#: Set once we have taken ownership of the thread variables. xdist workers
#: inherit the controller's environment, so a plain `setdefault` in a worker
#: finds the controller's whole-machine budget already there and leaves it --
#: which is how every worker ends up sized for the whole machine anyway. The
#: flag is inherited too, and tells a worker the value it sees is ours to
#: narrow rather than the caller's to respect.
_OWNED_FLAG = "_PYTEST_THREAD_BUDGET_OWNED"

_BUDGET = _thread_budget()


def _take_thread_budget(budget: int) -> bool:
    """Apply `budget` unless the caller set a budget of their own."""

    ours = os.environ.get(_OWNED_FLAG) == "1"
    if any(name in os.environ for name in _THREAD_VARS) and not ours:
        return False
    for name in _THREAD_VARS:
        os.environ[name] = str(budget)
    os.environ[_OWNED_FLAG] = "1"
    return True


_OWNED = _take_thread_budget(_BUDGET)


def pytest_configure(config):
    # Torch may already be imported (another conftest, a plugin), in which case
    # the environment variables above missed their window and only the runtime
    # call still binds.
    if not _OWNED:
        return
    try:
        import torch
    except Exception:
        return
    try:
        torch.set_num_threads(_BUDGET)
    except Exception:
        pass


# --- per-worker run log ------------------------------------------------------
#
# Written line by line and flushed, so a worker killed by the timeout -- or a
# machine that goes down mid-run -- still leaves a file whose last START without
# a matching END names the test that was in flight. The suite's own output
# cannot do this: `-q` prints a dot only after a test has finished.

_LOG_DIR = Path(os.environ.get("PYTEST_RUN_LOG_DIR", ".pytest_run_logs"))
_log = None


def _worker_id() -> str:
    return os.environ.get("PYTEST_XDIST_WORKER", "main")


def pytest_sessionstart(session):
    global _log
    if os.environ.get("PYTEST_RUN_LOG", "1") == "0":
        return
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        _log = (_LOG_DIR / f"{_worker_id()}.log").open("w", encoding="utf-8")
        _log.write(f"# started {time.strftime('%Y-%m-%d %H:%M:%S')} "
                   f"worker={_worker_id()} threads={_BUDGET}\n")
        _log.flush()
    except OSError:
        # A read-only checkout is not a reason to fail the run.
        _log = None


def pytest_runtest_logstart(nodeid, location):
    if _log is not None:
        _log.write(f"START {time.time():.3f} {nodeid}\n")
        _log.flush()


def pytest_runtest_logfinish(nodeid, location):
    if _log is not None:
        _log.write(f"END   {time.time():.3f} {nodeid}\n")
        _log.flush()


def pytest_sessionfinish(session, exitstatus):
    if _log is not None:
        _log.write(f"# finished exitstatus={exitstatus}\n")
        _log.close()
