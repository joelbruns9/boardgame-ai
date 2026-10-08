"""Stage 0's CPU visibility helpers against fixture /proc and /sys trees.

Review of d8aa2e3, finding 2: the spinner count came from the HOST topology
(64 spinners for a container pinned to 16 CPUs), the lscpu-less fallback used
the logical count, and the quota reader looked only at the cgroup root.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

BASH = shutil.which("bash")
SCRIPT = Path(__file__).resolve().parents[2] / "setup_cloud_7wd.sh"
pytestmark = pytest.mark.skipif(BASH is None, reason="no bash on PATH")


def _helpers() -> str:
    text = SCRIPT.read_text(encoding="utf-8")
    start = text.index('BOX_PROC="${BOX_PROC:-/proc}"')
    end = text.index("box_spin() {")
    return text[start:end]


def _posix(path: Path) -> str:
    # Git Bash on Windows takes forward slashes; POSIX paths pass unchanged.
    return path.as_posix()


def _call(tmp_path, call, *, lscpu=None, nproc=32):
    proc, sys_ = tmp_path / "proc", tmp_path / "sys"
    proc.mkdir(exist_ok=True)
    sys_.mkdir(exist_ok=True)
    lscpu_cmd = "/nonexistent-lscpu"
    if lscpu is not None:
        out = tmp_path / "lscpu.out"
        out.write_text(lscpu, encoding="utf-8", newline="\n")
        fake = tmp_path / "lscpu"
        fake.write_text(f'#!/usr/bin/env bash\ncat "{_posix(out)}"\n', encoding="utf-8", newline="\n")
        fake.chmod(0o755)
        lscpu_cmd = _posix(fake)
    script = (
        "set -euo pipefail\n"
        f"nproc() {{ echo {nproc}; }}\n"
        f'BOX_PROC="{_posix(proc)}"; BOX_SYS="{_posix(sys_)}"; BOX_LSCPU="{lscpu_cmd}"\n'
        + _helpers()
        + f"\n{call}\n"
    )
    done = subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def _status(tmp_path, allowed):
    (tmp_path / "proc" / "self").mkdir(parents=True, exist_ok=True)
    (tmp_path / "proc" / "self" / "status").write_text(
        f"Name:\tbash\nCpus_allowed_list:\t{allowed}\n", encoding="utf-8", newline="\n"
    )


def _host_lscpu(cores=64, threads=2):
    # CPU i sits on core i % cores (Linux numbers SMT siblings cores apart).
    rows = ["# CPU,Core,Socket"]
    rows += [f"{cpu},{cpu % cores},0" for cpu in range(cores * threads)]
    return "\n".join(rows) + "\n"


def test_a_pinned_container_counts_its_own_cores_not_the_hosts(tmp_path):
    (tmp_path / "proc").mkdir()
    _status(tmp_path, "0-15")
    assert _call(tmp_path, "box_physical_cores", lscpu=_host_lscpu()) == "16 lscpu"


def test_smt_siblings_in_the_allowed_set_are_one_core(tmp_path):
    (tmp_path / "proc").mkdir()
    _status(tmp_path, "0-7,64-71")
    assert _call(tmp_path, "box_physical_cores", lscpu=_host_lscpu()) == "8 lscpu"


def test_without_lscpu_sysfs_topology_counts_physical_cores(tmp_path):
    (tmp_path / "proc").mkdir()
    _status(tmp_path, "0-31")
    for cpu in range(64):
        topo = tmp_path / "sys" / "devices" / "system" / "cpu" / f"cpu{cpu}" / "topology"
        topo.mkdir(parents=True)
        (topo / "core_id").write_text(f"{cpu % 32}\n")
        (topo / "physical_package_id").write_text("0\n")
    # Allowed 0-31 are 32 distinct cores of this host; their siblings 32-63 are not allowed.
    assert _call(tmp_path, "box_physical_cores") == "32 sysfs"


def test_without_any_topology_the_count_assumes_smt(tmp_path):
    # The reviewer's second fixture: 32 logical CPUs, no lscpu. Not 32.
    assert _call(tmp_path, "box_physical_cores", nproc=32) == "16 assumed-smt"


def test_the_quota_on_an_ancestor_cgroup_is_found(tmp_path):
    (tmp_path / "proc" / "self").mkdir(parents=True)
    (tmp_path / "proc" / "self" / "cgroup").write_text("0::/a/b\n", encoding="utf-8")
    leaf = tmp_path / "sys" / "fs" / "cgroup" / "a" / "b"
    leaf.mkdir(parents=True)
    (leaf / "cpu.max").write_text("max 100000\n")
    (leaf.parent / "cpu.max").write_text("800000 100000\n")
    assert _call(tmp_path, "box_quota_cpus") == "8.00"


def test_the_tightest_of_several_quotas_wins(tmp_path):
    (tmp_path / "proc" / "self").mkdir(parents=True)
    (tmp_path / "proc" / "self" / "cgroup").write_text("0::/a\n", encoding="utf-8")
    cg = tmp_path / "sys" / "fs" / "cgroup"
    (cg / "a").mkdir(parents=True)
    (cg / "a" / "cpu.max").write_text("1600000 100000\n")
    (cg / "cpu.max").write_text("384000 100000\n")
    assert _call(tmp_path, "box_quota_cpus") == "3.84"


def test_a_v1_quota_is_read_without_a_namespace(tmp_path):
    # Path from /proc/self/cgroup does not exist under the mount: walk to root.
    (tmp_path / "proc" / "self").mkdir(parents=True)
    (tmp_path / "proc" / "self" / "cgroup").write_text(
        "4:cpu,cpuacct:/docker/abc\n", encoding="utf-8"
    )
    mount = tmp_path / "sys" / "fs" / "cgroup" / "cpu,cpuacct"
    mount.mkdir(parents=True)
    (mount / "cpu.cfs_quota_us").write_text("400000\n")
    (mount / "cpu.cfs_period_us").write_text("100000\n")
    assert _call(tmp_path, "box_quota_cpus") == "4.00"


def test_no_visible_quota_reads_empty(tmp_path):
    assert _call(tmp_path, "box_quota_cpus") == ""
