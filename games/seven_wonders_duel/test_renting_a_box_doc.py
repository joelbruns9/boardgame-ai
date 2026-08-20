"""`RENTING_A_BOX.md` names concrete helpers, scripts and flags.

A playbook that cites a function which has since been renamed is worse than no
playbook: it is read under time pressure, on a rented box, by someone who will
believe it. This keeps its citations true, and nothing more -- prose is not
testable and is not tested here.
"""

from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DOC = REPO_ROOT / "RENTING_A_BOX.md"


@pytest.fixture(scope="module")
def doc() -> str:
    return DOC.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "helper", ["common::self_checksum", "common::reexec_if_updated"]
)
def test_the_named_helpers_exist(doc, helper):
    """The self-update guard is the doc's most-repeated lesson; if the helper is
    renamed, the fix it points at stops being findable."""

    assert helper in doc
    common = (REPO_ROOT / "setup_cloud_common.sh").read_text(encoding="utf-8")
    assert f"{helper}()" in common, f"{helper} is cited by the playbook but gone"


def test_the_isolation_recipe_matches_what_the_scripts_do(doc):
    """Section 5's recipe is the one leaf_batch_test.sh and sweep_7wd.sh run."""

    assert "maturin build --release" in doc
    assert "--target" in doc
    for script in ("leaf_batch_test.sh", "sweep_7wd.sh"):
        text = (REPO_ROOT / script).read_text(encoding="utf-8")
        code = "\n".join(
            line for line in text.splitlines() if not line.lstrip().startswith("#")
        )
        assert "maturin build" in code, f"{script} no longer builds a wheel"
        assert "maturin develop" not in code, f"{script} installs in place again"


def test_the_remote_ref_checkout_is_what_the_scripts_use(doc):
    """Section 3.1. Fetch plus checkout is not pull -- the doc says so, and the
    scripts have to actually do it."""

    assert "checkout --quiet --detach origin/" in doc.replace(
        "git checkout --quiet --detach origin/main", "checkout --quiet --detach origin/"
    ) or "--detach origin" in doc
    for script in ("leaf_batch_test.sh", "sweep_7wd.sh"):
        text = (REPO_ROOT / script).read_text(encoding="utf-8")
        assert "--detach" in text and "origin/" in text, script


def test_every_script_the_playbook_relies_on_exists():
    for script in ("setup_cloud_common.sh", "setup_cloud_7wd.sh", "setup_cloud.sh",
                   "sweep_7wd.sh", "leaf_batch_test.sh"):
        assert (REPO_ROOT / script).is_file(), f"{script} is missing"


def test_version_markers_are_present_where_the_doc_promises_them(doc):
    """Section 1.3 exists because a CDN-cached copy cost a debugging round trip."""

    assert "version marker" in doc.lower()
    for script, marker in (
        ("sweep_7wd.sh", "SWEEP_SCRIPT_VERSION"),
        ("leaf_batch_test.sh", "LEAF_BATCH_SCRIPT_VERSION"),
    ):
        text = (REPO_ROOT / script).read_text(encoding="utf-8")
        assert marker in text, f"{script} lost its version marker"
