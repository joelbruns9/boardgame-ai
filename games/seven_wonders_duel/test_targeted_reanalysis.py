"""G8.2: targeted reanalysis -- selection, the overlay file, and its use in
derivation."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

swr = pytest.importorskip("seven_wonders_rust")
torch = pytest.importorskip("torch")

from . import phase_e as pe
from . import targeted_reanalysis as tr
from .buffer import append_records
from .dataset import derive_records_rust, examples_from_record
from .tactical_suite import sealed


@pytest.fixture(scope="module")
def records():
    return pe.fresh_bot_records(16, seed=8642)


def test_selection_follows_its_classes_caps_and_the_g0_reservation(records):
    rng = random.Random(0)
    found = []
    for record in records:
        targets = tr.select_record(record, rng, window=4, per_game_cap=6, cheap_rate=1.0)
        assert len(targets) <= 6
        assert len({t.move for t in targets}) == len(targets)
        if sealed(record.iteration, record.seed):
            assert targets == []
        found.extend(targets)
    reasons = {t.reason for t in found}
    assert reasons <= set(tr.REASONS)
    assert "pre_decisive" in reasons, reasons


def test_selection_is_random_within_reason_and_reproducible_per_game(records):
    def pick(record, seed):
        return tr.select_record(record, tr._record_rng(seed, record), window=4,
                                per_game_cap=2, cheap_rate=1.0)

    differs_from_earliest = False
    for record in records:
        assert pick(record, 0) == pick(record, 0)  # resume picks the same moves
        uncapped = tr.select_record(record, random.Random(0), window=4,
                                    per_game_cap=999, cheap_rate=1.0)
        pre = sorted(t.move for t in uncapped if t.reason == "pre_decisive")
        chosen = [t.move for t in pick(record, 0) if t.reason == "pre_decisive"]
        if len(pre) > 2 and sorted(chosen) != pre[:2]:
            differs_from_earliest = True
    assert differs_from_earliest


@pytest.fixture(scope="module")
def overlay_file(tmp_path_factory, records):
    from .train import build_model, make_checkpoint

    root = tmp_path_factory.mktemp("g82")
    checkpoint = root / "tiny.pt"
    torch.save(make_checkpoint(build_model("transformer", 32, 1),
                               {"model": "transformer", "d_model": 32, "layers": 1, "heads": 2}),
               checkpoint)
    buffer = root / "iter_0001.jsonl"
    append_records(buffer, records)
    out = root / "overlay.jsonl"
    summary = tr.run([buffer], out, checkpoint=str(checkpoint), sims=16, device="cpu",
                     precision="fp32", batch_positions=32, max_positions=40,
                     log=lambda *_: None)
    assert sum(summary["positions"].values()) >= 1
    return out, checkpoint, buffer


def test_the_overlay_is_well_formed_and_resumes(overlay_file):
    out, checkpoint, buffer = overlay_file
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["kind"] == "header"
    assert rows[0]["schema"] == tr.SCHEMA >= 2
    for row in rows[1:]:
        assert row["reason"] in tr.REASONS
        assert "root_outlook" in row  # None: the tiny test net has no W4 head
        assert sum(row["policy"].values()) == pytest.approx(1.0, abs=1e-6)
        assert -1.0 <= row["root_value"] <= 1.0
    # A second run skips everything already written.
    before = len(rows)
    tr.run([buffer], out, checkpoint=str(checkpoint), sims=16, device="cpu",
           precision="fp32", batch_positions=32, max_positions=40, log=lambda *_: None)
    keys = [(r["iteration"], r["seed"], r["move"]) for r in
            map(json.loads, out.read_text(encoding="utf-8").splitlines()[1:])]
    assert len(keys) == len(set(keys))
    assert len(keys) + 1 >= before


def test_derivation_applies_the_overlay_identically_in_both_backends(overlay_file, records):
    out, _checkpoint, _buffer = overlay_file
    overlay = tr.load_overlay(out)
    touched = [r for r in records if (r.iteration, r.seed) in overlay]
    assert touched
    rust = derive_records_rust(touched, reanalysis_overlay=overlay, tactic_labels=True,
                               batch_games=4)
    applied = 0
    for record, (rust_rows, _stats) in zip(touched, rust):
        python_rows = examples_from_record(record, reanalysis_overlay=overlay,
                                           tactic_labels=True)
        assert [e.move_index for e in python_rows] == [e.move_index for e in rust_rows]
        entries = overlay[(record.iteration, record.seed)]
        for a, b in zip(python_rows, rust_rows):
            assert a.reanalysed == b.reanalysed
            assert np.allclose(a.policy_target, b.policy_target)
            assert a.root_value == b.root_value
            if a.move_index in entries:
                applied += 1
                assert a.reanalysed and a.has_policy
                assert a.root_value == pytest.approx(entries[a.move_index]["root_value"])
        # Every re-searched move is emitted, cheap or not.
        assert set(entries) <= {e.move_index for e in rust_rows}
    assert applied == sum(len(overlay[(r.iteration, r.seed)]) for r in touched)


def test_w4_trains_toward_the_re_searched_outlook(records):
    """Review of 8014a6c, #3: W4's outcome target is built from the outlook,
    not the scalar, so a re-searched row must carry the new search's outlook
    -- and a row from a biased (lambda > 0) search must not refuse it."""

    from dataclasses import replace

    from .dataset import apply_reanalysis, collate, usable_root_outlook
    from .train import value_targets

    example = replace(examples_from_record(records[0])[6], search_lambda=0.4)
    policy = {int(a): 1.0 / len(example.legal) for a in example.legal}

    def outcome(outlook):
        row = apply_reanalysis(example, {"policy": policy, "root_value": 0.0,
                                         "root_outlook": outlook})
        assert row.search_lambda == 0.0
        assert usable_root_outlook(row) is not None
        targets = value_targets(collate([row]), outlook_bootstrap=0.5,
                                hierarchical=True)
        return targets["hier_outcome"][0]

    winning = outcome([0.3, 0.3, 0.3, 0.0, 0.0, 0.1, 0.0])
    losing = outcome([0.0, 0.1, 0.0, 0.3, 0.3, 0.3, 0.0])
    assert float(winning[0] - losing[0]) == pytest.approx(0.5 * 0.8, abs=1e-6)


def test_an_overlay_without_outlooks_is_refused(tmp_path):
    old = tmp_path / "old.jsonl"
    old.write_text(json.dumps({"kind": "header", "schema": 1}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema 1"):
        tr.load_overlay(old)
