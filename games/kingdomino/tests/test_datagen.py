"""NNUE data-generation harness gates.

Validates the replayable-source contract and the enhanced-Option-A plumbing on a
small batch: replay reproduces every outcome exactly (the source is lossless with
NO encoded features stored -> avoids the run10 encoder-lock), whole games are
split with no seed leakage, generation is deterministic per seed, and the stored
trajectory recovers a consistent (possibly non-alternating) actor sequence.
"""
import json

import pytest

pytest.importorskip("kingdomino_rust")
import kingdomino_rust as kr

from games.kingdomino.nnue import datagen as dg


@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    out = tmp_path_factory.mktemp("nnue_pilot")
    # depth-2 only keeps this fast; these gates test replay/split/determinism/actor
    # plumbing, not the depth-diversity mix (a generation feature, not a contract).
    cfg = dg.GenConfig(depth_choices=(2,), depth_weights=(1.0,))
    man = dg.generate(60, str(out), cfg, workers=1, seed_start=0, verify=True)
    recs = {}
    for split in ("train", "val", "test"):
        with open(out / f"{split}.jsonl") as f:
            recs[split] = [json.loads(l) for l in f]
    return {"out": out, "manifest": man, "recs": recs, "cfg": cfg}


def test_replay_reproduces_every_outcome(batch):
    # generate() already verified during writing; assert it and re-check independently.
    assert batch["manifest"]["verify_failures"] == 0
    for split_recs in batch["recs"].values():
        for rec in split_recs:
            s0, s1, oc = dg.replay(rec)
            assert [s0, s1] == rec["final_scores"], f"seed {rec['seed']}: replay scores differ"
            assert oc == rec["outcome_p0"], f"seed {rec['seed']}: replay outcome differs"


def test_no_stored_features_only_source(batch):
    """The record must be replayable SOURCE, not encoded features (run10 trap)."""
    rec = batch["recs"]["train"][0]
    assert set(rec) >= {"seed", "start_player", "deck", "current_row", "actions",
                        "final_scores", "outcome_p0", "provenance", "catalog_hash"}
    # no board tensors / flat vectors / feature arrays smuggled in
    assert not any(k in rec for k in ("flat", "my_board", "opp_board", "features", "sparse_idx"))
    assert len(rec["deck"]) + len(rec["current_row"]) == 48  # full catalog is the source


def test_whole_game_split_no_seed_leakage(batch):
    seeds = {split: {r["seed"] for r in recs} for split, recs in batch["recs"].items()}
    assert seeds["train"].isdisjoint(seeds["val"])
    assert seeds["train"].isdisjoint(seeds["test"])
    assert seeds["val"].isdisjoint(seeds["test"])
    total = sum(len(s) for s in seeds.values())
    assert total == 60
    # split is a pure function of the seed (reproducible, position-leak-free)
    for split, recs in batch["recs"].items():
        for r in recs:
            assert dg._split_of(r["seed"], batch["cfg"]) == split


def test_generation_deterministic(batch):
    """Same seed -> byte-identical record (reproducibility / provenance integrity)."""
    rec = batch["recs"]["train"][0]
    again = dg.play_one_game(rec["seed"], batch["cfg"])
    assert again["actions"] == rec["actions"]
    assert again["final_scores"] == rec["final_scores"]
    assert again["outcome_p0"] == rec["outcome_p0"]


def test_actor_recovery_and_nonalternating(batch):
    """Replaying the source recovers a consistent actor at each decision, and the
    dataset contains at least one non-alternating (consecutive same-actor) turn --
    the property that made run10 labels fragile."""
    GAME_OVER = dg.GAME_OVER
    saw_consecutive_same_actor = False
    both_actors_seen = False
    for rec in batch["recs"]["train"][:20]:
        rs = kr.RustGameState(rec["start_player"], list(rec["deck"]),
                              list(rec["current_row"]), rec["harmony"], rec["middle_kingdom"])
        actors = []
        for sa in rec["actions"]:
            actors.append(int(rs.current_actor()))
            p, pk = dg._deser_action(sa)
            rs = rs.step(p, pk)
        assert rs.phase == GAME_OVER
        if set(actors) == {0, 1}:
            both_actors_seen = True
        if any(actors[i] == actors[i + 1] for i in range(len(actors) - 1)):
            saw_consecutive_same_actor = True
    assert both_actors_seen
    assert saw_consecutive_same_actor, "no consecutive same-actor turn seen (Kingdomino has them)"


def test_manifest_integrity(batch):
    man = batch["manifest"]
    assert man["catalog_hash"] == dg.catalog_hash()
    assert man["engine_version"] == dg.ENGINE_VERSION
    assert man["format_version"] == dg.FORMAT_VERSION
    assert "git_commit" in man and "git_dirty" in man
    assert man["total_positions"] == sum(r["n_positions"]
                                         for recs in batch["recs"].values() for r in recs)


def test_loader_accepts_fresh_buffer(batch):
    recs = dg.load_records(str(batch["out"]), strict=True)
    assert len(recs) == 60
    # provenance stamped onto every record
    assert all("git_commit" in r and "format_version" in r for r in recs)


def test_loader_rejects_stale_buffers(tmp_path, batch):
    good = batch["recs"]["train"][0]
    for field, bad in [("engine_version", 999), ("format_version", 999),
                       ("catalog_hash", "deadbeef")]:
        rec = dict(good)
        rec[field] = bad
        p = tmp_path / f"stale_{field}.jsonl"
        p.write_text(json.dumps(rec) + "\n")
        with pytest.raises(dg.StaleBufferError):
            dg.load_records(str(p), strict=True)


def test_loader_rejects_mixed_rules(tmp_path, batch):
    a = dict(batch["recs"]["train"][0])
    b = dict(batch["recs"]["train"][1])
    b["harmony"] = not b["harmony"]
    p = tmp_path / "mixed_rules.jsonl"
    p.write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n")
    with pytest.raises(dg.StaleBufferError):
        dg.load_records(str(p), strict=True)
