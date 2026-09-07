"""Workstream 2: relational message passing over the printed tableau graph.

The graph is worth testing in three layers, because three different things
could be wrong and only the first is visible in a loss curve:

* the EDGES -- that `covers` really means covers, that the transitive edges
  reach as far as the layout does, and that the relation of j to i is the
  mirror of the relation of i to j;
* the PLUMBING -- that tokens land on the node their slot names, that absent
  slots send nothing, and that a message actually crosses an edge (a module
  that silently did nothing would pass every equivalence test in the file);
* the GATE -- that `graph_alpha = 0` is exactly inert, and that the parameters
  behind it are not zero-initialized, because a zeroed LayerNorm is a
  permanently dead module rather than a softly started one.
"""

import random

import pytest

torch = pytest.importorskip("torch")

from games.seven_wonders_duel.buffer import GameRecorder
from games.seven_wonders_duel.codec import legal_action_indices
from games.seven_wonders_duel.data import TABLEAU_LAYOUTS, covering_slots
from games.seven_wonders_duel.dataset import TOKEN_TYPES, collate, examples_from_record
from games.seven_wonders_duel.encoder import TokenType
from games.seven_wonders_duel.game import Phase
from games.seven_wonders_duel.net import SWDNet, TableauGraph
from games.seven_wonders_duel.slot_identity import (
    AGE_SLOT_IDS,
    MAX_COVER_DISTANCE,
    MAX_SLOTS_PER_AGE,
    RELATION_IDS,
    RELATION_NAMES,
    relation_matrix,
)
from games.seven_wonders_duel.train import migrate_state_dict

_TABLEAU = TOKEN_TYPES.index(TokenType.TABLEAU)


@pytest.fixture(scope="module")
def examples():
    recorder = GameRecorder(11, agents={"p0": "random", "p1": "random"})
    rng = random.Random(7)
    while recorder.game.phase is not Phase.COMPLETE:
        recorder.play(rng.choice(legal_action_indices(recorder.game)))
    return examples_from_record(recorder.finish())


@pytest.fixture(scope="module")
def batch(examples):
    return collate(examples)


def _model(**kwargs):
    torch.manual_seed(0)
    return SWDNet(d_model=32, layers=2, heads=4, **kwargs)


# --- the edges --------------------------------------------------------------


@pytest.mark.parametrize("age", sorted(TABLEAU_LAYOUTS))
def test_direct_edges_are_the_layout_cover_relation(age):
    """Checked against `data.covering_slots`, the engine's own definition."""

    layout = TABLEAU_LAYOUTS[age]
    matrix = relation_matrix(age)
    for index, slot in enumerate(layout):
        coverers = {layout.index(other) for other in covering_slots(layout, slot)}
        named = {
            j
            for j in range(len(layout))
            if RELATION_NAMES[matrix[index][j]] == "covered_by"
        }
        assert named == coverers, f"age {age} slot {slot}"


@pytest.mark.parametrize("age", sorted(TABLEAU_LAYOUTS))
def test_the_relation_is_mirrored(age):
    matrix = relation_matrix(age)
    size = len(TABLEAU_LAYOUTS[age])
    for i in range(size):
        for j in range(size):
            there = RELATION_NAMES[matrix[i][j]]
            back = RELATION_NAMES[matrix[j][i]]
            if there == "covers":
                assert back == "covered_by"
            elif there.startswith("descendant_"):
                assert back == there.replace("descendant", "ancestor")
            elif there in ("sibling", "none", "self"):
                assert back == there


def test_transitive_edges_span_the_deepest_age():
    """The plan says distance 2-5; Age III is seven rows deep.

    Stopping at 5 would declare the apex and the bottom row structurally
    unrelated, which is the one pair the transitive edges exist for.
    """

    assert MAX_COVER_DISTANCE == 6
    deepest = relation_matrix(3)
    names = {RELATION_NAMES[value] for row in deepest for value in row}
    assert "descendant_6" in names and "ancestor_6" in names


def test_a_distance_edge_means_a_real_chain():
    """Row difference alone is not the relation: the chain has to exist.

    Age III's row 3 holds only two slots, so plenty of pairs three rows apart
    are connected by nothing at all.
    """

    age, layout = 3, TABLEAU_LAYOUTS[3]
    matrix = relation_matrix(age)
    unrelated = [
        (i, j)
        for i in range(len(layout))
        for j in range(len(layout))
        if RELATION_NAMES[matrix[i][j]] == "none"
        and layout[i].row != layout[j].row
    ]
    assert unrelated, "a sparse Age should leave some cross-row pairs unrelated"


# --- the plumbing -----------------------------------------------------------


def test_tokens_land_on_the_node_their_slot_names(batch):
    """The scatter/gather is identity when the layers are.

    Runs the module with its layers replaced by a pass-through, so anything
    that survives is the addressing rather than the message passing.
    """

    model = _model(graph_module=True, graph_alpha=1.0)
    graph = model.graph
    graph.layers = torch.nn.ModuleList()  # no layers: update is exactly zero
    tokens = model.embedder(batch)
    out = graph(
        tokens, model.embedder.slot_ids(batch), model.embedder.row_ages(batch)
    )
    assert torch.equal(out, tokens)


def test_a_message_crosses_a_cover_edge_and_nothing_else():
    """One node carries a signal; only its structural neighbours may see it.

    Built by hand rather than from a game, because the point is to know
    exactly which nodes are connected. A module that did nothing at all would
    pass every other test here, so this is the one that proves it works.
    """

    width = 8
    graph = TableauGraph(width, layers=1, bases=4, alpha=1.0)
    torch.manual_seed(0)
    # A single message-passing layer with an identity-ish self term removed, so
    # a node's new value depends only on what its neighbours sent.
    layer = graph.layers[0]
    torch.nn.init.zeros_(layer.self_transform.weight)
    torch.nn.init.zeros_(layer.self_transform.bias)

    age, layout = 1, TABLEAU_LAYOUTS[1]
    source = 0  # (row 0, x 5)
    nodes = torch.zeros(1, MAX_SLOTS_PER_AGE, width)
    # Not a constant vector: the layer is pre-LayerNorm, and LayerNorm maps any
    # constant to exactly zero, so a vector of ones would send no message at
    # all and the test would pass on a module that does nothing.
    nodes[0, source] = torch.linspace(-1.0, 1.0, width)
    present = torch.ones(1, MAX_SLOTS_PER_AGE)
    relations = graph.relation_planes[torch.tensor([age])]

    updated = layer(nodes, relations, present)
    moved = (updated - nodes).abs().sum(-1)[0]

    matrix = relation_matrix(age)
    # `none` is a learned relation type too -- the plan lists "no structural
    # relation" as an edge -- so every node can hear every other. What must
    # differ is HOW: a coverer's message and a stranger's cannot be the same.
    # `matrix[listener][source]` is what the source is TO the listener, which
    # is the label the aggregation uses: a listener that covers the source
    # reads it as `covers`.
    coverers = [j for j in range(len(layout))
                if RELATION_NAMES[matrix[j][source]] == "covers"]
    strangers = [j for j in range(len(layout))
                 if RELATION_NAMES[matrix[j][source]] == "none"]
    assert coverers and strangers
    covered_signal = moved[coverers].mean()
    stranger_signal = moved[strangers].mean()
    assert not torch.isclose(covered_signal, stranger_signal, atol=1e-6), (
        "a cover edge and a non-edge produced the same message, so the "
        "relation type is not reaching the transform"
    )


def test_absent_slots_send_nothing():
    """A taken card is not a silent zero-vector neighbour; it is not there."""

    width = 8
    graph = TableauGraph(width, layers=1, bases=4, alpha=1.0)
    layer = graph.layers[0]
    torch.manual_seed(1)
    nodes = torch.randn(2, MAX_SLOTS_PER_AGE, width)
    relations = graph.relation_planes[torch.tensor([1, 1])]

    full = torch.ones(2, MAX_SLOTS_PER_AGE)
    partial = full.clone()
    partial[:, 5:] = 0.0

    # With slot 5+ absent, the surviving nodes must not depend on their values.
    a = layer(nodes, relations, partial)
    other = nodes.clone()
    other[:, 5:] = torch.randn_like(other[:, 5:])
    b = layer(other, relations, partial)
    assert torch.allclose(a[:, :5], b[:, :5], atol=1e-6)
    assert not torch.allclose(layer(nodes, relations, full)[:, :5], a[:, :5])


def test_non_tableau_tokens_pass_through_untouched(batch):
    model = _model(graph_module=True, graph_alpha=1.0)
    tokens = model.embedder(batch)
    slot_ids = model.embedder.slot_ids(batch)
    out = model.graph(tokens, slot_ids, model.embedder.row_ages(batch))
    untouched = slot_ids == 0
    assert untouched.any()
    assert torch.equal(out[untouched], tokens[untouched])
    assert not torch.equal(out[~untouched], tokens[~untouched])


# --- the gate ---------------------------------------------------------------


def test_a_zero_gate_is_exactly_inert(batch):
    plain = _model()
    plain.eval()
    with torch.no_grad():
        before = plain(batch)

    gated = _model(graph_module=True, graph_alpha=0.0)
    report = migrate_state_dict(plain.state_dict(), gated)
    assert not report["zeroed"], report["zeroed"]
    gated.eval()
    with torch.no_grad():
        after = gated(batch)
    for key, value in before.items():
        assert torch.equal(value, after[key]), key


def test_the_graph_parameters_are_not_zero_initialized(batch):
    """Zeroing this module would be permanent, not soft.

    A zeroed LayerNorm emits zeros whatever it is fed, so every activation
    downstream of it -- and every gradient that would revive them -- is zero
    too. Neutrality has to come from the gate, which lives outside the state
    dict, and the migration has to leave the weights alone.
    """

    plain = _model()
    gated = _model(graph_module=True, graph_alpha=1e-3)
    report = migrate_state_dict(plain.state_dict(), gated)
    assert all(key.startswith("graph.") for key in report["initialized"])
    assert gated.graph.layers[0].norm.weight.abs().sum() > 0
    assert gated.graph.layers[0].basis.abs().sum() > 0


def test_the_module_trains_at_the_default_gate(batch):
    model = _model(graph_module=True)
    assert model.graph_alpha == pytest.approx(1e-3)
    model.train()
    model(batch)["policy"].square().mean().backward()
    for name, parameter in model.graph.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.abs().sum() > 0, name


def test_w1_and_w2_are_independently_switchable(batch):
    """The reason they can share one training run.

    W2 orders its nodes by W1's slot identity but never reads W1's table, so
    each can be turned off without the other.
    """

    graph_only = _model(graph_module=True)
    assert graph_only.embedder.slot is None
    assert graph_only.embedder.slot_index
    graph_only.eval()
    with torch.no_grad():
        graph_only(batch)

    slots_only = _model(slot_embedding=True)
    assert slots_only.graph is None
    with torch.no_grad():
        slots_only.eval()(batch)


def test_the_fused_inference_path_agrees(batch):
    model = _model(graph_module=True, slot_embedding=True, graph_alpha=0.5)
    with torch.no_grad():
        torch.nn.init.normal_(model.embedder.slot.weight)
        model.embedder.slot.weight[0].zero_()
    model.eval()
    with torch.no_grad():
        loop = model(batch)
        model.embedder.fuse()
        fused = model(batch)
    for key, value in loop.items():
        assert torch.allclose(value, fused[key], atol=1e-5), key


def test_the_new_arms_survive_bf16(examples):
    """The lesson W5a taught, applied before it can be taught again.

    The W5a scorer crashed at its first forward under `--precision bf16` -- the
    setting every cloud run uses -- because autocast handed it bf16 while
    `policy` stayed fp32 and `scatter_add_` requires the two to match. Nothing
    caught it because the suite runs fp32. W1 and W2 both do index arithmetic
    on autocast tensors, which is the same class of hazard, so they are checked
    here rather than in a rented box's first minute.
    """

    _batch = collate(examples, contextual_actions=True)
    model = SWDNet(
        d_model=32,
        layers=2,
        heads=4,
        slot_embedding=True,
        graph_module=True,
        action_residual=True,
    )
    model.train()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = model(_batch)
        out["policy"].float().square().mean().backward()
    assert model.embedder.slot.weight.grad.abs().sum() > 0
    assert model.graph.layers[0].basis.grad.abs().sum() > 0


# --- what the checkpoint boundary must not do quietly ------------------------


def _graph_checkpoint(tmp_path, alpha, name="graph.pt"):
    from games.seven_wonders_duel.train import make_checkpoint

    model = _model(graph_module=True, graph_alpha=alpha)
    checkpoint = make_checkpoint(model, {"d_model": 32, "layers": 2, "heads": 4})
    path = tmp_path / name
    torch.save(checkpoint, path)
    return model, path, checkpoint


def _args(extra, stored=None):
    """Resolve the trainer's architecture flags without entering the trainer.

    Stops at flag resolution deliberately: the offline epoch trainer has a
    separate pre-existing launch defect, and this is a test about flags.
    """

    from games.seven_wonders_duel import train

    args = train.build_arg_parser().parse_args(["--buffer", "unused.jsonl", *extra])
    train.resolve_graph_args(args, stored)
    return args


def test_a_warm_start_inherits_the_gate_it_does_not_override():
    """The gate lives outside the state dict.

    Loading every weight successfully does not restore it, so an omitted
    `--graph-alpha` used to replace a checkpoint's saved value with the parser
    default -- turning a saved zero-gate ablation ON before the first step, in
    a run whose operator asked for no architecture change at all.
    """

    stored = {
        "graph_module": True,
        "graph_layers": 2,
        "graph_bases": 4,
        "graph_alpha": 0.5,
    }
    assert _args([], stored).graph_alpha == pytest.approx(0.5)
    # An explicit override still wins, INCLUDING zero, which is the ablation.
    assert _args(["--graph-alpha", "0"], stored).graph_alpha == 0.0
    assert _args(["--graph-alpha", "0.25"], stored).graph_alpha == pytest.approx(0.25)

    ablation = dict(stored, graph_alpha=0.0)
    assert _args([], ablation).graph_alpha == 0.0, (
        "a saved zero-gate ablation must not come back active"
    )


def test_a_fresh_graph_gets_the_default_gate():
    args = _args(["--graph-module"])
    assert args.graph_alpha == pytest.approx(1e-3)
    assert (args.graph_layers, args.graph_bases) == (2, 4)


def test_a_conflicting_shape_is_refused_rather_than_resolved():
    """Obeying the flag would fail to load; ignoring it would run another arm."""

    stored = {
        "graph_module": True,
        "graph_layers": 2,
        "graph_bases": 4,
        "graph_alpha": 0.5,
    }
    with pytest.raises(SystemExit, match="graph-layers"):
        _args(["--graph-layers", "3"], stored)
    with pytest.raises(SystemExit, match="graph-bases"):
        _args(["--graph-bases", "8"], stored)
    # Agreeing with the checkpoint is not a conflict.
    assert _args(["--graph-layers", "2"], stored).graph_layers == 2


def test_an_incomplete_graph_checkpoint_is_refused_by_the_evaluator(tmp_path):
    """Serving is not the boundary where a module gets added.

    `load_evaluator` rebuilds from the checkpoint's OWN config, so a parameter
    it has to invent is never a deliberate new branch -- it means the file does
    not carry the weights its config declares. Randomly initialized graph
    weights are the dangerous case: they serve plausible seed-dependent numbers
    rather than obvious nonsense.
    """

    from games.seven_wonders_duel.phase_e import load_evaluator

    _, path, checkpoint = _graph_checkpoint(tmp_path, 0.5)
    load_evaluator(path, "cpu", migrate=True)  # complete: fine

    del checkpoint["model_state"]["graph.layers.0.basis"]
    broken = tmp_path / "broken.pt"
    torch.save(checkpoint, broken)
    with pytest.raises(ValueError, match="does not carry"):
        load_evaluator(broken, "cpu", migrate=True)


def test_the_search_gain_probe_rebuilds_the_new_architecture(tmp_path):
    """It named every switch up to `action_residual` and would have stopped at
    `load_state_dict` on any W1/W2 checkpoint, before playing a game."""

    from games.seven_wonders_duel.phase_d import _model_from_spec, ModelAgentSpec

    model = _model(slot_embedding=True, graph_module=True, graph_alpha=0.25)
    spec = ModelAgentSpec(
        name="probe",
        model_state=model.state_dict(),
        d_model=32,
        layers=2,
        heads=4,
        slot_embedding=True,
        graph_module=True,
        graph_layers=2,
        graph_bases=4,
        graph_alpha=0.25,
        sims=1,
        mode="closed",
        top_k=2,
    )
    rebuilt = _model_from_spec(spec)
    assert rebuilt.graph_alpha == pytest.approx(0.25)
    assert rebuilt.embedder.slot is not None
