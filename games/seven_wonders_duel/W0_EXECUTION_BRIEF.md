# W0.3 + W0.1 execution brief

**For:** a fresh Claude Code session with elevated permissions, executing
unattended.
**Written:** 2026-07-28, by the session that produced
`CLOUD_TRAINING_PLAN.md` and shipped W0.2.
**Repo:** `C:\Users\joeld\projects\boardgame-ai`, branch
**`7wd-w0-2-attention-heads`** (do not create a new branch; commit onto this
one).

This brief is self-contained. You do not need to read the whole plan to
execute, but `CLOUD_TRAINING_PLAN.md` §W0 is the authority if anything here is
ambiguous. Everything numeric below was **measured**, not estimated, unless
labelled otherwise -- do not spend time re-deriving it.

---

## Ground rules

1. **Never rewrite an existing test's expected values to make it pass.** If an
   assertion moves, stop that thread of work, record it in the handoff notes,
   and continue with the other tasks. A precision change that shifts an
   existing expectation is a finding, not a chore.
2. **Commit to `7wd-w0-2-attention-heads`. Never push. Never open a PR.**
3. **`--precision` defaults to `fp32`.** bf16 is opt-in for this work. Whether
   it becomes the default is the user's call after reading the A/B.
4. Do not change model architecture, search parameters, or gate thresholds.
   W0 measures; it does not tune.
5. If you finish early, **stop**. Do not start W0.4 or any other workstream.
6. Report honestly. A negative result (bf16 is slower, wider is not better) is
   a successful outcome of this brief.

## Environment

- Windows 11, PowerShell primary; a Bash tool is also available.
- Python: **`.venv/Scripts/python.exe`** (3.12.10). Always use this, never bare
  `python`.
- GPU: RTX 3070 Laptop, 8 GB, CUDA available, torch 2.12.0+cu126. bf16 is
  supported (Ampere sm_86).
- Host RAM: 16 GB. **This is the binding constraint on W0.1 -- see §W0.1.2.**
- `pytest` is installed in the venv but is *not* in `requirements.txt`.

### Commands you will need (for pre-authorisation)

```
.venv/Scripts/python.exe -m pytest games/seven_wonders_duel/ -q
.venv/Scripts/python.exe -m pytest games/seven_wonders_duel/<file>.py -q
.venv/Scripts/python.exe -m games.seven_wonders_duel.<module> [args]
.venv/Scripts/python.exe <script path>
git status / diff / add / commit / log        (no push, no branch creation)
```

Writes are confined to `games/seven_wonders_duel/`, `games/az_loop/`, and the
session scratchpad. Long runs should go in the background; the full 7WD suite
takes **~7 minutes** (520 tests).

---

# W0.3 -- bf16 as an explicit, per-path precision config

## Why this is not a one-line change

The precision baseline is not uniform, and the plan's earlier wording ("bf16
everywhere") mis-stated it. Audited:

| path | file:line | precision today |
|---|---|---|
| training (`train_loop`) | `train.py:363-376` | **AMP fp16** -- `torch.autocast("cuda", enabled=use_amp)` with **no dtype** defaults to float16 on CUDA, wrapped in a `GradScaler` |
| training (`train_steps`) | `train.py:479-502` | same |
| validation (`evaluate`) | `train.py:96` | **fp32**, outside autocast |
| `Evaluator.evaluate` (advisor, python search) | `inference.py:73` | **fp32**, outside autocast |
| Rust flat adapter (self-play + gates) | `rust_bridge.py:368` | **fp32**, outside autocast |

Self-play generation is 77% of wall clock and gates are 10%; both are fp32
today. **That is where the entire measured speedup lives.** Training is ~2% of
wall clock and already uses AMP, so converting it buys no speed -- do it for
consistency and to drop the `GradScaler`, not for throughput.

## Design

Add a single `precision` value that flows to every model call.

**Source of truth: `Evaluator`.** The Rust adapter calls
`self.evaluator.model(batch)` directly (`rust_bridge.py:368`), bypassing
`Evaluator.evaluate`, so an autocast placed only inside `evaluate` would miss
87% of the work. Give `Evaluator` an autocast-context factory and use it at
**both** call sites:

```python
# inference.py
class Evaluator:
    def __init__(self, model, device="cpu", max_batch=512,
                 fuse_embedder=True, precision="fp32"):
        ...
        self.precision = precision

    def autocast(self):
        """Context for every forward through this evaluator's model.

        bf16 applies on CUDA only: on CPU the autocast path is not faster here
        and the advisor's latency budget does not need it.
        """
        if self.precision == "bf16" and str(self.device).startswith("cuda"):
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()
```

Then:
- `inference.py:73` -- wrap `self.model(batch)` in `with self.evaluator...`
  (it is `self` there).
- `rust_bridge.py:368` -- wrap `outputs = self.evaluator.model(batch)` in
  `with self.evaluator.autocast():`.
- `train.py:363-376` and `479-502` -- pass an explicit dtype
  (`torch.bfloat16` when precision is bf16, `torch.float16` otherwise so the
  existing default is preserved **bit-for-bit**), and set
  `GradScaler(enabled=use_amp and dtype is torch.float16)` -- bf16 needs no
  loss scaling.
- `train.py:96` (`evaluate`) -- wrap the forward so validation matches training.

**Config plumbing:**
- `PhaseDConfig.precision: str = "fp32"` (`phase_d.py`, the dataclass at
  line ~118; it is `slots=True` and **mutable**, which the A/B relies on).
- Validate in `PhaseDConfig.__post_init__`: one of `{"fp32","bf16"}`.
- CLI `--precision`, choices `fp32`/`bf16`, default `fp32`, threaded into the
  config construction near `phase_d.py:2422` (where `d_model`/`layers`/`heads`
  are passed).
- Pass `precision=self.config.precision` at **every** `Evaluator(...)`
  construction in `phase_d.py`. Find them with
  `grep -n "Evaluator(" games/seven_wonders_duel/phase_d.py` -- there are
  several (generation, both gate paths, the bot-anchor path, the process
  worker). Missing one silently leaves that path at fp32, which is exactly the
  bug class this task exists to prevent.
- `phase_e.load_evaluator` -- accept and forward a precision argument,
  defaulting to `fp32` (this is the advisor's path).

**Persistence and resume.** `run_manifest.json` already stores
`dataclasses.asdict(config)` (`games/az_loop/manifest.py:78`), so adding the
field persists it for free. Add a narrow resume check in `phase_d.py`: compare
the stored `config.precision` from the manifest against the current one and
raise a clear error on mismatch. `run_controller._validate_lifecycle_config`
(`run_controller.py:238`) is the model to imitate but do **not** add precision
to `lifecycle_config` -- that is the game-agnostic lifecycle contract, and
precision is training config.

## Tests (new file: `games/seven_wonders_duel/test_precision.py`)

The failure mode is **an autocast that silently does nothing**, so assert the
dtype actually in effect, not that the code ran:

1. `Evaluator(..., precision="bf16")` on CUDA yields `torch.bfloat16`
   activations; on CPU it yields fp32 (skip the CUDA half if unavailable, and
   say so in the handoff -- but CUDA is available on this box).
2. The Rust flat adapter path runs under bf16 when its evaluator is bf16.
   Register a forward hook on a submodule and assert the observed dtype;
   driving it through `rust_flat_batch_adapter` with a small payload is
   sufficient.
3. Default construction is fp32 everywhere (guards the opt-in rule).
4. `PhaseDConfig(precision="f16")` raises.
5. A bf16 resume against an fp32 manifest raises.
6. **Equivalence:** fp32 and bf16 policies agree on argmax for >= 99% of a
   real position sample, and max |dP(win)| < 1e-2. Reference measurement on 512
   real positions through `runs/laptop_training_03/checkpoints/current_best.pt`:
   mean policy KL **8.3e-6**, max |dP| **5.3e-3**, argmax agreement **99.61%**
   (2 flips, both on positions where the net's own top-1/top-2 priors differ by
   0.0016), mean |dP(win)| **1.1e-3**, max **5.3e-3**. Use loose thresholds --
   this test guards against a precision *regression*, not against the 2 flips.

Then run the full suite (`~7 min`). **Expect zero changes** -- precision
defaults to fp32 and the fp16 training path is preserved bit-for-bit. If
anything moves, ground rule 1 applies.

## Measurement

A/B on the real generation path via `f4_phase_d_ab.py`, which patches
`pd.Evaluator` and `pd.rust_flat_batch_adapter` at the module boundary and
**interleaves arms (A B B A)** so laptop thermal drift cannot masquerade as an
effect. Extend `make_arm_patches` / `arm_settings` with `fp32` and `bf16` arms
that set the precision, keeping the interleaving and the existing
fingerprinting.

```
.venv/Scripts/python.exe -m games.seven_wonders_duel.f4_phase_d_ab `
  --checkpoint games/seven_wonders_duel/runs/laptop_training_03/checkpoints/current_best.pt `
  --output <scratch>/w03_ab --games 64 --repetitions 2 --arms fp32,bf16
```

Expected ~35 min. **The harness fingerprints trajectories** (actions, sims,
digests) and bf16 *will* change some of them -- that is expected and is not a
failure. Report the divergence rate; do not assert byte-identity. Reference
expectation, from isolated forwards at batch 256: bf16 is **2.3-3.0x** fp32.
The plan's own history says benchmark gains only partly transfer (1.99x → 1.89x,
+48% → +21%), so a smaller end-to-end number is the expected result, not a bug.

## Acceptance

- `--precision` exists, defaults to fp32, is persisted, and is refused on a
  changed resume.
- Every `Evaluator` construction in `phase_d.py` passes it.
- Per-path dtype tests pass, including the Rust adapter path.
- Full suite green with no assertion edits.
- A/B numbers recorded with the trajectory-divergence rate.
- Committed to the branch.

---

# W0.1 -- the model-size experiment

**Goal:** a validation / strength / throughput Pareto over three widths, so the
user can choose the cloud run's model size. **You produce the data; you do not
choose the size.**

## W0.1.1 -- Arms

| arm | d_model | layers | heads | params | AMP step @b512 | VRAM |
|---|---|---|---|---|---|---|
| S | 128 | 4 | 4 | 1.03 M | 38.7 ms | 0.79 GB |
| M | 256 | 6 | 4 | 5.21 M | 122.8 ms | 2.19 GB |
| L | 384 | 8 | 6 | 14.9 M | 251.6 ms | 4.31 GB |

Head counts come from `default_heads(d_model) = max(4, d_model // 64)`, shipped
in W0.2 -- **pass `heads=None` and let it derive.** Do not hard-code.

## W0.1.2 -- Data (read this before writing any code)

Source: `games/seven_wonders_duel/runs/laptop_training_03/buffers/iter_*.jsonl`
(28,000 games across 70 files, plus `curriculum_seed.jsonl` -- **exclude the
seed file**, it is bot games).

**Do not load all 28,000 games.** Measured: 17.55 examples/game and **17.8 KB
per `Example`** → 28k games is **8.7 GB** of live Python objects, which on a
16 GB box reproduces the exact `MemoryError` that killed run 03 (that crash is
what this whole plan exists for).

**Use the most recent 12,000 games** (the highest-numbered `iter_*.jsonl`
files): ~210k examples, ~3.75 GB, still 1.5x the largest replay window run 03
ever trained on. Derivation costs ~0.137 s/game → **~27 min**, so derive once
and cache to disk (`torch.save` of the vectorized arrays, or `.npz`) rather
than re-deriving per arm. All arms must train on **byte-identical** data and
splits.

Use `games.seven_wonders_duel.dataset.examples_from_record` with
`record_fast_moves=False` (the production setting), and
`train.stable_game_split` for a **game-honest** train/val split so no game
appears on both sides.

## W0.1.3 -- LR sweep, then seeds

A single LR favours whichever width it was tuned for, so each arm gets its own.

- **Sweep:** LR in `{1e-4, 2e-4, 4e-4}` x 3 arms x **1 seed** x 4,000 steps at
  batch 512. Pick each arm's best by **validation total loss**. ~84 min.
- **Final:** each arm at its own best LR x **3 seeds** (0, 1, 2) x 4,000 steps.
  ~84 min.

4,000 steps at batch 512 over ~210k examples is ~10 epochs. Keep the existing
cosine schedule and warmup proportional to steps. If two LRs tie within seed
noise, take the lower.

Report per arm: best LR, and mean +/- spread across seeds for val total,
value_acc, joint7_acc, policy_top1.

## W0.1.4 -- Arena

Validation loss cannot rank playing strength; the arena decides the positive
case.

- **Pairings:** S-vs-M, M-vs-L, S-vs-L (3 matches), using each arm's **best
  seed by val total**.
- **Protocol:** the production gate path -- `gate_sims=64`,
  `eval_search_mode="gumbel"` (default), paired seeds with each model in both
  seats.
- **Heterogeneous matches work**: `_rust_model_gate_waves` builds each side
  from its own `ModelAgentSpec`, and W0.2 added `heads` to that spec.
  Construct the specs **manually** with each arm's own `d_model`/`layers`/
  `heads` -- do **not** use `PhaseDLoop._model_agent_spec`, which fills them
  from the single run config and would rebuild one side at the wrong width.
- **Stopping:** Wilson lower/upper bound at z=1.96 on **pairs** (outcome in
  {0, 0.5, 1}, draws = 0.5), checked only at completed seed pairs, minimum 60
  games, stop when the interval excludes 0.50 or at **800 games** per pairing.
  Reuse `games.kingdomino.promotion.wilson_lower_bound` -- it is already
  written and tested.
- **Cost:** *estimated* (not measured) at 1.74 s/game for S scaling to ~3.6 s
  for M and ~7.3 s for L, from the gate decomposition in
  `CLOUD_TRAINING_PLAN.md` A5. Budget 1-3.5 h and treat as +/-50%. If W0.3
  landed with bf16, run the arena with `--precision bf16` and expect 2-3x
  faster; **note in the report which precision the arena used.**

## W0.1.5 -- Output

Write `games/seven_wonders_duel/runs/w0_sizing/report.md` plus the raw JSON,
containing:

1. the LR sweep table (all 9 cells, val total);
2. the final table: per arm, best LR, val metrics mean +/- spread over 3 seeds;
3. the arena table: each pairing's score rate, Wilson interval, games played,
   stop reason;
4. throughput from the table in W0.1.1 plus the inference rows/s already
   measured (fp32 / bf16 @b256: S 14,118 / 32,182; M 3,645 / 9,835;
   L 1,430 / 4,356);
5. a **Pareto reading** -- which arms are dominated, which are on the frontier,
   and what each frontier point would cost per iteration at cloud scale. State
   the throughput cost of each step up in width against the strength gained.
   **Do not recommend a size.** Present the frontier and stop.

## Acceptance

- Harness committed (new module + tests), all arms trained on identical data.
- `report.md` complete with all five sections.
- Full suite still green.
- Committed to the branch.

---

## Traps this session hit -- do not rediscover them

1. **`SWDNet.heads` is the output-head bundle, not the attention count.**
   `getattr(model, "heads")` returns a `Heads` module. The attention count is
   `model.attention_heads`, or authoritatively
   `model.encoder.layers[0].self_attn.num_heads`. Unit tests passed while this
   was wrong; only loading a real checkpoint caught it.
2. **Head-count mismatches load silently.** Attention parameter shapes do not
   encode the head count, so a 4-head state dict loads into a 2-head model
   reporting "All keys matched successfully" and computes something else
   (measured: 6e-3 on the value head). Always rebuild with the checkpoint's
   recorded `heads`; `PhaseDLoop.load_model` now enforces this.
3. **`torch.autocast("cuda", enabled=x)` with no dtype is fp16, not bf16.**
   That is why the training path was already AMP without anyone deciding it.
4. **`collate_inputs` takes tuples, not `Example` objects** -- pass
   `(type_ids, entity_ids, aux_ids, features)`.
5. **Buffer and elo JSONL use compact separators** (`{"kind":"self_play"`, no
   space after `:`) -- regex extraction must not assume spaces.
6. **Checkpoints written before W0.2 have no `heads` key**; they resolve to
   `LEGACY_HEADS = 4`, *not* `default_heads`. These disagree from d_model 384
   up.
7. **`runs/` is gitignored.** Nothing you write under it will be committed;
   that is intended for the sizing artifacts, but copy `report.md` content into
   the handoff notes so it is not lost.

## Handoff

Leave a summary covering: what landed and what did not, the A/B and Pareto
numbers, any assertion that moved (ground rule 1), anything you had to decide
that the brief did not cover, and the commit SHAs. If a task was abandoned, say
which and why -- an honest partial result is more useful than a forced one.
