//! Chance layer (F3.1a): public-information prediction of the chance events an
//! action fires (`chance_signature`) and exact enumeration of their outcome
//! chains (`enumerate_chains`) — ports of the same functions in `search.py`.
//! Sampling (`sample_outcomes`) and the supplied-outcome apply path
//! (`make_with_chance`) land in F3.1b.

use crate::data::{back_type_of, card, layout, wonder_id};
use crate::engine::{Action, ActionUse};
use crate::pool::{unseen_pool, UnseenPool};
use crate::rng::Rng;
use crate::state::{coverers, GameState, Phase};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ChanceKind {
    CardReveal = 0,
    GreatLibraryDraw = 1,
    WonderGroupReveal = 2,
    AgeDeal = 3,
}

/// One predicted chance event. `context` mirrors `search.py`'s `ChanceSpec`
/// flattened to ints: CardReveal = `[row, x, back_id]`, AgeDeal = `[age]`,
/// GreatLibraryDraw / WonderGroupReveal = `[]`.
#[derive(Clone, Debug)]
pub struct ChanceSpec {
    pub kind: ChanceKind,
    pub context: Vec<i32>,
}

/// `(row, x, back_id)` of face-down cards a take from `taken` slot would expose,
/// sorted by `(row, x)`. Public topology only (mirrors
/// `_newly_accessible_after_take`); `taken` counts as already removed.
fn newly_accessible_after_take(g: &GameState, taken: usize) -> Vec<(i32, i32, i32)> {
    let age = g.tableau.age;
    let mut out = Vec::new();
    for j in 0..g.tableau.slots.len() {
        if j == taken {
            continue;
        }
        let sc = &g.tableau.slots[j];
        if !sc.present || sc.revealed {
            continue;
        }
        let covered = coverers(age, j)
            .iter()
            .any(|&c| c != taken && g.tableau.slots[c].present);
        if !covered {
            let (row, x) = g.tableau.slot_id(j);
            out.push((row, x, back_type_of(sc.card_id) as i32));
        }
    }
    out.sort_unstable();
    out
}

/// Does this action empty the tableau, so the next Age is dealt?
///
/// Decided by applying the action to a throwaway clone rather than by
/// re-deriving the rules here: whether the last take actually ends the Age
/// depends on victories and deferred choices that only the engine knows. A
/// cheap public precondition gates the clone, so it happens only on the one
/// take per Age that can empty the pyramid. Port of Python `_exhausts_the_age`.
fn exhausts_the_age(g: &GameState, action: &Action) -> bool {
    if g.age >= 3 {
        return false;
    }
    let present = g.tableau.slots.iter().filter(|s| s.present).count();
    if action.use_ == ActionUse::ResolvePendingChoice {
        if present != 0 {
            return false;
        }
    } else if present != 1 {
        return false;
    }
    let mut clone = g.clone();
    clone.apply_action(action);
    clone.phase == Phase::ChooseNextStartPlayer
}

pub fn chance_signature(g: &GameState, action: &Action) -> Vec<ChanceSpec> {
    match action.use_ {
        ActionUse::DraftWonder => {
            let picked: usize = g.cities.iter().map(|c| c.wonders.len()).sum();
            let mut specs = Vec::new();
            if picked == 3 {
                specs.push(ChanceSpec {
                    kind: ChanceKind::WonderGroupReveal,
                    context: vec![],
                });
            }
            if picked == 7 {
                specs.push(ChanceSpec {
                    kind: ChanceKind::AgeDeal,
                    context: vec![1],
                });
            }
            specs
        }
        // The Age was dealt when the previous one ran out; the choice itself
        // fires nothing.
        ActionUse::ChooseNextStartPlayer => vec![],
        ActionUse::ResolvePendingChoice => {
            if exhausts_the_age(g, action) {
                vec![ChanceSpec {
                    kind: ChanceKind::AgeDeal,
                    context: vec![g.age as i32 + 1],
                }]
            } else {
                vec![]
            }
        }
        _ => {
            let taken = action.slot.expect("primary action missing slot");
            let mut specs: Vec<ChanceSpec> = newly_accessible_after_take(g, taken)
                .into_iter()
                .map(|(row, x, back)| ChanceSpec {
                    kind: ChanceKind::CardReveal,
                    context: vec![row, x, back],
                })
                .collect();
            if action.use_ == ActionUse::ConstructWonder
                && action.wonder == Some(wonder_id("The Great Library"))
                && !unseen_pool(g).offboard_progress.is_empty()
            {
                specs.push(ChanceSpec {
                    kind: ChanceKind::GreatLibraryDraw,
                    context: vec![],
                });
            }
            if exhausts_the_age(g, action) {
                specs.push(ChanceSpec {
                    kind: ChanceKind::AgeDeal,
                    context: vec![g.age as i32 + 1],
                });
            }
            specs
        }
    }
}

/// k-combinations of `items` in ascending-index (lexicographic) order, matching
/// Python's `itertools.combinations` over the same ascending input.
fn combinations(items: &[usize], k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if k > items.len() {
        return out;
    }
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        out.push(idx.iter().map(|&i| items[i]).collect());
        // Advance the odometer, rightmost index that can still move.
        let mut i = k;
        loop {
            if i == 0 {
                return out;
            }
            i -= 1;
            if idx[i] != i + items.len() - k {
                break;
            }
        }
        idx[i] += 1;
        for j in i + 1..k {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

/// Observable signature of an AGE_DEAL (CODEC_SPEC §4.2): per layout slot, the
/// card id if face-up else a `NUM_CARDS + back_id` marker — two hidden deals with
/// the same signature are the same chance child. Port of `age_deal_key`.
pub fn age_deal_key(age: usize, deal: &[usize]) -> Vec<i32> {
    layout(age as u8)
        .iter()
        .zip(deal)
        .map(|(slot, &cid)| {
            if slot.face_up {
                cid as i32
            } else {
                crate::data::NUM_CARDS as i32 + back_type_of(cid) as i32
            }
        })
        .collect()
}

/// The observable key of a non-AGE outcome is the outcome ids themselves.
fn outcome_key(outcomes: &[Vec<usize>]) -> Vec<Vec<i32>> {
    outcomes
        .iter()
        .map(|o| o.iter().map(|&x| x as i32).collect())
        .collect()
}

/// All `(outcomes, joint_probability, observable_key)` chains for enumerable
/// specs. Each spec's outcome is an id list (CardReveal `[card_id]`,
/// GreatLibraryDraw `[p,p,p]`, WonderGroupReveal `[w,w,w,w]`); the key equals the
/// outcomes (no coalescing off AGE_DEAL). Sequential CardReveals condition later
/// pools on earlier picks. Panics on AgeDeal (sample-only), like Python.
pub fn enumerate_chains(
    g: &GameState,
    specs: &[ChanceSpec],
) -> Vec<(Vec<Vec<usize>>, f64, Vec<Vec<i32>>)> {
    let pool = unseen_pool(g);
    let mut used = vec![false; crate::data::NUM_CARDS];
    expand(&pool, specs, 0, &mut used)
        .into_iter()
        .map(|(outcomes, p)| {
            let key = outcome_key(&outcomes);
            (outcomes, p, key)
        })
        .collect()
}

fn expand(
    pool: &UnseenPool,
    specs: &[ChanceSpec],
    index: usize,
    used: &mut [bool],
) -> Vec<(Vec<Vec<usize>>, f64)> {
    if index == specs.len() {
        return vec![(vec![], 1.0)];
    }
    let spec = &specs[index];
    let mut results = Vec::new();
    match spec.kind {
        ChanceKind::CardReveal => {
            let back = spec.context[2] as usize;
            let names: Vec<usize> = pool.cards[back]
                .iter()
                .copied()
                .filter(|&c| !used[c])
                .collect();
            let len = names.len() as f64;
            for name in names {
                used[name] = true;
                for (tail, p) in expand(pool, specs, index + 1, used) {
                    let mut outcomes = Vec::with_capacity(tail.len() + 1);
                    outcomes.push(vec![name]);
                    outcomes.extend(tail);
                    results.push((outcomes, p / len));
                }
                used[name] = false;
            }
        }
        ChanceKind::GreatLibraryDraw => {
            let subsets = combinations(&pool.offboard_progress, 3);
            let p0 = 1.0 / subsets.len() as f64;
            for subset in subsets {
                for (tail, tp) in expand(pool, specs, index + 1, used) {
                    let mut outcomes = Vec::with_capacity(tail.len() + 1);
                    outcomes.push(subset.clone());
                    outcomes.extend(tail);
                    results.push((outcomes, p0 * tp));
                }
            }
        }
        ChanceKind::WonderGroupReveal => {
            let subsets = combinations(&pool.wonders, 4);
            let p0 = 1.0 / subsets.len() as f64;
            for subset in subsets {
                for (tail, tp) in expand(pool, specs, index + 1, used) {
                    let mut outcomes = Vec::with_capacity(tail.len() + 1);
                    outcomes.push(subset.clone());
                    outcomes.extend(tail);
                    results.push((outcomes, p0 * tp));
                }
            }
        }
        ChanceKind::AgeDeal => panic!("cannot enumerate AGE_DEAL (sample-only)"),
    }
    results
}

// --- balanced double-reveal support (CHANCE_ENUMERATION_PLAN.md Step 2) ------
// Port of `search.py::balanced_double_reveal_chains` and the two helpers it
// needs. Bit-for-bit: same FNV-1a seed derivation, same partial Fisher-Yates
// over the same `Rng`, same stratum-major emission order, same weight.

const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
const FNV_PRIME: u64 = 0x100_0000_01B3;
/// Domain separation: offsets are a function of the POSITION and the search
/// seed, drawn on a private stream so they never shift the main search RNG.
const OFFSET_DOMAIN_TAG: u64 = 0x0FF5_E75E_ED01_7D0B;

fn mix64(accumulator: u64, value: u64) -> u64 {
    (accumulator ^ value).wrapping_mul(FNV_PRIME)
}

/// Domain-separated seed for one edge's offset draw: search seed + chance
/// signature + reveal pools, and deliberately NOT the action index, so edges
/// sharing a signature share their support (common random numbers).
fn double_reveal_offset_seed(
    search_seed: u64,
    specs: &[ChanceSpec],
    pools: [&[usize]; 2],
) -> u64 {
    let mut accumulator = mix64(FNV_OFFSET, OFFSET_DOMAIN_TAG);
    accumulator = mix64(accumulator, search_seed);
    for spec in specs {
        accumulator = mix64(accumulator, spec.kind as u64);
        for &value in &spec.context {
            accumulator = mix64(accumulator, value as i64 as u64);
        }
    }
    for pool in pools {
        accumulator = mix64(accumulator, pool.len() as u64);
        for &id in pool {
            accumulator = mix64(accumulator, id as u64);
        }
    }
    accumulator
}

/// `count` distinct offsets in `[0, modulus)`, uniform over subsets, from a
/// partial Fisher-Yates draw on a private stream. Ascending, so the support
/// does not depend on draw order.
pub fn distinct_offsets(modulus: usize, count: usize, seed: u64) -> Vec<usize> {
    let mut rng = Rng::new(seed);
    let mut values: Vec<usize> = (0..modulus).collect();
    for k in 0..count {
        let j = k + rng.randrange((modulus - k) as u64) as usize;
        values.swap(k, j);
    }
    let mut chosen = values[..count].to_vec();
    chosen.sort_unstable();
    chosen
}

/// The balanced `n * offsets` support of a PURE double card-reveal edge whose
/// two slots share a back, in the `enumerate_chains` shape. `None` means the
/// construction does not apply (a different signature, two different backs,
/// `offsets == 0`, or an `offsets` that would retain the whole space) and the
/// caller must enumerate exhaustively.
///
/// Stratify on the first reveal — every hidden card leads exactly one stratum,
/// the marginal-coverage guarantee — then take the second by cyclic block:
/// stratum `i` pairs with `names[(i + 1 + t) % n]` over the drawn offsets `t`,
/// i.e. directed-pair distances `1..n-1`, so each card lands in second position
/// exactly `offsets` times and a self-pair is unreachable.
///
/// Two different backs mean disjoint pools and a full `n1 * n2` grid. A cyclic
/// support over the second pool would still be unbiased there, but only its
/// first margin could be exact (a both-margins-balanced subset needs a size
/// divisible by `lcm(n1, n2)`, usually the grid itself), and the retained count
/// would depend on which slot is listed first. Those edges are 2.9% of the
/// measured saving at 3-4x the Q error, so they stay exhaustive — see
/// `search.py::balanced_double_reveal_chains`.
pub fn balanced_double_reveal_chains(
    g: &GameState,
    specs: &[ChanceSpec],
    offsets: usize,
    search_seed: u64,
) -> Option<Vec<(Vec<Vec<usize>>, f64, Vec<Vec<i32>>)>> {
    if offsets == 0 || specs.len() != 2 {
        return None;
    }
    if specs.iter().any(|s| s.kind != ChanceKind::CardReveal) {
        return None;
    }
    let back = specs[0].context[2] as usize;
    if back != specs[1].context[2] as usize {
        return None;
    }
    let pool = unseen_pool(g);
    let names = &pool.cards[back];
    let n = names.len();
    let modulus = n.saturating_sub(1); // distances 1..n-1, never a self-pair
    if offsets >= modulus {
        return None;
    }
    let seed = double_reveal_offset_seed(search_seed, specs, [names, names]);
    let chosen = distinct_offsets(modulus, offsets, seed);
    let weight = 1.0 / (n * offsets) as f64;
    let mut chains = Vec::with_capacity(n * offsets);
    for (i, &name) in names.iter().enumerate() {
        for &offset in &chosen {
            let outcomes = vec![vec![name], vec![names[(i + 1 + offset) % n]]];
            let key = outcome_key(&outcomes);
            chains.push((outcomes, weight, key));
        }
    }
    Some(chains)
}

/// Card ids of a back type, ascending by card NAME — the order Python's
/// `sorted(pool.cards[back])` (a set of names) produces, which the AGE_DEAL
/// sampler shuffles.
fn pool_by_name(ids: &[usize]) -> Vec<usize> {
    let mut v = ids.to_vec();
    v.sort_by(|&a, &b| card(a).name.cmp(card(b).name));
    v
}

/// Sample one outcome per spec from `rng`, mirroring `search.py::sample_outcomes`
/// call-for-call. Returns `(outcomes, joint probability or None when any spec is
/// sample-only, observable key)`. Each outcome is an id list (CardReveal `[id]`,
/// GreatLibraryDraw `[p,p,p]`, WonderGroupReveal `[w,w,w,w]`, AgeDeal the 20-card
/// deal order); AgeDeal's key coalesces via `age_deal_key`, all others equal the
/// outcome.
pub fn sample_outcomes(
    g: &GameState,
    specs: &[ChanceSpec],
    rng: &mut Rng,
) -> (Vec<Vec<usize>>, Option<f64>, Vec<Vec<i32>>) {
    let pool = unseen_pool(g);
    let mut used = vec![false; crate::data::NUM_CARDS];
    let mut outcomes: Vec<Vec<usize>> = Vec::new();
    let mut prob: Option<f64> = Some(1.0);
    let mut key: Vec<Vec<i32>> = Vec::new();
    for spec in specs {
        match spec.kind {
            ChanceKind::CardReveal => {
                let back = spec.context[2] as usize;
                let names: Vec<usize> = pool.cards[back]
                    .iter()
                    .copied()
                    .filter(|&c| !used[c])
                    .collect();
                let choice = names[rng.randrange(names.len() as u64) as usize];
                used[choice] = true;
                outcomes.push(vec![choice]);
                key.push(vec![choice as i32]);
                if let Some(p) = prob.as_mut() {
                    *p *= 1.0 / names.len() as f64;
                }
            }
            ChanceKind::GreatLibraryDraw => {
                let subsets = combinations(&pool.offboard_progress, 3);
                let i = rng.randrange(subsets.len() as u64) as usize;
                key.push(subsets[i].iter().map(|&x| x as i32).collect());
                outcomes.push(subsets[i].clone());
                if let Some(p) = prob.as_mut() {
                    *p *= 1.0 / subsets.len() as f64;
                }
            }
            ChanceKind::WonderGroupReveal => {
                let subsets = combinations(&pool.wonders, 4);
                let i = rng.randrange(subsets.len() as u64) as usize;
                key.push(subsets[i].iter().map(|&x| x as i32).collect());
                outcomes.push(subsets[i].clone());
                if let Some(p) = prob.as_mut() {
                    *p *= 1.0 / subsets.len() as f64;
                }
            }
            ChanceKind::AgeDeal => {
                let age = spec.context[0] as usize;
                let mut names = pool_by_name(&pool.cards[age - 1]); // AgeI/II/III back
                rng.shuffle(&mut names);
                let deal = if age == 3 {
                    let mut guilds = pool_by_name(&pool.cards[3]); // Guild back
                    rng.shuffle(&mut guilds);
                    let mut deal: Vec<usize> = names[..17].to_vec();
                    deal.extend_from_slice(&guilds[..3]);
                    rng.shuffle(&mut deal);
                    deal
                } else {
                    names[..layout(age as u8).len()].to_vec()
                };
                key.push(age_deal_key(age, &deal));
                outcomes.push(deal);
                prob = None;
            }
        }
    }
    (outcomes, prob, key)
}
