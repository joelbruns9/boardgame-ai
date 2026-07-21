//! Chance layer (F3.1a): public-information prediction of the chance events an
//! action fires (`chance_signature`) and exact enumeration of their outcome
//! chains (`enumerate_chains`) — ports of the same functions in `search.py`.
//! Sampling (`sample_outcomes`) and the supplied-outcome apply path
//! (`make_with_chance`) land in F3.1b.

use crate::data::{back_type_of, card, layout, wonder_id};
use crate::engine::{Action, ActionUse};
use crate::pool::{unseen_pool, UnseenPool};
use crate::rng::Rng;
use crate::state::{coverers, GameState};

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
        ActionUse::ChooseNextStartPlayer => vec![ChanceSpec {
            kind: ChanceKind::AgeDeal,
            context: vec![g.age as i32 + 1],
        }],
        ActionUse::ResolvePendingChoice => vec![],
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
                let names: Vec<usize> =
                    pool.cards[back].iter().copied().filter(|&c| !used[c]).collect();
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
