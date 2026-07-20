"""Emit `src/data_gen.rs` from the authoritative Python `data.py` tables.

Run from the repo root:  python -m games.seven_wonders_duel.seven_wonders_rust.export_rust_data

The generated Rust `const` arrays match the struct shapes in `src/data.rs`.
Regenerate whenever `data.py` changes; the F2 codec/encoder gate and the F1
replay gate both assume Rust and Python agree on every card fact.
"""

from __future__ import annotations

import os

from games.seven_wonders_duel.data import (
    ALL_BUILDING_CARDS,
    PROGRESS_TOKENS,
    TABLEAU_LAYOUTS,
    WONDERS,
    CardColor,
    EffectKind,
    Resource,
    ScienceSymbol,
)

_RESOURCE = {
    Resource.WOOD: "Wood",
    Resource.CLAY: "Clay",
    Resource.STONE: "Stone",
    Resource.GLASS: "Glass",
    Resource.PAPYRUS: "Papyrus",
}
_COLOR = {
    CardColor.BROWN: "Brown",
    CardColor.GREY: "Grey",
    CardColor.BLUE: "Blue",
    CardColor.GREEN: "Green",
    CardColor.YELLOW: "Yellow",
    CardColor.RED: "Red",
    CardColor.PURPLE: "Purple",
}
_SCIENCE = {
    ScienceSymbol.ARMILLARY_SPHERE: "ArmillarySphere",
    ScienceSymbol.WHEEL: "Wheel",
    ScienceSymbol.SUNDIAL: "Sundial",
    ScienceSymbol.MORTAR_AND_PESTLE: "MortarAndPestle",
    ScienceSymbol.SET_SQUARE: "SetSquare",
    ScienceSymbol.QUILL_AND_INK: "QuillAndInk",
    ScienceSymbol.LAW: "Law",
}
_EFFECT = {
    EffectKind.IMMEDIATE_COINS: "ImmediateCoins",
    EffectKind.OPPONENT_LOSES_COINS: "OpponentLosesCoins",
    EffectKind.PLAY_AGAIN: "PlayAgain",
    EffectKind.COINS_PER_OWN_COLOR: "CoinsPerOwnColor",
    EffectKind.COINS_PER_OWN_WONDER: "CoinsPerOwnWonder",
    EffectKind.COINS_PER_MOST_COLOR: "CoinsPerMostColor",
    EffectKind.COINS_PER_MOST_BROWN_GREY: "CoinsPerMostBrownGrey",
    EffectKind.VP_PER_MOST_COLOR: "VpPerMostColor",
    EffectKind.VP_PER_MOST_WONDER: "VpPerMostWonder",
    EffectKind.VP_PER_RICHEST_COIN_SET: "VpPerRichestCoinSet",
    EffectKind.VP_PER_MOST_BROWN_GREY: "VpPerMostBrownGrey",
    EffectKind.DESTROY_OPPONENT_BROWN: "DestroyOpponentBrown",
    EffectKind.DESTROY_OPPONENT_GREY: "DestroyOpponentGrey",
    EffectKind.BUILD_FROM_DISCARD_FREE: "BuildFromDiscardFree",
    EffectKind.CHOOSE_UNUSED_PROGRESS: "ChooseUnusedProgress",
    EffectKind.FUTURE_WONDER_RESOURCE_DISCOUNT: "FutureWonderResourceDiscount",
    EffectKind.RECEIVE_OPPONENT_TRADE_SPEND: "ReceiveOpponentTradeSpend",
    EffectKind.FUTURE_BLUE_RESOURCE_DISCOUNT: "FutureBlueResourceDiscount",
    EffectKind.VP_PER_PROGRESS: "VpPerProgress",
    EffectKind.FUTURE_RED_EXTRA_SHIELD: "FutureRedExtraShield",
    EffectKind.FUTURE_WONDER_PLAY_AGAIN: "FutureWonderPlayAgain",
    EffectKind.COINS_PER_CHAIN_BUILD: "CoinsPerChainBuild",
}

# Chain tokens (printed icons) → stable small ids, sorted for determinism.
_CHAIN_TOKENS = sorted(
    {c.chain_from for c in ALL_BUILDING_CARDS if c.chain_from}
    | {c.chain_to for c in ALL_BUILDING_CARDS if c.chain_to}
)
_CHAIN_ID = {tok: i for i, tok in enumerate(_CHAIN_TOKENS)}


def _res_slice(resources) -> str:
    return "&[" + ", ".join(f"Resource::{_RESOURCE[r]}" for r in resources) + "]"


def _opt_science(s) -> str:
    return f"Some(ScienceSymbol::{_SCIENCE[s]})" if s is not None else "None"


def _opt_chain(tok) -> str:
    return f"Some({_CHAIN_ID[tok]})" if tok is not None else "None"


def _cost(cost) -> str:
    return (
        f"Cost {{ coins: {cost.coins}, wood: {cost.wood}, clay: {cost.clay}, "
        f"stone: {cost.stone}, glass: {cost.glass}, papyrus: {cost.papyrus} }}"
    )


def _effects(effects) -> str:
    parts = []
    for e in effects:
        color = f"Some(CardColor::{_COLOR[e.color]})" if e.color is not None else "None"
        parts.append(
            f"Effect {{ kind: EffectKind::{_EFFECT[e.kind]}, amount: {e.amount}, color: {color} }}"
        )
    return "&[" + ", ".join(parts) + "]"


def _card_literal(c) -> str:
    return (
        "    CardData {\n"
        f"        name: {c.name!r}, age: {c.age}, color: CardColor::{_COLOR[c.color]},\n"
        f"        cost: {_cost(c.cost)}, victory_points: {c.victory_points}, shields: {c.shields},\n"
        f"        fixed_production: {_res_slice(c.fixed_production)},\n"
        f"        choice_production: {_res_slice(c.choice_production)},\n"
        f"        trade_discount: {_res_slice(sorted(c.trade_discount, key=lambda r: list(Resource).index(r)))},\n"
        f"        science: {_opt_science(c.science)},\n"
        f"        chain_from: {_opt_chain(c.chain_from)}, chain_to: {_opt_chain(c.chain_to)},\n"
        f"        effects: {_effects(c.effects)},\n"
        "    },"
    ).replace("'", '"')


def _wonder_literal(w) -> str:
    cost = f"Some({_cost(w.cost)})" if w.cost is not None else "None"
    return (
        "    WonderData {\n"
        f"        name: {w.name!r}, cost: {cost}, victory_points: {w.victory_points}, shields: {w.shields},\n"
        f"        choice_production: {_res_slice(w.choice_production)},\n"
        f"        effects: {_effects(w.effects)},\n"
        "    },"
    ).replace("'", '"')


def _progress_literal(p) -> str:
    return (
        "    ProgressData {\n"
        f"        name: {p.name!r}, victory_points: {p.victory_points}, science: {_opt_science(p.science)},\n"
        f"        effects: {_effects(p.effects)},\n"
        "    },"
    ).replace("'", '"')


def _layout_literal(name: str, slots) -> str:
    body = "\n".join(
        f"    SlotDef {{ row: {s.row}, x: {s.x}, face_up: {str(s.face_up).lower()} }},"
        for s in slots
    )
    return f"pub static {name}: [SlotDef; {len(slots)}] = [\n{body}\n];\n"


def generate() -> str:
    lines = [
        "// @generated by export_rust_data.py from games/seven_wonders_duel/data.py",
        "// Do not edit by hand. Regenerate after any change to data.py.",
        "",
        f"pub static CARDS: [CardData; {len(ALL_BUILDING_CARDS)}] = [",
        "\n".join(_card_literal(c) for c in ALL_BUILDING_CARDS),
        "];",
        "",
        f"pub static WONDERS: [WonderData; {len(WONDERS)}] = [",
        "\n".join(_wonder_literal(w) for w in WONDERS),
        "];",
        "",
        f"pub static PROGRESS: [ProgressData; {len(PROGRESS_TOKENS)}] = [",
        "\n".join(_progress_literal(p) for p in PROGRESS_TOKENS),
        "];",
        "",
        _layout_literal("LAYOUT_AGE_1", TABLEAU_LAYOUTS[1]),
        _layout_literal("LAYOUT_AGE_2", TABLEAU_LAYOUTS[2]),
        _layout_literal("LAYOUT_AGE_3", TABLEAU_LAYOUTS[3]),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    out = os.path.join(os.path.dirname(__file__), "src", "data_gen.rs")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(generate())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
