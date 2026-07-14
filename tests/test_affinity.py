"""Anchor + Lift pairing model (no network needed)."""

import pandas as pd

from lilac.affinity import analyze, consonance, rank, register, suggest_dish
from lilac.ingredients import build_ingredient_signatures, idf_weights


def _world():
    # base is roasted/savory (pyrazine). roasty2 deepens it (shares pyrazine); herb
    # lifts it across register (adds terpene); sulfury adds a big note but stays in
    # the savory register (should be discounted vs the cross-register herb).
    df = pd.DataFrame({
        "ingredient": ["base", "roasty2", "herb", "sulfury"],
        "category": ["meat", "meat", "herb", "vegetable"],
        "smiles": [
            ["Cc1cnccn1"],                         # base: pyrazine (roasted)
            ["Cc1cnccn1", "CCC=O"],                # + aldehyde: deepener
            ["Cc1cnccn1", "CC1=CCC(CC1)C(=C)C"],   # + limonene: cross-register lift
            ["Cc1cnccn1", "CSSC"],                 # + sulfur: same-register lift
        ],
    })
    sigs = build_ingredient_signatures(df=df)
    return sigs, idf_weights(sigs), dict(zip(df["ingredient"], df["category"]))


def test_consonance_matrix_and_register():
    assert consonance("sulfur", "roasted/animalic") > consonance("sulfur", "fruity/creamy")
    assert consonance("terpene", "terpene") < consonance("terpene", "sulfur")
    assert register("terpene") == "fresh" and register("sulfur") == "savory"


def test_analyze_anchor_and_lift():
    sigs, idf, _ = _world()
    aff = analyze("base", "herb", sigs, idf)
    assert aff.anchor > 0                       # they share the pyrazine anchor
    assert aff.lift_family == "terpene"         # herb adds a terpene note the base lacks
    assert aff.crosses_register                 # savory base -> fresh lift


def test_cross_register_lift_beats_same_register():
    sigs, idf, _ = _world()
    order = [f.partner for f in rank("base", sigs, idf, mode="lifter", max_similarity=1.0)]
    assert order.index("herb") < order.index("sulfury")


def test_deepener_mode_prefers_shared_character():
    sigs, idf, _ = _world()
    order = [f.partner for f in rank("base", sigs, idf, mode="deepener", max_similarity=1.0)]
    assert order.index("roasty2") < order.index("herb")


def test_consonance_detects_clash_against_any_base_register():
    # base fires BOTH roasted (pyrazine) and sweet-fruity (ester); a sulfur lift is
    # consonant with the roasted side but clashes with the sweet side -> flagged low
    # (this is the garlic+chocolate failure the single-family version missed).
    df = pd.DataFrame({
        "ingredient": ["sweetsavory", "garlicky"],
        "category": ["x", "y"],
        "smiles": [["Cc1cnccn1", "CCOC(=O)C"],   # pyrazine + ester
                   ["CSSC", "CSC"]],             # sulfur
    })
    sigs = build_ingredient_signatures(df=df)
    idf = idf_weights(sigs)
    aff = analyze("sweetsavory", "garlicky", sigs, idf)
    assert aff.lift_family == "sulfur"
    assert aff.consonance <= 0.2               # worst-case clash against the sweet register


def test_suggest_dish_returns_a_deepener_and_a_lifter():
    sigs, idf, cats = _world()
    d = suggest_dish("base", sigs, idf, cats)
    assert d["deepener"] and d["lifter"]
    assert d["deepener"].partner != d["lifter"].partner
