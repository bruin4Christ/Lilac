"""Composition layer: coherence-led dish building + non-eliminating hedonics."""

import pandas as pd

from lilac.compose import CHALLENGE_WEIGHTS, compose
from lilac.ingredients import build_ingredient_signatures, idf_weights


def _world():
    """A small synthetic ingredient world with predictable chemistry.

    Each ingredient shares some distinctive sensor with others (so it can bridge)
    while adding a new one (so it has novelty).
    """
    df = pd.DataFrame({
        "ingredient": ["roasty", "roasty2", "fruity", "sulfury", "waxy"],
        "smiles": [
            ["Cc1cnccn1", "CCC=O"],          # pyrazine + aldehyde (roasted)
            ["c1cnccn1", "CCOC(=O)C"],       # pyrazine + ester  -> bridges via pyrazine
            ["CCOC(=O)C", "CCC=O"],          # ester + aldehyde  -> bridges via aldehyde
            ["CSC", "CSSC", "CCC=O"],        # sulfur + aldehyde (polarizing)
            ["CCCCCCCC", "CCC=O"],           # long alkyl + aldehyde
        ],
    })
    cats = {"roasty": "roast", "roasty2": "roast", "fruity": "fruit",
            "sulfury": "vegetable", "waxy": "fat"}
    sigs = build_ingredient_signatures(df=df)
    return sigs, idf_weights(sigs), cats


def test_base_first_and_size():
    sigs, idf, cats = _world()
    comp = compose("roasty", sigs, idf, cats, size=3)
    assert comp.members[0].ingredient == "roasty"
    assert comp.members[0].role == "base"
    assert len(comp.members) <= 3
    for m in comp.members[1:]:
        assert m.via is not None          # every pick names its bridge


def test_coherent_pick_beats_unrelated():
    # roasty2 bridges via the rare pyrazine; fruity/waxy only via common aldehyde.
    sigs, idf, cats = _world()
    comp = compose("roasty", sigs, idf, cats, size=2)
    assert comp.members[1].ingredient == "roasty2"
    assert comp.members[1].via == "pyrazine"


def test_challenge_warning_fires_on_polarizing_ingredient():
    sigs, idf, cats = _world()
    comp = compose("sulfury", sigs, idf, cats, size=2)
    assert comp.members[0].ingredient == "sulfury"
    assert comp.members[0].warning is not None
    assert any(s in comp.members[0].warning for s in CHALLENGE_WEIGHTS)


def test_diversity_prefers_distinct_over_near_duplicate():
    # After the forced first pick P1, `twin` is ~identical to P1 (adds only a small
    # extra note) while `fresh` is distinct. Diversity off takes the duplicate;
    # diversity on takes the distinct partner -- the redundancy penalty at work.
    df = pd.DataFrame({
        "ingredient": ["base", "P1", "twin", "fresh"],
        "smiles": [
            ["CSC", "CCC=O"],                          # base: sulfur + aldehyde
            ["CSC", "CCC=O", "Cc1cnccn1"],             # P1: + pyrazine (best first pick)
            ["CSC", "CCC=O", "Cc1cnccn1", "CCCCCCCC"], # twin: P1 + a small extra note
            ["CSC", "CCOC(=O)C"],                      # fresh: sulfur + ester (distinct)
        ],
    })
    sigs = build_ingredient_signatures(df=df)
    idf = idf_weights(sigs)
    off = [m.ingredient for m in compose("base", sigs, idf, size=3,
                                         diversity_weight=0.0).members]
    on = [m.ingredient for m in compose("base", sigs, idf, size=3,
                                        diversity_weight=1.0).members]
    assert "twin" in off and "fresh" not in off      # duplicate taken without penalty
    assert "fresh" in on and "twin" not in on        # distinct taken with penalty


def test_diversity_preserves_a_coherent_distinct_theme():
    # Three partners all bridging via the same rare note (pyrazine) but mutually
    # distinct: the redundancy penalty must NOT disrupt this (garlic-via-sulfur case).
    df = pd.DataFrame({
        "ingredient": ["roast", "t_sulfur", "t_ester", "t_alkyl"],
        "smiles": [
            ["Cc1cnccn1", "CCC=O"],       # base: pyrazine + aldehyde
            ["c1cnccn1", "CSC"],          # pyrazine + sulfur
            ["c1cnccn1", "CCOC(=O)C"],    # pyrazine + ester
            ["c1cnccn1", "CCCCCCCC"],     # pyrazine + long alkyl
        ],
    })
    sigs = build_ingredient_signatures(df=df)
    idf = idf_weights(sigs)
    off = {m.ingredient for m in compose("roast", sigs, idf, size=4,
                                         diversity_weight=0.0).members}
    on = {m.ingredient for m in compose("roast", sigs, idf, size=4,
                                        diversity_weight=1.0).members}
    assert off == on   # a genuinely diverse theme is left intact


def test_challenge_never_eliminates_the_only_coherent_pick():
    # Base bridges only to a sulfury (polarizing) partner; a huge challenge weight
    # must NOT filter it out -- the hedonic signal nudges ranking, never eliminates.
    df = pd.DataFrame({
        "ingredient": ["garlicish", "sulfury2", "inert"],
        "smiles": [["CSC", "CCC=O"],          # sulfur + aldehyde
                   ["CSSC", "CCOC(=O)C"],      # sulfur (bridge) + ester (novelty)
                   ["CCCCCCCC"]],             # shares nothing -> never qualifies
    })
    sigs = build_ingredient_signatures(df=df)
    comp = compose("garlicish", sigs, idf_weights(sigs), size=2,
                   challenge_weight=100.0)
    picks = [m.ingredient for m in comp.members]
    assert picks[1] == "sulfury2"             # still selected despite the penalty
    assert comp.members[1].warning is not None
