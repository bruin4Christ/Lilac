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
