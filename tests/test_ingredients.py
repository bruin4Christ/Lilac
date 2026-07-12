"""Ingredient superposition signatures and IDF weighting (no network needed)."""

import numpy as np
import pandas as pd

from lilac.ingredients import build_ingredient_signatures, idf_weights
from lilac.sensors import BIT_NAMES, active_names


def _df():
    # Two synthetic ingredients: a sulfury "allium" and a fruity-ester one.
    return pd.DataFrame({
        "ingredient": ["alliumish", "fruityish"],
        "smiles": [["CSC", "CCSCC", "CSSC"],
                   ["CCOC(=O)C", "CCCCOC(=O)C", "CCOC(=O)CC"]],
    })


def test_superposition_reflects_constituents():
    sigs = build_ingredient_signatures(df=_df())
    assert set(sigs) == {"alliumish", "fruityish"}
    # Every allium compound has sulfur -> the superposition keeps it crisp.
    assert "sulfur" in active_names(sigs["alliumish"].crisp)
    assert "ester" in active_names(sigs["fruityish"].crisp)
    # A soft signature is a fraction in [0, 1].
    assert float(sigs["alliumish"].soft.max()) <= 1.0


def test_idf_downweights_ubiquitous_sensors():
    # methyl fires for every molecule below; sulfur only for the allium one.
    df = pd.DataFrame({
        "ingredient": ["a", "b", "c"],
        "smiles": [["CSC"], ["CCO"], ["CCC"]],
    })
    sigs = build_ingredient_signatures(df=df)
    w = idf_weights(sigs)
    assert w[BIT_NAMES.index("sulfur")] > w[BIT_NAMES.index("methyl")]
