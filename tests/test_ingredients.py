"""Ingredient superposition signatures and IDF weighting (no network needed)."""

import numpy as np
import pandas as pd

from lilac.ingredients import (
    build_ingredient_signatures,
    compound_specificity,
    idf_weights,
)
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


def test_compound_specificity_favours_rare_compounds():
    # "CCO" is in all three ingredients (ubiquitous); "CSSC" in just one (rare).
    df = pd.DataFrame({
        "ingredient": ["a", "b", "c"],
        "smiles": [["CCO", "CSSC"], ["CCO"], ["CCO"]],
    })
    spec = compound_specificity(df)
    assert spec["CSSC"] > spec["CCO"]


def test_specificity_weighting_upweights_distinctive_compound():
    # An ingredient of one ubiquitous + one rare-but-sulfury compound: under
    # specificity weighting the rare sulfur compound carries more weight, so the
    # sulfur sensor's on-fraction rises vs uniform weighting.
    df = pd.DataFrame({
        "ingredient": ["target", "filler1", "filler2"],
        "smiles": [["CCO", "CSSC"], ["CCO"], ["CCO"]],
    })
    s_uniform = build_ingredient_signatures(df=df, weighting="uniform")
    s_spec = build_ingredient_signatures(df=df, weighting="specificity")
    sulfur = BIT_NAMES.index("sulfur")
    assert s_spec["target"].soft[sulfur] > s_uniform["target"].soft[sulfur]


def test_explicit_concentrations_override_weighting():
    df = pd.DataFrame({
        "ingredient": ["x"],
        "smiles": [["CCO", "CSSC"]],
    })
    # Force the sulfur compound to dominate via an explicit concentration.
    sigs = build_ingredient_signatures(
        df=df, concentrations={"x": {"CSSC": 9.0, "CCO": 1.0}})
    sulfur = BIT_NAMES.index("sulfur")
    assert sigs["x"].soft[sulfur] >= 0.9


def test_idf_downweights_ubiquitous_sensors():
    # methyl fires for every molecule below; sulfur only for the allium one.
    df = pd.DataFrame({
        "ingredient": ["a", "b", "c"],
        "smiles": [["CSC"], ["CCO"], ["CCC"]],
    })
    sigs = build_ingredient_signatures(df=df)
    w = idf_weights(sigs)
    assert w[BIT_NAMES.index("sulfur")] > w[BIT_NAMES.index("methyl")]
