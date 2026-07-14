"""Shared-compound pairing (compound-level, no network needed)."""

import pandas as pd

from lilac.shared import compound_idf, shared_compounds, shared_pairings


def _df():
    # Compound tokens (stand in for SMILES): 'common' is everywhere (low IDF),
    # 'rareX'/'rareY' are distinctive. alpha shares both rares with beta, only the
    # common one with gamma, and rareX + common with delta.
    return pd.DataFrame({
        "ingredient": ["alpha", "beta", "gamma", "delta"],
        "smiles": [
            ["rareX", "rareY", "common"],
            ["rareX", "rareY", "common", "o1"],
            ["common", "o2", "o3"],
            ["rareX", "common", "o4"],
        ],
    })


def test_compound_idf_favours_rare_compounds():
    idf, dfreq = compound_idf(_df())
    assert dfreq["common"] == 4 and dfreq["rareY"] == 2
    assert idf["rareY"] > idf["rareX"] > idf["common"]


def test_ranks_by_distinctive_overlap():
    rows = shared_pairings("alpha", df=_df(), top=10)
    order = [r["ingredient"] for r in rows]
    # beta shares both distinctive compounds -> first; gamma shares only 'common' -> last.
    assert order[0] == "beta"
    assert order[-1] == "gamma"
    # delta (shares rareX) must outrank gamma (shares only the ubiquitous compound).
    assert order.index("delta") < order.index("gamma")


def test_reports_shared_set_and_metrics():
    rows = {r["ingredient"]: r for r in shared_pairings("alpha", df=_df(), top=10)}
    beta = rows["beta"]
    assert beta["n_shared"] == 3
    assert set(beta["shared"][:2]) == {"rareX", "rareY"}   # distinctive first
    assert 0.0 < beta["weighted_jaccard"] <= 1.0
    # alpha's whole distinctive mass is inside beta -> coverage is 1.0
    assert beta["query_coverage"] == 1.0


def test_min_shared_filter():
    # gamma shares exactly one compound with alpha; require >=2 drops it.
    rows = shared_pairings("alpha", df=_df(), top=10, min_shared=2)
    assert "gamma" not in [r["ingredient"] for r in rows]


def test_shared_compounds_ordered_by_idf():
    idf, _ = compound_idf(_df())
    got = shared_compounds("alpha", "beta", df=_df(), idf=idf,
                           names={"rareX": "Rare X", "rareY": "Rare Y", "common": "Common"})
    assert [g[0] for g in got] == ["Rare Y", "Rare X", "Common"]   # rarest first
