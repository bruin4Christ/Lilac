"""Flavor triangles: closed A–B–C bridge cycles (no network needed)."""

import pandas as pd

from lilac.ingredients import build_ingredient_signatures, idf_weights
from lilac.triangles import find_triangles, note_family


def _world():
    # A, B, C each carry two single-note components so every pair bridges on a
    # DIFFERENT note family: A∩B = terpene, B∩C = roasted, C∩A = sulfur. D is inert.
    df = pd.DataFrame({
        "ingredient": ["a", "b", "c", "d"],
        "category": ["cat1", "cat2", "cat3", "cat4"],
        "smiles": [
            ["CSC", "CC1=CCC(CC1)C(=C)C"],        # sulfur + limonene(terpene)
            ["CC1=CCC(CC1)C(=C)C", "c1cnccn1"],   # limonene(terpene) + pyrazine(roasted)
            ["c1cnccn1", "CSC"],                  # pyrazine(roasted) + sulfur
            ["CCCCCCCC"],                         # inert alkane
        ],
    })
    sigs = build_ingredient_signatures(df=df)
    return sigs, idf_weights(sigs), dict(zip(df["ingredient"], df["category"]))


def test_note_family_map():
    assert note_family("sulfur") == "sulfur"
    assert note_family("pyrazine") == "roasted/animalic"
    assert note_family("ester") == "fruity/creamy"
    assert note_family("multi_isoprene") == "terpene"


def test_finds_the_magical_triangle():
    sigs, idf, cats = _world()
    tris = find_triangles(sigs, idf, categories=cats, edge_min=0.1,
                          max_similarity=0.999, min_families=1)
    assert tris, "expected at least one triangle"
    top = tris[0]
    assert set(top.members) == {"a", "b", "c"}   # d is inert, never in a triangle
    assert len(top.families) == 3                # three distinct note families
    assert len(top.categories) == 3


def test_magical_filter_and_closure():
    sigs, idf, cats = _world()
    strict = find_triangles(sigs, idf, categories=cats, edge_min=0.1,
                            max_similarity=0.999, min_families=3, min_categories=3)
    assert len(strict) == 1
    t = strict[0]
    # every edge names a real bridge sensor, and the three form a closed loop
    pairs = {frozenset((e.a, e.b)) for e in t.edges}
    assert pairs == {frozenset(("a", "b")), frozenset(("b", "c")), frozenset(("a", "c"))}
    assert all(e.strength > 0 for e in t.edges)


def test_anchor_restricts_membership():
    sigs, idf, cats = _world()
    tris = find_triangles(sigs, idf, categories=cats, anchor="a", edge_min=0.1,
                          max_similarity=0.999)
    assert tris and all("a" in t.members for t in tris)


def test_molecular_level_triangle():
    # Three ingredients linked pairwise by three DIFFERENT shared compounds:
    # A∩B = m_ab, B∩C = m_bc, C∩A = m_ca. D shares nothing.
    df = pd.DataFrame({
        "ingredient": ["a", "b", "c", "d"],
        "category": ["cat1", "cat2", "cat3", "cat4"],
        "smiles": [["m_ab", "m_ca", "fa"],
                   ["m_ab", "m_bc", "fb"],
                   ["m_bc", "m_ca", "fc"],
                   ["fd1", "fd2", "fd3"]],
    })
    names = {"m_ab": "compound AB", "m_bc": "compound BC", "m_ca": "compound CA"}
    tris = find_triangles(level="molecular", df=df,
                          categories=dict(zip(df["ingredient"], df["category"])),
                          compound_names=names, edge_min=0.01, max_similarity=0.99,
                          min_families=1)
    assert tris
    top = tris[0]
    assert set(top.members) == {"a", "b", "c"}
    # each edge carried by a different molecule -> 3 distinct groups (the smiles)
    assert top.families == {"m_ab", "m_bc", "m_ca"}
    assert {e.via for e in top.edges} == {"compound AB", "compound BC", "compound CA"}


def test_unknown_anchor_raises():
    sigs, idf, cats = _world()
    try:
        find_triangles(sigs, idf, categories=cats, anchor="nope")
        assert False, "expected KeyError"
    except KeyError:
        pass
