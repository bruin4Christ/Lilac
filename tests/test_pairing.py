"""Bespoke signatures and flavor-pairing ranking (no network needed)."""

import numpy as np

from lilac.pairing import REFERENCE_AROMAS, rank_pairings, reference_signature
from lilac.sensors import N_BITS, active_names
from lilac.signatures import FlavorSignature, signature_from_smiles


def test_blueberry_reference_builds():
    sig, dropped = reference_signature("blueberry")
    assert dropped == []
    assert sig.n_molecules == len(REFERENCE_AROMAS["blueberry"])
    assert sig.soft.shape == (N_BITS,)
    # Blueberry is terpenoid-rich (linalool, geraniol, limonene, citral, ...).
    assert "multi_isoprene" in active_names(sig.crisp)


def test_signature_from_smiles_reports_dropped():
    sig, dropped = signature_from_smiles("x", ["CCO", "not_a_molecule"])
    assert dropped == ["not_a_molecule"]
    assert sig.n_molecules == 1  # only ethanol survived


def test_weights_shift_the_soft_signature():
    # Up-weighting the sulfur molecule should raise the sulfur on-fraction.
    smiles = ["CSC", "CCO"]  # dimethyl sulfide, ethanol
    even, _ = signature_from_smiles("even", smiles, weights=[1, 1])
    heavy, _ = signature_from_smiles("heavy", smiles, weights=[9, 1])
    s = active_names(np.ones(N_BITS))  # all names, just to locate index
    sulfur_idx = s.index("sulfur")
    assert heavy.soft[sulfur_idx] > even.soft[sulfur_idx]


def test_rank_modes_pick_expected_extremes():
    query, _ = reference_signature("blueberry")
    same = FlavorSignature("same", 10, query.soft.copy(), query.crisp.copy())
    # A flavor with a single bit the query does NOT have -> minimal overlap.
    off = int(np.where(query.crisp == 0)[0][0])
    other_soft = np.zeros(N_BITS)
    other_crisp = np.zeros(N_BITS, dtype=np.uint8)
    other_soft[off] = 1.0
    other_crisp[off] = 1
    other = FlavorSignature("other", 10, other_soft, other_crisp)
    sigs = {"same": same, "other": other}

    assert rank_pairings(query, sigs, mode="reinforce", top=2)[0]["flavor"] == "same"
    assert rank_pairings(query, sigs, mode="contrast", top=2)[0]["flavor"] == "other"
