"""Spot-check that known aroma molecules light up the expected sensors."""

import numpy as np
import pytest

from lilac.sensors import (
    BIT_NAMES,
    LARGE_STRUCTURE_BITS,
    N_BITS,
    active_names,
    encode,
    encode_frame,
)


def test_bit_count_and_names_unique():
    assert N_BITS == 55
    assert len(BIT_NAMES) == N_BITS
    assert len(set(BIT_NAMES)) == N_BITS  # no duplicate sensor names
    assert set(LARGE_STRUCTURE_BITS).issubset(set(BIT_NAMES))


def test_unparseable_smiles_returns_none():
    assert encode("this is not a molecule") is None
    assert encode("") is None


def test_encode_shape_and_dtype():
    bits = encode("CCO")  # ethanol
    assert bits is not None
    assert bits.shape == (N_BITS,)
    assert bits.dtype == np.uint8
    assert set(np.unique(bits)).issubset({0, 1})


@pytest.mark.parametrize(
    "name, smiles, expected",
    [
        ("limonene", "CC1=CCC(CC1)C(=C)C", {"terpene_isoprene", "alkene", "aliphatic_ring"}),
        ("vanillin", "COc1cc(C=O)ccc1O", {"aldehyde", "methoxy_phenol", "phenol"}),
        ("ethyl_acetate", "CCOC(=O)C", {"ester", "acetyl"}),
        ("dimethyl_sulfide", "CSC", {"sulfur"}),
        ("citral", "CC(=CCCC(=CC=O)C)C", {"aldehyde", "terpene_isoprene"}),
        ("acetic_acid", "CC(=O)O", {"carboxylic_acid"}),
    ],
)
def test_known_molecules_hit_expected_bits(name, smiles, expected):
    bits = encode(smiles)
    assert bits is not None, f"{name} failed to parse"
    on = set(active_names(bits))
    missing = expected - on
    assert not missing, f"{name}: expected sensors not on: {missing} (got {on})"


@pytest.mark.parametrize(
    "name, smiles, expected",
    [
        # The whole-shape motifs that local "corner" detectors miss.
        ("muscone", "CC1CCCCCCCCCCCCCC1=O", {"macrocycle", "large_scaffold"}),
        ("pentadecanolide", "O=C1CCCCCCCCCCCCCCO1", {"macrocycle", "macrolactone"}),
        ("coumarin", "O=c1ccc2ccccc2o1", {"coumarin", "fused_ring_sys"}),
        ("indole", "c1ccc2[nH]ccc2c1", {"indole", "fused_ring_sys"}),
        ("limonene", "CC1=CCC(CC1)C(=C)C", {"multi_isoprene"}),
    ],
)
def test_large_structure_sensors(name, smiles, expected):
    bits = encode(smiles)
    assert bits is not None, f"{name} failed to parse"
    on = set(active_names(bits))
    missing = expected - on
    assert not missing, f"{name}: expected large sensors not on: {missing} (got {on})"


def test_tiny_molecule_has_no_large_structure_bits():
    # Acetone is too small to trip any large-structure sensor.
    on = set(active_names(encode("CC(=O)C")))
    assert on.isdisjoint(set(LARGE_STRUCTURE_BITS))


def test_encode_frame_masks_bad_smiles():
    smiles = ["CCO", "garbage", "CC(=O)C"]
    codes, ok = encode_frame(smiles)
    assert ok.tolist() == [True, False, True]
    assert codes.shape == (2, N_BITS)
