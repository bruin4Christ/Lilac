"""The 40-bit "nose".

Every molecule is reduced to 40 on/off sensors. The design goal is *legibility*:
each bit has a name and a chemical/perceptual reason, so a flavor's signature can
be read directly ("lemon = aldehydic + terpene + ester + ...").

Two kinds of sensor:

* **Structural bits** — presence of a substructure, detected with a SMARTS pattern.
  These are the "facets" of the molecule's shape (methyl, acetyl, ester ring, ...).
* **Descriptor bits** — a physicochemical property crossing a threshold (size,
  greasiness, polarity, ...). Smell depends on volatility and how a molecule
  partitions into the mucus/receptor, not just which groups it carries.

The public surface is small:

    from lilac.sensors import encode, BIT_NAMES, N_BITS
    bits = encode("CC(=O)OCC")      # -> np.uint8 array of length 40
    dict(zip(BIT_NAMES, bits))      # -> readable {name: 0/1}

`encode` returns None for SMILES RDKit cannot parse, so callers can drop them.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

# ---------------------------------------------------------------------------
# Structural sensors: (name, SMARTS). Order defines the bit index.
# Patterns are deliberately simple and readable; a few (e.g. "methyl") match
# very broadly on purpose -- they are meant as coarse facet detectors, not as
# strict IUPAC group definitions.
# ---------------------------------------------------------------------------
_STRUCTURAL: list[tuple[str, str]] = [
    ("hydroxyl",        "[OX2H]"),                       # -OH of any kind
    ("primary_alcohol", "[CX4;H2][OX2H]"),               # -CH2-OH
    ("phenol",          "[c][OX2H]"),                    # aromatic -OH
    ("carboxylic_acid", "[CX3](=O)[OX2H1]"),             # -COOH (sour/acidic)
    ("ester",           "[CX3](=O)[OX2][#6]"),           # fruity workhorse
    ("lactone",         "[CX3](=O)[OX2][#6;R]"),         # cyclic ester (coconut/creamy)
    ("aldehyde",        "[CX3H1](=O)[#6]"),              # -CHO (aldehydic, citrus)
    ("ketone",          "[#6][CX3](=O)[#6]"),            # C=O between carbons
    ("ether",           "[OD2]([#6])[#6]"),              # C-O-C
    ("acetal",          "[CX4]([OX2][#6])[OX2][#6]"),    # C(OR)(OR)
    ("methyl",          "[CH3]"),                        # methyl facet
    ("gem_dimethyl",    "[CX4]([CH3])[CH3]"),            # isopropyl / gem-dimethyl
    ("benzene_ring",    "c1ccccc1"),                     # aromatic 6-ring
    ("fused_aromatic",  "c1ccc2ccccc2c1"),               # naphthalene-like
    ("aliphatic_ring",  "[R;!a]"),                       # any non-aromatic ring atom
    ("alkene",          "[CX3]=[CX3]"),                  # C=C
    ("conjugated_diene","[CX3]=[CX3][CX3]=[CX3]"),       # diene
    ("terpene_isoprene","[CH3]C(=C)"),                   # isoprene-ish unit (terpenes)
    ("amine",           "[NX3;!$(NC=O)]"),               # amine, not amide (fishy/animalic)
    ("sulfur",          "[#16]"),                        # any S (alliaceous/savory/sulfurous)
    ("pyrazine",        "c1cnccn1"),                     # roasted / nutty / earthy
    ("furan",           "c1ccoc1"),                      # furan ring (bready/sweet)
    ("halogen",         "[F,Cl,Br,I]"),                  # halogen
    ("acetyl",          "[CH3][CX3](=O)"),               # CH3-C=O
    ("methoxy",         "[OX2]([CH3])[#6]"),             # -OCH3
    ("methoxy_phenol",  "[c][OX2][CH3]"),                # guaiacol/vanillin-like
    ("nitrogen_hetero", "[n]"),                          # aromatic N (besides pyrazine)
    ("long_alkyl_chain","[CH2][CH2][CH2][CH2][CH2][CH2]"),  # >= 6 CH2 in a row (fatty/waxy)
    ("branched_chain",  "[CX4]([#6])([#6])([#6])[#6]"),  # quaternary-ish branch point
]

# ---------------------------------------------------------------------------
# Descriptor sensors: (name, predicate over an RDKit Mol -> bool).
# Thresholds are starting points, tuned against the sanity checks in validate.py.
# ---------------------------------------------------------------------------
_DESCRIPTOR: list[tuple[str, Callable[[Chem.Mol], bool]]] = [
    ("mw_low",          lambda m: Descriptors.MolWt(m) < 120),
    ("mw_high",         lambda m: Descriptors.MolWt(m) > 200),
    ("logp_low",        lambda m: Crippen.MolLogP(m) < 1.0),       # hydrophilic
    ("logp_high",       lambda m: Crippen.MolLogP(m) > 3.0),       # greasy / fatty
    ("high_tpsa",       lambda m: rdMolDescriptors.CalcTPSA(m) > 40),   # polar surface
    ("flexible",        lambda m: rdMolDescriptors.CalcNumRotatableBonds(m) >= 5),
    ("hbond_donor",     lambda m: rdMolDescriptors.CalcNumHBD(m) >= 1),
    ("hbond_acceptors", lambda m: rdMolDescriptors.CalcNumHBA(m) >= 3),
    ("aromatic_rich",   lambda m: _aromatic_fraction(m) > 0.5),
    ("multi_ring",      lambda m: rdMolDescriptors.CalcNumRings(m) >= 2),
    ("has_stereocenter",lambda m: _has_stereocenter(m)),
]


def _aromatic_fraction(mol: Chem.Mol) -> float:
    heavy = mol.GetNumHeavyAtoms()
    if heavy == 0:
        return 0.0
    aromatic = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    return aromatic / heavy


def _has_stereocenter(mol: Chem.Mol) -> bool:
    return len(Chem.FindMolChiralCenters(mol, useLegacyImplementation=False,
                                         includeUnassigned=True)) > 0


# Compile SMARTS once at import time. A pattern that fails to compile is a bug
# in the table above, so surface it loudly rather than silently skipping a bit.
_STRUCT_PATTERNS: list[tuple[str, Chem.Mol]] = []
for _name, _smarts in _STRUCTURAL:
    _patt = Chem.MolFromSmarts(_smarts)
    if _patt is None:
        raise ValueError(f"Bad SMARTS for sensor {_name!r}: {_smarts!r}")
    _STRUCT_PATTERNS.append((_name, _patt))

BIT_NAMES: list[str] = [name for name, _ in _STRUCTURAL] + [name for name, _ in _DESCRIPTOR]
N_BITS: int = len(BIT_NAMES)


def encode(smiles: str) -> Optional[np.ndarray]:
    """SMILES -> uint8 array of length N_BITS, or None if it cannot be parsed."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    bits = np.zeros(N_BITS, dtype=np.uint8)
    for i, (_name, patt) in enumerate(_STRUCT_PATTERNS):
        if mol.HasSubstructMatch(patt):
            bits[i] = 1
    offset = len(_STRUCT_PATTERNS)
    for j, (_name, predicate) in enumerate(_DESCRIPTOR):
        try:
            if predicate(mol):
                bits[offset + j] = 1
        except Exception:
            # A descriptor that blows up on an odd molecule leaves its bit at 0
            # rather than sinking the whole encoding.
            pass
    return bits


def encode_frame(smiles_list) -> tuple[np.ndarray, np.ndarray]:
    """Encode many SMILES.

    Returns (codes, ok_mask): `codes` is an (n_ok, N_BITS) uint8 matrix for the
    SMILES that parsed, and `ok_mask` is a boolean array over the input marking
    which ones survived. Keeping the mask lets callers realign labels.
    """
    rows: list[np.ndarray] = []
    ok = np.zeros(len(smiles_list), dtype=bool)
    for i, smi in enumerate(smiles_list):
        bits = encode(smi)
        if bits is not None:
            ok[i] = True
            rows.append(bits)
    codes = np.vstack(rows) if rows else np.empty((0, N_BITS), dtype=np.uint8)
    return codes, ok


def bits_to_string(bits: np.ndarray) -> str:
    """Render a 40-bit code as its compact binary string, e.g. '0101...'."""
    return "".join(str(int(b)) for b in bits)


def active_names(bits: np.ndarray) -> list[str]:
    """Names of the sensors that are switched on for this code."""
    return [BIT_NAMES[i] for i, b in enumerate(bits) if b]
