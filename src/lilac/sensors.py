"""The sensor "nose".

Every molecule is reduced to a fixed panel of `N_BITS` on/off sensors (55 today,
but the panel is a design knob meant to grow). The design goal is *legibility*:
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
    bits = encode("CC(=O)OCC")      # -> np.uint8 array of length N_BITS
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


# ---------------------------------------------------------------------------
# Larger-structure sensors, part 1: whole named scaffolds via SMARTS.
# Unlike the "corner" detectors above, these fire on multi-ring / composite
# frameworks -- the kind of large motif that carries a smell as a whole.
# ---------------------------------------------------------------------------
_LARGE_STRUCTURAL: list[tuple[str, str]] = [
    ("indole",        "c1ccc2c(c1)cc[nH]2"),   # jasmine/animalic (indole, skatole)
    ("coumarin",      "O=c1ccc2ccccc2o1"),      # sweet / hay / tonka
    ("benzofuran",    "c1ccc2occc2c1"),         # smoky / phenolic bicyclic
    ("quinoline",     "c1ccc2ncccc2c1"),        # leathery / animalic N-bicyclic
    ("thiazole",      "c1cscn1"),               # roasted / meaty S,N ring
    ("thiophene",     "c1cccs1"),               # savoury / alliaceous S ring
    ("decalin",       "C1CCC2CCCCC2C1"),        # woody / ambery fused saturated bicyclic
    ("oxane_ring",    "[#6]1[#6][#6][#6][#6][OX2]1"),  # tetrahydropyran (rose oxide, sugars)
    ("polyene",       "C=CC=CC=C"),             # extended conjugation (carotenoid-like)
    # celery / lovage / angelica. A 5-membered lactone fused to a 6-ring; the
    # any-bond (~) fusion catches aromatic phthalides AND the dihydro forms
    # (ligustilide, sedanolide), while excluding phthalic anhydride (its position-3
    # ring atom is a second carbonyl). Fires on ~0.17% of the odorant library.
    ("phthalide",     "O=C1O[#6;!$([CX3]=O)][#6]2~[#6]1~[#6]~[#6]~[#6]~[#6]~2"),
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


def _largest_ring(mol: Chem.Mol) -> int:
    rings = mol.GetRingInfo().AtomRings()
    return max((len(r) for r in rings), default=0)


# Isoprene unit (C5 building block of terpenes); >= 2 of them ~ a real terpenoid
# skeleton rather than an incidental branch.
_ISOPRENE = Chem.MolFromSmarts("C(=C)C")
_LACTONE = Chem.MolFromSmarts("[CX3](=O)[OX2][#6;R]")


def _is_macrolactone(mol: Chem.Mol) -> bool:
    return mol.HasSubstructMatch(_LACTONE) and _largest_ring(mol) >= 10


def _is_fused(mol: Chem.Mol) -> bool:
    ri = mol.GetRingInfo()
    return any(ri.NumAtomRings(a.GetIdx()) >= 2 for a in mol.GetAtoms())


# ---------------------------------------------------------------------------
# Larger-structure sensors, part 2: whole-molecule topology predicates.
# These describe the scaffold at a scale no local substructure can: ring size,
# fusion, poly-cyclicity, overall heavy-atom count.
# ---------------------------------------------------------------------------
_LARGE_TOPO: list[tuple[str, Callable[[Chem.Mol], bool]]] = [
    ("macrocycle",     lambda m: _largest_ring(m) >= 12),   # macrocyclic musks
    ("macrolactone",   _is_macrolactone),                   # musk lactones
    ("multi_isoprene", lambda m: len(m.GetSubstructMatches(_ISOPRENE)) >= 2),  # terpenoid
    ("fused_ring_sys", _is_fused),                          # any shared-edge ring system
    ("polycyclic",     lambda m: m.GetRingInfo().NumRings() >= 3),
    ("large_scaffold", lambda m: m.GetNumHeavyAtoms() >= 16),
]


def _compile(table: list[tuple[str, str]]) -> list[tuple[str, Chem.Mol]]:
    """Compile a (name, SMARTS) table, failing loudly on a bad pattern."""
    out = []
    for name, smarts in table:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            raise ValueError(f"Bad SMARTS for sensor {name!r}: {smarts!r}")
        out.append((name, patt))
    return out


# Compile SMARTS once at import time. Small "corner" detectors and large scaffold
# detectors are matched the same way, just kept in separate tables for clarity.
_STRUCT_PATTERNS = _compile(_STRUCTURAL)
_LARGE_PATTERNS = _compile(_LARGE_STRUCTURAL)

# Bit order: small structural | large scaffolds | descriptors | large topology.
_SMARTS_SENSORS = _STRUCT_PATTERNS + _LARGE_PATTERNS
_PREDICATE_SENSORS = _DESCRIPTOR + _LARGE_TOPO

BIT_NAMES: list[str] = (
    [name for name, _ in _STRUCTURAL]
    + [name for name, _ in _LARGE_STRUCTURAL]
    + [name for name, _ in _DESCRIPTOR]
    + [name for name, _ in _LARGE_TOPO]
)
N_BITS: int = len(BIT_NAMES)

# Names of the larger-structure sensors, exposed for reporting / analysis.
LARGE_STRUCTURE_BITS: list[str] = (
    [name for name, _ in _LARGE_STRUCTURAL] + [name for name, _ in _LARGE_TOPO]
)


def encode(smiles: str) -> Optional[np.ndarray]:
    """SMILES -> uint8 array of length N_BITS, or None if it cannot be parsed."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    bits = np.zeros(N_BITS, dtype=np.uint8)
    # All SMARTS sensors first (order: small structural, then large scaffolds).
    for i, (_name, patt) in enumerate(_SMARTS_SENSORS):
        if mol.HasSubstructMatch(patt):
            bits[i] = 1
    # Then all predicate sensors (order: descriptors, then large topology).
    offset = len(_SMARTS_SENSORS)
    for j, (_name, predicate) in enumerate(_PREDICATE_SENSORS):
        try:
            if predicate(mol):
                bits[offset + j] = 1
        except Exception:
            # A predicate that blows up on an odd molecule leaves its bit at 0
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
    """Render an N_BITS code as its compact binary string, e.g. '0101...'."""
    return "".join(str(int(b)) for b in bits)


def active_names(bits: np.ndarray) -> list[str]:
    """Names of the sensors that are switched on for this code."""
    return [BIT_NAMES[i] for i, b in enumerate(bits) if b]
