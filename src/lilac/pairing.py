"""Flavor pairing over signatures.

Given a query flavor's signature and the dataset's flavor signatures, rank other
flavors three ways:

* **reinforce** -- most signature overlap. Flavors that smell alike; pairing by
  shared character (the "shared-compound" food-pairing hypothesis).
* **bridge**    -- a target *middle* overlap (default 20%). Flavors that keep a
  thread of the query but add a new layer -- often the most interesting pairings.
* **contrast**  -- least overlap. Pairing by opposition.

Overlap is reported two ways: crisp **Jaccard** over the 0/1 signatures (what
"bitmask overlap" means literally) and soft **cosine** over the on-fraction
vectors (more reliable when a signature is sparse). Bridge/contrast rank on
Jaccard; reinforce ranks on cosine, since crisp Jaccard saturates at the top.

Run it:

    python -m lilac.pairing blueberry --mode reinforce
    python -m lilac.pairing blueberry --mode bridge --target 0.2
    python -m lilac.pairing "CCOC(=O)C,CC(C)=CCCC(C)(O)C=C" --mode contrast
"""

from __future__ import annotations

import argparse

import numpy as np

from .data import load_dataset
from .sensors import active_names
from .signatures import (
    FlavorSignature,
    build_signatures,
    signature_from_smiles,
)
from .similarity import cosine_distance, weighted_cosine_distance

# ---------------------------------------------------------------------------
# Reference aroma profiles: a flavor defined by its character-impact compounds.
# (name, SMILES, weight) -- weight ~ how odour-defining the compound is (rough,
# not an odour-activity value). Extend this dict to add more bespoke flavors.
# ---------------------------------------------------------------------------
REFERENCE_AROMAS: dict[str, list[tuple[str, str, float]]] = {
    "blueberry": [
        ("linalool",                "CC(C)=CCCC(C)(O)C=C",        3),
        ("(E)-2-hexenal",           "CCC/C=C/C=O",                3),
        ("ethyl 2-methylbutanoate", "CCC(C)C(=O)OCC",             3),
        ("methyl cinnamate",        "O=C(OC)/C=C/c1ccccc1",       3),
        ("1,8-cineole",             "CC12CCC(CC1)C(C)(C)O2",      2),
        ("geraniol",                "CC(C)=CCC/C(C)=C/CO",        2),
        ("ethyl cinnamate",         "O=C(OCC)/C=C/c1ccccc1",      2),
        ("hexanal",                 "CCCCCC=O",                   1),
        ("(E)-2-hexen-1-ol",        "CCC/C=C/CO",                 1),
        ("nerol",                   "CC(C)=CCC/C(C)=C\\CO",       1),
        ("alpha-terpineol",         "CC(C)(O)C1CCC(C)=CC1",       1),
        ("limonene",                "CC1=CCC(CC1)C(=C)C",         1),
        ("ethyl hexanoate",         "CCCCCC(=O)OCC",              1),
        ("(E)-2-hexenyl acetate",   "CCC/C=C/COC(C)=O",           1),
        ("citral",                  "CC(=CCCC(=CC=O)C)C",         1),
    ],
}


def reference_signature(name: str) -> tuple[FlavorSignature, list[str]]:
    """Build a signature from a REFERENCE_AROMAS entry."""
    if name not in REFERENCE_AROMAS:
        raise KeyError(f"no reference aroma {name!r}; known: {sorted(REFERENCE_AROMAS)}")
    compounds = REFERENCE_AROMAS[name]
    smiles = [s for _, s, _ in compounds]
    weights = [w for _, _, w in compounds]
    return signature_from_smiles(name, smiles, weights=weights)


def jaccard_bits(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def rank_pairings(
    query: FlavorSignature,
    flavor_sigs: dict[str, FlavorSignature],
    mode: str = "reinforce",
    target: float | None = None,
    target_pct: float = 65.0,
    top: int = 10,
    weights: np.ndarray | None = None,
    band_on: str = "jaccard",
) -> list[dict]:
    """Rank `flavor_sigs` against `query` under the chosen mode.

    `weights` (optional, length N_BITS) scales the soft-cosine per sensor -- pass
    IDF weights to stop ubiquitous bits from saturating the similarity.

    `band_on` chooses what the reinforce/bridge/contrast bands sort on: ``jaccard``
    (crisp bitmask overlap -- good for sparse single-flavor signatures) or
    ``cosine`` (soft similarity -- needed for dense ingredient mixtures whose crisp
    signatures collapse).

    Bridge target
    -------------
    An interesting "bridge" partner sits at a *middle* overlap -- but the absolute
    overlap that counts as "middle" is completely different from one query to the
    next (a distinctive ingredient like garlic is far from everything; a central
    one like blueberry is close to everything). So the bridge centre defaults to a
    *percentile of this query's own* partner-similarity distribution
    (`target_pct`, default 65), which self-calibrates per query. Pass an explicit
    absolute `target` to override with a fixed overlap level instead.
    """
    def cos_sim(a, b):
        d = (weighted_cosine_distance(a, b, weights) if weights is not None
             else cosine_distance(a, b))
        return 1.0 - d

    scored = []
    for name, sig in flavor_sigs.items():
        if name == query.descriptor:
            continue
        scored.append({
            "flavor": name,
            "n": sig.n_molecules,
            "jaccard": jaccard_bits(query.crisp, sig.crisp),
            "cosine": cos_sim(query.soft, sig.soft),
        })

    key = "cosine" if band_on == "cosine" else "jaccard"
    if mode == "reinforce":
        scored.sort(key=lambda r: (-r["cosine"], -r["jaccard"]))
    elif mode == "contrast":
        scored.sort(key=lambda r: (r[key], r["cosine"]))
    elif mode == "bridge":
        center = _bridge_center(scored, key, target, target_pct)
        scored.sort(key=lambda r: abs(r[key] - center))
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return scored[:top]


def _bridge_center(scored: list[dict], key: str, target: float | None,
                   target_pct: float) -> float:
    """Where the bridge band centres.

    An explicit absolute `target` wins; otherwise take the `target_pct` percentile
    of this query's own partner values on `key` -- a robust, self-calibrating
    "middle overlap" that adapts to how central or distinctive the query is.
    """
    if target is not None:
        return target
    values = [r[key] for r in scored]
    return float(np.percentile(values, target_pct)) if values else 0.0


def _resolve_query(spec: str) -> tuple[FlavorSignature, list[str]]:
    """A query is a reference-aroma name, or a comma-separated SMILES list."""
    if spec in REFERENCE_AROMAS:
        return reference_signature(spec)
    if "," in spec or any(c in spec for c in "()=#"):
        smiles = [s.strip() for s in spec.split(",") if s.strip()]
        return signature_from_smiles("custom", smiles)
    raise KeyError(
        f"{spec!r} is not a known reference aroma and doesn't look like SMILES. "
        f"Known: {sorted(REFERENCE_AROMAS)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Flavor pairing over Lilac signatures.")
    ap.add_argument("query", help="reference aroma name (e.g. blueberry) or comma-separated SMILES")
    ap.add_argument("--mode", choices=["reinforce", "bridge", "contrast", "all"],
                    default="all")
    ap.add_argument("--target", type=float, default=None,
                    help="explicit absolute bridge overlap (overrides --target-pct)")
    ap.add_argument("--target-pct", type=float, default=65.0,
                    help="bridge centre as a percentile of the query's own partners")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--min-molecules", type=int, default=8,
                    help="ignore dataset flavors rarer than this")
    args = ap.parse_args()

    query, dropped = _resolve_query(args.query)
    if dropped:
        print(f"(dropped unparseable: {dropped})")

    # Build dataset flavor signatures from the labelled Leffingwell molecules.
    from .sensors import encode

    df, descriptors = load_dataset()
    enc = [encode(s) for s in df["smiles"]]
    codes = np.vstack([b for b in enc if b is not None])
    labels = [labs for labs, b in zip(df["labels"].tolist(), enc) if b is not None]
    flavor_sigs = build_signatures(codes, labels, descriptors,
                                   min_molecules=args.min_molecules)

    print(f"\nQuery: {query.descriptor}  ({query.n_molecules} compounds)")
    print(f"  active sensors: {', '.join(active_names(query.crisp))}\n")

    bridge_label = (f"~{args.target:.0%} overlap" if args.target is not None
                    else f"p{args.target_pct:.0f} of own partners")
    modes = ["reinforce", "bridge", "contrast"] if args.mode == "all" else [args.mode]
    titles = {"reinforce": "MOST overlap (reinforce)",
              "bridge": f"MIDDLE overlap (bridge, {bridge_label})",
              "contrast": "MINIMAL overlap (contrast)"}
    for mode in modes:
        print(f"### {titles[mode]}")
        for r in rank_pairings(query, flavor_sigs, mode=mode, target=args.target,
                               target_pct=args.target_pct, top=args.top):
            print(f"  {r['flavor']:14} n={r['n']:4}  "
                  f"jaccard={r['jaccard']:.2f}  cosine={r['cosine']:.2f}")
        print()


if __name__ == "__main__":
    main()
