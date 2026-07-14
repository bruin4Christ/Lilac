"""Real culinary ingredients as superpositions of molecules.

A raw ingredient (blueberry, coffee, garlic) is a *mixture* of aroma compounds,
so its "smell number" is the superposition of its constituent molecules on the
sensor panel: per bit, the fraction of the ingredient's compounds that fire it.
This module builds those signatures from the Ahn Flavor-Network data and pairs
ingredients against each other.

Because complex mixtures light up the common bits (methyl, ether) almost by
default, plain cosine similarity saturates near 1.0 for any two foods. We fix
that with **IDF weighting**: sensors that fire across many ingredients are
down-weighted, so distinctive sensors (sulfur, pyrazine, macrocycle) drive the
match. IDF is on by default.

    python -m lilac.ingredients blueberry --mode all
    python -m lilac.ingredients coffee --mode reinforce --top 10
    python -m lilac.ingredients garlic --mode contrast --no-idf
"""

from __future__ import annotations

import argparse

import numpy as np

from .data import load_flavor_network
from .pairing import _resolve_query, rank_pairings
from .sensors import BIT_NAMES, active_names
from .signatures import FlavorSignature, signature_from_smiles


def build_ingredient_signatures(
    df=None,
    min_compounds: int = 5,
    threshold: float = 0.5,
) -> dict[str, FlavorSignature]:
    """One superimposed signature per ingredient.

    Pass a preloaded ingredients DataFrame (columns ``ingredient``/``smiles``) to
    avoid the network fetch; otherwise it is loaded via `load_flavor_network`.
    """
    if df is None:
        df = load_flavor_network(min_compounds=min_compounds)
    sigs: dict[str, FlavorSignature] = {}
    for _, row in df.iterrows():
        sig, _ = signature_from_smiles(row["ingredient"], row["smiles"],
                                       threshold=threshold)
        sigs[row["ingredient"]] = sig
    return sigs


def idf_weights(sigs: dict[str, FlavorSignature]) -> np.ndarray:
    """Inverse-document-frequency weight per sensor across the ingredient set.

    A sensor firing (soft > 0) in few ingredients gets a high weight; one firing
    in nearly all gets a low weight.
    """
    mat = np.array([s.soft for s in sigs.values()])
    n = len(sigs)
    doc_freq = (mat > 0).sum(axis=0)
    return np.log((n + 1) / (doc_freq + 1)) + 1.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Pair culinary ingredients over Lilac signatures.")
    ap.add_argument("query", help="ingredient name (e.g. blueberry), reference aroma, or SMILES list")
    ap.add_argument("--mode", choices=["reinforce", "bridge", "contrast", "all"],
                    default="all")
    ap.add_argument("--target", type=float, default=None,
                    help="explicit absolute bridge similarity (overrides --target-pct)")
    ap.add_argument("--target-pct", type=float, default=65.0,
                    help="bridge centre as a percentile of the query's own partners")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--min-compounds", type=int, default=5)
    ap.add_argument("--no-idf", action="store_true", help="disable IDF sensor weighting")
    args = ap.parse_args()

    sigs = build_ingredient_signatures(min_compounds=args.min_compounds)
    weights = None if args.no_idf else idf_weights(sigs)

    # Query: an ingredient in the library, else a reference aroma / SMILES set.
    if args.query in sigs:
        query = sigs[args.query]
    else:
        query, dropped = _resolve_query(args.query)
        if dropped:
            print(f"(dropped unparseable: {dropped})")

    print(f"\nLibrary: {len(sigs)} ingredients. "
          f"Query: {query.descriptor} ({query.n_molecules} compounds)")
    print(f"  active sensors: {', '.join(active_names(query.crisp)) or '(none above threshold)'}")
    print(f"  weighting: {'IDF' if weights is not None else 'plain cosine'}\n")

    bridge_label = (f"~{args.target:.0%} sim" if args.target is not None
                    else f"p{args.target_pct:.0f} of own partners")
    modes = ["reinforce", "bridge", "contrast"] if args.mode == "all" else [args.mode]
    titles = {"reinforce": "MOST alike (reinforce)",
              "bridge": f"MIDDLE overlap (bridge, {bridge_label})",
              "contrast": "MOST contrast (opposition)"}
    for mode in modes:
        print(f"### {titles[mode]}")
        for r in rank_pairings(query, sigs, mode=mode, target=args.target,
                               target_pct=args.target_pct, top=args.top,
                               weights=weights, band_on="cosine"):
            print(f"  {r['flavor']:22} n={r['n']:4}  sim={r['cosine']:.3f}  jac={r['jaccard']:.2f}")
        print()


if __name__ == "__main__":
    main()
