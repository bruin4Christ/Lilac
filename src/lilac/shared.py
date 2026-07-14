"""Shared-compound pairing -- the fourth lens.

`reinforce` / `bridge` / `contrast` compare ingredients through their **sensor
bits**: an abstraction of structure, so two foods can "match" without sharing a
single molecule. This module compares them through the actual **molecules they
have in common** -- the original food-pairing hypothesis (Ahn *et al.*, *Flavor
Network*, 2011): foods that literally share aroma compounds have a concrete
chemical reason to sit together.

Not every shared compound is equally telling. Nearly everything contains a few
ubiquitous volatiles (acetic acid, a common fatty alcohol); sharing one of those
says little. Sharing a *distinctive* compound -- a specific pyrazine, a particular
lactone -- says a lot. So overlap is weighted by each compound's **inverse
ingredient-frequency** (the same IDF idea used for sensors, one level down at the
molecule): rare shared compounds count for more.

The headline metric is a weighted **Jaccard** over the two compound sets (shared
distinctive mass / total distinctive mass), which is symmetric and size-fair. We
also report `query_coverage` -- the fraction of the *query's* own distinctive mass
that the partner covers -- which is the more natural "what shares this one's
character" reading when ingredient sizes differ a lot (hazelnut has 9 compounds,
cocoa 146).

    python -m lilac.shared cocoa
    python -m lilac.shared hazelnut --top 12
    python -m lilac.shared cocoa --with hazelnut   # just the shared molecules
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from .data import load_flavor_network, load_odorant_library


def compound_idf(df) -> tuple[dict[str, float], Counter]:
    """Per-compound inverse ingredient-frequency weight + raw document frequencies.

    A compound in few ingredients (a character-impact molecule) gets a high weight;
    one in nearly every ingredient (a ubiquitous background volatile) gets ~1.0.
    """
    n = len(df)
    doc_freq: Counter = Counter()
    for smiles in df["smiles"]:
        for s in set(smiles):
            doc_freq[s] += 1
    idf = {s: float(np.log((n + 1) / (c + 1)) + 1.0) for s, c in doc_freq.items()}
    return idf, doc_freq


def _compound_names() -> dict[str, str]:
    """SMILES -> a human-readable compound name, from the odorant library."""
    lib = load_odorant_library()
    return dict(zip(lib["smiles"], lib["name"].astype(str)))


def shared_pairings(
    query: str,
    df=None,
    top: int = 10,
    min_shared: int = 1,
    idf: dict[str, float] | None = None,
) -> list[dict]:
    """Rank ingredients by how many *distinctive* aroma compounds they share with `query`.

    Returns a list of dicts sorted by weighted Jaccard (then raw shared count), each:
    ``ingredient, n_shared, weighted_jaccard, query_coverage, shared`` -- where
    `shared` is the shared SMILES, most-distinctive first.
    """
    if df is None:
        df = load_flavor_network(min_compounds=5)
    if idf is None:
        idf, _ = compound_idf(df)

    sets = {row["ingredient"]: set(row["smiles"]) for _, row in df.iterrows()}
    if query not in sets:
        raise KeyError(f"{query!r} is not a known ingredient")

    qset = sets[query]
    qmass = sum(idf[c] for c in qset) or 1.0

    out: list[dict] = []
    for name, cset in sets.items():
        if name == query:
            continue
        inter = qset & cset
        if len(inter) < min_shared:
            continue
        inter_mass = sum(idf[c] for c in inter)
        union_mass = sum(idf[c] for c in (qset | cset)) or 1.0
        out.append({
            "ingredient": name,
            "n_shared": len(inter),
            "weighted_jaccard": inter_mass / union_mass,
            "query_coverage": inter_mass / qmass,
            "shared": sorted(inter, key=lambda c: -idf[c]),
        })
    out.sort(key=lambda r: (-r["weighted_jaccard"], -r["n_shared"]))
    return out[:top]


def shared_compounds(a: str, b: str, df=None,
                     idf: dict[str, float] | None = None,
                     names: dict[str, str] | None = None) -> list[tuple[str, str, float]]:
    """The compounds two ingredients share, as (name, smiles, idf), distinctive first."""
    if df is None:
        df = load_flavor_network(min_compounds=5)
    if idf is None:
        idf, _ = compound_idf(df)
    if names is None:
        names = _compound_names()
    sets = {row["ingredient"]: set(row["smiles"]) for _, row in df.iterrows()}
    for name in (a, b):
        if name not in sets:
            raise KeyError(f"{name!r} is not a known ingredient")
    inter = sorted(sets[a] & sets[b], key=lambda c: -idf[c])
    return [(names.get(s, s), s, idf[s]) for s in inter]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Shared-compound pairing: rank ingredients by the aroma molecules they share.")
    ap.add_argument("query", help="ingredient name (e.g. cocoa)")
    ap.add_argument("--with", dest="partner", default=None,
                    help="just list the compounds shared with this one ingredient")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--min-shared", type=int, default=1)
    ap.add_argument("--min-compounds", type=int, default=5)
    args = ap.parse_args()

    df = load_flavor_network(min_compounds=args.min_compounds)
    idf, _ = compound_idf(df)
    names = _compound_names()

    if args.partner:
        rows = shared_compounds(args.query, args.partner, df=df, idf=idf)
        print(f"\n{args.query} ∩ {args.partner}: {len(rows)} shared compounds "
              f"(most distinctive first)\n")
        for nm, smi, w in rows:
            print(f"  {nm:44} idf={w:.2f}")
        return

    print(f"\nShared-compound partners for {args.query}:\n")
    print(f"  {'ingredient':24} {'shared':>6}  {'wJacc':>6}  {'cover':>6}   top shared compounds")
    for r in shared_pairings(args.query, df=df, top=args.top,
                             min_shared=args.min_shared, idf=idf):
        tops = ", ".join(names.get(s, s) for s in r["shared"][:3])
        print(f"  {r['ingredient']:24} {r['n_shared']:6}  "
              f"{r['weighted_jaccard']:.3f}  {r['query_coverage']:.3f}   {tops}")
    print()


if __name__ == "__main__":
    main()
