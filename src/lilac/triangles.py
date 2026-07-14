"""Flavor triangles -- closed A–B–C bridge cycles.

A pairing is an edge; a *triangle* is three ingredients where every pair bridges,
and -- the magic -- each edge is carried by a **different kind of note**: A meets B
on one distinctive sensor, B meets C on another, C loops back to A on a third. The
trio then spans three aroma families at once, a balanced little chord rather than
three variations on one theme.

Two ingredients "bridge" on the distinctive (IDF-weighted) sensor they most share,
exactly as in :mod:`lilac.compose`. An edge counts only if that bridge is strong
(top-quantile) and the two ingredients aren't near-duplicates. A triangle is
*magical* when its three bridge sensors fall in three different **note families**
(terpene / sulfur / roasted / fruity / phenolic / oxygenated) and, by default, its
three ingredients come from different culinary categories.

    python -m lilac.triangles                 # the best magical triangles overall
    python -m lilac.triangles tarragon        # triangles built around one ingredient
    python -m lilac.triangles --any-note       # drop the distinct-note-family rule
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from .compose import _CHAR_MASK
from .sensors import BIT_NAMES

# Which *kind* of note a bridge sensor carries. Distinct families across the three
# edges is what makes a triangle a genuine three-way complement (not three terpenes).
NOTE_FAMILIES: dict[str, str] = {}
for _s in ("terpene_isoprene", "multi_isoprene", "alkene", "conjugated_diene",
           "gem_dimethyl", "aliphatic_ring", "polyene", "decalin", "oxane_ring"):
    NOTE_FAMILIES[_s] = "terpene"
for _s in ("sulfur", "thiophene", "thiazole"):
    NOTE_FAMILIES[_s] = "sulfur"
for _s in ("pyrazine", "nitrogen_hetero", "quinoline", "indole", "furan", "amine"):
    NOTE_FAMILIES[_s] = "roasted/animalic"
for _s in ("ester", "lactone", "acetyl", "acetal"):
    NOTE_FAMILIES[_s] = "fruity/creamy"
for _s in ("benzene_ring", "fused_aromatic", "phenol", "methoxy", "methoxy_phenol",
           "benzofuran", "coumarin", "phthalide"):
    NOTE_FAMILIES[_s] = "phenolic/balsamic"
for _s in ("hydroxyl", "primary_alcohol", "ether", "aldehyde", "ketone",
           "carboxylic_acid", "halogen"):
    NOTE_FAMILIES[_s] = "oxygenated"


def note_family(sensor: str) -> str:
    return NOTE_FAMILIES.get(sensor, "other")


@dataclass
class Edge:
    a: str
    b: str
    sensor: str          # the distinctive sensor bridging the pair
    family: str          # its note family
    strength: float


@dataclass
class Triangle:
    members: tuple[str, str, str]
    edges: list[Edge]
    weakest: float                 # weakest of the three bridge strengths
    families: set = field(default_factory=set)
    categories: set = field(default_factory=set)

    def describe(self) -> str:
        head = " + ".join(self.members)
        lines = [f"{head}"]
        for e in self.edges:
            lines.append(f"    {e.a} – {e.b}  via {e.sensor} ({e.family})  [{e.strength:.2f}]")
        return "\n".join(lines)


def _bridge_matrices(soft: np.ndarray, cw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For every pair, the strongest shared characterful sensor and its strength."""
    n = soft.shape[0]
    strength = np.zeros((n, n))
    bridge = np.zeros((n, n), dtype=int)
    for i in range(n):
        m = np.minimum(soft[i], soft) * cw      # (n, N_BITS) shared, char-weighted
        strength[i] = m.max(axis=1)
        bridge[i] = m.argmax(axis=1)
    np.fill_diagonal(strength, 0.0)
    return strength, bridge


def find_triangles(
    sigs: dict,
    idf: np.ndarray,
    categories: dict[str, str] | None = None,
    anchor: str | None = None,
    top: int = 12,
    edge_percentile: float = 85.0,
    edge_min: float | None = None,
    max_similarity: float = 0.88,
    min_families: int = 1,
    min_categories: int = 1,
) -> list[Triangle]:
    """Find closed A–B–C bridge triangles.

    Parameters
    ----------
    sigs, idf : ingredient signatures and per-sensor IDF (see `lilac.ingredients`).
    categories : ingredient -> culinary category (for the category-diversity rule).
    anchor : if given, only triangles containing this ingredient.
    edge_percentile / edge_min : an edge exists when its bridge strength clears this
        quantile of all bridge strengths (or the absolute `edge_min`, if given).
    max_similarity : skip edges between near-duplicate ingredients (trivial cliques).
    min_families : minimum distinct note families among the three bridges (1 = no
        filter; 3 = fully "magical"). Results are always *ranked* by family diversity
        regardless, so the most complementary triangles come first.
    min_categories : minimum distinct culinary categories among the three ingredients.
    """
    categories = categories or {}
    names = list(sigs)
    idx = {n: i for i, n in enumerate(names)}
    if anchor is not None and anchor not in idx:
        raise KeyError(f"{anchor!r} is not a known ingredient")
    soft = np.array([sigs[n].soft for n in names])
    cw = idf * _CHAR_MASK

    strength, bridge = _bridge_matrices(soft, cw)
    W = soft * idf
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    cos = Wn @ Wn.T

    # Self-calibrate the edge bar: globally, off all bridge strengths; anchored, off
    # the anchor's OWN bridges, so a weakly-bridged hub (roasted cocoa) still surfaces
    # its best triangles instead of being frozen out by terpene-heavy ingredients.
    if edge_min is not None:
        thr = edge_min
    else:
        pop = strength[idx[anchor]] if anchor is not None else strength
        pop = pop[pop > 0]
        thr = float(np.percentile(pop, edge_percentile)) if pop.size else 0.0

    n = len(names)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if strength[i, j] >= thr and cos[i, j] < max_similarity:
                adj[i].add(j)
                adj[j].add(i)

    def make_edge(i, j) -> Edge:
        s = BIT_NAMES[bridge[i, j]]
        return Edge(names[i], names[j], s, note_family(s), float(strength[i, j]))

    def consider(i, j, k, out):
        e = [make_edge(i, j), make_edge(j, k), make_edge(i, k)]
        fams = {x.family for x in e}
        cats = {categories.get(names[t], "") for t in (i, j, k)} - {""}
        if len(fams) < min_families or len(cats) < min_categories:
            return
        out.append(Triangle(
            members=(names[i], names[j], names[k]), edges=e,
            weakest=min(x.strength for x in e), families=fams, categories=cats))

    tris: list[Triangle] = []
    if anchor is not None:
        a = idx[anchor]
        nb = sorted(adj[a])
        for x in range(len(nb)):
            for y in range(x + 1, len(nb)):
                j, k = nb[x], nb[y]
                if k in adj[j]:
                    consider(a, j, k, tris)
    else:
        for i in range(n):
            ni = [j for j in adj[i] if j > i]
            for x in range(len(ni)):
                for y in range(x + 1, len(ni)):
                    j, k = ni[x], ni[y]
                    if k in adj[j]:
                        consider(i, j, k, tris)

    # Most complementary first: distinct note families, then categories, then the
    # strength of the weakest edge (so the closed loop is solid all the way round).
    tris.sort(key=lambda t: (-len(t.families), -len(t.categories), -t.weakest))
    return tris[:top]


def main() -> None:
    from .data import load_flavor_network
    from .ingredients import build_ingredient_signatures, idf_weights

    ap = argparse.ArgumentParser(description="Find closed A–B–C flavor-bridge triangles.")
    ap.add_argument("anchor", nargs="?", default=None,
                    help="optional ingredient to build triangles around")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--magical", action="store_true",
                    help="require the strict version: 3 distinct note families + 3 categories")
    ap.add_argument("--weighting", choices=["uniform", "specificity"], default="uniform")
    args = ap.parse_args()

    df = load_flavor_network(min_compounds=5)
    sigs = build_ingredient_signatures(df=df, weighting=args.weighting)
    idf = idf_weights(sigs)
    cats = dict(zip(df["ingredient"], df["category"]))

    mf = 3 if args.magical else 1
    mc = 3 if args.magical else 1
    tris = find_triangles(sigs, idf, categories=cats, anchor=args.anchor, top=args.top,
                          min_families=mf, min_categories=mc)
    where = f" around {args.anchor}" if args.anchor else ""
    print(f"\nFlavor triangles{where} — closed A–B–C bridge cycles"
          f"{' (three distinct note families)' if args.magical else ', best first'}:\n")
    if not tris:
        print("  (none found — try --magical off, or a different ingredient)\n")
    for t in tris:
        print(t.describe())
        print()


if __name__ == "__main__":
    main()
