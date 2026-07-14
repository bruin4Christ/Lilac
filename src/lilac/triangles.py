"""Flavor triangles -- closed A–B–C bridge cycles, at two levels.

A pairing is an edge; a *triangle* is three ingredients where every pair bridges,
and -- the magic -- each edge is carried by a **different kind of link**: A meets B
one way, B meets C another, C loops back to A a third. The trio then spans three
things at once, a balanced little chord rather than three variations on one theme.

Two levels, mirroring the pairing lenses:

* **bit** -- each edge is the distinctive (IDF-weighted) *sensor* the pair most
  shares (as in :mod:`lilac.compose`); "different link" means a different **note
  family** (terpene / sulfur / roasted / fruity / phenolic / oxygenated). This is
  an abstraction of structure -- two foods can bridge without sharing a molecule.
* **molecular** -- each edge is an actual *shared compound* (as in
  :mod:`lilac.shared`), weighted by how distinctive that molecule is; "different
  link" means a different **molecule** on each edge (not all three leaning on one
  ubiquitous compound). This is the literal food-pairing hypothesis, closed into a
  loop.

Edges are kept only when strong (top-quantile) and between non-duplicate
ingredients. Results are ranked by link diversity, so the most complementary
triangles come first; a hard `min_families` / `min_categories` gives the strict
"magical" version.

    python -m lilac.triangles                       # best bit-level triangles
    python -m lilac.triangles --level molecular      # shared-compound triangles
    python -m lilac.triangles tarragon --magical     # strict, around one ingredient
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from .compose import _CHAR_MASK
from .sensors import BIT_NAMES

# Which *kind* of note a bridge sensor carries (bit level). Distinct families across
# the three edges is what makes a triangle a genuine three-way complement.
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
    via: str             # what carries the link: a sensor name (bit) or compound name
    group: str           # the distinctness key: note family (bit) or molecule (molecular)
    strength: float

    # bit-level readers used `sensor`/`family`; keep them as aliases.
    @property
    def sensor(self) -> str:
        return self.via

    @property
    def family(self) -> str:
        return self.group


@dataclass
class Triangle:
    members: tuple[str, str, str]
    edges: list[Edge]
    weakest: float                 # weakest of the three bridge strengths
    families: set = field(default_factory=set)   # distinct link groups
    categories: set = field(default_factory=set)

    def describe(self) -> str:
        lines = [" + ".join(self.members)]
        for e in self.edges:
            lines.append(f"    {e.a} – {e.b}  via {e.via} ({e.group})  [{e.strength:.2f}]")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Edge providers: each returns (strength matrix, too_similar(i,j), edge_of(i,j)).
# ---------------------------------------------------------------------------
def _bit_edges(sigs, idf, max_similarity):
    names = list(sigs)
    soft = np.array([sigs[n].soft for n in names])
    cw = idf * _CHAR_MASK
    n = len(names)
    strength = np.zeros((n, n))
    bridge = np.zeros((n, n), dtype=int)
    for i in range(n):
        m = np.minimum(soft[i], soft) * cw
        strength[i] = m.max(axis=1)
        bridge[i] = m.argmax(axis=1)
    np.fill_diagonal(strength, 0.0)
    W = soft * idf
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    cos = Wn @ Wn.T

    def too_similar(i, j):
        return cos[i, j] >= max_similarity

    def edge_of(i, j):
        s = BIT_NAMES[bridge[i, j]]
        return Edge(names[i], names[j], s, note_family(s), float(strength[i, j]))

    return names, strength, too_similar, edge_of


def _molecular_edges(df, max_similarity, compound_names):
    from .shared import compound_idf, _compound_names
    names = [row["ingredient"] for _, row in df.iterrows()]
    csets = [set(row["smiles"]) for _, row in df.iterrows()]
    cidf, _ = compound_idf(df)
    cname = compound_names if compound_names is not None else _compound_names()

    # Weighted Jaccard + overlap coefficient via a binary ingredient×compound matrix.
    vocab = {s: k for k, s in enumerate(sorted({s for cs in csets for s in cs}))}
    n, m = len(names), len(vocab)
    M = np.zeros((n, m))
    wv = np.zeros(m)
    for s, k in vocab.items():
        wv[k] = cidf[s]
    for i, cs in enumerate(csets):
        for s in cs:
            M[i, vocab[s]] = 1.0
    Mw = M * wv[None, :]
    num = Mw @ M.T                      # Σ_c w_c · a_c · b_c  (shared distinctive mass)
    mass = Mw.sum(axis=1)
    denom = mass[:, None] + mass[None, :] - num
    wj = np.where(denom > 0, num / denom, 0.0)
    np.fill_diagonal(wj, 0.0)
    counts = M @ M.T                    # shared compound counts
    sizes = M.sum(axis=1)
    overlap = counts / np.maximum(np.minimum(sizes[:, None], sizes[None, :]), 1)

    def too_similar(i, j):
        return overlap[i, j] >= max_similarity   # one ingredient ~ subset of the other

    def edge_of(i, j):
        inter = csets[i] & csets[j]
        top = max(inter, key=lambda s: cidf[s])  # most distinctive shared molecule
        return Edge(names[i], names[j], cname.get(top, top), top, float(wj[i, j]))

    return names, wj, too_similar, edge_of


def _search(names, strength, too_similar, edge_of, categories, anchor, top,
            edge_percentile, edge_min, min_families, min_categories):
    idx = {n: i for i, n in enumerate(names)}
    if anchor is not None and anchor not in idx:
        raise KeyError(f"{anchor!r} is not a known ingredient")

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
            if strength[i, j] >= thr and not too_similar(i, j):
                adj[i].add(j)
                adj[j].add(i)

    out: list[Triangle] = []

    def consider(i, j, k):
        e = [edge_of(i, j), edge_of(j, k), edge_of(i, k)]
        fams = {x.group for x in e}
        cats = {categories.get(names[t], "") for t in (i, j, k)} - {""}
        if len(fams) < min_families or len(cats) < min_categories:
            return
        out.append(Triangle((names[i], names[j], names[k]), e,
                            min(x.strength for x in e), fams, cats))

    if anchor is not None:
        a = idx[anchor]
        nb = sorted(adj[a])
        for x in range(len(nb)):
            for y in range(x + 1, len(nb)):
                if nb[y] in adj[nb[x]]:
                    consider(a, nb[x], nb[y])
    else:
        for i in range(n):
            ni = [j for j in adj[i] if j > i]
            for x in range(len(ni)):
                for y in range(x + 1, len(ni)):
                    if ni[y] in adj[ni[x]]:
                        consider(i, ni[x], ni[y])

    out.sort(key=lambda t: (-len(t.families), -len(t.categories), -t.weakest))
    return out[:top]


def find_triangles(
    sigs=None,
    idf=None,
    categories: dict[str, str] | None = None,
    level: str = "bit",
    df=None,
    anchor: str | None = None,
    top: int = 12,
    edge_percentile: float = 85.0,
    edge_min: float | None = None,
    max_similarity: float = 0.88,
    min_families: int = 1,
    min_categories: int = 1,
    compound_names: dict[str, str] | None = None,
) -> list[Triangle]:
    """Find closed A–B–C bridge triangles at the ``bit`` or ``molecular`` level.

    ``bit`` needs `sigs` + `idf`; ``molecular`` needs `df` (compound sets). See the
    module docstring for what an edge and a "family" mean at each level. Results are
    ranked by link diversity; `min_families` / `min_categories` hard-filter (set both
    to 3 for the strict "magical" version).
    """
    categories = categories or {}
    if level == "bit":
        if sigs is None or idf is None:
            raise ValueError("bit-level triangles need `sigs` and `idf`")
        names, strength, too_similar, edge_of = _bit_edges(sigs, idf, max_similarity)
    elif level == "molecular":
        if df is None:
            from .data import load_flavor_network
            df = load_flavor_network(min_compounds=5)
        names, strength, too_similar, edge_of = _molecular_edges(
            df, max_similarity, compound_names)
    else:
        raise ValueError(f"unknown level {level!r} (use 'bit' or 'molecular')")

    return _search(names, strength, too_similar, edge_of, categories, anchor, top,
                   edge_percentile, edge_min, min_families, min_categories)


def main() -> None:
    from .data import load_flavor_network
    from .ingredients import build_ingredient_signatures, idf_weights

    ap = argparse.ArgumentParser(description="Find closed A–B–C flavor-bridge triangles.")
    ap.add_argument("anchor", nargs="?", default=None,
                    help="optional ingredient to build triangles around")
    ap.add_argument("--level", choices=["bit", "molecular"], default="bit",
                    help="bit = shared sensor / note family; molecular = shared compound")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--magical", action="store_true",
                    help="require 3 distinct links + 3 categories")
    ap.add_argument("--weighting", choices=["uniform", "specificity"], default="uniform")
    args = ap.parse_args()

    df = load_flavor_network(min_compounds=5)
    sigs = build_ingredient_signatures(df=df, weighting=args.weighting)
    idf = idf_weights(sigs)
    cats = dict(zip(df["ingredient"], df["category"]))

    mf = mc = 3 if args.magical else 1
    tris = find_triangles(sigs=sigs, idf=idf, categories=cats, level=args.level, df=df,
                          anchor=args.anchor, top=args.top,
                          min_families=mf, min_categories=mc)
    where = f" around {args.anchor}" if args.anchor else ""
    kind = "shared-compound" if args.level == "molecular" else "sensor-bit"
    print(f"\n{kind.capitalize()} triangles{where} — closed A–B–C cycles"
          f"{' (3 distinct links + categories)' if args.magical else ', best first'}:\n")
    if not tris:
        print("  (none found — drop --magical, or try a different ingredient)\n")
    for t in tris:
        print(t.describe())
        print()


if __name__ == "__main__":
    main()
