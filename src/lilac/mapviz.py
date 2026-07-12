"""The 2-D olfactory map.

Project the 40-bit codes into two dimensions and colour each molecule by its
dominant odor label. Nearby points share sensors, so odor families should form
visible neighbourhoods -- the picture your intuition was reaching for.

UMAP gives the nicest layout but is an optional dependency; if it isn't
installed we fall back to scikit-learn's t-SNE, and if there are too few points
for t-SNE, to PCA. The Jaccard metric is used for UMAP because the codes are
sparse 0/1 vectors.
"""

from __future__ import annotations

import collections

import numpy as np


def embed_2d(codes: np.ndarray, random_state: int = 0) -> tuple[np.ndarray, str]:
    """Return (xy, method_name) with xy an (n, 2) float array."""
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(n_components=2, metric="jaccard",
                            random_state=random_state)
        return reducer.fit_transform(codes.astype(float)), "UMAP(jaccard)"
    except Exception:
        pass

    n = len(codes)
    if n >= 10:
        from sklearn.manifold import TSNE

        perplexity = min(30, max(5, n // 4))
        tsne = TSNE(n_components=2, metric="hamming", init="random",
                    perplexity=perplexity, random_state=random_state)
        return tsne.fit_transform(codes.astype(float)), "t-SNE(hamming)"

    from sklearn.decomposition import PCA

    return PCA(n_components=2, random_state=random_state).fit_transform(
        codes.astype(float)), "PCA"


# Major, perceptually distinct odor families used to colour the map. Ordered
# most-specific first so a molecule tagged both "citrus" and "fruity" is coloured
# by the sharper family (citrus). Anything matching none becomes "other".
_ODOR_FAMILIES = [
    "sulfurous", "meaty", "minty", "citrus", "nutty", "roasted", "spicy",
    "woody", "vanilla", "rose", "floral", "herbal", "waxy", "fatty",
    "green", "sweet", "fruity",
]


def dominant_labels(
    labels: list[list[str]],
    families: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Colour each molecule by the first major odor family it carries.

    Families are tried in `families` order (most specific first), so a molecule
    labelled both 'fruity' and 'citrus' is coloured 'citrus'. Molecules matching
    no family are bucketed as 'other'. This keeps the palette informative instead
    of drowning the map in rare one-off descriptors.
    """
    families = families or _ODOR_FAMILIES

    def pick(ls: list[str]) -> str:
        s = set(ls)
        for fam in families:
            if fam in s:
                return fam
        return "other"

    chosen = [pick(ls) for ls in labels]
    present = collections.Counter(chosen)
    # Palette keeps families in the canonical order, dropping any that never occur.
    palette = [fam for fam in families if present.get(fam)]
    if present.get("other"):
        palette.append("other")
    return chosen, palette


def plot_map(codes: np.ndarray, labels: list[list[str]], out_path: str,
             title: str = "Lilac olfactory map") -> str:
    """Render the map to `out_path`. Returns the embedding method used."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xy, method = embed_2d(codes)
    colour_labels, palette = dominant_labels(labels)

    cmap = plt.get_cmap("tab10" if len(palette) <= 10 else "tab20")
    colour_of = {name: cmap(i % cmap.N) for i, name in enumerate(palette)}

    fig, ax = plt.subplots(figsize=(11, 9))
    for name in palette:
        pts = np.array([xy[i] for i, c in enumerate(colour_labels) if c == name])
        if len(pts) == 0:
            continue
        ax.scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.7,
                   color=colour_of[name], label=name)
    ax.set_title(f"{title}\n({len(codes)} molecules, embedding: {method})")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
              fontsize=8, title="dominant odor")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return method
