"""Distances and nearest-neighbour lookups over sensor codes.

Two molecules (or two flavor signatures) are compared bit-for-bit. Hamming
distance counts differing sensors; Jaccard distance ignores the many bits that
are off in both, which suits sparse codes. Both operate on 0/1 arrays.
"""

from __future__ import annotations

import numpy as np


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    """Number of sensors on which the two codes disagree."""
    return int(np.count_nonzero(a != b))


def jaccard_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - |A & B| / |A | B| over the 'on' bits.

    An empty signature carries no signal, so a pair with no 'on' bits between
    them is treated as maximally *distant* (1.0), never as an identical match --
    otherwise two odourless/unencodable codes would spuriously pair perfectly.
    """
    ab = np.logical_and(a, b).sum()
    aub = np.logical_or(a, b).sum()
    if aub == 0:
        return 1.0
    return 1.0 - ab / aub


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine similarity between two real vectors (e.g. soft signatures).

    More sensitive than crisp Hamming for comparing flavor signatures, because it
    uses each sensor's on-fraction rather than a 0/1 threshold. A zero vector has
    no direction and carries no signal, so it is treated as maximally *distant*
    (1.0) rather than identical -- a zero query must not "reinforce" with
    everything.
    """
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / (na * nb))


def weighted_cosine_distance(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    """Cosine distance after scaling each dimension by `weights`.

    Used to down-weight ubiquitous sensors (methyl, ether) that fire for almost
    every ingredient, so discriminative sensors (sulfur, pyrazine, macrocycle)
    drive the similarity. A scaled vector with no signal is treated as maximally
    *distant* (1.0), never as an identical match.
    """
    aw = a * weights
    bw = b * weights
    na = np.linalg.norm(aw)
    nb = np.linalg.norm(bw)
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - float(np.dot(aw, bw) / (na * nb))


def nearest(query: np.ndarray, codes: np.ndarray, k: int = 5,
            metric: str = "jaccard") -> list[tuple[int, float]]:
    """Indices and distances of the k codes closest to `query`.

    `codes` is an (n, N_BITS) matrix. Returns (row_index, distance) sorted by
    ascending distance. Ties are broken by row order.
    """
    if metric == "hamming":
        dists = np.array([hamming(query, c) for c in codes], dtype=float)
    elif metric == "jaccard":
        dists = np.array([jaccard_distance(query, c) for c in codes], dtype=float)
    else:
        raise ValueError(f"unknown metric {metric!r}")
    order = np.argsort(dists, kind="stable")[:k]
    return [(int(i), float(dists[i])) for i in order]


def signature_distance_matrix(signatures: dict, metric: str = "hamming"):
    """Pairwise distances between flavor crisp signatures.

    Returns (names, matrix) where matrix[i][j] is the distance between the crisp
    signatures of names[i] and names[j].
    """
    names = list(signatures.keys())
    crisps = [signatures[n].crisp for n in names]
    n = len(names)
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if metric == "hamming":
                d = hamming(crisps[i], crisps[j])
            elif metric == "jaccard":
                d = jaccard_distance(crisps[i], crisps[j])
            else:
                raise ValueError(f"unknown metric {metric!r}")
            mat[i, j] = mat[j, i] = d
    return names, mat
