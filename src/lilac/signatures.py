"""Per-flavor signatures.

Given every molecule's 40-bit code and its odor labels, a *flavor signature* is
what the "nose" tends to report for that flavor. For each descriptor we look at
all molecules carrying it and, per bit, measure how often that sensor fires:

* **soft signature** -- a 40-d vector of on-fractions in [0, 1]
* **crisp signature** -- the soft vector thresholded at 0.5, i.e. the flavor's
  own 40-bit "number"

This is deliberately simple and readable. `top_bits` turns a signature back into
a ranked list of named sensors so "lemon" can be described in words.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sensors import BIT_NAMES, N_BITS


@dataclass
class FlavorSignature:
    descriptor: str
    n_molecules: int
    soft: np.ndarray            # float64[N_BITS], on-fraction per sensor
    crisp: np.ndarray           # uint8[N_BITS], soft >= threshold

    def top_bits(self, k: int = 6) -> list[tuple[str, float]]:
        """The k most characteristic sensors, as (name, on-fraction) pairs."""
        order = np.argsort(-self.soft)
        return [(BIT_NAMES[i], float(self.soft[i])) for i in order[:k]]

    def describe(self, k: int = 6) -> str:
        parts = [f"{name}({frac:.0%})" for name, frac in self.top_bits(k)]
        return f"{self.descriptor} [n={self.n_molecules}]: " + " + ".join(parts)


def build_signatures(
    codes: np.ndarray,
    labels: list[list[str]],
    descriptors: list[str],
    min_molecules: int = 5,
    threshold: float = 0.5,
) -> dict[str, FlavorSignature]:
    """Aggregate 40-bit codes into one signature per descriptor.

    Parameters
    ----------
    codes : (n_molecules, N_BITS) uint8 matrix, aligned row-for-row with `labels`.
    labels : per-molecule list of descriptor names.
    descriptors : the descriptors to build signatures for.
    min_molecules : skip descriptors with fewer than this many examples (too noisy).
    threshold : on-fraction cut for the crisp signature.
    """
    assert codes.shape[1] == N_BITS, "codes width must match N_BITS"
    label_sets = [set(ls) for ls in labels]

    signatures: dict[str, FlavorSignature] = {}
    for desc in descriptors:
        idx = [i for i, s in enumerate(label_sets) if desc in s]
        if len(idx) < min_molecules:
            continue
        subset = codes[idx]
        soft = subset.mean(axis=0)
        crisp = (soft >= threshold).astype(np.uint8)
        signatures[desc] = FlavorSignature(desc, len(idx), soft, crisp)
    return signatures
