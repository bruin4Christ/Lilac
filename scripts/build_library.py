"""Build the large odorant library and attach a code to every molecule.

    python scripts/build_library.py

Unions the molecule lists of several pyrfume archives (GoodScents, Leffingwell,
IFRA, Sigma, AromaDb, FlavorNet), deduplicates by structure, encodes each with the
sensor panel, and writes ``data/odorant_library_encoded.pkl``. These molecules are
all *known odorants* even where we have no descriptor for them -- coverage beyond
the labelled Leffingwell set.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from lilac.data import LIBRARY_SOURCES, load_odorant_library  # noqa: E402
from lilac.sensors import LARGE_STRUCTURE_BITS, N_BITS, BIT_NAMES, encode  # noqa: E402


def main() -> None:
    print(f"Sources: {', '.join(LIBRARY_SOURCES)}")
    lib = load_odorant_library()
    print(f"Distinct odorant molecules: {len(lib)}")

    codes = [encode(smi) for smi in lib["smiles"]]
    lib["code"] = codes
    dropped = lib["code"].isna().sum()
    lib = lib[lib["code"].notna()].reset_index(drop=True)

    out = Path("data/odorant_library_encoded.pkl")
    out.parent.mkdir(parents=True, exist_ok=True)
    lib.to_pickle(out)

    matrix = np.vstack(lib["code"].to_numpy())
    print(f"Encoded: {len(lib)} ({dropped} unparseable). Code width: {N_BITS} bits.")
    print(f"Mean sensors on per molecule: {matrix.sum(axis=1).mean():.2f}")

    # How much the new large-structure tier actually fires across the library.
    large_idx = [BIT_NAMES.index(n) for n in LARGE_STRUCTURE_BITS]
    any_large = (matrix[:, large_idx].sum(axis=1) > 0).mean()
    print(f"Molecules tripping >=1 large-structure sensor: {any_large:.1%}")
    print("Large-structure sensor hit-rates:")
    for n, i in sorted(zip(LARGE_STRUCTURE_BITS, large_idx),
                       key=lambda t: -matrix[:, t[1]].mean()):
        print(f"    {n:16} {matrix[:, i].mean():6.2%}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
