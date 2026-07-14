"""Load the odor dataset and attach a sensor code to every molecule.

    python scripts/build_dataset.py

Writes ``data/molecules_encoded.pkl`` (the joined dataset plus a `code` column,
one uint8[N_BITS] array per row) and prints a short report. Unparseable SMILES are
dropped and counted.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from lilac.data import load_dataset  # noqa: E402
from lilac.sensors import N_BITS, encode  # noqa: E402


def main() -> None:
    df, descriptors = load_dataset()
    n_before = len(df)

    codes = [encode(smi) for smi in df["smiles"]]
    df["code"] = codes
    dropped = df["code"].isna().sum()
    df = df[df["code"].notna()].reset_index(drop=True)

    out = Path("data/molecules_encoded.pkl")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out)

    code_matrix = np.vstack(df["code"].to_numpy())
    print(f"Molecules: {n_before} loaded, {dropped} unparseable, {len(df)} encoded.")
    print(f"Descriptors: {len(descriptors)}. Code width: {N_BITS} bits.")
    print(f"Mean sensors on per molecule: {code_matrix.sum(axis=1).mean():.2f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
