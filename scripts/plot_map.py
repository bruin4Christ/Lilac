"""Render the 2-D olfactory map.

    python scripts/plot_map.py

Reads ``data/molecules_encoded.pkl`` and writes ``outputs/odor_map.png``:
every molecule as a point, coloured by its dominant odor family.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lilac.mapviz import plot_map  # noqa: E402


def main() -> None:
    enc_path = Path("data/molecules_encoded.pkl")
    if not enc_path.exists():
        raise SystemExit("Run scripts/build_dataset.py first.")
    df = pd.read_pickle(enc_path)

    codes = np.vstack(df["code"].to_numpy())
    labels = df["labels"].tolist()

    out = Path("outputs/odor_map.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    method = plot_map(codes, labels, str(out))
    print(f"Wrote {out} using {method} on {len(codes)} molecules.")


if __name__ == "__main__":
    main()
