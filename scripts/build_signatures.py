"""Aggregate the 40-bit codes into one signature per flavor.

    python scripts/build_signatures.py

Reads ``data/molecules_encoded.pkl`` (run build_dataset.py first) and writes
``outputs/flavor_signatures.csv`` -- one row per descriptor with its molecule
count, crisp 40-bit signature, and top sensors. Also prints a few readable
signatures.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lilac.sensors import BIT_NAMES, bits_to_string  # noqa: E402
from lilac.signatures import build_signatures  # noqa: E402


def main() -> None:
    enc_path = Path("data/molecules_encoded.pkl")
    if not enc_path.exists():
        raise SystemExit("Run scripts/build_dataset.py first.")
    df = pd.read_pickle(enc_path)

    codes = np.vstack(df["code"].to_numpy())
    labels = df["labels"].tolist()
    descriptors = [c for c in df.columns
                   if c not in {"cid", "name", "smiles", "labels", "code"}]

    sigs = build_signatures(codes, labels, descriptors, min_molecules=5)

    rows = []
    for desc, sig in sorted(sigs.items(), key=lambda kv: -kv[1].n_molecules):
        rows.append({
            "descriptor": desc,
            "n_molecules": sig.n_molecules,
            "signature_bits": bits_to_string(sig.crisp),
            "top_sensors": ", ".join(f"{n}:{f:.2f}" for n, f in sig.top_bits(6)),
        })
    out = Path("outputs/flavor_signatures.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)

    print(f"Built {len(sigs)} flavor signatures across {len(BIT_NAMES)} sensors.")
    print(f"Wrote {out}\n")
    for desc in ["fruity", "citrus", "lemon", "floral", "meaty", "minty",
                 "sulfurous", "vanilla"]:
        if desc in sigs:
            print("  " + sigs[desc].describe(6))


if __name__ == "__main__":
    main()
