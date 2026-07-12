"""Build the ingredient library and its superimposed signatures.

    python scripts/build_ingredients.py

Fetches the Ahn Flavor-Network ingredient-compound data, maps compounds to
structures via the odorant library, builds one superimposed signature per
ingredient, and writes ``outputs/ingredient_signatures.csv``. Prints library
stats and a couple of example pairings.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from lilac.data import load_flavor_network  # noqa: E402
from lilac.ingredients import build_ingredient_signatures, idf_weights  # noqa: E402
from lilac.pairing import rank_pairings  # noqa: E402
from lilac.sensors import active_names, bits_to_string  # noqa: E402


def main() -> None:
    df = load_flavor_network(min_compounds=5)
    print(f"Ingredients (>=5 resolved compounds): {len(df)}")
    print("Top categories:", df["category"].value_counts().head(6).to_dict())
    print(f"Median compounds per ingredient: {int(df['n_compounds'].median())}\n")

    sigs = build_ingredient_signatures(df=df)
    weights = idf_weights(sigs)

    rows = []
    for name, sig in sigs.items():
        rows.append({
            "ingredient": name,
            "n_compounds": sig.n_molecules,
            "signature_bits": bits_to_string(sig.crisp),
            "top_sensors": ", ".join(f"{n}:{f:.2f}" for n, f in sig.top_bits(6)),
        })
    out = Path("outputs/ingredient_signatures.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}\n")

    for q in ["blueberry", "coffee", "garlic"]:
        if q not in sigs:
            continue
        top = rank_pairings(sigs[q], sigs, mode="reinforce", top=5,
                            weights=weights, band_on="cosine")
        print(f"  {q:10} ~ " + ", ".join(f"{r['flavor']}({r['cosine']:.2f})" for r in top))


if __name__ == "__main__":
    main()
