"""Validation: does the 40-bit nose actually carry smell information?

Two checks, both runnable as ``python -m lilac.validate``:

1. **Held-out odor prediction.** Split the molecules, and for each test molecule
   predict its odor descriptors by majority vote among its k nearest neighbours
   in the training set. We score this for the interpretable 40-bit code and, as a
   ceiling, for a full 2048-bit Morgan fingerprint. The gap tells you how much
   legibility costs in accuracy.

2. **Perceptual sanity.** Flavor signatures that should feel close (lemon/lime)
   ought to sit closer in Hamming distance than ones that shouldn't (lemon/rose).
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from .data import load_dataset
from .sensors import encode_frame
from .signatures import build_signatures
from .similarity import cosine_distance, hamming


def morgan_codes(smiles_list, radius: int = 2, n_bits: int = 2048):
    """Morgan/ECFP fingerprints as a (n, n_bits) uint8 matrix + ok-mask.

    Aligned to `encode_frame`'s convention so the two feature sets cover the same
    molecules.
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    rows, ok = [], np.zeros(len(smiles_list), dtype=bool)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            continue
        ok[i] = True
        rows.append(np.frombuffer(gen.GetFingerprint(mol).ToBitString().encode(),
                                  "u1") - ord("0"))
    codes = np.vstack(rows).astype(np.uint8) if rows else np.empty((0, n_bits), np.uint8)
    return codes, ok


def _knn_predict(train_x, train_y, test_x, k: int, metric: str) -> np.ndarray:
    """Multi-label majority-vote kNN. Returns a 0/1 prediction matrix for test_x."""
    from sklearn.neighbors import NearestNeighbors

    sk_metric = "hamming" if metric == "hamming" else "jaccard"
    nn = NearestNeighbors(n_neighbors=k, metric=sk_metric)
    nn.fit(train_x)
    _, idx = nn.kneighbors(test_x)
    # A label is predicted if it appears in more than half of the k neighbours.
    preds = np.zeros((len(test_x), train_y.shape[1]), dtype=np.uint8)
    for i, neighbours in enumerate(idx):
        votes = train_y[neighbours].sum(axis=0)
        preds[i] = (votes > k / 2).astype(np.uint8)
    return preds


def evaluate_knn(codes, label_matrix, k: int = 5, metric: str = "jaccard",
                 seed: int = 0) -> dict:
    """Train/test split + kNN; report micro/macro F1 over the descriptor set."""
    x_tr, x_te, y_tr, y_te = train_test_split(
        codes, label_matrix, test_size=0.2, random_state=seed)
    preds = _knn_predict(x_tr, y_tr, x_te, k=k, metric=metric)
    return {
        "micro_f1": float(f1_score(y_te, preds, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_te, preds, average="macro", zero_division=0)),
        "samples_f1": float(f1_score(y_te, preds, average="samples", zero_division=0)),
        "n_train": len(x_tr),
        "n_test": len(x_te),
    }


def perceptual_checks(signatures: dict, pairs_close, pairs_far) -> list[dict]:
    """Compare distances of 'should be close' vs 'should be far' flavor pairs.

    Reports both crisp Hamming and soft-signature cosine distance; the cosine
    distance is the sensitive one because it uses on-fractions, not a 0/1 cut.
    """
    results = []
    for kind, pairs in (("close", pairs_close), ("far", pairs_far)):
        for a, b in pairs:
            if a in signatures and b in signatures:
                results.append({
                    "kind": kind, "a": a, "b": b,
                    "hamming": hamming(signatures[a].crisp, signatures[b].crisp),
                    "cosine": cosine_distance(signatures[a].soft, signatures[b].soft),
                })
    return results


def main() -> None:
    df, descriptors = load_dataset()
    smiles = df["smiles"].tolist()
    labels = df["labels"].tolist()

    # 40-bit interpretable codes
    codes40, ok40 = encode_frame(smiles)
    label_matrix = df[descriptors].to_numpy().astype(np.uint8)
    y40 = label_matrix[ok40]

    print(f"Loaded {len(df)} molecules; {ok40.sum()} encoded, "
          f"{(~ok40).sum()} unparseable. {len(descriptors)} descriptors.\n")

    print("== Held-out odor prediction (5-NN) ==")
    res40 = evaluate_knn(codes40, y40, k=5, metric="jaccard")
    print(f"  40-bit nose     micro-F1={res40['micro_f1']:.3f}  "
          f"macro-F1={res40['macro_f1']:.3f}  samples-F1={res40['samples_f1']:.3f}")

    codesM, okM = morgan_codes(smiles)
    yM = label_matrix[okM]
    resM = evaluate_knn(codesM, yM, k=5, metric="jaccard")
    print(f"  Morgan-2048     micro-F1={resM['micro_f1']:.3f}  "
          f"macro-F1={resM['macro_f1']:.3f}  samples-F1={resM['samples_f1']:.3f}")
    print(f"  (baseline uses {codesM.shape[1]} bits vs the nose's {codes40.shape[1]})\n")

    print("== Perceptual sanity (distance between flavor signatures) ==")
    sigs = build_signatures(codes40, [labels[i] for i in np.where(ok40)[0]],
                            descriptors)
    checks = perceptual_checks(
        sigs,
        pairs_close=[("lemon", "lime"), ("apple", "pear"), ("rose", "floral")],
        pairs_far=[("lemon", "rose"), ("meaty", "floral"), ("sulfurous", "fruity")],
    )
    close_cos = [c["cosine"] for c in checks if c["kind"] == "close"]
    far_cos = [c["cosine"] for c in checks if c["kind"] == "far"]
    for c in checks:
        tag = "closer" if c["kind"] == "close" else "farther"
        print(f"  [{tag:7}] {c['a']:10} vs {c['b']:10} -> "
              f"cosine {c['cosine']:.3f}  (Hamming {c['hamming']})")
    if close_cos and far_cos:
        print(f"  mean cosine: close pairs {np.mean(close_cos):.3f} < "
              f"far pairs {np.mean(far_cos):.3f}  "
              f"-> {'PASS' if np.mean(close_cos) < np.mean(far_cos) else 'FAIL'}")


if __name__ == "__main__":
    main()
