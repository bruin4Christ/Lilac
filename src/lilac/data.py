"""Load the odor dataset.

We use the **Leffingwell** collection (~3.5k aroma molecules, each tagged with a
set of odor descriptors), curated by the Pyrfume project. The `pyrfume` Python
package has heavy native dependencies that don't build everywhere, so we fetch
the two CSVs it would have downloaded straight from the public `pyrfume-data`
GitHub archive and cache them locally.

    behavior.csv   Stimulus (CID) + one binary column per odor descriptor
    molecules.csv  CID + IsomericSMILES + names

`load_dataset()` joins them and returns one tidy DataFrame:

    cid | name | smiles | labels (list[str]) | <one 0/1 column per descriptor>
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import pandas as pd

_RAW_BASE = "https://raw.githubusercontent.com/pyrfume/pyrfume-data/main/leffingwell"
_DATA_DIR = Path(os.environ.get("LILAC_DATA_DIR", "data"))

# Non-label columns in behavior.csv / molecules.csv, excluded from the descriptor set.
_ID_COL = "Stimulus"
_MOL_ID_COL = "CID"


def _download(filename: str) -> Path:
    """Fetch a raw CSV into the cache dir if not already present; return its path."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = _DATA_DIR / f"leffingwell_{filename}"
    if not dest.exists():
        url = f"{_RAW_BASE}/{filename}"
        urllib.request.urlretrieve(url, dest)
    return dest


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the raw (behavior, molecules) DataFrames, downloading if needed."""
    behavior = pd.read_csv(_download("behavior.csv"))
    molecules = pd.read_csv(_download("molecules.csv"))
    return behavior, molecules


def descriptor_columns(behavior: pd.DataFrame) -> list[str]:
    """The odor-descriptor label columns (everything but the id column)."""
    return [c for c in behavior.columns if c != _ID_COL]


def load_dataset(cache: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Load and join the dataset.

    Returns (df, descriptors) where `df` has columns
    ``cid, name, smiles, labels`` plus one 0/1 column per descriptor, and
    `descriptors` is the ordered list of descriptor names.

    A joined pickle cache is written to speed up repeated runs; delete
    ``data/leffingwell_joined.pkl`` to force a rebuild.
    """
    cache_path = _DATA_DIR / "leffingwell_joined.pkl"
    if cache and cache_path.exists():
        df = pd.read_pickle(cache_path)
        descriptors = [c for c in df.columns
                       if c not in {"cid", "name", "smiles", "labels"}]
        return df, descriptors

    behavior, molecules = load_raw()
    descriptors = descriptor_columns(behavior)

    mols = molecules.rename(
        columns={_MOL_ID_COL: _ID_COL, "IsomericSMILES": "smiles"}
    )[[_ID_COL, "smiles", "name"]]

    merged = behavior.merge(mols, on=_ID_COL, how="inner")
    merged = merged.dropna(subset=["smiles"])
    merged = merged[merged["smiles"].str.len() > 0].copy()  # defragment before insert

    label_values = merged[descriptors].to_numpy() == 1
    merged["labels"] = [
        [descriptors[j] for j in range(len(descriptors)) if row[j]]
        for row in label_values
    ]

    df = merged.rename(columns={_ID_COL: "cid"})
    ordered = ["cid", "name", "smiles", "labels"] + descriptors
    df = df[ordered].reset_index(drop=True).copy()  # defragment

    if cache:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache_path)
    return df, descriptors
