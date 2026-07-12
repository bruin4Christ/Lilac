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
import re
import urllib.request
from pathlib import Path

import pandas as pd

_PYRFUME_BASE = "https://raw.githubusercontent.com/pyrfume/pyrfume-data/main"
_DATA_DIR = Path(os.environ.get("LILAC_DATA_DIR", "data"))

# Non-label columns in behavior.csv / molecules.csv, excluded from the descriptor set.
_ID_COL = "Stimulus"
_MOL_ID_COL = "CID"

# Pyrfume archives that ship a molecules.csv of the same schema. Every molecule in
# these fragrance/flavor databases is a known odorant, so their union is a large
# "known to smell like something" library even where per-molecule descriptors vary.
LIBRARY_SOURCES = [
    "leffingwell", "goodscents", "ifra_2019", "sigma_2014", "aromadb", "flavornet",
]


def _fetch(archive: str, filename: str) -> Path:
    """Fetch a raw CSV from a pyrfume-data archive into the cache; return its path."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = _DATA_DIR / f"{archive}_{filename}"
    if not dest.exists():
        urllib.request.urlretrieve(f"{_PYRFUME_BASE}/{archive}/{filename}", dest)
    return dest


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the raw Leffingwell (behavior, molecules) DataFrames."""
    behavior = pd.read_csv(_fetch("leffingwell", "behavior.csv"))
    molecules = pd.read_csv(_fetch("leffingwell", "molecules.csv"))
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


def _canonical_smiles(smiles: str) -> str | None:
    """RDKit canonical SMILES for de-duplication, or None if unparseable."""
    from rdkit import Chem  # local import keeps data.py importable without RDKit

    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles else None
    return Chem.MolToSmiles(mol) if mol is not None else None


def load_odorant_library(
    sources: list[str] | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    """Union the molecule lists of several pyrfume archives into one library.

    Returns a DataFrame ``cid | name | smiles | canonical_smiles | sources`` where
    every row is a distinct odorant (deduplicated on canonical SMILES, falling back
    to CID). Molecules carry no odor descriptors here -- the point is coverage: a
    large set of compounds *known to smell like something*, beyond the ~3.5k labelled
    Leffingwell molecules. Merge with ``load_dataset`` when you need the labels.
    """
    sources = sources or LIBRARY_SOURCES
    cache_path = _DATA_DIR / f"odorant_library_{'_'.join(sources)}.pkl"
    if cache and cache_path.exists():
        return pd.read_pickle(cache_path)

    frames = []
    for src in sources:
        try:
            m = pd.read_csv(_fetch(src, "molecules.csv"))
        except Exception as exc:  # a missing/unreachable archive shouldn't sink the rest
            print(f"  ! skipping {src}: {exc}")
            continue
        m = m.rename(columns={_MOL_ID_COL: "cid", "IsomericSMILES": "smiles"})
        m = m[["cid", "smiles", "name"]].dropna(subset=["smiles"])
        m["source"] = src
        frames.append(m)

    lib = pd.concat(frames, ignore_index=True)
    lib["canonical_smiles"] = lib["smiles"].map(_canonical_smiles)
    lib = lib.dropna(subset=["canonical_smiles"])

    # Dedupe on canonical structure; remember every source a molecule came from.
    sources_by_mol = (lib.groupby("canonical_smiles")["source"]
                      .agg(lambda s: ",".join(sorted(set(s)))))
    lib = lib.drop_duplicates(subset="canonical_smiles", keep="first").copy()
    lib["sources"] = lib["canonical_smiles"].map(sources_by_mol)
    lib = lib.drop(columns="source").reset_index(drop=True)

    if cache:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        lib.to_pickle(cache_path)
    return lib


# ---------------------------------------------------------------------------
# Ingredient layer: real culinary ingredients as superpositions of molecules.
# Source: the Ahn et al. 2011 "Flavor Network" (Sci. Rep.) ingredient-compound
# data. Compounds there are named (+ CAS) but carry no structures, so we map
# each to a SMILES by matching its name against the odorant library.
# ---------------------------------------------------------------------------
_FLAVORNET_BASE = "https://raw.githubusercontent.com/lingcheng99/Flavor-Network/master/data"


def _fetch_flavornet(filename: str) -> Path:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = _DATA_DIR / f"ahn_{filename}"
    if not dest.exists():
        urllib.request.urlretrieve(f"{_FLAVORNET_BASE}/{filename}", dest)
    return dest


def _norm_name(s: str) -> str:
    """Loose normalisation for matching compound names across sources."""
    s = str(s).lower().strip()
    s = re.sub(r"\(.*?\)", "", s)                 # drop parenthetical stereo/notes
    s = s.replace("alpha", "a").replace("beta", "b").replace("gamma", "g")
    return re.sub(r"[^a-z0-9]", "", s)            # keep alnum only


def load_flavor_network(min_compounds: int = 3, cache: bool = True) -> pd.DataFrame:
    """Ingredients as lists of constituent-molecule SMILES.

    Returns a DataFrame ``ingredient | category | n_compounds | smiles`` where
    `smiles` is the list of the ingredient's aroma compounds that we could resolve
    to a structure. Only ingredients with at least `min_compounds` resolved
    molecules are kept (fewer than that is too thin to superimpose meaningfully).

    Coverage note: ~65% of the ~1,100 named compounds map to a structure by name,
    and the matched ones are the common aroma molecules, so well-studied
    ingredients retain a representative profile. There are no concentrations, so a
    signature weights every compound equally -- a known simplification.
    """
    cache_path = _DATA_DIR / f"flavor_network_min{min_compounds}.pkl"
    if cache and cache_path.exists():
        return pd.read_pickle(cache_path)

    comp = pd.read_csv(_fetch_flavornet("comp_info.tsv"), sep="\t",
                       names=["id", "name", "cas"], skiprows=1)
    ingr = pd.read_csv(_fetch_flavornet("ingr_info.tsv"), sep="\t",
                       names=["id", "name", "category"], skiprows=1)
    pairs = pd.read_csv(_fetch_flavornet("ingr_comp.tsv"), sep="\t",
                        names=["iid", "cid"], skiprows=1)

    lib = load_odorant_library()
    name_to_smiles = dict(zip(lib["name"].map(_norm_name), lib["smiles"]))
    comp["smiles"] = comp["name"].map(_norm_name).map(name_to_smiles)
    cid_to_smiles = dict(zip(comp["id"], comp["smiles"]))

    pairs["smiles"] = pairs["cid"].map(cid_to_smiles)
    grouped = (pairs.dropna(subset=["smiles"]).groupby("iid")["smiles"]
               .apply(lambda s: sorted(set(s))))

    out = ingr.assign(smiles=ingr["id"].map(grouped)).dropna(subset=["smiles"])
    out["n_compounds"] = out["smiles"].map(len)
    out = out[out["n_compounds"] >= min_compounds]
    out = (out.rename(columns={"name": "ingredient"})
           [["ingredient", "category", "n_compounds", "smiles"]]
           .reset_index(drop=True))

    if cache:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        out.to_pickle(cache_path)
    return out
