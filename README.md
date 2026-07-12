# Lilac — an interpretable olfactory map

Model the nose as a fixed panel of **55 on/off sensors**. A molecule's structural
"facets" (methyl, acetyl, ester, aromatic ring, sulfur, …) *and* its larger scaffolds
(macrocyclic musk, coumarin, indole, terpenoid skeleton, …) switch on a *subset* of
those sensors, so every molecule gets a compact **bit-code**. Aggregate the codes for
all molecules that smell "lemon", and you get lemon's own signature — the compact
"smell number" for a flavor. (The panel width is a design knob — it started at 40 and
grew to 55 when a larger-structure tier was added; see below.)

This is a small, deliberately *legible* take on real olfactory science:

| Idea in Lilac | Established science |
|---|---|
| On/off sensors, a subset fires per molecule | **Combinatorial receptor coding** (Buck & Axel) — ~400 human receptors, the *combination* is the percept |
| Structural facets → bits | **Molecular fingerprints** (Morgan/ECFP) — bits mark substructures |
| Each flavor → its own bit-signature | **Principal Odor Map** (Google/Osmo, *Science* 2023) — a learned map where odors are regions |

The twist Lilac adds: the code is small (55 bits) and named enough that you can *read*
it — every bit has a name and a reason.

## What it does

1. **Loads real data** — the Leffingwell odor collection (~3,500 aroma molecules, each
   tagged with odor descriptors) for supervised work, plus a **6,300-molecule odorant
   library** unioned across six pyrfume archives (GoodScents, Leffingwell, IFRA, Sigma,
   AromaDb, FlavorNet), all fetched from the public
   [pyrfume-data](https://github.com/pyrfume/pyrfume-data) archive.
2. **Encodes each molecule** into a 55-bit code (`src/lilac/sensors.py`) — 29 SMARTS
   "corner" detectors + 9 larger scaffolds + 11 physicochemical + 6 whole-molecule
   topology sensors, all via RDKit.
3. **Builds per-flavor signatures** — the characteristic bit pattern of each odor.
4. **Draws the map** — a 2-D layout of all molecules, coloured by odor family.
5. **Validates** — held-out odor prediction (vs. a 2048-bit Morgan baseline) and a
   perceptual sanity check.

## Results (out of the box)

```
Held-out odor prediction (5-NN):        micro-F1   macro-F1
  55-bit nose                             0.393      0.235   <- beats baseline on macro-F1
  Morgan-2048  (51x more bits, opaque)    0.403      0.227
```

The compact, human-readable 55-bit code lands within ~1 micro-F1 point of a full
2048-bit fingerprint — and *ahead* of it on macro-F1, where the larger-structure
sensors (indole→jasmine, coumarin→hay) help the rare odor classes. Legibility is
nearly free.

Readable signatures fall out directly:

```
lemon     : methyl(100%) + alkene(84%) + terpene_isoprene(76%) + logp_high(64%) + ...
meaty     : sulfur(90%)  + methyl(85%)  + hbond_donor(44%) + ...
vanilla   : methyl(97%)  + benzene_ring(92%) + methoxy_phenol(66%) + ...   # the vanillin skeleton
```

Perceptual sanity (cosine distance between soft signatures):

```
close pairs (apple/pear, rose/floral)          mean 0.032
far   pairs (lemon/rose, meaty/floral, ...)     mean 0.287   -> PASS
```

## Install & run

```bash
pip install -e .            # rdkit, numpy, pandas, scikit-learn, matplotlib
# optional, nicer map layout:  pip install -e ".[map]"   (umap-learn)

python scripts/build_dataset.py       # -> data/molecules_encoded.pkl  (labelled set)
python scripts/build_library.py       # -> data/odorant_library_encoded.pkl  (6.3k odorants)
python scripts/build_signatures.py    # -> outputs/flavor_signatures.csv
python scripts/plot_map.py            # -> outputs/odor_map.png
python -m lilac.validate              # prints prediction + sanity metrics
```

Run the tests with `pytest`.

## Flavor pairing

Signatures make flavor pairing a one-liner. Query by a dataset label, a bespoke
aroma profile, or raw SMILES:

```bash
python -m lilac.pairing blueberry --mode reinforce   # smells alike (shared-compound pairing)
python -m lilac.pairing blueberry --mode bridge       # ~20% overlap: some shared, some new
python -m lilac.pairing blueberry --mode contrast     # pairing by opposition
python -m lilac.pairing "CCOC(=O)C,CC(C)=CCCC(C)(O)C=C" --mode all   # any molecule set
```

`blueberry` isn't a dataset label, so it's defined as a **bespoke signature** from its
character-impact compounds (linalool, ethyl 2-methylbutanoate, (E)-2-hexenal, methyl
cinnamate, …) in `pairing.REFERENCE_AROMAS` — sharper than leaning on a generic `berry`
label. Build your own with `signatures.signature_from_smiles(name, smiles, weights)`.

## Ingredient library

A real ingredient isn't one molecule — it's a *mixture*, so its signature is the
**superposition** of its constituent volatiles on the sensor panel (per bit, the
fraction of the ingredient's compounds that fire it). Lilac builds **590 ingredients**
from the [Ahn et al. *Flavor Network*](https://www.nature.com/articles/srep00196)
ingredient–compound data, mapping each compound to a structure via the odorant library
(~65% of compounds resolve; the matched ones are the common aroma molecules).

```bash
python scripts/build_ingredients.py                     # -> outputs/ingredient_signatures.csv
python -m lilac.ingredients blueberry --mode all        # pair one ingredient vs the other 589
python -m lilac.ingredients coffee   --mode reinforce   # coffee ~ cocoa, roasted peanut, beef
python -m lilac.ingredients garlic   --mode contrast --no-idf
```

Because complex mixtures light up the common bits by default, plain cosine saturates
near 1.0 for any two foods; **IDF weighting** (on by default) down-weights ubiquitous
sensors so distinctive ones (sulfur, pyrazine, macrocycle) drive the match. Results are
culinarily sensible:

```
coffee  ~ cocoa, roasted peanut, peanut butter, roasted beef   (Maillard cluster)
garlic  ~ chive, shallot, onion, cabbage                        (allium / sulfur)
blueberry contrast: goat milk, sour milk, brussels sprout       (fruity vs dairy/savory)
```

Caveat: the data carries no concentrations, so every compound is weighted equally — a
known simplification (trace character-impact compounds are under-counted).

## Layout

```
src/lilac/
  sensors.py      # THE CORE: encode(smiles) -> uint8[55] + named bit table
  data.py         # fetch the Leffingwell set + union the 6.3k odorant library
  signatures.py   # per-flavor signatures (soft + crisp); bespoke from SMILES
  similarity.py   # hamming / jaccard / cosine, nearest-neighbour lookup
  mapviz.py       # 2-D map (UMAP -> t-SNE -> PCA fallback)
  validate.py     # kNN odor prediction + Morgan baseline + sanity checks
  pairing.py      # flavor pairing: reinforce / bridge / contrast + CLI
  ingredients.py  # 590 real ingredients as superimposed mixtures + IDF pairing CLI
scripts/          # thin CLI entry points for the steps above
tests/            # known molecules light up the expected sensors
```

## The 55 sensors

29 **structural "corners"** (SMARTS): hydroxyl, primary alcohol, phenol, carboxylic
acid, ester, lactone, aldehyde, ketone, ether, acetal, methyl, gem-dimethyl, benzene
ring, fused aromatic, aliphatic ring, alkene, conjugated diene, terpene/isoprene, amine,
**sulfur**, pyrazine, furan, halogen, acetyl, methoxy, methoxy-phenol, aromatic-N, long
alkyl chain, branched chain.

9 **larger scaffolds** (SMARTS): indole (jasmine/animalic), coumarin (hay/tonka),
benzofuran, quinoline (leathery), thiazole (roasted), thiophene, decalin (woody/ambery),
oxane ring, polyene.

11 **physicochemical** (RDKit descriptors): MW low/high, logP low/high, high TPSA,
flexible, H-bond donor, ≥3 H-bond acceptors, aromatic-rich, multi-ring, has stereocenter.

6 **whole-molecule topology**: macrocycle (musk), macrolactone (musk lactone),
multi-isoprene (terpenoid skeleton), fused-ring system, polycyclic, large scaffold.
These read the molecule at a scale no local "corner" can — e.g. muscone (a 15-membered
macrocyclic musk) now trips `macrocycle`, where before it was indistinguishable from a
small ketone.

The panel is a starting point, not frozen — it's meant to be tuned against the sanity
checks in `validate.py`.

## Where this goes next (Phase 2)

The interpretable panel above is Phase 1. Phase 2 replaces the hand-picked bits with a
*learned* 40-unit bottleneck (a small net mapping a full fingerprint → odor labels,
squeezed through 40 binary "receptors"), then compares its accuracy and its map against
the interpretable one. Kept as a follow-on so the readable prototype stands on its own
first.

## Data & references

- Leffingwell / Pyrfume — https://pyrfume.org
- Principal Odor Map (Google/Osmo, *Science* 2023) — https://www.science.org/doi/10.1126/science.ade4401
- M2OR receptor↔molecule dataset — https://github.com/chemosim-lab/M2OR
- RDKit — https://www.rdkit.org
