# Lilac — an interpretable olfactory map

Model the nose as a fixed panel of **40 on/off sensors**. A molecule's structural
"facets" (methyl, acetyl, ester, aromatic ring, sulfur, …) switch on a *subset* of
those sensors, so every molecule gets a **40-bit code**. Aggregate the codes for all
molecules that smell "lemon", and you get lemon's own signature — the compact
"40-bit number" for a flavor.

This is a small, deliberately *legible* take on real olfactory science:

| Idea in Lilac | Established science |
|---|---|
| 40 on/off sensors, a subset fires per molecule | **Combinatorial receptor coding** (Buck & Axel) — ~400 human receptors, the *combination* is the percept |
| Structural facets → bits | **Molecular fingerprints** (Morgan/ECFP) — bits mark substructures |
| Each flavor → its own bit-signature | **Principal Odor Map** (Google/Osmo, *Science* 2023) — a learned map where odors are regions |

The twist Lilac adds: the code is small enough (40 bits) and named enough that you can
*read* it — every bit has a name and a reason.

## What it does

1. **Loads real data** — the Leffingwell odor collection (~3,500 aroma molecules, each
   tagged with odor descriptors), fetched from the public
   [pyrfume-data](https://github.com/pyrfume/pyrfume-data) archive.
2. **Encodes each molecule** into a 40-bit code (`src/lilac/sensors.py`) — ~29 SMARTS
   substructure detectors + ~11 physicochemical threshold sensors, all via RDKit.
3. **Builds per-flavor signatures** — the characteristic 40-bit pattern of each odor.
4. **Draws the map** — a 2-D layout of all molecules, coloured by odor family.
5. **Validates** — held-out odor prediction (vs. a 2048-bit Morgan baseline) and a
   perceptual sanity check.

## Results (out of the box)

```
Held-out odor prediction (5-NN):
  40-bit nose    micro-F1 = 0.390
  Morgan-2048    micro-F1 = 0.403     <- 51x more bits, opaque
```

The compact, human-readable 40-bit code lands within ~1 F1 point of a full 2048-bit
fingerprint. Legibility is nearly free.

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

python scripts/build_dataset.py       # -> data/molecules_encoded.pkl
python scripts/build_signatures.py    # -> outputs/flavor_signatures.csv
python scripts/plot_map.py            # -> outputs/odor_map.png
python -m lilac.validate              # prints prediction + sanity metrics
```

Run the tests with `pytest`.

## Layout

```
src/lilac/
  sensors.py      # THE CORE: encode(smiles) -> uint8[40] + named bit table
  data.py         # fetch + join the Leffingwell dataset, cache locally
  signatures.py   # per-flavor 40-bit signatures (soft + crisp)
  similarity.py   # hamming / jaccard / cosine, nearest-neighbour lookup
  mapviz.py       # 2-D map (UMAP -> t-SNE -> PCA fallback)
  validate.py     # kNN odor prediction + Morgan baseline + sanity checks
scripts/          # thin CLI entry points for the steps above
tests/            # known molecules light up the expected sensors
```

## The 40 sensors

29 **structural** (SMARTS): hydroxyl, primary alcohol, phenol, carboxylic acid, ester,
lactone, aldehyde, ketone, ether, acetal, methyl, gem-dimethyl, benzene ring, fused
aromatic, aliphatic ring, alkene, conjugated diene, terpene/isoprene, amine, **sulfur**,
pyrazine, furan, halogen, acetyl, methoxy, methoxy-phenol, aromatic-N, long alkyl chain,
branched chain.

11 **physicochemical** (RDKit descriptors): MW low/high, logP low/high, high TPSA,
flexible, H-bond donor, ≥3 H-bond acceptors, aromatic-rich, multi-ring, has stereocenter.

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
