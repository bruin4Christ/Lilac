# Lilac — an interpretable olfactory map

Model the nose as a fixed panel of **72 on/off sensors**. A molecule's structural
"facets" (methyl, acetyl, ester, aromatic ring, sulfur, …), its larger scaffolds
(macrocyclic musk, coumarin, indole, terpenoid skeleton, …), *and* its coarse
composition (carbon/oxygen count, chain length, how many methyls) switch on a *subset*
of those sensors, so every molecule gets a compact **bit-code**. Aggregate the codes for
all molecules that smell "lemon", and you get lemon's own signature — the compact
"smell number" for a flavor. (The panel width is a design knob — it started at 40 and
grew to 72 as new tiers were added: larger structures, then a composition tier that
makes the code more *assemblable* back into a structure; see below.)

## Launch the apps

One command builds the three web apps + a hub and serves them locally:

```bash
python scripts/launch.py          # build all + serve at http://127.0.0.1:8000
#   make app                      # same thing, if you prefer make
#   python scripts/launch.py --build-only   # just regenerate the HTML files
```

It opens a **🏠 hub** (`index.html`) that links the three:

- **🍽 Composition studio** — `lilac_compose.html` — grow a whole dish from one base
- **🌸 Pairing explorer** — `lilac_pairings.html` — reinforce / bridge / contrast partners
- **🔬 Molecule inspector** — `lilac_molecules.html` — the molecules inside an ingredient

(First run fetches + caches the datasets, so give it a few seconds.) Every page is
self-contained with no external requests, so they also work opened straight from disk.

---

This is a small, deliberately *legible* take on real olfactory science:

| Idea in Lilac | Established science |
|---|---|
| On/off sensors, a subset fires per molecule | **Combinatorial receptor coding** (Buck & Axel) — ~400 human receptors, the *combination* is the percept |
| Structural facets → bits | **Molecular fingerprints** (Morgan/ECFP) — bits mark substructures |
| Each flavor → its own bit-signature | **Principal Odor Map** (Google/Osmo, *Science* 2023) — a learned map where odors are regions |

The twist Lilac adds: the code is compact (72 bits) and named enough that you can *read*
it — every bit has a name and a reason.

## What it does

1. **Loads real data** — the Leffingwell odor collection (~3,500 aroma molecules, each
   tagged with odor descriptors) for supervised work, plus a **6,300-molecule odorant
   library** unioned across six pyrfume archives (GoodScents, Leffingwell, IFRA, Sigma,
   AromaDb, FlavorNet), all fetched from the public
   [pyrfume-data](https://github.com/pyrfume/pyrfume-data) archive.
2. **Encodes each molecule** into a 72-bit code (`src/lilac/sensors.py`) — 29 SMARTS
   "corner" detectors + 10 larger scaffolds + 11 physicochemical + 6 whole-molecule
   topology + 16 composition (counts / atom budget / chain length) sensors, all via RDKit.
3. **Builds per-flavor signatures** — the characteristic bit pattern of each odor.
4. **Draws the map** — a 2-D layout of all molecules, coloured by odor family.
5. **Validates** — held-out odor prediction (vs. a 2048-bit Morgan baseline) and a
   perceptual sanity check.

## Results (out of the box)

```
Held-out odor prediction (5-NN):        micro-F1   macro-F1
  72-bit nose                             0.397      0.235   <- beats baseline on macro-F1
  Morgan-2048  (28x more bits, opaque)    0.403      0.227
```

The compact, human-readable 72-bit code lands within ~1 micro-F1 point of a full
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
python -m lilac.pairing blueberry --mode bridge       # middle overlap: some shared, some new
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

Caveat — concentrations: the Ahn data carries no proportions, so by default every
compound is weighted equally, which under-counts trace character-impact molecules. Two
levers address this:

- `--weighting specificity` weights each compound by its **inverse ingredient-frequency**
  (a distinctive compound found in few ingredients counts for more than a background one
  found in hundreds) — a coarse *impact* proxy that needs no extra data, e.g. it sharpens
  `coffee` onto its roasted/Maillard cluster.
- When you *do* have real proportions, pass them straight through:
  `build_ingredient_signatures(concentrations={"coffee": {"<smiles>": weight, ...}})`
  overrides the estimate per compound; anything missing falls back to the chosen weighting.

## Composition — building a dish, not just a pair

Pairing answers "what goes with X?"; a recipe is a *composition* problem. `compose`
grows a small ensemble from a base ingredient, greedily adding the partner that
brings the most **new distinctive aroma** while still sharing a **bridge** with
what is already on the plate:

```bash
python -m lilac.compose coffee --size 4      # coffee + popcorn + filbert + ...  (roasted/nutty)
python -m lilac.compose garlic --size 4      # garlic ~ durian, grape brandy, boiled beef (sulfur family)
python -m lilac.compose blueberry --size 5   # fruit + herb/terpene bridges
```

Each addition is scored on four interpretable, tunable forces:

- **complementarity + coherence** — coherence to the palette so far leads (keeping
  the dish connected and base-specific), with a bonus for the single best *new*
  distinctive note. Scoring runs on the smell-carrying sensors only (the 11 gross
  physicochemical descriptors are excluded so broad ingredients don't dominate).
- **bridge** — every pick names the distinctive sensor it joins on ("cocoa *via*
  pyrazine"), and reports the new sensors it introduces.
- **surprise** — a bonus for a bridging pick from a *different culinary category*
  (the food-pairing hypothesis: unexpected foods sharing a key aroma), e.g.
  beef → roasted shrimp *via pyrazine*.
- **diversity** — an adaptive redundancy penalty (`--diversity`, default on): each
  pick is docked for resembling an ingredient already on the plate, so a dish spans
  the palette instead of stacking near-duplicates. It's *base-agnostic* — it breaks
  up a monotone dish (olive's four fermented fruits) yet leaves a coherent
  single-theme dish intact when its partners are mutually distinct (garlic → durian,
  grape brandy, boiled beef — all allium, none a duplicate of another).
- **challenge** — a *non-eliminating* hedonic flag. Ingredients leaning on
  polarizing notes (sulfur, amine, indole…) get a ⚠ warning, and *optionally* a
  gentle ranking nudge (`--challenge-weight`, default **0** = warn only). It never
  filters: a bold pairing (durian, blue cheese, garlic) is always reachable.

```
$ python -m lilac.compose garlic --size 3
Dish built on garlic:
  [base     ] garlic
      ⚠ leans challenging (sulfur, thiazole, phenol)
  [reinforce] durian        via sulfur; adds ester, ether, methoxy; vegetable→fruit leap
      ⚠ leans challenging (sulfur, amine, thiazole)
  [reinforce] boiled_beef   via sulfur; adds pyrazine, ketone, carboxylic_acid; vegetable→meat leap
```

### Composition studio (HTML app)

```bash
python scripts/build_compose_app.py    # -> outputs/lilac_compose.html (self-contained)
```

The composition layer made clickable: pick a base and a full **dish** appears — each
partner as a card showing its role (reinforce / bridge / accent), the distinctive sensor
it bridges on, the new notes it brings, any culinary-category leap, and a ⚠ hedonic
caution (never a filter). A **Harmonious ↔ Adventurous** toggle dials how far the dish
reaches, and a **dish palette** strip shows the combined signature across all sensors.
Click any partner to grow a new dish from it. All 590 ingredients' dishes are precomputed
and embedded.

### Pairing explorer (HTML app)

```bash
python scripts/build_app.py            # -> outputs/lilac_pairings.html (self-contained)
```

Generates a single static page: pick a base ingredient and its **reinforce / bridge /
contrast** lists appear side by side, each row showing the partner's category, an
IDF-weighted similarity meter, and the distinctive sensor that bridges the two
("garlic ~ chive *via sulfur*"). Click any partner to re-center; 🎲 jumps at random. No
external requests — all 590 ingredients' pairings are precomputed and embedded.

### Molecule inspector (HTML app)

```bash
python scripts/build_molecule_widget.py    # -> outputs/lilac_molecules.html
```

The other side of the coin: pick a raw ingredient and see the **molecules inside it**,
each rendered as its 72-bit sensor signature (a colour-coded strip + hex) with its SMILES,
plus the ingredient's **superimposed signature** — each sensor shaded by the fraction of
its molecules that fire it. The 72 sensors are colour-grouped (structural / large scaffold
/ physicochemical / topology / composition). Click a molecule to name its active sensors, or open the
built-in **Bit index** for a plain-English gloss of what all 72 sensors detect.

## Layout

```
src/lilac/
  sensors.py      # THE CORE: encode(smiles) -> uint8[72] + named bit table
  data.py         # fetch the Leffingwell set + union the 6.3k odorant library
  signatures.py   # per-flavor signatures (soft + crisp); bespoke from SMILES
  similarity.py   # hamming / jaccard / cosine, nearest-neighbour lookup
  mapviz.py       # 2-D map (UMAP -> t-SNE -> PCA fallback)
  validate.py     # kNN odor prediction + Morgan baseline + sanity checks
  pairing.py      # flavor pairing: reinforce / bridge / contrast + CLI
  ingredients.py  # 590 real ingredients as superimposed mixtures + IDF pairing CLI
  compose.py      # build a dish: coherence-led ensemble + surprise + hedonic warnings
scripts/          # thin CLI entry points for the steps above
tests/            # known molecules light up the expected sensors
```

## The 72 sensors

29 **structural "corners"** (SMARTS): hydroxyl, primary alcohol, phenol, carboxylic
acid, ester, lactone, aldehyde, ketone, ether, acetal, methyl, gem-dimethyl, benzene
ring, fused aromatic, aliphatic ring, alkene, conjugated diene, terpene/isoprene, amine,
**sulfur**, pyrazine, furan, halogen, acetyl, methoxy, methoxy-phenol, aromatic-N, long
alkyl chain, branched chain.

10 **larger scaffolds** (SMARTS): indole (jasmine/animalic), coumarin (hay/tonka),
benzofuran, quinoline (leathery), thiazole (roasted), thiophene, decalin (woody/ambery),
oxane ring, polyene, **phthalide** (celery/lovage/angelica).

11 **physicochemical** (RDKit descriptors): MW low/high, logP low/high, high TPSA,
flexible, H-bond donor, ≥3 H-bond acceptors, aromatic-rich, multi-ring, has stereocenter.

6 **whole-molecule topology**: macrocycle (musk), macrolactone (musk lactone),
multi-isoprene (terpenoid skeleton), fused-ring system, polycyclic, large scaffold.
These read the molecule at a scale no local "corner" can — e.g. muscone (a 15-membered
macrocyclic musk) now trips `macrocycle`, where before it was indistinguishable from a
small ketone.

16 **composition** (counts / atom budget / chain length): graded longest-carbon-chain
(C2–3 / C4–5 / C6–9 / C10+), group counts (≥2 or ≥3 methyls, ≥2 hydroxyls, ≥2 esters),
carbon budget (≤4 / 5–7 / 8–11 / 12+), and heteroatom budget (≥2/≥3 oxygens, ≥1/≥2
nitrogens). Where the bits above ask *"which motifs are present?"*, these ask *"how many,
and how big?"* — so the code carries a coarse molecular formula and starts to be
*assemblable* back into a structure. Concretely, this is what finally separates the
fruit-ester series (ethyl → isoamyl → n-hexyl acetate), which shared one code before.

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
