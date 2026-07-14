"""Anchor + Lift -- the pairing model reverse-engineered from the canon.

Comparing ingredients by how much they *overlap* (reinforce / bridge / contrast)
turns out not to predict classic pairings: across a set of timeless pairs, overlap
sits at chance. What *is* consistent is a two-part structure every good pair shares:

* **anchor** -- the strongest distinctive note the two hold *in common* (their
  common ground; this is the old "bridge").
* **lift**   -- the strongest distinctive note *one partner brings that the other
  lacks* (what it *adds*).

The reinforce/contrast split is just the ratio: a **deepener** is mostly anchor with
a small lift (strawberry+vanilla), a **lifter** is a modest anchor with a big lift
(garlic transforming basil; tarragon lifting chicken). A great dish wants one of
each around the base -- a deepener for depth, a lifter for a new dimension (exactly
tomato + tarragon around beef).

A lift only helps if it *belongs*: garlic's sulfur completes savory beef but wrecks a
custard. So each lift is scored for **consonance** with the base's own character -- a
hand-set prior over note families (like a musical interval being consonant or not).

    python -m lilac.affinity beef --mode deepener   # partners that deepen beef
    python -m lilac.affinity beef --mode lifter      # partners that lift it (consonantly)
    python -m lilac.affinity beef --mode dish         # a deepener + a lifter around beef
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from .compose import _CHAR_MASK
from .sensors import BIT_NAMES
from .triangles import note_family

FAMILIES = ["terpene", "sulfur", "roasted/animalic", "fruity/creamy",
            "phenolic/balsamic", "oxygenated", "other"]

# Consonance of adding a LIFT of family R (row) onto a BASE whose character is
# family C (col): 1.0 = completes it, ~0.15 = clashes. A hand-set prior from flavor
# lore -- tunable, like the hedonic weights. The low diagonal captures redundancy
# (adding a terpene lift onto an already-terpene base is the anise-stacking trap).
_C = {
    #                      base: terpene sulfur roast fruity phenol oxygen other
    "terpene":            dict(terpene=.45, sulfur=.85, **{"roasted/animalic":.7,  "fruity/creamy":.7,  "phenolic/balsamic":.6,  "oxygenated":.75, "other":.6}),
    "sulfur":             dict(terpene=.7,  sulfur=.45, **{"roasted/animalic":.9,  "fruity/creamy":.15, "phenolic/balsamic":.55, "oxygenated":.8,  "other":.55}),
    "roasted/animalic":   dict(terpene=.6,  sulfur=.85, **{"roasted/animalic":.5,  "fruity/creamy":.75, "phenolic/balsamic":.75, "oxygenated":.7,  "other":.6}),
    "fruity/creamy":      dict(terpene=.75, sulfur=.6,  **{"roasted/animalic":.75, "fruity/creamy":.5,  "phenolic/balsamic":.7,  "oxygenated":.8,  "other":.65}),
    "phenolic/balsamic":  dict(terpene=.5,  sulfur=.6,  **{"roasted/animalic":.75, "fruity/creamy":.7,  "phenolic/balsamic":.5,  "oxygenated":.65, "other":.6}),
    "oxygenated":         dict(terpene=.65, sulfur=.65, **{"roasted/animalic":.65, "fruity/creamy":.7,  "phenolic/balsamic":.6,  "oxygenated":.55, "other":.6}),
    "other":              dict(terpene=.55, sulfur=.55, **{"roasted/animalic":.55, "fruity/creamy":.6,  "phenolic/balsamic":.55, "oxygenated":.6,  "other":.5}),
}


def consonance(lift_family: str, base_family: str) -> float:
    return _C.get(lift_family, _C["other"]).get(base_family, 0.55)


# Coarser than families: a lift that stays in the base's own register deepens rather
# than lifts (sulfur onto savory beef is just *more savory*); a lift that crosses to a
# new register is a true lift (tarragon's fresh anise onto savory beef).
_REGISTER = {"terpene": "fresh", "sulfur": "savory", "roasted/animalic": "savory",
             "fruity/creamy": "sweet", "phenolic/balsamic": "aromatic",
             "oxygenated": "neutral", "other": "neutral"}


def register(family: str) -> str:
    return _REGISTER.get(family, "neutral")


@dataclass
class Affinity:
    partner: str
    anchor_sensor: str
    anchor_family: str
    anchor: float                 # strength of the shared distinctive note
    lift_sensor: str
    lift_family: str
    lift: float                   # strength of the note the partner ADDS to the base
    consonance: float             # does that lift belong with this base?
    base_family: str = "other"    # the base's own dominant note family
    role: str = ""                # deepener / lifter / complete / weak

    @property
    def crosses_register(self) -> bool:
        return register(self.lift_family) != register(self.base_family)

    @property
    def lift_score(self) -> float:
        """A lift counts to the extent it (a) is strong, (b) *consonantly* completes
        the base, and (c) takes it to a *new register* rather than deepening its own."""
        novelty = 1.0 if self.crosses_register else 0.5
        return self.lift * self.consonance * novelty


def _base_family(base_soft: np.ndarray, cw: np.ndarray) -> str:
    return note_family(BIT_NAMES[int(np.argmax(base_soft * cw))])


def analyze(base: str, partner: str, sigs: dict, idf: np.ndarray) -> Affinity:
    """The anchor + lift (+ consonance) between a base and one partner."""
    cw = idf * _CHAR_MASK
    a, b = sigs[base].soft, sigs[partner].soft
    shared = np.minimum(a, b) * cw                 # common distinctive notes
    ai = int(np.argmax(shared))
    adds = np.maximum(0.0, b - a) * cw             # notes the partner has, base lacks
    li = int(np.argmax(adds))
    lfam = note_family(BIT_NAMES[li])
    bfam = _base_family(a, cw)
    return Affinity(
        partner=partner,
        anchor_sensor=BIT_NAMES[ai], anchor_family=note_family(BIT_NAMES[ai]),
        anchor=float(shared[ai]),
        lift_sensor=BIT_NAMES[li], lift_family=lfam, lift=float(adds[li]),
        consonance=consonance(lfam, bfam), base_family=bfam)


def rank(base: str, sigs: dict, idf: np.ndarray, mode: str = "deepener",
         top: int = 12, max_similarity: float = 0.985) -> list[Affinity]:
    """Rank partners for `base` as deepeners, lifters, or balanced ('dish' scores both).

    mode: ``deepener`` (by anchor), ``lifter`` (by consonant lift), or ``both``
    (anchor x consonant-lift, for partners that do a bit of each).
    """
    if base not in sigs:
        raise KeyError(f"{base!r} is not a known ingredient")
    cw = idf * _CHAR_MASK
    a = sigs[base].soft
    out = []
    for p in sigs:
        if p == base:
            continue
        # skip near-identical partners (a deepener should add depth, not be a clone)
        x, y = a * idf, sigs[p].soft * idf
        if float(x @ y / ((np.linalg.norm(x) * np.linalg.norm(y)) + 1e-9)) >= max_similarity:
            continue
        out.append(analyze(base, p, sigs, idf))

    # relative labels from this base's own spread
    anchors = np.array([f.anchor for f in out]) if out else np.array([0.0])
    lifts = np.array([f.lift_score for f in out]) if out else np.array([0.0])
    a_hi, l_hi = np.percentile(anchors, 70), np.percentile(lifts, 70)
    for f in out:
        strong_a, strong_l = f.anchor >= a_hi, f.lift_score >= l_hi
        f.role = ("complete" if strong_a and strong_l else "deepener" if strong_a
                  else "lifter" if strong_l else "weak")

    key = {"deepener": lambda f: f.anchor,
           "lifter": lambda f: f.lift_score,
           "both": lambda f: f.anchor * f.lift_score}.get(mode)
    if key is None:
        raise ValueError(f"unknown mode {mode!r} (deepener / lifter / both)")
    out.sort(key=key, reverse=True)
    return out[:top]


def suggest_dish(base: str, sigs: dict, idf: np.ndarray,
                 categories: dict[str, str] | None = None) -> dict:
    """A base + its best deepener + its best consonant lifter (different categories)."""
    categories = categories or {}
    deep = rank(base, sigs, idf, mode="deepener", top=8)
    lift = rank(base, sigs, idf, mode="lifter", top=8)
    deepener = deep[0] if deep else None
    # pick the top lifter that isn't the deepener and isn't the same category as it
    lifter = None
    for f in lift:
        if deepener and f.partner == deepener.partner:
            continue
        if deepener and categories.get(f.partner) == categories.get(deepener.partner):
            continue
        lifter = f
        break
    return {"base": base, "deepener": deepener, "lifter": lifter or (lift[0] if lift else None)}


def main() -> None:
    from .data import load_flavor_network
    from .ingredients import build_ingredient_signatures, idf_weights

    ap = argparse.ArgumentParser(description="Anchor + Lift pairing (deepeners, lifters, dishes).")
    ap.add_argument("base", help="base ingredient (e.g. beef)")
    ap.add_argument("--mode", choices=["deepener", "lifter", "both", "dish"], default="dish")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--weighting", choices=["uniform", "specificity"], default="uniform")
    args = ap.parse_args()

    df = load_flavor_network(min_compounds=5)
    sigs = build_ingredient_signatures(df=df, weighting=args.weighting)
    idf = idf_weights(sigs)
    cats = dict(zip(df["ingredient"], df["category"]))
    cap = lambda s: s.replace("_", " ")

    if args.base not in sigs:
        raise SystemExit(f"{args.base!r} is not a known ingredient")

    if args.mode == "dish":
        d = suggest_dish(args.base, sigs, idf, cats)
        print(f"\nA dish around {cap(args.base)}:")
        for role in ("deepener", "lifter"):
            f = d[role]
            if not f:
                continue
            note = (f"deepens on {f.anchor_sensor}" if role == "deepener"
                    else f"lifts with {f.lift_sensor} ({f.lift_family}), consonance {f.consonance:.2f}")
            print(f"  {role:9} {cap(f.partner):20} — {note}")
        print()
        return

    rows = rank(args.base, sigs, idf, mode=args.mode, top=args.top)
    print(f"\n{args.mode.capitalize()}s for {cap(args.base)}:\n")
    for f in rows:
        print(f"  {cap(f.partner):20} [{f.role:8}] anchor {f.anchor_sensor}({f.anchor:.2f})  "
              f"lift {f.lift_sensor}({f.lift:.2f}) ×cons {f.consonance:.2f} = {f.lift_score:.2f}")
    print()


if __name__ == "__main__":
    main()
