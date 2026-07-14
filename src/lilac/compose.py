"""Composition: build a small ensemble of ingredients, not just a pair.

Pairing answers "what goes with X?"; a *recipe* is a composition problem -- a
handful of ingredients chosen together so the whole covers a coherent-but-
interesting slice of the sensor palette. This module grows a dish greedily from a
base ingredient, at each step adding the partner that brings the most **new
distinctive aroma** while still sharing a **bridge** with what is already on the
plate.

Four forces shape each pick (all interpretable, all tunable):

* **complementarity** -- how much *new* distinctive (high-IDF) sensor coverage the
  candidate adds over what the current palette already fires. This is the engine:
  it stops the dish piling onto the same few bits.
* **bridge / coherence** -- the strongest distinctive sensor the candidate *shares*
  with the current palette. A pick with no bridge is a non-sequitur, not a dish;
  every addition names the sensor it joins on ("cocoa *via* pyrazine").
* **surprise** -- a bonus when the candidate comes from a *different culinary
  category* yet still bridges (the food-pairing hypothesis: unexpected foods that
  share a key aroma). This is what makes a suggestion *inspiring* rather than
  obvious.
* **challenge** -- a *soft, non-eliminating* signal that a candidate leans on
  polarizing notes (sulfur, amine, indole...). It is surfaced as a **warning** and
  can *optionally* nudge ranking, but it NEVER removes a candidate. A legitimately
  bold pairing (blue cheese, durian, garlic) must always be reachable. The ranking
  nudge defaults to **off** (weight 0): warnings show, ranking is untouched, until
  you opt in.

    python -m lilac.compose coffee --size 4
    python -m lilac.compose blueberry --size 5 --surprise 1.5
    python -m lilac.compose garlic --size 4 --challenge-weight 0.3   # opt-in nudge
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from .sensors import _COMPOSITION, BIT_NAMES
from .signatures import FlavorSignature

# ---------------------------------------------------------------------------
# Hedonic prior: a rough, hand-set "challenge" weight per sensor -- how much a
# sensor reads as polarizing / acquired-taste in isolation. This is deliberately
# a *prior*, not a truth: sulfur is glorious in garlic and coffee. It exists only
# to raise a flag, never to veto. Sensors not listed contribute 0.
# ---------------------------------------------------------------------------
CHALLENGE_WEIGHTS: dict[str, float] = {
    "sulfur":     1.0,   # alliaceous / rotten in excess
    "amine":      1.0,   # fishy / animalic
    "indole":     0.9,   # fecal / mothball at strength
    "quinoline":  0.7,   # leathery / tarry
    "thiazole":   0.5,   # meaty-roasted, can turn rubbery
    "thiophene":  0.5,   # savoury-sulfurous
    "benzofuran": 0.4,   # smoky / phenolic
    "phenol":     0.3,   # medicinal / smoky
}

# Ubiquitous sensors that carry no bridging information (fire almost everywhere).
_UNINFORMATIVE = {"methyl"}

# The 11 physicochemical descriptor bits read gross properties (size, greasiness,
# polarity), not a smell-carrying substructure. They fire broadly, so counting them
# as "distinctive coverage" rewards big, promiscuous ingredients regardless of the
# base. Composition scores on the *characterful* sensors -- structural corners,
# named scaffolds, whole-molecule topology -- and ignores these for novelty/bridging.
_PHYSICOCHEMICAL = {
    "mw_low", "mw_high", "logp_low", "logp_high", "high_tpsa", "flexible",
    "hbond_donor", "hbond_acceptors", "aromatic_rich", "multi_ring", "has_stereocenter",
}

# The composition bits (counts / atom budget / chain length) sharpen the *code* for
# reconstruction, but "bridges via carbon_5_7" is not a meaningful aroma link, so
# they are likewise excluded from the characterful sensors compose scores on.
_COMPOSITION_BITS = {name for name, _ in _COMPOSITION}


def _character_mask() -> np.ndarray:
    """1.0 for smell-carrying (structural/scaffold/topology) sensors, else 0.0."""
    drop = _PHYSICOCHEMICAL | _COMPOSITION_BITS | _UNINFORMATIVE
    return np.array([0.0 if b in drop else 1.0 for b in BIT_NAMES])


_CHAR_MASK = _character_mask()


@dataclass
class Member:
    """One ingredient in the composed dish (the base is included as the first)."""
    ingredient: str
    role: str                      # base / reinforce / bridge / accent
    thread: float                  # weighted-cosine similarity to the base
    via: str | None                # distinctive sensor bridging it to the palette
    adds: list[str] = field(default_factory=list)   # new distinctive sensors it brings
    category: str = ""
    surprise: bool = False         # different culinary category from the base
    challenge: float = 0.0         # normalized polarizing-note load in [0, ~1]
    warning: str | None = None     # non-eliminating hedonic caution, if any
    rationale: str = ""


@dataclass
class Composition:
    base: str
    members: list[Member]
    palette: np.ndarray            # combined soft signature (per-sensor max)

    def describe(self) -> str:
        lines = [f"Dish built on {self.base}:"]
        for m in self.members:
            head = f"  [{m.role:9}] {m.ingredient}"
            if m.role != "base":
                head += f"  (thread={m.thread:.2f})"
            lines.append(head)
            if m.rationale:
                lines.append(f"      {m.rationale}")
            if m.warning:
                lines.append(f"      ⚠ {m.warning}")
        return "\n".join(lines)


def _challenge_score(soft: np.ndarray) -> tuple[float, list[str]]:
    """Weighted polarizing-note load of a signature + the sensors driving it."""
    total, drivers = 0.0, []
    for name, w in CHALLENGE_WEIGHTS.items():
        v = float(soft[BIT_NAMES.index(name)])
        if v > 0:
            total += w * v
            drivers.append((name, w * v))
    # Normalize so a single fully-firing top-weight sensor (e.g. sulfur) already
    # reads as a strong challenge (~1.0), rather than being diluted by the full
    # panel of weights -- the point is to flag boldness, not to average it away.
    norm = min(1.0, total / (max(CHALLENGE_WEIGHTS.values()) or 1.0))
    drivers.sort(key=lambda t: -t[1])
    return norm, [n for n, _ in drivers[:3]]


def _weighted_cosine(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    aw, bw = a * w, b * w
    na, nb = np.linalg.norm(aw), np.linalg.norm(bw)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(aw, bw) / (na * nb))


def _role(thread: float, base_partner_sims: np.ndarray) -> str:
    """Label a pick by where its base-similarity sits in the base's own spread."""
    p40, p75 = np.percentile(base_partner_sims, [40, 75])
    if thread >= p75:
        return "reinforce"
    if thread >= p40:
        return "bridge"
    return "accent"


def compose(
    base: str,
    sigs: dict[str, FlavorSignature],
    idf: np.ndarray,
    categories: dict[str, str] | None = None,
    size: int = 4,
    surprise_weight: float = 0.25,
    novelty_weight: float = 0.3,
    diversity_weight: float = 0.5,
    challenge_weight: float = 0.0,
    challenge_flag: float = 0.35,
) -> Composition:
    """Greedily grow a dish of `size` ingredients (including the base).

    Parameters
    ----------
    base : ingredient name; must be a key of `sigs`.
    sigs : ingredient signatures (see `ingredients.build_ingredient_signatures`).
    idf  : per-sensor IDF weights (see `ingredients.idf_weights`); distinctive
           sensors count for more, so complementarity/bridges track real character.
    categories : ingredient -> culinary category, for the surprise bonus.
    size : total ingredients in the dish, base included.
    surprise_weight : bonus for a bridging pick from a different category.
    novelty_weight : weight on the (normalized) best new distinctive note. Kept
        secondary to coherence so picks stay base-specific rather than collapsing
        onto whichever ingredients happen to own the rarest sensors.
    diversity_weight : discourage the dish from stacking *near-duplicate* picks --
        adaptively and base-agnostically. Each candidate is penalized by its
        similarity to the most-similar ingredient already on the plate (not to the
        base). This fixes monotone dishes (olive → four near-identical fermented
        fruits) while *preserving* a coherent single-theme dish whose partners are
        mutually distinct (garlic → durian, grape brandy, boiled beef -- all allium,
        none a duplicate of another). Set 0 to disable.
    challenge_weight : OPTIONAL down-nudge for polarizing picks. Default 0 -- the
        hedonic signal only warns and never eliminates. Raise it to gently reorder.
    challenge_flag : challenge score above which a (non-eliminating) warning shows.

    Each step scores candidates on **coherence** (char-weighted cosine to the
    palette so far -- the lead term, keeping the dish connected and base-specific),
    plus a normalized **novelty** bonus and a **surprise** bonus, minus a
    **redundancy** penalty (similarity to the nearest existing pick) and the optional
    challenge nudge. A per-step coherence floor drops near-unrelated picks so a rare
    shared sensor alone can't drag in an incoherent ingredient.
    """
    if base not in sigs:
        raise KeyError(f"{base!r} is not a known ingredient")
    categories = categories or {}
    names = [n for n in sigs if n != base]

    # IDF restricted to characterful sensors: distinctive substructures drive the
    # scoring, gross physicochemical descriptors do not.
    cw = idf * _CHAR_MASK

    chosen_softs: list[np.ndarray] = []    # soft vectors of picks (excl. base), for
                                           # the redundancy (near-duplicate) penalty

    base_sig = sigs[base]
    base_cat = categories.get(base, "")
    # Base-similarity spread, used to label each pick's role.
    base_partner_sims = np.array(
        [_weighted_cosine(base_sig.soft, sigs[n].soft, cw) for n in names])

    covered = base_sig.soft.copy()          # best on-fraction seen per sensor so far
    chosen: set[str] = set()
    bch, bdrivers = _challenge_score(base_sig.soft)
    members = [Member(
        ingredient=base, role="base", thread=1.0, via=None,
        category=base_cat, challenge=bch,
        warning=(f"leans challenging ({', '.join(bdrivers)})"
                 if bch >= challenge_flag else None),
        rationale=f"base — anchors the dish in category '{base_cat or '?'}'",
    )]

    for _ in range(max(0, size - 1)):
        # Pass 1: gather size-invariant metrics for every remaining candidate.
        cand = []
        for n in names:
            if n in chosen:
                continue
            soft = sigs[n].soft
            new_contrib = cw * np.maximum(0.0, soft - covered)    # distinctive novelty
            shared = cw * np.minimum(soft, covered)               # distinctive overlap
            novelty = float(new_contrib.max())                    # best single new note
            bridge_strength = float(shared.max())
            if bridge_strength <= 0 or novelty <= 0:
                continue                                          # no thread, or nothing new
            coherence = _weighted_cosine(soft, covered, cw)       # 0..1, base-specific
            surprise = categories.get(n, "") != base_cat and base_cat != ""
            ch, _ = _challenge_score(soft)
            # Redundancy: closeness to the most-similar pick already on the plate
            # (base excluded). High -> this candidate duplicates an existing pick.
            redundancy = max((_weighted_cosine(soft, ps, cw) for ps in chosen_softs),
                             default=0.0)
            cand.append({"n": n, "new": new_contrib, "shared": shared, "soft": soft,
                         "novelty": novelty, "coherence": coherence,
                         "surprise": surprise, "ch": ch, "redundancy": redundancy})
        if not cand:
            break

        # Coherence floor: keep only the better-connected half so a lone rare shared
        # sensor can't pull in an otherwise-unrelated ingredient.
        coh = np.array([c["coherence"] for c in cand])
        floor = float(np.percentile(coh, 50)) if len(cand) > 2 else -np.inf
        eligible = [c for c in cand if c["coherence"] >= floor] or cand
        nmax = max(c["novelty"] for c in eligible) or 1.0          # normalize novelty

        best, best_score = None, -np.inf
        for c in eligible:
            score = (c["coherence"]
                     + novelty_weight * (c["novelty"] / nmax)
                     + surprise_weight * c["surprise"]
                     - diversity_weight * c["redundancy"]   # penalize near-duplicates
                     - challenge_weight * c["ch"])
            if score > best_score:
                best_score, best = score, c

        n, new_contrib, shared, surprise, ch = (
            best["n"], best["new"], best["shared"], best["surprise"], best["ch"])
        chosen.add(n)
        chosen_softs.append(best["soft"])
        via = BIT_NAMES[int(np.argmax(shared))]
        adds = [BIT_NAMES[i] for i in np.argsort(-new_contrib)
                if new_contrib[i] > 0 and BIT_NAMES[i] != via][:3]
        thread = float(base_partner_sims[names.index(n)])
        _, drivers = _challenge_score(sigs[n].soft)
        cat = categories.get(n, "")
        jump = f"; {base_cat}→{cat} leap" if surprise else ""
        rationale = (f"via {via}; adds {', '.join(adds) or '(texture only)'}{jump}")
        members.append(Member(
            ingredient=n, role=_role(thread, base_partner_sims), thread=thread,
            via=via, adds=adds, category=cat, surprise=bool(surprise), challenge=ch,
            warning=(f"leans challenging ({', '.join(drivers)})"
                     if ch >= challenge_flag else None),
            rationale=rationale,
        ))
        covered = np.maximum(covered, sigs[n].soft)

    return Composition(base=base, members=members, palette=covered)


def main() -> None:
    from .ingredients import build_ingredient_signatures, idf_weights
    from .data import load_flavor_network

    ap = argparse.ArgumentParser(description="Compose a dish of ingredients over Lilac signatures.")
    ap.add_argument("base", help="base ingredient name (e.g. coffee)")
    ap.add_argument("--size", type=int, default=4, help="ingredients in the dish (base included)")
    ap.add_argument("--surprise", type=float, default=0.25, dest="surprise_weight",
                    help="bonus for cross-category bridging picks")
    ap.add_argument("--novelty", type=float, default=0.3, dest="novelty_weight",
                    help="weight on the best new distinctive note (secondary to coherence)")
    ap.add_argument("--diversity", type=float, default=0.5, dest="diversity_weight",
                    help="adaptive penalty for reusing a common bridge sensor (0 = off)")
    ap.add_argument("--challenge-weight", type=float, default=0.0,
                    help="OPTIONAL down-nudge for polarizing picks (0 = warn only, never rank)")
    ap.add_argument("--min-compounds", type=int, default=5)
    ap.add_argument("--weighting", choices=["uniform", "specificity"], default="uniform")
    args = ap.parse_args()

    df = load_flavor_network(min_compounds=args.min_compounds)
    sigs = build_ingredient_signatures(df=df, weighting=args.weighting)
    idf = idf_weights(sigs)
    cats = dict(zip(df["ingredient"], df["category"]))

    if args.base not in sigs:
        raise SystemExit(f"{args.base!r} is not a known ingredient "
                         f"(try one of {', '.join(list(sigs)[:8])}, ...)")

    comp = compose(args.base, sigs, idf, categories=cats, size=args.size,
                   surprise_weight=args.surprise_weight,
                   novelty_weight=args.novelty_weight,
                   diversity_weight=args.diversity_weight,
                   challenge_weight=args.challenge_weight)
    print()
    print(comp.describe())
    print()


if __name__ == "__main__":
    main()
