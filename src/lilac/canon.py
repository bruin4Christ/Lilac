"""The reverse-engineering corpus: labelled *good* and *bad* pairings.

These are the ground-truth pairs the anchor+lift / consonance model is tuned and
tested against. `GOOD_PAIRS` are timeless, cross-cuisine combinations; `BAD_PAIRS`
are combinations most cooks would call a clash. Names are dataset ingredients
(so chocolate → ``cocoa``, fish → ``salmon``/``tuna``).

Grow these lists as you meet more tried-and-true (or genuinely awful) combinations
-- `tests/test_affinity.py` asserts the model keeps the two sets apart, so the
corpus doubles as a regression guard. Analysis lives in the affinity docstring;
the headline finding: overlap doesn't predict pairings, but a *lift* that clashes
with any register the base strongly holds does predict a bad one.
"""

from __future__ import annotations

GOOD_PAIRS: list[tuple[str, str]] = [
    ("tomato", "basil"), ("beef", "mushroom"), ("chicken", "tarragon"),
    ("lamb", "rosemary"), ("cocoa", "hazelnut"), ("strawberry", "vanilla"),
    ("pork", "apple"), ("salmon", "dill"), ("carrot", "ginger"),
    ("coffee", "vanilla"), ("apple", "cinnamon"), ("blue_cheese", "pear"),
    ("cocoa", "orange"), ("potato", "rosemary"), ("peach", "almond"),
    ("cherry", "almond"), ("tomato", "garlic"), ("pea", "mint"),
    ("cucumber", "dill"), ("lamb", "mint"), ("pork", "sage"),
    ("chicken", "lemon"), ("beef", "onion"), ("mushroom", "thyme"),
    ("coconut", "lime"), ("cocoa", "coffee"), ("cardamom", "coffee"),
    ("fig", "honey"), ("pear", "cinnamon"), ("banana", "peanut"),
    ("ginger", "honey"), ("orange", "almond"), ("rosemary", "lemon"),
    ("basil", "garlic"), ("beef", "red_wine"),
]

BAD_PAIRS: list[tuple[str, str]] = [
    ("garlic", "cocoa"), ("garlic", "strawberry"), ("garlic", "peach"),
    ("garlic", "vanilla"), ("garlic", "banana"), ("onion", "banana"),
    ("onion", "strawberry"), ("onion", "peach"), ("onion", "cocoa"),
    ("onion", "vanilla"), ("salmon", "cocoa"), ("salmon", "strawberry"),
    ("salmon", "vanilla"), ("salmon", "caramel"), ("tuna", "banana"),
    ("tuna", "vanilla"), ("coffee", "garlic"), ("broccoli", "cocoa"),
    ("cabbage", "vanilla"), ("cabbage", "cocoa"), ("cauliflower", "vanilla"),
    ("beef", "banana"), ("beef", "strawberry"), ("shrimp", "vanilla"),
    ("shrimp", "cocoa"), ("mushroom", "strawberry"), ("mustard", "strawberry"),
    ("blue_cheese", "banana"), ("sauerkraut", "cocoa"), ("horseradish", "cocoa"),
]
