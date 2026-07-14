"""Lilac -- an interpretable olfactory map.

Molecules become a sensor "nose" (see :mod:`lilac.sensors`), which aggregates into
per-flavor signatures (:mod:`lilac.signatures`) and a 2-D map (:mod:`lilac.mapviz`).
"""

from .sensors import BIT_NAMES, N_BITS, encode, encode_frame

__all__ = ["encode", "encode_frame", "BIT_NAMES", "N_BITS"]
__version__ = "0.1.0"
