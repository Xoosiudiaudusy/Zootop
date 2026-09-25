"""Abstraction: shrinking No-Limit Hold'em to something CFR can solve.

The real game is far too big to tabulate.  Two independent reductions make it
tractable, and both are what Pluribus (and every solver) does:

* **Card abstraction** (``buckets.py``, ``potential.py``): hands that play the same are
  merged.  Preflop: 169 lossless classes.  Postflop: equity buckets (E[HS]) or
  potential-aware buckets (EMD clusters of the next-street equity histogram).  Suit
  isomorphism (``canonical.py``) is applied first because it is *lossless*.
* **Action abstraction** (``actions.py``): instead of every legal chip amount
  the bot considers a small grid of bet sizes (fractions of the pot + all-in).
  Opponents' off-grid sizes are mapped back with the pseudo-harmonic rule.

``infoset.py`` combines both into the string key that CFR keeps regrets for.
"""
from .actions import ALL_IN, CALL_NAME, FOLD_NAME, BetGrid, pseudo_harmonic
from .buckets import EquityBucketer
from .canonical import canonical_form, canonical_key
from .infoset import history_string, infoset_key, starting_stacks
from .potential import BUCKET_KINDS, PotentialAwareBucketer, bucketer_kind, load_bucketer, make_bucketer

__all__ = [
    "ALL_IN",
    "BUCKET_KINDS",
    "CALL_NAME",
    "FOLD_NAME",
    "BetGrid",
    "EquityBucketer",
    "PotentialAwareBucketer",
    "bucketer_kind",
    "canonical_form",
    "canonical_key",
    "history_string",
    "infoset_key",
    "load_bucketer",
    "make_bucketer",
    "pseudo_harmonic",
    "starting_stacks",
]
