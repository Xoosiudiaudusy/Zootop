"""Equilibrium-finding layer (the future "GTO blueprint").

Roadmap for this package:

* ``kuhn.py``      – vanilla CFR on Kuhn poker: the whole algorithm in ~100 lines,
                     converges in a second, exploitability printed.  Read this first.
* ``strategy.py``  – the ``Strategy`` interface every blueprint must implement so
                     the table/agents don't care whether it is tabular, MCCFR or a net.
* ``game.py``      – ``GameSpec``: the (reduced) game a blueprint is solved for.
* ``mccfr.py``     – external-sampling MCCFR (optionally Linear CFR) on the abstracted game.
* ``exploit.py``   – exact best response / exploitability (2-player preflop games) and the
                     LBR lower bound (games with postflop betting): the ruler for everything else.
* ``search.py``    – depth-limited re-solving with 4 biased continuation strategies
                     (the Pluribus trick); ``ContinuationPolicy.bias_factors`` and
                     ``RangeSampler`` are where opponent stats will plug in.
"""
from .exploit import ClassEquity, exact_exploitability, lbr_lower_bound
from .game import GameSpec
from .mccfr import MCCFRTrainer
from .search import ContinuationPolicy, SearchConfig, SubgameSolver
from .strategy import BlueprintStrategy, Strategy, TabularStrategy

__all__ = [
    "BlueprintStrategy", "ClassEquity", "ContinuationPolicy", "GameSpec", "MCCFRTrainer",
    "SearchConfig", "Strategy", "SubgameSolver", "TabularStrategy",
    "exact_exploitability", "lbr_lower_bound",
]
