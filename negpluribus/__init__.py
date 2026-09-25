"""NegativePluribus: a 6-max NLHE research sandbox.

Layers (bottom-up):

* ``cards`` / ``evaluator``    – primitives, 7-card hand strength
* ``engine``                   – full 6-max No-Limit Hold'em rules (side pots, min-raise, all-ins)
* ``agents``                   – Agent interface + parametrised archetype bots (nit, station, maniac, …)
* ``stats``                    – HUD-style opponent statistics with Bayesian smoothing
* ``eval``                     – duplicate-deal (paired seed) evaluation in bb/100 with confidence intervals
"""

__version__ = "0.1.0"
