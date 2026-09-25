"""SearchAgent: blueprint preflop, depth-limited search postflop (Pluribus layout)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..abstraction import EquityBucketer
from ..cfr.game import GameSpec
from ..cfr.search import ContinuationPolicy, SearchConfig, SubgameSolver
from ..cfr.strategy import BlueprintStrategy
from ..engine import Action, Observation, Street
from .blueprint import BlueprintAgent


class SearchAgent(BlueprintAgent):
    name = "search"

    def __init__(
        self,
        spec: GameSpec,
        strategy: BlueprintStrategy,
        bucketer: EquityBucketer,
        config: SearchConfig = SearchConfig(),
        continuations: Optional[ContinuationPolicy] = None,
        name: Optional[str] = None,
        seed: Optional[int] = None,
        search_from: Street = Street.FLOP,
        opponent_model=None,
        models: Optional[Dict[int, object]] = None,
    ):
        """``opponent_model`` (one model for every other seat) or ``models`` (seat -> model) plug the
        exploitation layer's opponent model into the search: ranges and continuation strategies of
        the modelled seats come from the model instead of the blueprint (step 5b).  ``strategy``
        may itself be an exploit strategy (RNR) - it is what we play preflop and read at the root."""
        super().__init__(strategy, bucketer, spec.grid, name=name, seed=seed)
        self.spec = spec
        self.config = config
        self.opponent_model = opponent_model
        self.models: Dict[int, object] = dict(models or {})
        self.solver = SubgameSolver(spec, strategy, bucketer, config, continuations, seed=(seed or 0) + 1,
                                    models=self.models)
        self.search_from = search_from
        self.n_searches = 0
        self.n_search_fallback = 0
        self.last: Optional[Tuple[str, List[str], Optional[List[float]], List[float]]] = None  # (key, legal, blueprint, search)

    def act(self, obs: Observation) -> Action:
        if obs.street < self.search_from:
            return super().act(obs)
        if self._new_hand or self._nonce is None:
            self._nonce = self.rng.getrandbits(32)
            self._new_hand = False
        legal = self.grid.abstract_actions(obs)
        if self.opponent_model is not None:
            others = {s: self.opponent_model for s in range(obs.n_players) if s != obs.seat}
            self.solver.models = others
            self.solver.ranges.models = others
            self.solver.cont.models = others
        probs = self.solver.solve(obs, event_rng=self._event_rng)
        self.n_searches += 1
        self.n_decisions += 1
        bp = self.strategy.policy(self.solver.last_root_key or "", legal)
        if probs is None:
            self.n_search_fallback += 1
            self._new_hand = False
            return super().act(obs) if bp is not None else self.grid.to_concrete(obs, "c")
        self.last = (self.solver.last_root_key or "", legal, bp, probs)
        r = self.rng.random()
        acc = 0.0
        choice = legal[-1]
        for a, p in zip(legal, probs):
            acc += p
            if r < acc:
                choice = a
                break
        return self.grid.to_concrete(obs, choice)
