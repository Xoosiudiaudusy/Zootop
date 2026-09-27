"""The real-time search agent at the table: the C++ subgame search behind the ``Agent`` interface.

``CoreSearchAgent`` plays as Pluribus does (docs/search_design.md, docs/search_core.md part 3), with the
C++ core of csrc/search.h (``SubgameSearch``); the Python ``SearchAgent`` of ``agents/search.py``
stays the reference and is unchanged.

* **Preflop**: the blueprint (an inner ``BlueprintAgent``).  A search only when an opponent's raise
  of this preflop is far off the grid (``preflop_offgrid``: the relative distance between its raise
  increment and the nearest abstract size at that node), when the blueprint has no strategy at our
  key (it would check/call blind: "off the map"), or when an earlier preflop decision of this hand
  was searched.  A preflop search starts at the start of the hand and stops at the flop
  (continuation leaves, rollouts).
* **From the flop**: a search at every decision, from the start of the current round, the round's
  real actions as the path (ours fixed for our hand, off-grid sizes inserted), depth ``depth``
  ("pluribus": heads-up to the end of the hand; three or more players at the flop: leaves at the
  turn or right after the second raise).
* **Ranges**: Bayes over the blueprint for rounds played without a search; for a searched round,
  the last search's average strategy over the round's real actions (``SubgameSearch.likelihood``,
  each factor at least ``range_floor``), passed as overrides to the later rounds' searches.
* **Play**: the average strategy of our hand at our decision (``play="average"``, the default: as a
  whole profile the final iteration was 4.4 times as exploitable at turn roots, docs/search_core.md),
  or the final iteration (``"final"``, what Pluribus played).

Budget: ``time_budget`` seconds per decision, or per street (``street_budgets``), on ``threads``
threads; ``iterations`` > 0 replaces the clock (tests, reproducible runs).  ``stats`` counts the
decisions by street and kind, the searches (seconds, iterations), why preflop decisions were
searched, and ``off_map``: decisions played by a default (check/call) because nothing had a
strategy there.
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..abstraction import load_bucketer
from ..cards import Deck
from ..cfr.game import GameSpec
from ..engine import CALL, FOLD, Action, ActionType, Event, HandRecord, HandState, Observation, Street, raise_to
from .base import Agent
from .blueprint import BlueprintAgent

DEFAULT_CACHE_CAPS = (8_000_000, 64_000_000, 4_000_000)  # holds a whole precomputed flop + turn cache (0 evictions)
STREET_NAMES = {"preflop": Street.PREFLOP, "flop": Street.FLOP, "turn": Street.TURN, "river": Street.RIVER}


def parse_street_budgets(text: str) -> Dict[int, float]:
    """``"preflop=2,flop=1.5,turn=1,river=0.5"`` -> {street: seconds} (streets not named keep the default)."""
    out: Dict[int, float] = {}
    for part in (p.strip() for p in (text or "").split(",")):
        if not part:
            continue
        name, _, value = part.partition("=")
        name = name.strip().lower()
        if name not in STREET_NAMES or not value:
            raise ValueError(f"street budget {part!r}: expected <preflop|flop|turn|river>=<seconds>")
        out[int(STREET_NAMES[name])] = float(value)
    return out


@dataclass
class SearchConfig:
    time_budget: float = 2.0                       # seconds per decision
    street_budgets: Dict[int, float] = field(default_factory=dict)  # {street: seconds}, overrides time_budget
    iterations: int = 0                            # > 0: this many iterations instead of the clock
    street_iterations: Dict[int, int] = field(default_factory=dict)  # {street: iterations}, overrides iterations
    search_from_street: int = 1                    # rounds after the preflop and before this one: the blueprint
    search_to_street: int = 3                      # rounds after this one: the blueprint
    threads: int = 14
    play: str = "average"                          # "average" or "final"
    depth: str = "pluribus"                        # SubgameSearch depth rule
    preflop_offgrid: float = 0.3                   # search preflop above this relative distance to the grid
    preflop_unknown: bool = True                   # search preflop where the blueprint has no strategy
    search_ranges: bool = True                     # later rounds' ranges from the searched rounds' likelihoods
    range_floor: float = 1e-3                      # floor of each factor of those likelihoods
    rollouts: int = 3
    bias: float = 5.0
    min_prob: float = 1e-3
    focus: float = 0.5

    def __post_init__(self) -> None:
        if self.play not in ("average", "final"):
            raise ValueError(f"play must be 'average' or 'final', got {self.play!r}")
        if self.iterations <= 0 and self.time_budget <= 0 and not self.street_budgets and not self.street_iterations:
            raise ValueError("set time_budget, street_budgets or iterations")
        if not 1 <= int(self.search_from_street) <= 3:
            raise ValueError("search_from_street: 1 (flop), 2 (turn) or 3 (river)")
        if not int(self.search_from_street) <= int(self.search_to_street) <= 3:
            raise ValueError("search_to_street: search_from_street..3 (river)")

    def budget(self, street: int) -> float:
        return float(self.street_budgets.get(int(street), self.time_budget))

    def iters(self, street: int) -> int:
        return int(self.street_iterations.get(int(street), self.iterations))


@dataclass
class SearchResources:
    """What every search agent of one game shares (load once per process): the game spec, the
    Python bucketer (the inner blueprint agent's keys), its C++ twin with the saved bucket cache,
    the C++ blueprint lookup and the ``SearchGame``."""

    spec: GameSpec
    bucketer: object
    core_bucketer: object
    blueprint: object
    game: object
    cache_loaded: Optional[list] = None
    load_seconds: float = 0.0

    @classmethod
    def load(cls, spec: GameSpec, blueprint_path: str, buckets_path: Optional[str] = None, bucketer=None,
             cache_path: Optional[str] = None, cache_caps: Sequence[int] = DEFAULT_CACHE_CAPS,
             presample_seed: Optional[int] = None) -> "SearchResources":
        """From files: the blueprint (binary or JSON; binary loads in about 0.1 s), the buckets JSON (or
        an already loaded ``bucketer``) and the saved bucket cache (scripts/precompute_buckets.py)."""
        from ..fast.blueprint import load_blueprint

        t0 = time.perf_counter()
        bk = bucketer if bucketer is not None else load_bucketer(buckets_path)
        bp = load_blueprint(blueprint_path, backend="cpp", n_players=spec.n_players)
        res = cls.build(spec, bk, bp, cache_path=cache_path, cache_caps=cache_caps, presample_seed=presample_seed)
        res.load_seconds = time.perf_counter() - t0
        return res

    @classmethod
    def build(cls, spec: GameSpec, bucketer, blueprint, cache_path: Optional[str] = None,
              cache_caps: Sequence[int] = DEFAULT_CACHE_CAPS, presample_seed: Optional[int] = None) -> "SearchResources":
        """From loaded objects: a Python bucketer and a ``CppBlueprint`` (``load_blueprint``, or a C++
        trainer's ``blueprint()``)."""
        from ..fast import core
        from ..fast.trainer import core_bucketer, spec_to_dict

        c = core()
        if c is None or not hasattr(c, "SubgameSearch"):
            raise RuntimeError("the C++ core with the subgame search is not built (python scripts/build_fast.py)")
        if not hasattr(blueprint, "lookup"):
            raise TypeError("the search needs the C++ blueprint lookup (load_blueprint(path, backend='cpp'))")
        t0 = time.perf_counter()
        cbk = core_bucketer(bucketer, tuple(cache_caps))
        loaded = cbk.load_cache(cache_path) if cache_path else None
        game = c.SearchGame(spec_to_dict(spec), cbk, blueprint.lookup)
        if presample_seed is not None:
            game.presample(presample_seed)
        return cls(spec, bucketer, cbk, blueprint, game, loaded, time.perf_counter() - t0)


@dataclass
class SearchStats:
    decisions: Counter = field(default_factory=Counter)        # by street
    blueprint: int = 0                                         # preflop decisions the blueprint played
    searches: Counter = field(default_factory=Counter)         # by street
    seconds: Counter = field(default_factory=Counter)          # search seconds by street (build + solve)
    iterations: Counter = field(default_factory=Counter)       # by street
    preflop_reasons: Counter = field(default_factory=Counter)  # why a preflop decision was searched
    inserted: int = 0                                          # searches whose path held an off-grid size
    off_map: int = 0                                           # decisions played by a default without a strategy
    errors: Counter = field(default_factory=Counter)

    @property
    def n_decisions(self) -> int:
        return sum(self.decisions.values())

    @property
    def n_searches(self) -> int:
        return sum(self.searches.values())

    def off_map_rate(self) -> float:
        return self.off_map / self.n_decisions if self.n_decisions else 0.0

    def summary(self) -> str:
        names = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}
        parts = [f"decisions {self.n_decisions:,} (preflop blueprint {self.blueprint:,}); searches {self.n_searches:,}"]
        for s in sorted(self.searches):
            n = self.searches[s]
            parts.append(f"{names.get(s, s)} {n:,} x {self.seconds[s] / n:.2f}s, {self.iterations[s] / n:,.0f} it")
        if self.preflop_reasons:
            parts.append("preflop searched: " + ", ".join(f"{k} {v}" for k, v in sorted(self.preflop_reasons.items())))
        parts.append(f"inserted sizes {self.inserted:,}; off the map {self.off_map:,} ({100 * self.off_map_rate():.2f}%)")
        if self.errors:
            parts.append("errors: " + "; ".join(f"{k} x{v}" for k, v in self.errors.most_common(3)))
        return "; ".join(parts)


def add_search_args(ap) -> None:
    """The search agent's command-line flags (scripts/play_slumbot.py, scripts/eval_archetypes.py)."""
    ap.add_argument("--cache", default=None, help="saved bucket cache (scripts/precompute_buckets.py); without it the "
                    "first searches on a board compute their flop / turn buckets (slow)")
    ap.add_argument("--search-budget", type=float, default=2.0, help="seconds per searched decision")
    ap.add_argument("--search-street-budgets", default="", help='per street, e.g. "flop=3,turn=2,river=1" (others: --search-budget)')
    ap.add_argument("--search-iterations", type=int, default=0, help="a fixed number of iterations per search instead of the clock")
    ap.add_argument("--search-street-iterations", default="",
                    help='fixed iterations per street, e.g. "flop=340000,turn=1150000,river=3600000" (others: --search-iterations)')
    ap.add_argument("--search-from-street", choices=("flop", "turn", "river"), default="flop",
                    help="rounds after the preflop and before this one are played by the blueprint (default flop: search everywhere)")
    ap.add_argument("--search-to-street", choices=("flop", "turn", "river"), default="river",
                    help="rounds after this one are played by the blueprint (default river: search to the end)")
    ap.add_argument("--search-threads", type=int, default=14)
    ap.add_argument("--search-play", choices=("average", "final"), default="average",
                    help="play the average strategy of our hand (default) or the final iteration")
    ap.add_argument("--search-depth", default="pluribus", help="depth rule (pluribus, hu_flop_limit, end, next_street)")
    ap.add_argument("--preflop-offgrid", type=float, default=0.3,
                    help="search preflop when an opponent's raise is farther than this (relative) from every grid size")
    ap.add_argument("--no-preflop-search", action="store_true",
                    help="never search preflop: the blueprint translates every size (unknown keys: its check/call)")
    ap.add_argument("--presample", action="store_true", help="rollouts play pre-sampled blueprint actions (leaves only)")


def search_config_from_args(args) -> "SearchConfig":
    no_pre = getattr(args, "no_preflop_search", False)
    return SearchConfig(time_budget=args.search_budget, street_budgets=parse_street_budgets(args.search_street_budgets),
                        iterations=args.search_iterations,
                        street_iterations={k: int(v) for k, v in parse_street_budgets(args.search_street_iterations).items()},
                        search_from_street={"flop": 1, "turn": 2, "river": 3}[args.search_from_street],
                        search_to_street={"flop": 1, "turn": 2, "river": 3}[args.search_to_street],
                        threads=args.search_threads, play=args.search_play,
                        depth=args.search_depth, preflop_offgrid=float("inf") if no_pre else args.preflop_offgrid,
                        preflop_unknown=not no_pre)


def raise_offgrid_distance(obs: Observation, amount: int, grid) -> float:
    """How far a raise to ``amount`` at the node of ``obs`` is from the grid: the relative distance
    between its raise increment and the nearest abstract raise's increment (the grid's sizes as the
    engine clamps them there, all-in included); 0 on the grid."""
    level = obs.street_bets_max()
    inc = amount - level
    best = float("inf")
    for name in grid.abstract_actions(obs):
        if name in ("f", "c"):
            continue
        a = grid.to_concrete(obs, name)
        if a.type != ActionType.RAISE:
            continue
        g = a.amount - level
        if g > 0:
            best = min(best, abs(inc - g) / g)
    return best


class CoreSearchAgent(Agent):
    """See the module docstring.  ``resources``: ``SearchResources.load(...)`` (shared by all agents of
    the game); ``config``: ``SearchConfig``."""

    name = "search"

    def __init__(self, resources: SearchResources, config: Optional[SearchConfig] = None, name: Optional[str] = None,
                 seed: Optional[int] = None, randomize_translation: bool = True):
        super().__init__(name=name, seed=seed)
        from ..fast import core

        self.core = core()
        self.res = resources
        self.cfg = config or SearchConfig()
        self.spec = resources.spec
        self.grid = resources.spec.grid
        self.blueprint_agent = BlueprintAgent(resources.blueprint, resources.bucketer, self.grid, name=f"{self.name}_bp",
                                              seed=self.rng.getrandbits(32), randomize_translation=randomize_translation)
        self.stats = SearchStats()
        self.last: Optional[dict] = None   # the last search: result dict, reason, street, seconds
        self._last_info: Optional[dict] = None
        self._new_hand()

    # ------------------------------------------------------------------ what the tools read
    @property
    def n_decisions(self) -> int:
        return self.stats.n_decisions

    @property
    def n_fallback(self) -> int:
        """Decisions played by a default (check/call) without a strategy: "off the map"."""
        return self.stats.off_map

    @property
    def fallback_rate(self) -> float:
        return self.stats.off_map_rate()

    @property
    def n_searches(self) -> int:
        return self.stats.n_searches

    def decision_info(self) -> dict:
        """The last decision, for logs (the Slumbot explainer): how it was made and, for a search,
        its actions, our strategy (final iteration and average) and effort."""
        d = self._last_info
        if d is None:
            return {}
        return dict(d)

    # ------------------------------------------------------------------ per hand
    def _new_hand(self) -> None:
        self._searches: Dict[int, object] = {}       # street -> the last SubgameSearch of that round
        self._likelihoods: Dict[int, Dict[int, List[float]]] = {}  # street -> seat -> 1326 likelihoods
        self._preflop_searched = False
        self._seen: List[Event] = []
        self._hole: Tuple[int, ...] = ()

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed)
        # the inner blueprint agent draws exactly what a BlueprintAgent with this seed draws, so its
        # preflop decisions are the same as a blueprint hero's on the same deal (paired comparisons)
        self.blueprint_agent.reset(seed)
        self._new_hand()

    def end_hand(self, record: HandRecord, my_seat: int) -> None:
        self.blueprint_agent.end_hand(record, my_seat)
        self._new_hand()

    def _track(self, obs: Observation) -> None:
        # a new hand when the events seen so far are not a prefix of these (robust without end_hand)
        hole = tuple(obs.hole)
        n = len(self._seen)
        if hole != self._hole or len(obs.events) < n or list(obs.events[:n]) != self._seen:
            self._new_hand()
            self._hole = hole
        self._seen = list(obs.events)

    # ------------------------------------------------------------------ decisions
    def act(self, obs: Observation) -> Action:
        self._track(obs)
        street = int(obs.street)
        self.stats.decisions[street] += 1
        self._last_info = None
        reason = "every decision"
        if street == Street.PREFLOP:
            if self._preflop_searched:
                reason = "after a preflop search"
            else:
                reason = self._preflop_offgrid(obs)
                if reason is None:
                    before = self.blueprint_agent.n_fallback
                    a = self.blueprint_agent.act(obs)
                    if self.blueprint_agent.n_fallback == before:
                        self.stats.blueprint += 1
                        self._last_info = {"played": "blueprint", "off_map": False}
                        return a
                    if not self.cfg.preflop_unknown:
                        self.stats.off_map += 1
                        self._last_info = {"played": "blueprint", "off_map": True}
                        return a
                    reason = "no blueprint strategy"
            self.stats.preflop_reasons[reason] += 1
        elif not int(self.cfg.search_from_street) <= street <= int(self.cfg.search_to_street):  # the blueprint's round
            before = self.blueprint_agent.n_fallback
            a = self.blueprint_agent.act(obs)
            off = self.blueprint_agent.n_fallback != before
            if off:
                self.stats.off_map += 1
            else:
                self.stats.blueprint += 1
            self._last_info = {"played": "blueprint", "off_map": off}
            return a
        try:
            return self._search(obs, reason)
        except Exception as e:  # never stall a match: the blueprint (or check/call) instead
            msg = f"{type(e).__name__}: {str(e)[:100]}"
            self.stats.errors[msg] += 1
            before = self.blueprint_agent.n_fallback
            a = self.blueprint_agent.act(obs)
            off = self.blueprint_agent.n_fallback != before
            if off:
                self.stats.off_map += 1
            self._last_info = {"played": "blueprint after a search error", "error": msg, "off_map": off}
            return a

    def _preflop_offgrid(self, obs: Observation) -> Optional[str]:
        """"off-grid size" when an opponent's raise of this preflop is farther than preflop_offgrid
        from every abstract size at its node (the hand replayed on the engine)."""
        if not any(e.street == Street.PREFLOP and e.seat != obs.seat and e.action.type == ActionType.RAISE for e in obs.events):
            return None
        st = HandState(list(self._starting_stacks(obs)), obs.button, self.spec.sb, self.spec.bb, self.spec.ante,
                       deck=Deck.from_order(list(range(52))), max_street=self.spec.max_street)
        for e in obs.events:
            if e.street != Street.PREFLOP:
                break
            if e.seat != obs.seat and e.action.type == ActionType.RAISE:
                if raise_offgrid_distance(st.observe(e.seat), e.action.amount, self.grid) > self.cfg.preflop_offgrid:
                    return "off-grid size"
            st.apply(e.action)
        return None

    def _starting_stacks(self, obs: Observation) -> List[int]:
        if obs.starting_stacks:
            return list(obs.starting_stacks)
        return [self.spec.stack_bb * self.spec.bb] * obs.n_players

    def _overrides(self, obs: Observation) -> List[tuple]:
        """(street, seat, likelihoods) for every searched earlier round and every live seat."""
        if not self.cfg.search_ranges:
            return []
        out = []
        for street in range(int(obs.street)):
            s = self._searches.get(street)
            if s is None:
                continue
            if street not in self._likelihoods:
                acts = [(int(e.action.type), int(e.action.amount) if e.action.type == ActionType.RAISE else 0)
                        for e in obs.events if int(e.street) == street]
                self._likelihoods[street] = {}
                for seat in range(obs.n_players):
                    if not obs.folded[seat]:
                        self._likelihoods[street][seat] = s.likelihood(seat, acts, floor=self.cfg.range_floor)[0]
            for seat, w in self._likelihoods[street].items():
                if not obs.folded[seat]:
                    out.append((street, seat, w))
        return out

    def _search(self, obs: Observation, reason: str) -> Action:
        street = int(obs.street)
        t0 = time.perf_counter()
        actions = [(int(e.action.type), int(e.action.amount) if e.action.type == ActionType.RAISE else 0) for e in obs.events]
        overrides = self._overrides(obs)
        budget = self.cfg.budget(street)
        iters = self.cfg.iters(street)
        s = self.core.SubgameSearch(
            self.res.game, self._starting_stacks(obs), obs.button, actions, list(obs.board), obs.seat, list(obs.hole),
            iterations=iters, time_budget=0.0 if iters > 0 else budget,
            threads=self.cfg.threads, seed=self.rng.getrandbits(32), focus=self.cfg.focus, min_prob=self.cfg.min_prob,
            linear=True, overrides=overrides or None, depth=self.cfg.depth, rollouts=self.cfg.rollouts, bias=self.cfg.bias)
        r = s.solve()
        probs = r["average"] if self.cfg.play == "average" else r["final"]
        # the uniform draw comes from the inner blueprint agent's stream, where a BlueprintAgent reset
        # with the same seed takes its draw for this decision: common random numbers (the actions are
        # the grid's, in the same order), so a paired comparison with the blueprint agent on the same
        # deals picks the same action wherever the two strategies put the draw in the same place
        x = self.blueprint_agent.rng.random() * sum(probs)
        i, acc = len(probs) - 1, 0.0
        for k, p in enumerate(probs):
            acc += p
            if x < acc:
                i = k
                break
        t, amount = r["types"][i], r["amounts"][i]
        action = raise_to(amount) if t == int(ActionType.RAISE) else (CALL if t == int(ActionType.CALL) else FOLD)
        self._searches[street] = s
        self._likelihoods.pop(street, None)
        if street == Street.PREFLOP:
            self._preflop_searched = True
        dt = time.perf_counter() - t0
        self.stats.searches[street] += 1
        self.stats.seconds[street] += dt
        self.stats.iterations[street] += r["iterations"]
        inserted = [p["amount"] for p in s.path() if p["inserted"]]
        if inserted:
            self.stats.inserted += 1
        self.last = dict(result=r, street=street, reason=reason, seconds=dt, choice=i, overrides=len(overrides),
                         search=s)
        self._last_info = {
            "played": "search", "off_map": False, "reason": reason, "actions": r["actions"],
            "final": [round(p, 4) for p in r["final"]], "average": [round(p, 4) for p in r["average"]],
            "iterations": r["iterations"], "search_s": round(dt, 3), "inserted": inserted,
            "range_overrides": len(overrides),
        }
        return action
