"""StackGridAgent: one agent per stack depth, chosen per hand by the hero's effective stack.

docs/stack_grid_design.md (sections 2, 3 and the review in 6).  A wrapper over any ``Agent``: the
engine, the table, the blueprint and the search are unchanged; what sits at a point (a blueprint
agent, a search agent, an exploit agent) is the ``inner_factory``'s business.

* **When**: on the first decision of a hand (after ``reset`` / ``end_hand``, or when the hole cards or
  the events show a new hand), from the stacks at the start of the hand.  Every later decision of the
  hand goes to the same inner agent: stacks change only between hands.
* **Effective stack** (big blinds): min(own starting stack, max(starting stacks of the opponents not
  folded at that decision)), from ``obs.starting_stacks`` (the stacks before the blinds; our engine
  puts them in every observation), else ``fallback_stack_bb`` for every seat.  Heads-up: min(own,
  other).  With three or more players it is an approximation (docs/stack_depth.md section 6); the two
  alternatives of the review (6.4), min(own, min(opponents)) and min(own, median(opponents)), and the
  points they would pick are reported with the first decision, so a log can compare the rules later
  without replaying.
* **Point**: the manifest's rule (``StackRule``, the manifest's ``kind`` decides): ``boundaries``, an
  inclusive upper depth per point (the first point whose upper depth is >= the stack), or
  ``nearest_log``, the nearest point by ratio (between neighbours p < q the boundary is sqrt(p q), on it
  the deeper point); below the grid ``below_grid``, above it ``above_grid`` (by default the lowest and
  the deepest point); no interpolation between points (review 6.5(d)).  All in exact arithmetic.
* **Inner agents**: made by ``inner_factory(point)`` on the first hand at their point; at most
  ``max_loaded`` are kept, the least recently used evicted (a pot16 point is its blueprint lookup,
  6-234 MB).  ``n_decisions`` / ``n_fallback`` / ``fallback_rate`` add up every inner agent, evicted
  ones included; ``hands_by_point`` / ``loads`` / ``evictions`` say what was played and loaded.
* **Seeds**: ``reset(seed)`` resets every loaded inner agent with that same seed, and an inner agent
  made later is reset with it when it is made.  So the agent that plays a hand starts exactly as the
  point's own agent reset with that seed (bit for bit the single agent at a grid point: acceptance 1 of
  the design), and a hand depends only on (seed, point, observations), not on the points loaded or
  played before.  One inner agent plays a hand, so they need no separate streams (the search agent
  passes its seed to its inner blueprint agent the same way).
* **Logs**: ``decision_info()`` = the inner agent's own (if it has one) plus ``point`` (the point's
  stack in bb) and ``blueprint`` (its file as the manifest writes it) on every decision; the first
  decision of a hand adds ``eff_bb``, ``eff_min_bb``, ``eff_median_bb``, ``point_min`` and
  ``point_median``.  ``end_hand`` goes to the inner agent that played the hand (none when we had no
  decision in it: heads-up, the small blind folded to us).

``StackGrid`` reads a manifest (``data/stack_grid_pot16_s0.json``): the points, the rule, the bet grid
and the shared bucketer; ``blueprint_factory`` / ``search_factory`` make the inner agents (every point
on one bucketer; the search agents also on one C++ bucketer with its cache, ``SearchResources``
sharing), ``point_for_hand`` reads back which point played a logged hand (scripts/aivat_eval.py).
"""
from __future__ import annotations

import json
import os
from collections import Counter, OrderedDict
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..engine import Action, HandRecord, Observation
from .base import Agent

RULE_KINDS = ("nearest_log", "boundaries")


def _num(x) -> Fraction:
    """Exact value of an int, a float (its binary value), a decimal string (a JSON key) or a Fraction."""
    return x if isinstance(x, Fraction) else Fraction(x)


def _plain(x: float):
    """A stack for JSON: 50 rather than 50.0 when whole."""
    f = float(x)
    return int(f) if f.is_integer() else f


# ------------------------------------------------------------------------------------------ rule
@dataclass(frozen=True)
class StackRule:
    """How an effective stack S (bb) picks a point; the thresholds come from the manifest, never from code.

    ``boundaries`` (the manifest's ``"upper": {"<stack_bb>": <upper depth in bb>, ...}``, one entry per
    point; here ``upper[i]`` of point i, points sorted by stack): S goes to the first point whose upper
    depth is >= S (inclusive), each point's upper in [its stack, the next point's stack).
    ``nearest_log``: the nearest point by ratio; between neighbours p < q, S goes to q when S * S >= p * q
    (the boundary sqrt(p q) belongs to the deeper point).
    Either way S below the lowest point gets ``below_grid`` and S above the grid (above the last upper
    depth; for nearest_log above the deepest point) gets ``above_grid``, both points of the grid, by
    default the lowest and the deepest point.  Exact arithmetic (a float in the manifest is its binary value)."""

    kind: str = "nearest_log"
    upper: Tuple[float, ...] = ()
    below_grid: Optional[float] = None
    above_grid: Optional[float] = None

    def validate(self, stacks: Sequence[float]) -> None:
        if self.kind not in RULE_KINDS:
            raise ValueError(f"rule kind must be one of {RULE_KINDS}, got {self.kind!r}")
        for what, v in (("below_grid", self.below_grid), ("above_grid", self.above_grid)):
            if v is not None and all(_num(v) != _num(s) for s in stacks):
                raise ValueError(f"rule {what} {v} is not a point of the grid ({list(stacks)})")
        if self.kind == "nearest_log":
            if self.upper:
                raise ValueError("a nearest_log rule takes no 'upper' depths (kind 'boundaries' does)")
            return
        if len(self.upper) != len(stacks):
            raise ValueError(f"'boundaries' needs an upper depth for each of the {len(stacks)} points, got {len(self.upper)}")
        for i, (p, u) in enumerate(zip(stacks, self.upper)):
            nxt = stacks[i + 1] if i + 1 < len(stacks) else None
            if _num(u) < _num(p) or (nxt is not None and _num(u) >= _num(nxt)):
                raise ValueError(f"the upper depth {u} of the {p}bb point must be in [{p}, {nxt if nxt is not None else 'inf'})")

    def _point(self, stacks: Sequence[float], v: Optional[float], default: int) -> int:
        return default if v is None else [_num(s) for s in stacks].index(_num(v))

    def pick(self, stacks: Sequence[float], eff_bb) -> int:
        """The index of the point (``stacks`` sorted increasing) for an effective stack in bb."""
        x = _num(eff_bb)
        if x < _num(stacks[0]):
            return self._point(stacks, self.below_grid, 0)
        if self.kind == "nearest_log":
            if x > _num(stacks[-1]):
                return self._point(stacks, self.above_grid, len(stacks) - 1)
            i = 0
            while i + 1 < len(stacks) and x * x >= _num(stacks[i]) * _num(stacks[i + 1]):
                i += 1
            return i
        for i, u in enumerate(self.upper):
            if x <= _num(u):
                return i
        return self._point(stacks, self.above_grid, len(stacks) - 1)

    @classmethod
    def from_manifest(cls, d: Optional[dict], stacks: Sequence[float], absent: Sequence[float] = ()) -> "StackRule":
        """The manifest's "rule" for the points at ``stacks`` (present, sorted); entries of ``absent`` points
        (``"present": false``) are left out, so their depths go to the next deeper point."""
        d = dict(d or {})
        kind = d.get("kind", "nearest_log")
        upper: Tuple[float, ...] = ()
        if kind == "boundaries":
            m = d.get("upper")
            if not isinstance(m, dict):
                raise ValueError("rule 'upper' must map every point's stack_bb to its inclusive upper depth in bb")
            by = {_num(k): v for k, v in m.items()}
            missing = [_plain(s) for s in stacks if _num(s) not in by]
            if missing:
                raise ValueError(f"rule 'upper' has no entry for the point(s) {missing} (bb)")
            known = {_num(s) for s in stacks} | {_num(s) for s in absent}
            unknown = [k for k in m if _num(k) not in known]
            if unknown:
                raise ValueError(f"rule 'upper' names depths that are no points of the grid: {unknown}")
            upper = tuple(by[_num(s)] for s in stacks)
        return cls(kind=kind, upper=upper, below_grid=d.get("below_grid"), above_grid=d.get("above_grid"))

    def describe(self, stacks: Sequence[float]) -> str:
        out = self.kind
        if self.upper:
            out += " (upper, inclusive: " + ", ".join(f"{_plain(p)}bb <= {_plain(u)}" for p, u in zip(stacks, self.upper)) + ")"
        below = _plain(stacks[self._point(stacks, self.below_grid, 0)])
        above = _plain(stacks[self._point(stacks, self.above_grid, len(stacks) - 1)])
        return out + f"; below the grid {below}bb, above it {above}bb"


# ------------------------------------------------------------------------------------------ depth
def hand_stacks(obs: Observation, fallback_stack_bb: Optional[float] = None) -> List[int]:
    """Every seat's stack at the start of the hand (before the blinds), in chips."""
    if obs.starting_stacks:
        return list(obs.starting_stacks)
    if fallback_stack_bb is None:
        raise ValueError("the observation has no starting stacks and there is no fallback stack")
    return [int(round(fallback_stack_bb * obs.bb))] * obs.n_players


def depths(stacks: Sequence[int], seat: int, live: Sequence[bool], bb: int) -> Dict[str, Fraction]:
    """Effective stacks of ``seat`` in bb (exact): ``eff`` = min(own, max(live opponents)), the rule's
    input; the review's alternatives ``min`` = min(own, min(live opponents)) and ``median`` = min(own,
    median(live opponents)).  Heads-up all three are min(own, other)."""
    own = stacks[seat]
    opp = sorted(stacks[i] for i in range(len(stacks)) if i != seat and live[i])
    if not opp:  # nobody else in the hand (cannot happen at a decision): our own depth
        opp = [own]
    n = len(opp)
    median = Fraction(opp[(n - 1) // 2] + opp[n // 2], 2)
    return {"eff": Fraction(min(own, opp[-1]), bb), "min": Fraction(min(own, opp[0]), bb),
            "median": min(Fraction(own), median) / bb}


# ------------------------------------------------------------------------------------------ agent
@dataclass(frozen=True)
class GridPoint:
    stack_bb: float                 # the depth the point's agent is for (its blueprint's training depth)
    blueprint: str                  # the blueprint file as the manifest writes it (logged per decision)
    buckets: Optional[str] = None   # its buckets file (a grid shares one bucketer: checked on load)
    trained: str = ""               # free text from the manifest

    @property
    def label(self) -> str:
        return f"{_plain(self.stack_bb)}bb"


class StackGridAgent(Agent):
    """See the module docstring.  ``points``: sorted by stack; ``inner_factory(point) -> Agent``;
    ``rule``: a ``StackRule``; ``max_loaded``: inner agents kept at once (LRU);
    ``fallback_stack_bb``: every seat's stack when an observation has no starting stacks."""

    name = "grid"

    def __init__(self, points: Sequence[GridPoint], inner_factory: Callable[[GridPoint], Agent],
                 rule: Optional[StackRule] = None, max_loaded: int = 4, fallback_stack_bb: Optional[float] = None,
                 name: Optional[str] = None, seed: Optional[int] = None):
        super().__init__(name=name, seed=seed)
        self.points = list(points)
        if not self.points:
            raise ValueError("a stack grid needs at least one point")
        stacks = [p.stack_bb for p in self.points]
        if any(not _num(a) < _num(b) for a, b in zip(stacks, stacks[1:])):
            raise ValueError(f"points must be sorted by stack, each deeper than the last: {stacks}")
        self.rule = rule or StackRule()
        self.rule.validate(stacks)
        if max_loaded < 1:
            raise ValueError("max_loaded must be at least 1")
        self.inner_factory = inner_factory
        self.max_loaded = int(max_loaded)
        self.fallback_stack_bb = fallback_stack_bb
        self.hands_by_point: Counter = Counter()   # stack_bb -> hands played there (with a decision of ours)
        self.loads: Counter = Counter()            # stack_bb -> inner agents made there
        self.evictions = 0
        self._stacks = stacks
        self._seed = seed
        self._inner: "OrderedDict[int, Agent]" = OrderedDict()  # point index -> agent, least recently used first
        self._evicted_decisions = 0
        self._evicted_fallback = 0
        self._last_info: Optional[dict] = None
        self._start_hand()

    # ------------------------------------------------------------------ what the tools read
    def point_for(self, eff_bb) -> GridPoint:
        return self.points[self.rule.pick(self._stacks, eff_bb)]

    @property
    def current_point(self) -> Optional[GridPoint]:
        return None if self._current is None else self.points[self._current]

    @property
    def loaded_points(self) -> List[float]:
        """Stacks of the loaded points, least recently used first."""
        return [self._stacks[i] for i in self._inner]

    def inner_agents(self) -> List[Agent]:
        return list(self._inner.values())

    @property
    def n_decisions(self) -> int:
        return self._evicted_decisions + sum(getattr(a, "n_decisions", 0) for a in self._inner.values())

    @property
    def n_fallback(self) -> int:
        return self._evicted_fallback + sum(getattr(a, "n_fallback", 0) for a in self._inner.values())

    @property
    def fallback_rate(self) -> float:
        n = self.n_decisions
        return self.n_fallback / n if n else 0.0

    def decision_info(self) -> dict:
        return dict(self._last_info) if self._last_info else {}

    def summary(self) -> str:
        hands = ", ".join(f"{_plain(s)}bb {self.hands_by_point[s]:,}" for s in self._stacks if self.hands_by_point[s])
        loads = ", ".join(f"{_plain(s)}bb x{self.loads[s]}" for s in self._stacks if self.loads[s])
        now = ", ".join(f"{_plain(s)}bb" for s in self.loaded_points)
        return (f"hands by point: {hands or 'none'}; loads: {loads or 'none'}; evictions {self.evictions}; "
                f"loaded now (max {self.max_loaded}): {now or 'none'}")

    # ------------------------------------------------------------------ per hand
    def _start_hand(self) -> None:
        self._current: Optional[int] = None
        self._seen: list = []
        self._hole: Tuple[int, ...] = ()
        self._hand_info: Optional[dict] = None

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed)
        self._seed = seed
        for agent in self._inner.values():
            agent.reset(seed)
        self._start_hand()

    def end_hand(self, record: HandRecord, my_seat: int) -> None:
        if self._current is not None and self._current in self._inner:
            self._inner[self._current].end_hand(record, my_seat)
        self._start_hand()

    def _is_new_hand(self, obs: Observation) -> bool:
        """None open (reset / end_hand), other hole cards, or events that do not continue those of our last decision
        (between two decisions of one hand at least our own action is added): robust without reset and end_hand."""
        n = len(self._seen)
        return (self._current is None or tuple(obs.hole) != self._hole or len(obs.events) <= n
                or list(obs.events[:n]) != self._seen)

    def _load(self, i: int) -> Agent:
        agent = self._inner.get(i)
        if agent is not None:
            self._inner.move_to_end(i)
            return agent
        while len(self._inner) >= self.max_loaded:
            _, old = self._inner.popitem(last=False)
            self._evicted_decisions += getattr(old, "n_decisions", 0)
            self._evicted_fallback += getattr(old, "n_fallback", 0)
            self.evictions += 1
        agent = self.inner_factory(self.points[i])
        agent.reset(self._seed)
        self._inner[i] = agent
        self.loads[self._stacks[i]] += 1
        return agent

    def _choose(self, obs: Observation) -> None:
        self._start_hand()
        stacks = hand_stacks(obs, self.fallback_stack_bb)
        d = depths(stacks, obs.seat, [not f for f in obs.folded], obs.bb)
        i = self.rule.pick(self._stacks, d["eff"])
        self._load(i)
        self._current = i
        self._hole = tuple(obs.hole)
        self.hands_by_point[self._stacks[i]] += 1
        self._hand_info = {
            "eff_bb": round(float(d["eff"]), 4), "eff_min_bb": round(float(d["min"]), 4),
            "eff_median_bb": round(float(d["median"]), 4),
            "point_min": _plain(self._stacks[self.rule.pick(self._stacks, d["min"])]),
            "point_median": _plain(self._stacks[self.rule.pick(self._stacks, d["median"])]),
        }

    # ------------------------------------------------------------------ decisions
    def act(self, obs: Observation) -> Action:
        first = self._is_new_hand(obs)
        if first:
            self._choose(obs)
        agent = self._inner[self._current]
        self._seen = list(obs.events)
        action = agent.act(obs)
        info = getattr(agent, "decision_info", None)
        d = dict(info()) if callable(info) else {}
        p = self.points[self._current]
        d.update(point=_plain(p.stack_bb), blueprint=p.blueprint)
        if first:
            d.update(self._hand_info)
        self._last_info = d
        return action


# ------------------------------------------------------------------------------------------ manifest
@dataclass
class StackGrid:
    """A depth grid from its manifest (e.g. data/stack_grid_pot16_s0.json, docs/stack_grid_design.md
    section 3): the points (those marked ``"present": false`` are left out), the rule, the bet grid
    and the shared bucketer.  Paths in the manifest are relative to ``root``: by default the parent of
    the manifest's directory (manifests live in data/ and write "data/...")."""

    points: List[GridPoint]
    rule: StackRule
    bucketer: str
    preflop_fracs: Tuple[float, ...]
    postflop_fracs: Tuple[float, ...]
    max_raises: int
    root: str
    path: Optional[str] = None
    absent: Tuple[float, ...] = ()

    @classmethod
    def load(cls, path: str, root: Optional[str] = None, check_files: bool = True) -> "StackGrid":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        root = root if root is not None else os.path.dirname(os.path.dirname(os.path.abspath(path)))
        return cls.from_dict(d, root, path=path, check_files=check_files)

    @classmethod
    def from_dict(cls, d: dict, root: str, path: Optional[str] = None, check_files: bool = True) -> "StackGrid":
        pts, absent = [], []
        for p in d.get("points", []):
            if not p.get("present", True):
                absent.append(_plain(p["stack_bb"]))
                continue
            pts.append(GridPoint(stack_bb=_plain(p["stack_bb"]), blueprint=p["blueprint"], buckets=p.get("buckets"),
                                 trained=str(p.get("trained", ""))))
        if not pts:
            raise ValueError(f"{path or 'manifest'}: no present points")
        pts.sort(key=lambda p: _num(p.stack_bb))
        stacks = [p.stack_bb for p in pts]
        if any(not _num(a) < _num(b) for a, b in zip(stacks, stacks[1:])):
            raise ValueError(f"{path or 'manifest'}: two points at one stack: {stacks}")
        g = d.get("grid") or {}
        bucketer = d.get("bucketer") or pts[0].buckets
        if not bucketer:
            raise ValueError(f"{path or 'manifest'}: no 'bucketer' and no 'buckets' at the points")
        try:
            rule = StackRule.from_manifest(d.get("rule"), stacks, absent)
            rule.validate(stacks)
        except ValueError as e:
            raise ValueError(f"{path or 'manifest'}: {e}") from None
        grid = cls(points=pts, rule=rule, bucketer=bucketer,
                   preflop_fracs=tuple(float(x) for x in g.get("preflop", (1.0,))),
                   postflop_fracs=tuple(float(x) for x in g.get("postflop", (0.5, 1.0))),
                   max_raises=int(g.get("max_raises", 3)), root=root, path=path, absent=tuple(absent))
        if check_files:
            missing = [f for f in [grid.bucketer] + [p.blueprint for p in pts] if not os.path.exists(grid.resolve(f))]
            if missing:
                raise FileNotFoundError(f"{path or 'manifest'}: missing {', '.join(grid.resolve(f) for f in missing)}")
        return grid

    # ------------------------------------------------------------------ reading
    @property
    def stacks(self) -> List[float]:
        return [p.stack_bb for p in self.points]

    def resolve(self, p: str) -> str:
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(self.root, p))

    def pick(self, eff_bb) -> GridPoint:
        return self.points[self.rule.pick(self.stacks, eff_bb)]

    def point_at(self, stack_bb) -> GridPoint:
        for p in self.points:
            if _num(p.stack_bb) == _num(stack_bb):
                return p
        raise KeyError(f"no point at {stack_bb}bb in the grid ({self.stacks})")

    def find(self, blueprint: str) -> GridPoint:
        """The point of a logged blueprint (its manifest path; else the same file name)."""
        for p in self.points:
            if p.blueprint == blueprint:
                return p
        base = [p for p in self.points if os.path.basename(p.blueprint) == os.path.basename(blueprint)]
        if len(base) == 1:
            return base[0]
        raise KeyError(f"blueprint {blueprint!r} is not a point of the grid ({[p.blueprint for p in self.points]})")

    def point_for_hand(self, rec: dict, default_stack_bb: Optional[float] = None) -> GridPoint:
        """The point that played a hand of a log (scripts/eval_archetypes.py --log-hands, the grid agent as the
        hero): the hero's first decision names it ("blueprint", else "point"); a hand without a decision of the
        hero (heads-up: the small blind folded to it) gets the point the rule gives at its starting stacks
        ("stacks" in chips; else ``default_stack_bb`` for both seats), as the agent would have picked it."""
        hero = rec.get("hero") or []
        if hero:
            first = hero[0]
            if first.get("blueprint"):
                return self.find(first["blueprint"])
            if first.get("point") is not None:
                return self.point_at(first["point"])
            raise KeyError("the hero's first decision names no grid point (was the hero a grid agent?)")
        bb = int(rec.get("bb", 100))
        stacks = rec.get("stacks")
        if not stacks:
            if default_stack_bb is None:
                raise KeyError("a hand without a hero decision and without stacks: give the default stack")
            stacks = [int(round(default_stack_bb * bb))] * len(rec["holes"])
        seat = int(rec["hero_seat"])
        d = depths(stacks, seat, [True] * len(stacks), bb)
        return self.pick(d["eff"])

    def bet_grid(self):
        from ..abstraction import BetGrid

        return BetGrid(preflop_fracs=self.preflop_fracs, postflop_fracs=self.postflop_fracs, allow_all_in=True,
                       max_raises_per_street=self.max_raises)

    def describe(self) -> str:
        pts = ", ".join(p.label for p in self.points)
        absent = f"; absent: {', '.join(f'{_plain(s)}bb' for s in self.absent)}" if self.absent else ""
        return (f"stack grid {os.path.basename(self.path or '')}: points {pts}{absent}; "
                f"rule {self.rule.describe(self.stacks)}; bets preflop {list(self.preflop_fracs)}, "
                f"postflop {list(self.postflop_fracs)}, {self.max_raises} raises; buckets {self.bucketer}")

    def check_blueprint(self, point: GridPoint, blueprint, bucketer_identity: Optional[dict] = None) -> None:
        """A point's blueprint must be of this grid: trained at the point's stack, on the grid's bets and (when
        ``bucketer_identity`` is given) on the shared bucketer.  Blueprints without a stored game (old JSON) pass."""
        ident = getattr(getattr(blueprint, "lookup", None), "identity", None)
        if not ident or "stack_bb" not in ident:
            return
        want = {"stack_bb": point.stack_bb, "preflop_fracs": list(self.preflop_fracs),
                "postflop_fracs": list(self.postflop_fracs), "max_raises_per_street": self.max_raises}
        if bucketer_identity:
            want.update(bucketer_kind=bucketer_identity["kind"], bucketer_n_buckets=bucketer_identity["n_buckets"],
                        bucketer_fingerprint=bucketer_identity["fingerprint"])
        bad = {k: (ident.get(k), v) for k, v in want.items()
               if (_num(ident.get(k)) != _num(v) if k == "stack_bb" else ident.get(k) != v)}
        if bad:
            raise ValueError(f"{point.blueprint} is not the {point.label} point of this grid: " +
                             "; ".join(f"{k} {got} (the grid: {w})" for k, (got, w) in bad.items()))

    # ------------------------------------------------------------------ inner agents
    def blueprint_factory(self, bucketer, grid=None, name: str = "grid", bucketer_identity: Optional[dict] = None
                          ) -> Callable[[GridPoint], Agent]:
        """Makes a ``BlueprintAgent`` per point on one ``bucketer`` (the one the agents use at the table: e.g. a
        C++ bucketer on its bucket table, as scripts/eval_archetypes.py --tables) and ``grid`` (default: the
        manifest's).  Each call loads the point's blueprint afresh (an evicted point frees its memory) and checks it
        against the grid (``check_blueprint``; ``bucketer_identity``: the shared bucketer's C++ identity, default:
        the given bucketer's own when it has one)."""
        from ..fast.blueprint import load_blueprint
        from .blueprint import BlueprintAgent

        bet_grid = grid if grid is not None else self.bet_grid()
        ident = bucketer_identity or getattr(bucketer, "identity", None)

        def make(point: GridPoint) -> Agent:
            bp = load_blueprint(self.resolve(point.blueprint))
            self.check_blueprint(point, bp, ident)
            return BlueprintAgent(bp, bucketer, bet_grid, name=f"{name}_{point.label}")

        return make

    def search_factory(self, spec, bucketer, config=None, name: str = "grid", cache_path: Optional[str] = None,
                       search_bucketer=None, search_tables: Optional[str] = None, search_cache_path: Optional[str] = None,
                       cache_caps=None) -> Callable[[GridPoint], Agent]:
        """Makes a ``CoreSearchAgent`` per point: ``SearchResources`` of the point's blueprint (``spec`` at the point's
        stack), every point on one C++ bucketer with the cache ``cache_path`` and one search bucketer with its tables /
        cache, built with the first point and shared by the later ones (``SearchResources.build(core_bucketer=...)``).
        The factory hook for the next step (search inside the points); ``bucketer``: the Python bucketer."""
        from ..fast.blueprint import load_blueprint
        from .core_search import DEFAULT_CACHE_CAPS, CoreSearchAgent, SearchResources

        shared: Dict[str, object] = {}
        caps = DEFAULT_CACHE_CAPS if cache_caps is None else cache_caps

        def make(point: GridPoint) -> Agent:
            sp = replace(spec, stack_bb=point.stack_bb)
            bp = load_blueprint(self.resolve(point.blueprint), backend="cpp", n_players=sp.n_players)
            if not shared:
                res = SearchResources.build(sp, bucketer, bp, cache_path=cache_path, cache_caps=caps,
                                            search_bucketer=search_bucketer, search_tables=search_tables,
                                            search_cache_path=search_cache_path)
                shared.update(core=res.core_bucketer, search=res.search_core_bucketer)
            else:
                res = SearchResources.build(sp, bucketer, bp, cache_caps=caps, core_bucketer=shared["core"],
                                            search_bucketer=search_bucketer if shared["search"] is not None else None,
                                            search_core_bucketer=shared["search"])
            self.check_blueprint(point, bp, res.core_bucketer.identity)
            return CoreSearchAgent(res, config, name=f"{name}_{point.label}")

        return make

    def agent(self, inner_factory: Callable[[GridPoint], Agent], **kw) -> StackGridAgent:
        return StackGridAgent(self.points, inner_factory, self.rule, **kw)
