"""Parametrised heuristic bots ("opponent archetypes").

These are *not* meant to be strong.  They exist so that:

1. the exploitative layer has systematically-flawed opponents to learn against
   (StratFormer / AlphaExploitem both train against a pool of such archetypes);
2. evaluation has stable, reproducible baselines;
3. you can feel how a HUD lights up against each style.

All decisions come from two crude strength signals:

* preflop: percentile of the 169-class starting hand (6-max equity table)
* postflop: Monte-Carlo equity vs ``n_active-1`` random hands

and a ``Style`` that says how loose/aggressive/sticky the bot is.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional

from ..engine import CALL, FOLD, Action, Observation, Street
from ..equity import equity_vs_random, hole_percentile
from .base import Agent

# position multipliers on the opening range (6-max)
POS_FACTOR: Dict[str, float] = {
    "UTG": 0.65,
    "HJ": 0.8,
    "CO": 1.0,
    "BTN": 1.35,
    "SB": 0.9,
    "BB": 1.0,
    "BTN/SB": 1.2,
}


@dataclass(frozen=True)
class Style:
    # ---- preflop
    vpip: float = 0.24  # fraction of hands willing to put money in (CO baseline)
    pfr_ratio: float = 0.8  # of the hands played unraised, fraction that open-raise (rest limp)
    three_bet: float = 0.07  # fraction of all hands we 3-bet when facing one raise
    call_vs_raise: float = 0.12  # fraction of all hands we flat a raise with
    continue_vs_3bet: float = 0.06  # fraction of hands we continue (call) vs a 3-bet
    four_bet: float = 0.025  # fraction of hands we 4-bet/shove
    open_size_bb: float = 2.5  # open raise size in bb (+1bb per limper)
    # ---- postflop
    cbet: float = 0.65  # c-bet frequency as preflop aggressor (in addition to value bets)
    value_bet: float = 0.62  # equity above which we bet/raise for value
    raise_threshold: float = 0.78  # equity above which we raise a bet
    bluff: float = 0.08  # random bet frequency when checked to, regardless of equity
    bluff_raise: float = 0.02  # random raise frequency when facing a bet
    fold_threshold: float = 0.42  # facing a bet: fold if equity (vs random hands) is below this…
    size_sensitivity: float = 0.5  # …adjusted by +size_sensitivity*(pot_odds-0.3): bigger bets need more equity
    bet_size: float = 0.66  # bet as fraction of pot
    # ---- misc
    samples: int = 120  # MC samples for postflop equity


class HeuristicAgent(Agent):
    name = "heuristic"

    def __init__(self, style: Style, name: Optional[str] = None, seed: Optional[int] = None):
        super().__init__(name=name, seed=seed)
        self.style = style

    # -------------------------------------------------------------- preflop
    def _preflop(self, obs: Observation) -> Action:
        st = self.style
        pct = hole_percentile(obs.hole[0], obs.hole[1])
        pos = POS_FACTOR.get(obs.position, 1.0)
        r = obs.raises_this_street
        rnd = self.rng.random()

        if r == 0:
            play_range = min(1.0, st.vpip * pos)
            if obs.to_call == 0:  # BB option (or everyone limped and we're BB)
                if pct > 1.0 - play_range * st.pfr_ratio and rnd < 0.8:
                    return self._open(obs)
                return CALL
            if pct > 1.0 - play_range:
                if pct > 1.0 - play_range * st.pfr_ratio:
                    return self._open(obs)
                return CALL  # limp / complete
            return FOLD

        if r == 1:
            if pct > 1.0 - st.four_bet * 0.6 or pct > 1.0 - st.three_bet * min(1.6, pos + 0.3):
                return obs.clamp_raise(obs.max_raise_to if pct > 0.995 and rnd < 0.15 else 3.2 * obs.street_bets_max())
            if pct > 1.0 - st.call_vs_raise * pos or (obs.position == "BB" and pct > 1.0 - st.call_vs_raise * 2.2):
                return CALL
            return FOLD

        # facing a 3-bet or more
        if pct > 1.0 - st.four_bet:
            return obs.clamp_raise(2.4 * obs.street_bets_max() if r == 2 else obs.max_raise_to)
        if pct > 1.0 - st.continue_vs_3bet:
            return CALL
        # cheap all-in call with decent hand
        if obs.to_call <= 0.04 * obs.stack + 1 and pct > 0.8:
            return CALL
        return FOLD

    def _open(self, obs: Observation) -> Action:
        limpers = sum(1 for e in obs.events if e.street == Street.PREFLOP and e.paid > 0 and not e.is_aggressive)
        size = (self.style.open_size_bb + max(0, limpers - (1 if obs.position in ("SB", "BB") else 0))) * obs.bb
        return obs.clamp_raise(size)

    # ------------------------------------------------------------- postflop
    def _postflop(self, obs: Observation) -> Action:
        st = self.style
        n_opp = max(1, obs.n_active - 1)
        eq = equity_vs_random(obs.hole, obs.board, n_opp, st.samples, self.rng)
        rnd = self.rng.random()
        pot_total = obs.pot + obs.to_call

        if obs.to_call == 0:
            i_am_aggressor = obs.aggressor == obs.seat and obs.street == Street.FLOP
            want_bet = (
                eq > st.value_bet
                or rnd < st.bluff
                or (i_am_aggressor and rnd < st.cbet)
            )
            if want_bet and obs.can_raise:
                return obs.clamp_raise(obs.street_bets_max() + st.bet_size * pot_total)
            return CALL

        # facing a bet
        if eq > st.raise_threshold or rnd < st.bluff_raise:
            if obs.can_raise:
                return obs.clamp_raise(obs.street_bets_max() + obs.to_call + st.bet_size * (pot_total + obs.to_call))
            return CALL
        needed = st.fold_threshold + st.size_sensitivity * (obs.pot_odds - 0.3)
        needed += 0.06 * max(0, obs.raises_this_street - 1)  # raises after a bet mean strength
        if eq >= needed:
            return CALL
        return FOLD

    def act(self, obs: Observation) -> Action:
        if obs.street == Street.PREFLOP:
            return self._preflop(obs)
        return self._postflop(obs)


# ----------------------------------------------------------------------------
# Archetype library
# ----------------------------------------------------------------------------
TAG = Style()  # "reasonable regular" baseline

NIT = replace(
    TAG,
    vpip=0.13,
    pfr_ratio=0.85,
    three_bet=0.035,
    call_vs_raise=0.05,
    continue_vs_3bet=0.03,
    four_bet=0.015,
    cbet=0.5,
    value_bet=0.7,
    raise_threshold=0.85,
    bluff=0.01,
    bluff_raise=0.0,
    fold_threshold=0.55,
)

STATION = replace(  # over-caller: never folds, rarely raises
    TAG,
    vpip=0.55,
    pfr_ratio=0.2,
    three_bet=0.02,
    call_vs_raise=0.45,
    continue_vs_3bet=0.3,
    four_bet=0.01,
    cbet=0.25,
    value_bet=0.75,
    raise_threshold=0.9,
    bluff=0.0,
    bluff_raise=0.0,
    fold_threshold=0.12,
    size_sensitivity=0.2,
)

MANIAC = replace(  # hyper-aggressive
    TAG,
    vpip=0.7,
    pfr_ratio=0.95,
    three_bet=0.35,
    call_vs_raise=0.2,
    continue_vs_3bet=0.3,
    four_bet=0.15,
    open_size_bb=3.5,
    cbet=0.9,
    value_bet=0.45,
    raise_threshold=0.6,
    bluff=0.4,
    bluff_raise=0.15,
    fold_threshold=0.3,
    bet_size=0.9,
)

LAG = replace(
    TAG,
    vpip=0.34,
    pfr_ratio=0.9,
    three_bet=0.13,
    call_vs_raise=0.14,
    continue_vs_3bet=0.1,
    four_bet=0.04,
    cbet=0.8,
    value_bet=0.55,
    raise_threshold=0.72,
    bluff=0.2,
    bluff_raise=0.06,
    fold_threshold=0.38,
    bet_size=0.75,
)

PASSIVE = replace(  # loose-passive fish: limps a lot, never raises without the nuts
    TAG,
    vpip=0.45,
    pfr_ratio=0.1,
    three_bet=0.015,
    call_vs_raise=0.3,
    continue_vs_3bet=0.12,
    four_bet=0.01,
    cbet=0.3,
    value_bet=0.85,
    raise_threshold=0.93,
    bluff=0.0,
    bluff_raise=0.0,
    fold_threshold=0.28,
    bet_size=0.5,
)

TIGHT_PASSIVE = replace(
    NIT,
    vpip=0.16,
    pfr_ratio=0.4,
    cbet=0.35,
    value_bet=0.8,
    raise_threshold=0.9,
    fold_threshold=0.5,
    size_sensitivity=0.3,
)

STYLES: Dict[str, Style] = {
    "tag": TAG,
    "nit": NIT,
    "station": STATION,
    "maniac": MANIAC,
    "lag": LAG,
    "passive": PASSIVE,
    "tight_passive": TIGHT_PASSIVE,
}


def make_agent(name: str, seed: Optional[int] = None, label: Optional[str] = None) -> Agent:
    """Factory by archetype name (also accepts 'random' and 'caller')."""
    from .base import CallingAgent, RandomAgent

    key = name.lower()
    if key == "random":
        return RandomAgent(name=label or "random", seed=seed)
    if key == "caller":
        return CallingAgent(name=label or "caller", seed=seed)
    if key not in STYLES:
        raise KeyError(f"unknown archetype {name!r}; choose from {sorted(STYLES) + ['random', 'caller']}")
    return HeuristicAgent(STYLES[key], name=label or key, seed=seed)
