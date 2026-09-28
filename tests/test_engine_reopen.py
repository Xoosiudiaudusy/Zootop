"""An all-in raise smaller than a full raise does not reopen the betting for players who already acted (no-limit rules:
a player who has acted and then faces less than a full raise may only call or fold; Robert's Rules / TDA).

M4 (defect hunter, confirmed here in both engines): negpluribus/engine.py HandState.apply and csrc/engine.h reset
``acted`` of every other player on ANY raise, so a short all-in lets the original raiser re-raise.  3-handed, button 0,
stacks 100bb / 100bb / 11bb: the BTN raises to 1000, the SB calls, the BB goes all-in to 1100 (a raise of 100 against
a min-raise of 900): the BTN can then raise to 2000 and more.  Heads-up with equal stacks it never happens; with
carried stacks (web table, mozg matches, depth-grid duels) and in 3-max it does.  Changing it changes the trainer's
game (both engines), so the fix needs a decision; the tests below state the rule.
"""
from __future__ import annotations

import pytest

from negpluribus import fast
from negpluribus.engine import CALL, FOLD, HandState, Street, raise_to

core = fast.core()
M4 = "M4: engine.py / csrc/engine.h reset every player's 'acted' on any raise, so a short all-in reopens the betting"


def _python(stacks, actions):
    h = HandState(list(stacks), 0)
    for a in actions:
        h.apply(a)
    return h


def _cpp(stacks, actions):
    c = core.Hand(list(stacks), 0, 50, 100, 0, list(range(52)), int(Street.RIVER))
    for a in actions:
        c.apply(int(a.type), a.amount)
    return c


SHORT = ([10_000, 10_000, 1100], [raise_to(1000), CALL, raise_to(1100)])  # the BB's all-in: +100, min-raise 900
FULL = ([10_000, 10_000, 3000], [raise_to(1000), CALL, raise_to(3000)])   # the BB's all-in: +2000, a full raise


def test_a_full_all_in_raise_reopens_the_betting():
    h = _python(*FULL)
    obs = h.observe()
    assert obs.seat == 0 and obs.can_raise and obs.min_raise_to == 5000
    if core is not None:
        c = _cpp(*FULL)
        o = c.observe(0)
        assert c.current_player == 0 and o["can_raise"] and o["min_raise_to"] == 5000


@pytest.mark.xfail(strict=True, reason=M4)
def test_a_short_all_in_does_not_reopen_the_betting_python():
    h = _python(*SHORT)
    obs = h.observe()
    assert obs.seat == 0 and obs.to_call == 100
    assert not obs.can_raise and obs.legal_actions() == [FOLD, CALL]
    h.apply(CALL)
    assert not h.observe().can_raise  # the SB, who called 1000, may only call the extra 100 too


@pytest.mark.skipif(core is None, reason="C++ core not built")
@pytest.mark.xfail(strict=True, reason=M4 + " (csrc/engine.h)")
def test_a_short_all_in_does_not_reopen_the_betting_cpp():
    c = _cpp(*SHORT)
    o = c.observe(0)
    assert c.current_player == 0 and o["to_call"] == 100
    assert not o["can_raise"]
