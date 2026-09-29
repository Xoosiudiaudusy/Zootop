"""Rollout noise of flop-state branches (x's flop decisions whose action keeps the flop going): total vs with the
turn and river fixed (the action part)."""
import os, sys, random
import numpy as np
sys.path.insert(0, "/home/user/avbench")
os.environ.setdefault("AVN", "600")
from avenv import *  # noqa
from negpluribus.engine import Street, ActionType, raise_to, CALL, FOLD
K = 64
evs = [FastAivat(game, (4, K, 8), 2000, s, rt) for s in (0, 1)]
rng = random.Random(3)
tot_var, act_var = [], []
def act(t, a):
    return raise_to(a) if t == 2 else (CALL if t == 1 else FOLD)
for h in hands:
    order = list(h["holes"][0]) + list(h["holes"][1]) + list(h["board"])
    order += [c for c in range(52) if c not in order]
    st = spec.new_hand(order, button=h["button"])
    x = h["known_seat"]
    for k, (t, a) in enumerate(h["actions"]):
        if st.is_terminal: break
        if st.current_player == x and st.street == Street.FLOP:
            probe = st.clone(); probe.apply(act(t, a))
            if not probe.is_terminal and probe.street == Street.FLOP:
                v = np.array(evs[0].branch_values(h, k, None, [], 11, list(range(1326))))
                combos = [c for c in np.nonzero(v)[0]][:60]
                if combos:
                    b = [e.branch_values(h, k, None, [], 11, combos) for e in evs]
                    tot_var.append(np.mean((b[1] - b[0]) ** 2 / 2 * K))
                    used = set(h["holes"][0]) | set(h["holes"][1]) | set(h["board"][:3])
                    acc = []
                    for rep in range(5):
                        tr2 = rng.sample([c for c in range(52) if c not in used], 2)
                        cs = [c for c in combos if not ({c // 1 for c in []})]
                        bb = [e.branch_values(h, k, None, tr2, 11, combos) for e in evs]
                        acc.append(np.mean((bb[1] - bb[0]) ** 2 / 2 * K))
                    act_var.append(np.mean(acc))
        st.apply(act(t, a))
    if len(tot_var) >= 30: break
print(f"{len(tot_var)} flop branches; mean rollout variance (chips^2): total {np.mean(tot_var):.3g}, turn and river fixed {np.mean(act_var):.3g}; action share {np.mean(act_var) / np.mean(tot_var):.2f}")
