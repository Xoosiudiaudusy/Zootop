import os, sys
import numpy as np
sys.path.insert(0, "/home/user/avbench")
from avenv import *  # noqa
K = int(os.environ.get("AVK", "2000"))
ex = FastAivat(game, (4, 8, 8), 2000, 0, rt, turn_exact=True)
v1 = FastAivat(game, (4, 8, 8), 2000, 0, rt)
mc0 = FastAivat(game, (4, 8, K), 2000, 0, rt)
mc1 = FastAivat(game, (4, 8, K), 2000, 1, rt)
de, dm = [], []
nv = 0
for h in hands:
    a = ex.evaluate(h, True)["trace"]; b0 = mc0.evaluate(h, True)["trace"]; b1 = mc1.evaluate(h, True)["trace"]; base = v1.evaluate(h, True)["trace"]
    for (na, va), (_, v0), (_, vv1), (_, vb) in zip(a, b0, b1, base):
        va, v0, vv1, vb = map(np.array, (va, v0, vv1, vb))
        if np.array_equal(va, vb):  # not a turn-exact vector
            continue
        if na.endswith(":before") and not np.array_equal(v0, vv1) and np.array_equal(va, vb): continue
        m = va != 0
        # turn-exact vectors: the exact values differ from v1 (8 rollouts); flop-closing 'before' ones use rollouts in both
        if not np.allclose(va[m], va[m]): continue
        if (na.endswith(":before")): continue
        nv += 1
        de.append((v0 - va)[m]); dm.append((v1 - v0)[m] / np.sqrt(2)) if False else dm.append((vv1 - v0)[m] / np.sqrt(2))
de = np.concatenate(de); dm = np.concatenate(dm)
print(f"{nv} turn-exact vectors, {len(de)} combo values")
print(f"MC({K}) - exact: mean {de.mean():+.3f} (se {de.std() / np.sqrt(len(de)):.3f}), sd {de.std():.2f}")
print(f"MC noise alone (two seeds / sqrt2): mean {dm.mean():+.3f}, sd {dm.std():.2f}")
print(f"ratio of sds (1 = exact is the rollouts' mean): {de.std() / dm.std():.3f}")
