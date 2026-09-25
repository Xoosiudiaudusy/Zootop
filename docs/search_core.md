# Real-time search in C++ (2026-09-25): part 1, the subgame core; part 2, depth limits and leaves

The design is `docs/search_design.md` (Pluribus, checked against the supplement) with
`docs/search_research.md` sections 1, 2 and 7.  Part 1 builds the subgame and its solver in
C++ (`csrc/search.h`, bindings in `csrc/bindings.cpp`); part 2 (below, from "Part 2") adds the
depth rules, continuation leaves with rollouts, warm bucket caches and the exact evaluator for turn
roots; the agent is part 3.  The Python search (`negpluribus/cfr/search.py`, `agents/search.py`)
stays as the reference and is unchanged.

## What a search is

```python
from negpluribus import fast
from negpluribus.fast.blueprint import load_blueprint
from negpluribus.fast.trainer import core_bucketer, spec_to_dict
core = fast.core()

bp = load_blueprint("data/blueprint_hunl200w3_pot16_s0.json")          # the C++ lookup
game = core.SearchGame(spec_to_dict(spec), core_bucketer(bucketer), bp.lookup)   # once per match
s = core.SubgameSearch(game, stacks, button, actions, board, seat, hole,          # once per decision
                       time_budget=2.0, threads=15, iterations=0, seed=0, focus=0.5, min_prob=1e-3)
r = s.solve()   # {"actions": [...], "final": [...], "average": [...], "iterations", "seconds", ...}
```

`actions` are the real actions of the hand so far as `(type, raise-to amount)` (0 fold, 1 check/
call, 2 raise), `board` the real board so far, `seat` / `hole` ours; it must be our turn.

* **Root**: the public state at the start of the current betting round, rebuilt by replaying the
  real actions from the start of the hand: exact pot, stacks and bets whatever sizes were used
  (`s.root_info()`).  The actions of the current round are the real path (`s.path()`).
* **Ranges**: for every seat still in the hand, weights over all 1326 hole combos: the product
  over that seat's actions before the root of `max(sigma(action), min_prob)`, sigma = the
  blueprint as `BlueprintStrategy.policy` gives it at the key the blueprint agent would build
  (bucket of the combo on that street's board, history translated deterministically), uniform
  when the key is unknown; combos on the board get 0 (`s.ranges()`; `conditioned=True` also
  removes our cards from the opponents' ranges, for display).  `min_prob` (default 1e-3, as in the
  Python reference) keeps a range alive when a real opponent does what the blueprint never does.
  For any (street, seat) before the root the caller can pass per-combo likelihoods instead
  (`overrides=[(street, seat, [1326 floats])]`), e.g. from the previous round's search:
  `s.likelihood(seat)` gives, per combo, the probability of that seat's actions of this round under
  the search's average strategy.
* **Actions**: at every node the blueprint grid (+ all-in).  Each real action of this round that
  the grid does not produce at its node (another size) is **inserted** there as one more action,
  named `x<amount>` in `path()`; a new off-grid action means a new `SubgameSearch` from the same
  root (the root and the ranges are the same; `test_an_off_grid_action_is_inserted...`).  Limit:
  8 actions per node (`MAX_ACTIONS`), so a grid with 5 raise fractions has no room for an insertion
  (a clear error); the HU grids have 4 postflop and 3 preflop fractions.
* **Fixed**: our actions already taken in this round are forced for our actual hole only (exact
  cards); our other holes and every opponent decide freely.
* **Cards**: the current round is lossless: one infoset per canonical form of hole + board (the
  169 classes preflop); later rounds use the blueprint's buckets.
* **Solver**: Linear external-sampling MCCFR on `threads` threads to the end of the hand, stopping
  at `iterations` or `time_budget` seconds.  A deal draws every live hole independently from its
  range and redraws on a shared card (exactly the joint distribution); the rest of the board is
  uniform over the unseen cards.  In a share `focus` of our traversals our hole is our actual hole
  (the others drawn conditionally) and the opponents' real actions of this round are followed,
  the traversal weighted by their current probabilities of those actions: unbiased, and our
  current infoset is updated in every such traversal instead of only when sampled opponents
  happen to repeat the real path.  Average strategies are accumulated in the other traversals.
  `solve()` returns, at our current decision and for our actual hole, the strategy of the final
  iteration (`final`, what Pluribus plays) and the average (`average`); the node table stays for
  `likelihood()`.

## Checks (all measured 2026-09-25)

**Unit tests** (`tests/test_search_core.py`, 10 tests, with the key test mode on):

| test | against |
|---|---|
| root = start of the round | the Python engine replaying the same actions: pot, stacks, bets, invested, folded, all-in, to act, active players, events (24 random hands, 2 and 3 players, random off-grid sizes) |
| ranges | a brute-force Python loop over the 1326 combos with `infoset_key_for_bucket` and `BlueprintStrategy.policy`: **bit-identical** (9 hands, 2 and 3 players, flop / turn / river roots, off-grid sizes before the root), more than 1000 informative weights |
| overrides | the brute force with the flop factors replaced |
| insertion | the off-grid bet appears as `x<amount>` at its node, after the grid's actions; the subgame plays it; a second insertion in the same round keeps the same root |
| fixing | on a flop where every combo is its own class, the node of our actual hole at our earlier action is never updated (regrets and strategy sums stay 0) while our other holes and the opponent's holes are |
| deals | 40,000 three-player deals share no card; focused deals hold our hole; our hole's frequencies match the exact joint distribution (card removal against the opponent's range) within 5 standard deviations |
| river subgame | exact exploitability below: decreasing with iterations, under a quarter of the blueprint's |
| budgets | iteration and time limits, distributions, `likelihood`, "not our turn" |

**Exact exploitability, preflop** (`scripts/search_exploitability.py`).  Our exact best response
(`cfr/exploit.py`) exists for 2-player preflop-only games.  There the root of every search (start
of the round) is the start of the hand, so each search solves the whole game from uniform ranges;
the blueprint's only possible disadvantage is how it handles actions it does not have.  Setup:
push/fold (the HRC game), blueprint = 1M iterations of the C++ trainer; the search-derived
strategy = a search at every decision node for each of the 169 classes as our actual hand
(representative combo), final iteration or average.  (1) the blueprint's own game; (2) the same
game with a min-raise available to both players (grid fraction 0.01, clamped to the minimum raise
at every node), which the blueprint answers through its translation (as `BlueprintAgent`:
deterministic mapping, check/call when the key is unknown) and the search by insertion.  bb/100,
average over the two seats of best response minus self-play:

| game, strategy | 10bb, 50k it/search | 10bb, 400k | 20bb, 50k | 20bb, 400k |
|---|---:|---:|---:|---:|
| own game: blueprint (1M iterations) | 0.53 | 0.52 | 1.02 | 1.07 |
| own game: search, final iteration | 0.71 | **0.07** | 3.04 | **0.40** |
| own game: search, average | 1.63 | 0.18 | 4.60 | 0.90 |
| + min-raise: blueprint with translation | 30.01 | 30.12 | 46.45 | 46.51 |
| + min-raise: search with insertion, final | **1.40** | **0.32** | **6.38** | **4.69** |
| + min-raise: search with insertion, average | 1.92 | 0.48 | 7.01 | 4.98 |
| + min-raise: blueprint trained on that game (1M), reference | 0.82 | 0.92 | 1.78 | 1.71 |

(The blueprint rows differ between runs because the 8-thread trainer is not deterministic.)
With enough iterations per search the search-derived strategy is less exploitable than the
blueprint in its own game; with an off-grid size it removes nearly all of the translation's
exploitability (a best-responding SB wins 60-92 bb/100 by min-raising against the translation,
0.6-8.6 against the search).  At 20bb the search stays above the blueprint trained with the
size (4.69 vs 1.71): each search solves the SB's min-raise range at equilibrium and answers it,
while a best responder may min-raise with another range - the known price of unsafe subgame
solving (Safe and Nested Subgame Solving, 2017).  338 / 1352 searches took 3 / 16 s at 50k
iterations and 22-24 / 111-112 s at 400k (8 threads).

**Exact exploitability inside a river subgame** (`SubgameSearch.river_exploitability`).  A root
on the river has no chance left, so the exact best response over all hole pairs (card removal,
exact showdowns, the root ranges as the players' distributions) measures a strategy inside the
subgame: the search's average, its final iteration, or the blueprint as the agent plays it there.
In bb per deal of the subgame:

| spot | search iterations (threads) | average | final iteration | blueprint |
|---|---|---:|---:|---:|
| test game 2p 30bb, river after check-downs, pot 6bb, 6k-iteration blueprint | 2k / 200k / 1M / 5M (8) | 5.28 / 0.45 / 0.16 / 0.062 | 4.88 / 0.37 / 0.29 / 0.53 | 3.98 |
| same spot, 2M-iteration blueprint | 2k / 200k / 1M / 5M (8) | 6.50 / 0.35 / 0.13 / 0.055 | 6.38 / 0.40 / 0.31 / 0.99 | 1.03 |
| HU 200bb blueprint, river after check-downs, pot 12bb | 0.35M / 1.5M / 6.4M (12; 0.25 / 1 / 4 s) | 1.48 / 0.45 / 0.17 | 2.12 / 2.57 / 2.18 | 2.13 |

The average strategy converges (1% of the pot after 5M iterations, 2.3 s on 8 threads); **the
final iteration does not**: as a profile it stays at 0.3-2.6 bb per deal, no better than the
blueprint in the HU spot.  The core returns both; whether the agent plays the final iteration (as
Pluribus), the average, or something between is a decision for part 3.

**Next to the Python reference** (`scripts/search_compare.py`, the 2p 20bb flop game, blueprint
`blueprint_2p_20bb_flop.json`, 6 hands per spot, C++ 1 s on 4 threads).  The two searches differ by
design (root, ranges, card abstraction, final vs average, the reference's blueprint prior), so this
compares distributions.  Mean probabilities fold / check-call / bet-raise:

| spot | blueprint | Python (prior 300, 800 it) | Python (no prior, 4000 it) | C++ final | C++ average |
|---|---|---|---|---|---|
| BB first to act | 0 / .93 / .07 | 0 / .94 / .06 | 0 / .69 / .31 | 0 / .83 / .17 | 0 / .84 / .16 |
| SB vs check | 0 / .17 / .83 | 0 / .29 / .71 | 0 / .37 / .63 | 0 / 0 / 1.00 | 0 / 0 / 1.00 |
| SB vs half-pot bet | .34 / .37 / .29 | .47 / .48 / .05 | .50 / .50 / 0 | .33 / .67 / 0 | .33 / .66 / .01 |
| BB vs bet after check | .15 / .63 / .22 | .19 / .49 / .32 | .27 / .55 / .18 | .33 / .33 / .33 | .32 / .29 / .39 |

(C++ columns from the run next to the prior-300 reference; the second run's C++ numbers differ by
up to .17 in the last row, the 4-thread runs not being deterministic.)  The same kind of answer in
three spots of four: first to act, mostly checking; facing a half-pot bet, folding about a third
with almost no raises; facing a bet after a check, a mix of all three.  After a check the C++
search bets every one of the six hands where the blueprint and the reference check 17-37%.  The
C++ strategies are nearly pure; the reference with its blueprint prior stays close to the
blueprint (mean TV 0.09-0.35), and without the prior it moves away from it in the same direction
as the C++ search in two spots (first to act: more bets; vs half-pot: no raises).

**Timing on the HU 200bb blueprint** (`scripts/search_timing.py`: `blueprint_hunl200w3_pot16_s0.json`,
3.04M infosets, potential-16 buckets, grid preflop 0.5/1/3, postflop 0.5/1/2/4, 3 raises; 14 threads,
2 s per search, 3 hands per spot, 5 seeds each; the live Slumbot match was the only other load):

| spot | iterations per second | iterations per 2 s | table (nodes) | build (ranges), first on the board / then |
|---|---:|---:|---:|---:|
| turn, BB first to act | 463k-604k | 0.90-1.23M | 85k-124k | 69-88 ms / 2-3 ms |
| turn, BB faces a pot bet | 542k-665k | 1.00-1.36M | 83k-122k | 46-84 ms / 3 ms |
| river, BB first to act | 1.58M-1.86M | 3.15-3.78M | 69k-108k | 87-146 ms / 3 ms |
| river, BB faces a pot bet | 1.90M-2.21M | 3.55-4.45M | 106k-111k | 102-129 ms / 3 ms |

(One iteration = one traversal per player who can act.)  The first build on a board computes the
blueprint buckets of all 1326 combos for the earlier streets (potential-aware flop and turn
buckets); the bucketer's caches make the next searches on that board 2-3 ms.  **Stability across
seeds**: in all 12 hands the final-iteration strategy of our hole was the same pure action for the
5 seeds (total variation 0.000); the average strategy of our hole varied more (mean TV between
seeds 0.000-0.236, the largest on the turn) because it accumulates only when the opponents'
traversals deal our hole.

## What part 2 needed (written after part 1; done in part 2, below)

1. The depth rule in the traversal (Pluribus): the first round searches to its end; the second
   round, with more than two players at its start, to the start of the third round or right after
   the second raise, whichever comes first; otherwise to the end of the hand.  Heads-up that means
   leaves only in preflop searches (flop onward to the end of the hand); 3-max and 6-max flops get
   leaves too.
2. A leaf decision per live player: one of four continuations (the blueprint, and copies with the
   probability of fold, of call, or of every raise multiplied by 5 and renormalised), chosen per
   the player's infoset at the leaf (the same choice at every leaf of that infoset), a node of the
   subgame with its own regrets.
3. Leaf values: the mean of 3 rollouts to the end of the hand (Depth-Limited Solving 2018: three
   was their best trade-off), each player playing its chosen continuation from the C++ blueprint
   lookup: keys from the bucket and the history translated deterministically, which also covers
   leaves reached through an inserted off-grid action (pseudo-harmonic mapping, as Pluribus).
   The keys can be built from the `HistHash` the ranges already use; chance cards sampled.
4. Optionally the compressed continuations: one pre-sampled action per abstract infoset and
   continuation (4 x 3.04M bytes for the HU blueprint) so a rollout is a chain of lookups.
5. Bucket costs: rollouts from a preflop leaf need flop buckets (potential-aware, about 0.5 ms each
   uncached); the flop cache holds all 1.29M canonical flops, so a one-off fill (about 11 CPU-minutes)
   or a cache saved to disk would keep rollouts at lookup speed.
6. Decisions for part 3 from the measurements above: final iteration or average (the final
   iteration's profile does not converge in river subgames), ranges across rounds through
   `likelihood()` overrides, and the time budget per decision.

# Part 2: depth limits and continuation leaves (2026-09-25)

## What a search does now

```python
s = core.SubgameSearch(game, stacks, button, actions, board, seat, hole, time_budget=2.0, threads=14,
                       depth="pluribus", rollouts=3, bias=5.0)
r = s.solve()   # as in part 1, plus "leaves", "leaf_evals", "rollouts", "rollout_steps"
```

* **Depth rule** (`depth`; `s.root_info()` shows `limit_street` and `raise_limit`):
  * `"end"` (the default, as in part 1): to the end of the hand;
  * `"pluribus"`: a preflop root searches to the end of the preflop (leaves at the start of the
    flop); a flop root with more than two players at the start of the flop stops at the start of
    the turn or right after the second raise of the flop, whichever comes first; every other root
    goes to the end of the hand (heads-up, only preflop searches have leaves);
  * `"hu_flop_limit"` (Modicum, Depth-Limited Solving 2018; `docs/search_research.md` section 7):
    preflop roots stop at the start of the flop and flop roots at the start of the turn; turn and
    river roots go to the end of the hand;
  * `"next_street"` (for measurements): every root stops at the start of the next round.

  The real path up to our decision is never cut: when we act after the second raise of a flop, the
  leaves come right after our action.
* **Leaves**: at a leaf every player who can still act chooses one of four continuations: the
  blueprint, or its copies with the probability of fold, of call, or of every raise (all-in
  included) multiplied by `bias` = 5 and renormalised (`core.apply_continuation`).  The choice is a
  node of the subgame with its own regrets and average (so mixtures are allowed).  Its key is the
  player's seat, its hole's lossless class on the root's round and the public path, so every leaf
  of that infoset shares one choice, whatever card comes next and whatever the others hold.  The
  traverser evaluates its four choices on common random numbers; the other choosers sample theirs.
* **Leaf value**: the mean of `rollouts` = 3 rollouts to the end of the hand.  A rollout redraws the
  board cards the choosers have not seen, then every player plays its chosen continuation of the
  blueprint as `BlueprintAgent` plays it: the bucket key with the history translated
  deterministically (pseudo-harmonic, so leaves behind inserted off-grid sizes are covered),
  check/call when the key is unknown.
* **Pre-sampled actions** (optional; Pluribus' compression of the continuations):
  `game.presample(seed)` draws one action per blueprint record and continuation (one byte each);
  rollouts then play that action whenever it is legal at the node.  `game.clear_presampled()`.
* **Buckets**: `bucketer.precompute(street, threads)`, `save_cache(path, streets)` and
  `load_cache(path)` (`csrc/bucketcache.h`; the file starts with the tag NPBKCH01 and the bucketer's
  identity, and a cache of other buckets is refused); `scripts/precompute_buckets.py` makes the
  file.  A search from a flop or turn root also builds a table of the river buckets of every board
  it can reach (1,176 boards from a flop, 48 from a turn; potential-aware river buckets are computed
  per board in one batch, `river_buckets_all`), and a search from a flop root fills a table of turn
  buckets per turn card on first use.  Tables and batch give the bucketer's own numbers (tests).
* **Measurement tools**: `s.subgame_exploitability(kind, br_street)` for 2-player turn and river
  roots (the river card a chance node; for a search with leaves at the river start, its river play
  is the continuation mixes); `br_street=7` measures inside the search's own model (the best
  responder deviates on the turn and picks its best continuation at a leaf before the river card,
  then the river is played by the continuations); `s.freeze_round(src, kind)` makes every player
  play the root's round as the solved search `src` of the same spot does, so a new `solve()` learns
  only the later rounds (a river re-solve after a turn search); `s.path_strategies(k, kind)` gives
  the strategy of every combo at a node of the real path.

## How part 2 is checked

The acceptance rule of 2026-09-25: identical numbers where the logic is deterministic, equivalence
within noise where there is randomness (sampling, thread interleaving, summation order in new code).
Nothing in part 2 reproduces an old random stream.

| deterministic: checked for identity | against |
|---|---|
| continuation arithmetic | the same formula in Python, `==` |
| rollout policy at real states (60 random hands with off-grid sizes, 4 continuations) | `BlueprintStrategy.policy` at the key `BlueprintAgent` builds, then the continuation: within 1e-15 |
| turn and river bucket tables of a search; batch river buckets | the bucketer's `bucket()`: `==` |
| bucket cache save / load | the values before saving; nothing computed after loading; a cache of other buckets refused |
| canonical boards | 1,755 flops, 16,432 turns |
| where the leaves are, per depth rule | the rule, over the whole public tree of each subgame (2 and 3 players, every root street; for `"end"` at preflop and flop roots only the limit, those trees being too big to enumerate) |
| exact evaluator | pairwise showdowns: within 1e-9 relative (the summation order differs); the same numbers on 1 and 4 threads; zero-sum values; a best response on the river alone gains no more than one on the whole subgame |
| frozen round | on a river root the frozen search's profile, strategies and exploitability are the source's: `==` |

| random: checked statistically | how |
|---|---|
| a leaf choice is shared by the player's indistinguishable leaves | 20,000 logged choices: one key per (seat, class, public path) whatever the turn card and the other hole; suit-isomorphic holes share it |
| 3 rollouts per leaf | rollouts = 3 x leaf evaluations (also 1 and 5) |
| the search solves its own model | the exploitability inside the model falls with iterations (test game: 2k vs 100k iterations; the HU game below) |
| pre-sampled actions | always a legal action of positive blueprint probability, the same on every call; quality against seed noise below |
| our action across seeds | the timing table below |

`tests/test_search_core.py` has 22 tests: 10 from part 1 and 12 for part 2 (the ones above, the
exact evaluator, the raise limit on the real path, the frozen round and the per-iteration average).

**Our hole's average** (a fix found by these measurements): `solve()["average"]` now adds up our
decision's strategy once per iteration (weight t, as Linear MCCFR weights the average).  Our actual
hole's own reach there is fixed (its actions of the round are forced), so this is exactly the
average it plays.  The node's accumulated average, the one the profile, `likelihood()` and the
evaluator use, is still returned as `average_table`; for our class it accumulates only when the
opponents' traversals deal it and follow the real path, and in the quality run below it stayed
uniform (never accumulated) in 8 of 20 turn searches; the test reproduces that by giving our hole a
tiny weight in our range.

## Quality at turn roots (verification b)

`scripts/search_quality.py` on the HU 200bb blueprint (16 potential-aware buckets): 10 turn hands, 5
with BB first to act and 5 with BB facing a pot bet after checking, pot 12 bb (a half-pot flop bet
called).  Each search 2 s on 14 threads (the live Slumbot match was the only other load).  The
exploitability is exact: both players' best responses
over all 1326 x 1326 hole pairs with card removal, the river card a chance node; in bb per deal of
the subgame, mean ± standard error over the 10 hands.

* A: search to the end of the hand (the river on the blueprint's buckets); A2: another seed;
  Aeq: A stopped at B's number of iterations.
* B: leaves at the start of the river (`depth="next_street"`, each player's continuation mix, 3
  rollouts per leaf); as a profile, its river play is the continuation mixes.  B2: another seed.
  Bp: B with pre-sampled actions.
* X~: every player plays the turn exactly as X's average says (`freeze_round`) and the river is
  searched again, to the end, for 4 s.  This is how the agent plays: it searches again at its next
  decision, so B's own river play (the continuations) is never played.
* Columns: "subgame": the best responder deviates on the turn and the river; "river": on the river
  only, the turn played as the profile says; "model": inside B's own model (deviations on the turn
  and in the continuation choice, the river played by the continuations).

| profile (average strategy) | iterations | subgame | river | model |
|---|---:|---:|---:|---:|
| blueprint, as the agent plays it | | 6.19 ± 0.43 | 3.70 ± 0.30 | |
| A, to the end | 1.21M | **3.31 ± 0.17** | 1.88 ± 0.10 | |
| A~, A's turn, the river searched again | | 3.39 ± 0.15 | 1.93 ± 0.09 | |
| Aeq, to the end, B's iterations | 0.36M | 5.04 ± 0.23 | 1.55 ± 0.09 | |
| B, leaves at the river, river = continuations | 0.36M | 10.27 ± 0.82 | 3.29 ± 0.21 | 2.95 ± 0.22 |
| B~, B's turn, the river searched again | | **5.47 ± 0.19** | 1.62 ± 0.10 | |
| Bp, B with pre-sampled actions, river = continuations | 0.43M | 16.94 ± 0.87 | 6.69 ± 0.24 | 2.67 ± 0.20 |
| Bp~, Bp's turn, the river searched again | | 5.22 ± 0.20 | 1.67 ± 0.07 | |

Paired differences of the subgame exploitability (same hands):

| difference | bb per deal | reading |
|---|---:|---|
| B~ − A~ | +2.08 ± 0.15 | leaves at the river start cost 2.1 bb per deal (17% of the pot) at the same 2 s, even with the river searched again |
| B~ − Aeq | +0.42 ± 0.10 | at equal iterations the leaf model costs at least about 0.4 bb (Aeq's river had only its 0.36M joint iterations, B~'s river a 4 s search of its own); the rest is the rollouts' price: B runs 3.3 times fewer iterations in 2 s |
| B~ − blueprint | −0.72 ± 0.40 | the depth-limited search is barely better than the blueprint here |
| A − blueprint | −2.88 ± 0.46 | the search to the end halves the blueprint's exploitability |
| A~ − A | +0.08 ± 0.07 | freezing the turn and searching the river again reproduces the joint solution (the tool measures what it should) |
| A2 − A, B2 − B, B2~ − B~ | +0.12 ± 0.05, −0.05 ± 0.13, −0.03 ± 0.05 | noise between seeds |
| Bp~ − B~ | −0.24 ± 0.11 | pre-sampled actions: equivalent within about two standard errors (if anything slightly better), 19% more iterations |

* **The search solves its own model**: B's exploitability inside its model is 2.95 at 0.36M
  iterations, and it falls with iterations (two turn hands, one per spot, `depth="next_street"`, 14
  threads: 17.6 / 8.1 / 3.2 / 1.06 and 19.6 / 8.8 / 3.7 / 0.91 at 30k / 100k / 300k / 1M iterations), while its
  exploitability in the real subgame stays at 7.8-8.1 at 1M iterations.  The gap is the model's error:
  the continuations are the blueprint's river play (3.3-4.1 bb per deal exploitable in these spots,
  28-34% of the pot) with only four biases, and the turn strategy B learns against them is worse than
  the one learned against a solved river.
* **Distance between root strategies**: our hole's final-iteration action differed between A and B
  in 2 of the 10 hands (B bet where A checked; B raised where A folded); A and A2 agreed in all 10.
  Over our whole range (weights: the root range times the likelihood of our path actions) the total
  variation between A and B was 0.31 ± 0.02, but between two seeds of the same search it was already
  0.22 (A) and 0.30 (B): range-wide strategies are dominated by rarely sampled combos, so the exact
  exploitability is the measure to read.
* **Final iteration against average** (as whole profiles): the final iteration is about 4.4 times as
  exploitable (A: 14.49 ± 0.98 against 3.31; B~ with B's final turn and the re-search's final river:
  16.39 ± 0.89).  In play each decision takes one hole's strategy from a fresh search, so the agent's
  effective strategy mixes over searches; which one to play stays the part-3 duel.
* **Why rollouts are expensive**: at the same 1M iterations on 14 threads a turn root took 4.0-6.1 s
  with leaves and 1.4-2.0 s to the end of the hand.

The iteration sweep is `search_quality.py --hands 1 --sweep 30000,100000,300000,1000000` (numbers
vary a little between runs: 14 threads are not deterministic).

## Timing on the HU 200bb blueprint (verification c)

`scripts/search_timing.py`, blueprint `blueprint_hunl200w3_pot16_s0` (binary, 3.04M infosets), the
bucket cache loaded (0 evictions), 14 threads, 2 s per search, 3 hands per spot, 5 seeds each; the
live Slumbot match was the only other load.  Spots: SB opens pot, BB calls; the flop with BB first to
act and with SB facing a half-pot bet; turn and river after a called half-pot flop bet and checks
(BB first to act; BB facing a pot bet after checking).  Turn and river roots are searched to the end
of the hand by every depth rule.

| spot | depth rule | iterations / s | iterations in 2 s | table (nodes) | nodes touched / it | rollouts in 2 s | same action in all 5 seeds |
|---|---|---:|---:|---:|---:|---:|---:|
| flop, BB first to act | hu_flop_limit (leaves at the turn) | 92k-95k | 183k-190k | 228k-360k | 58-61 | 18.9M-19.5M | 3 of 3 hands |
| flop, SB vs half pot | hu_flop_limit | 90k-121k | 156k-245k | 118k-349k | 44-54 | 15.8M-19.1M | 1 of 3 (4/5 in the others) |
| flop, BB first to act | hu_flop_limit + pre-sampled actions | 103k-108k | 205k-227k | 231k-362k | 57-61 | 21.2M-22.5M | 3 of 3 |
| flop, SB vs half pot | hu_flop_limit + pre-sampled actions | 115k-143k | 230k-288k | 119k-356k | 43-53 | 21.0M-21.3M | 1 of 3 (3/5, 2/5) |
| flop, BB first to act | pluribus (heads-up: to the end) | 166k-174k | 317k-359k | 308k-378k | 181-189 | 0 | 3 of 3 |
| flop, SB vs half pot | pluribus | 191k-209k | 378k-421k | 249k-375k | 154-164 | 0 | 1 of 3 (4/5, 4/5) |
| turn, BB first to act | (to the end) | 499k-629k | 0.99M-1.27M | 83k-85k | 56-69 | 0 | 2 of 3 (4/5) |
| turn, BB vs pot bet | (to the end) | 614k-659k | 1.22M-1.33M | 122k-124k | 52-56 | 0 | 3 of 3 |
| river, BB first to act | (to the end) | 1.55M-1.71M | 3.07M-3.47M | 106k-110k | 18-20 | 0 | 2 of 3 (3/5) |
| river, BB vs pot bet | (to the end) | 1.93M-2.06M | 3.80M-4.15M | 104k-110k | 15-17 | 0 | 1 of 3 (3/5, 4/5) |

(One iteration = one traversal per player who can act.)  "Same action": the final iteration's most
likely action of our hole was the same for all 5 seeds; where it was not, the hole was close to
indifferent (for example 3 seeds bet and 2 check on the river: an equilibrium mix, of which the final
iteration shows one side).  Heads-up flops searched to the end run 1.7-2.1 times as many iterations
as with leaves at the turn: with warm caches, rollouts cost more than solving the turn and the river.
Pre-sampled actions give 11-33% more iterations at flop roots.

**Build cost per search** (warm cache): 30-38 ms at flop roots (the river table: 1,176 boards, 25-31
ms), 4-10 ms at turn roots (48 boards, 2-3 ms), 3-4 ms on the river; 0-4 turn buckets computed per
flop search (the rest come from the cache).

**Bucket caches, one-time and per-search costs** (potential-aware 16 buckets, measured this session
while the Slumbot match ran):

| step | cost |
|---|---|
| precompute flop forms: 2,063,880 (board, hole) pairs, 1,286,766 forms | 88 s on 14 threads |
| precompute turn forms: 18,535,296 pairs, 13,960,023 forms | 553 s on 14 threads |
| save the cache file | 121,974,381 bytes, 0.2 s |
| load it (`core_bucketer(bk, (8_000_000, 64_000_000, 4_000_000))`, then `load_cache`) | 0.3-0.45 s, 0 evictions, +577 MB private memory |
| without the cache: the first flop searches on a new board | 29,228 and 51,174 turn buckets computed inside the 2 s; 42,862 and 5,836 iterations against 104,704 and 99,555 for the next seed (hu_flop_limit, before the turn table) |
| river buckets | not cached: per search, one batch per board (0.22 ms per board; 1,176 boards from a flop) |

The other startup costs of a match: the binary blueprint loads in 0.11 s (+105 MB private memory);
`SearchGame` is free; `presample` takes 0.2 s and 12,171,064 bytes.

**Pre-sampled actions: memory and speed** (optional, verification 4): 12.17 MB for the 3.04M-infoset
HU blueprint (4 continuations x 1 byte per record), drawn in 0.2 s; 11-33% more iterations at flop
roots with leaves at the turn, 19% more at turn roots with leaves at the river; quality equivalent to
sampling the blueprint within about two standard errors (Bp~ − B~ above).  It pays a little, only
where there are rollouts.

## What part 3 needs (the agent)

1. **Preflop**: the blueprint (`BlueprintAgent`), with a search only when an opponent's size is far
   off the grid (Pluribus: "sufficiently different" from the grid's sizes; here a threshold parameter).
   Such a search has its root at the start of the hand and leaves at the start of the flop (every
   depth rule): rollouts then meet new flops, turns and rivers, so the bucket cache matters most there
   (river buckets are computed per call there: no river table for preflop roots; to measure).
2. **From the flop, every decision is a search** from the start of the current round: the real
   actions of the round as the path (our own fixed for our hole, off-grid sizes inserted; a new
   off-grid size means a new search from the same root).  Heads-up, `depth="pluribus"` (to the end of
   the hand from the flop) looks better than `"hu_flop_limit"`: 1.7-2.1 times the iterations with
   warm caches, and at turn roots leaves cost 2.1 bb per deal of exploitability (at least about 0.4
   at equal iterations).  The flop itself cannot be checked exactly with this evaluator (a turn and a river
   card to enumerate), so the choice between the two at the flop is for the duel.  Multiway flops
   (3-max) get leaves by the Pluribus rule; there the leaf model's error depends on the blueprint's
   quality, and nothing exact measures it (the evaluator is 2-player).
3. **Ranges across rounds**: a new round's search takes `overrides=[(street, seat, likelihoods)]`
   for the rounds already searched, the likelihoods from the previous search's `likelihood(seat)`
   (its average strategy over the actions really taken); rounds not searched keep the blueprint's
   Bayes.  Not yet measured: how much this changes play against using the blueprint's ranges.
4. **Play mode**: `solve()` returns the final iteration and the average (now per iteration, see
   above).  Pluribus played the final iteration and updated ranges with the average
   (`docs/search_research.md` section 1).  As whole profiles the final iteration is about 4 times as
   exploitable as the average at turn roots in 2 s; in play each decision takes a fresh search's
   strategy for one hole, so the duel decides (with seed control, AIVAT when it exists).
5. **Time budget**: 2 s gives 0.32M-0.42M iterations at heads-up flops (to the end), 1.0M-1.3M at
   turns and 3.1M-4.2M at rivers on 14 threads; the river's final iteration still flips between the
   sides of mixed spots.  A budget per street (less on the river, more on the flop) is an option to
   measure in the duel.
6. **At startup**: load the binary blueprint, `core_bucketer(bk, (8_000_000, 64_000_000,
   4_000_000))` + `load_cache(<122 MB file>)` (0.5 s, about 0.7 GB with the blueprint), one
   `SearchGame` per match; `presample` optional.  The bucket cache has to be made once per bucketer
   (`scripts/precompute_buckets.py`, about 11 minutes on 14 threads).
7. **Known limits**: at most 8 actions per node (a grid with 5 raise sizes leaves no room for an
   inserted size); ranges and rollouts translate off-grid history deterministically (a randomized
   translation in play would need the same in the ranges); the exact evaluator covers 2 players at
   turn and river roots only.
8. **Possible speed-ups under the new acceptance rule** (proposals, not done): the per-node locks of
   the search could go (updates without locks give statistically equivalent results, to be shown
   with the exact evaluator's curves); rollouts could stop at the river with a precomputed river
   value table instead of playing it out.
