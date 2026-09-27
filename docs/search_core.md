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

# Part 3: the search agent at the table (2026-09-25)

## Using it

```python
from negpluribus.agents.core_search import CoreSearchAgent, SearchConfig, SearchResources
res = SearchResources.load(spec, "data/blueprint_hunl200w3_pot16_s0.bin", "data/buckets_hunl200w3_pot16_s0.json",
                           cache_path="data/bucketcache_hunl200w3_pot16_s0.bin")   # once per process, 0.4-0.5 s
agent = CoreSearchAgent(res, SearchConfig(time_budget=0.5, threads=14), seed=1)   # a drop-in Agent
```

```
# Slumbot (its own log; the grid must be the blueprint's)
python scripts/play_slumbot.py --agent search --blueprint data/blueprint_hunl200w3_pot16_s0.bin \
    --buckets data/buckets_hunl200w3_pot16_s0.json --cache data/bucketcache_hunl200w3_pot16_s0.bin \
    --preflop-fracs 0.5,1.0,3.0 --search-budget 2 --hands 1000 --log data/slumbot/search_hunl200w3.jsonl
# duels and archetypes: the search hero against the blueprint, the overbettor, random, the archetypes
python scripts/eval_archetypes.py --spec 2p_200bb_river --preflop-fracs 0.5,1.0,3.0 --postflop-fracs 0.5,1.0,2.0,4.0 \
    --blueprint data/blueprint_hunl200w3_pot16_s0.bin --buckets data/buckets_hunl200w3_pot16_s0.json \
    --cache data/bucketcache_hunl200w3_pot16_s0.bin --agent search --search-budget 0.5 \
    --opponents blueprint,overbettor,random --deals 1000 --progress 50 --log data/duel_search05.jsonl
python scripts/duel_log.py data/duel_search05.jsonl [--paired data/duel_blueprint.jsonl]   # also while it runs
# where a hero wins or loses: add --log-hands data/hands.jsonl to the duel, then
python scripts/hand_log_report.py data/hands.jsonl [--paired data/hands_blueprint.jsonl]
```

**The data files** (in `data/`, not in git), made once per blueprint and per bucketer:

```
python scripts/export_json.py data/blueprint_hunl200w3_pot16_s0.json data/blueprint_hunl200w3_pot16_s0.bin
    # 3.2 s; 233,759,108 bytes; the binary loads in 0.1 s instead of the JSON's minutes
python scripts/precompute_buckets.py --buckets data/buckets_hunl200w3_pot16_s0.json \
    --out data/bucketcache_hunl200w3_pot16_s0.bin --threads 14
    # about 11 minutes; 121,974,381 bytes; every flop and turn bucket (part 2)
```

Flags of `--agent search` (both scripts): `--cache`, `--search-budget` (seconds per decision,
default 2), `--search-street-budgets "preflop=8,flop=3,turn=2,river=1"`, `--search-iterations` (a
fixed count instead of the clock), `--search-threads` (default 14), `--search-play average|final`
(default average), `--search-depth` (default pluribus), `--preflop-offgrid` (default 0.3),
`--no-preflop-search` (the blueprint translates every preflop size), `--presample` (off by default).

## How it plays

* **Preflop: the blueprint**, through an inner `BlueprintAgent` (randomized translation, as on
  Slumbot).  A search instead when (1) an opponent's raise of this preflop is far off the grid: the
  relative distance between its raise increment and the nearest abstract size at its node (the grid's
  sizes as the engine clamps them there, all-in included) is above `preflop_offgrid` = 0.3; (2) the
  blueprint has no strategy at our key (it would check/call blind: "off the map"); (3) an earlier
  preflop decision of this hand was searched.  A preflop search has its root at the start of the hand
  and leaves at the start of the flop (continuation leaves, 3 rollouts).
* **From the flop: a search at every decision**, root at the start of the round, the round's real
  actions as the path (our own fixed for our hand, off-grid sizes inserted), `depth="pluribus"`
  (heads-up: to the end of the hand; 3 or more players at the flop: leaves at the turn or after the
  second raise).
* **Ranges across rounds**: Bayes over the blueprint for rounds played without a search; for a
  searched round, the last search's `likelihood(seat, round actions, floor=1e-3)` of every live seat,
  passed as `overrides` to the later rounds' searches (tested: the turn search gets exactly the flop
  search's likelihoods).
* **Play**: the average strategy of our hand (default; `solve()["average"]`, accumulated every
  iteration), or the final iteration.  The chosen action is sampled from it.
* **Never stalls**: a search error falls back to the blueprint and is counted (none in any run below).
* **What it reports**: `stats` (decisions by street; preflop decisions of the blueprint; searches,
  seconds and iterations by street; why preflop decisions were searched; searches with an inserted
  size; off-map decisions), `fallback_rate` like `BlueprintAgent`, and `decision_info()` per
  decision, which the Slumbot log records (searched or not, actions, final and average strategy,
  iterations, seconds, inserted sizes).

The duel tool (`negpluribus/eval/duel.py`) deals and seeds exactly as `duplicate_match` (tested:
the same raw numbers) and adds per hand the heads-up **card-luck correction**: at every card deal
(the hole cards, then each street reached), (equity after − equity before) × the pot at the deal,
with exact equities (`equity_vs_hand` enumerates every board completion; the preflop 1.7M boards on
8 threads in 21 ms).  Equity is a martingale over the cards to come (tested exactly), so the
correction has mean zero whatever the players do and removing it keeps the estimate unbiased.

## Checks (measured 2026-09-25, HU 200bb blueprint, 14 threads, the live Slumbot match the only other load)

Game: `blueprint_hunl200w3_pot16_s0` (binary, 3.04M infosets; preflop 0.5/1/3, postflop
0.5/1/2/4 pot + all-in, 3 raises, 16 potential-aware buckets), the bucket cache loaded.  Duels:
`scripts/eval_archetypes.py --agent search ... --seed 0`, duplicate deals (each deck twice, seats
swapped), 95% CI with one deal as one sample; "corrected" = minus the card luck.

**a. Decisions off the map, against the random agent** (it raises to any chip amount):

| hero | hands | off-map decisions | bb/100 raw | luck-corrected |
|---|---:|---:|---:|---:|
| blueprint agent | 100,000 | 6.8% | +147.3 ± 21.8 | |
| search agent, 0.5 s | 300 | **0 of 453 (0.0%)** | +1531 ± 873 | +1485 ± 526 |

The blueprint's 6.8% (10-11% in the HU 100bb game of docs/scale_4street.md) are the check/call
fallbacks of unknown keys.  The search agent has none: 101 of its 453 decisions were preflop
searches (89 for a size farther than 0.3 from the grid, 12 after a preflop search), 166 searches
held an inserted size.  Its preflop searches are slow: 4,200 iterations in 0.5 s (a preflop root's
leaves are at the flop, and every leaf value plays 3 rollouts through the flop, turn and river).

**c. The duel: search agent against the blueprint agent** (the blueprint on both sides):

| budget | deals / hands | bb/100 raw | luck-corrected | hero s / hand | searches / hand | iterations: flop / turn / river |
|---|---:|---:|---:|---:|---:|---:|
| 0.5 s | 2,000 / 4,000 | +16.6 ± 61.0 | **−1.7 ± 49.2** | 0.81 | 1.56 | 74k / 228k / 802k |
| 2 s | 150 / 300 | +94.8 ± 112.2 | +84.8 ± 92.9 | 3.22 | 1.60 | 339k / 985k / 3.44M |

At 0.5 s per decision the search agent is **not measurably better than the blueprint**: the
luck-corrected estimate is −1.7 with a 95% interval of ±49 bb/100 over 4,000 hands (the correction
cut the interval from ±61).  Expected before the run: the search agent wins, size unknown; at this
budget it does not show.  At 2 s the 300 hands (run for the timing of d) give +85 ± 93: too few
hands to tell.  A reading, measured in part 2 at turn roots: the exact exploitability of the search's
play falls below the blueprint's (6.2 bb per deal) only after about 200-300k iterations (100k:
9.6-10; 300k: 4.9-5.1; 1M: 2.4-3.4), and at 0.5 s the flop search runs 74k and the turn 228k, at 2 s
339k and 985k.

**b. The value overbettor** (`negpluribus/agents/overbettor.py`, the test of docs/scale_4street.md
rebuilt: the blueprint agent, but when it raises with a strong hand, preflop a hole in the top 15%
of `hole_percentile`, later an equity of at least 0.8 against one random hand, it raises 2.5 x the
grid's largest size instead, 7.5 pots preflop and 10 pots later, at most 80% of the way to its
all-in; pure value, off the grid):

| hero | deals | hands | bb/100 raw | luck-corrected | overbets |
|---|---|---:|---:|---:|---:|
| blueprint agent (pseudo-harmonic translation) | 0..49,999 | 100,000 | +24.0 ± 9.8 | | |
| blueprint agent, logged | 0..2,499 | 5,000 | +35.6 ± 44.0 | +39.2 ± 34.6 | 821 |
| search agent, 0.5 s | 0..1,499 | 3,000 | −288.3 ± 141.6 | −222.5 ± 92.6 | 510 |
| search agent, 0.5 s, logged | 1,500..2,499 | 2,000 | −307.6 ± 188.9 | −311.0 ± 122.0 | 354 |
| search agent, 2 s on every street, logged (R5) | 0..1,499 | 3,000 | −161.5 ± 121.1 | −114.5 ± 78.4 | 498 |

The old clamping translation lost 54 bb/100 to this opponent on the narrow HU 100bb grid; with the
paper's translation the blueprint agent beats it.  **The search agent at 0.5 s loses to it**: −258 ±
74 bb/100 luck-corrected over the 5,000 hands of both runs, −249 ± 106 and −369 ± 123 per deal against
the blueprint agent on the same decks and seeds.  The check fails.

**Where, and the mechanism** (`--log-hands`, `scripts/hand_log_report.py`, deals 1,500..2,499, per
100 hands of each kind, without the card luck; "paired": the search agent minus the blueprint agent
in the same hand, same deal and seat, raw):

| hands | number | search agent | paired with the blueprint agent | share of the −308 bb/100 |
|---|---:|---:|---:|---:|
| the overbettor overbet preflop, the hero searched preflop | 187 | −2,663 ± 956 | −2,978 ± 1,721 | −288 |
| no overbet | 1,685 | +52 ± 57 | −49 ± 77 | +72 |
| overbet on the flop | 44 | −3,096 ± 1,952 | −1,923 ± 2,477 | −49 |
| overbet on the turn | 42 | −1,562 ± 1,723 | −213 ± 2,433 | −32 |
| overbet on the river | 42 | −226 ± 1,324 | +2 ± 1,308 | −10 |

The loss is in the hands where the overbettor raised preflop off the grid (7.5 pots: an open to
16 bb, a 3-bet to 32 or 48 bb) and the hero answered with a preflop search: 187 hands of 2,000,
27-31 bb lost in each.  What the hero did at its first preflop search, by the strength of its hand:
raised with 73 of 110 hands in the bottom half of `hole_percentile` (called 8, folded 29), with 60
of 61 between 50% and 85%, with all 16 in the top 15%; the opponent holds the top 15% only.  Typical
hands: 3♦8♠ and T♠4♣ re-raised a 16 bb open and then shoved into AA and AK; 6♠J♣ shoved over a
32 bb 3-bet into AK; 7♣6♦ 4-bet a 32 bb 3-bet and called KK's shove.  The blueprint agent folded all
four at its first decision.  The preflop searches ran 3,649 iterations on average
in 0.5 s (a preflop root has leaves at the flop, each leaf value 3 rollouts through the flop, turn
and river).

**Under-convergence, measured on the spot itself** (`SubgameSearch` with the BB holding 3♦8♠ after
the overbettor's 16 bb open; our average strategy, fold / call / 3-bet to 32 bb; two seeds each;
the 0.5 s rows ran while another 14-thread search was running, so they got 1,400-1,800 iterations
instead of the 3,600 of the duel):

| solve | iterations | fold | call | 3-bet to 32 bb |
|---|---:|---:|---:|---:|
| 14 threads, 0.5 s | 1,365-1,763 | 0.32-0.77 | 0.14-0.32 | 0.00-0.46 |
| 14 threads, fixed | 3,600 | 0.59 / 0.91 | 0.23 / 0.05 | 0.08 / 0.01 |
| 1 thread, fixed | 3,600 | 0.96 / 0.90 | 0.01 / 0.08 | 0.00 / 0.01 |
| 14 threads, fixed (7 s) | 30,000 | **1.00** | 0.00 | 0.00 |

At the duel's budget the average strategy of a trash hand still calls or re-raises a 7.5-pot open
much of the time; at 30,000 iterations it folds.  At the same number of iterations one thread
converged further than fourteen here (a hypothesis: the fourteen threads' first iterations all read
the same near-uniform regrets).  What the
search believes about the open (the SB's range times its probability of the open, from the
search's average; one thread): at 3,600 iterations 16.1% of the SB's hands open 16 bb, half of them
in the bottom half of `hole_percentile`; at 30,000, 9.7%, 29% in the bottom half and 24.5% in the
top 15%.  The overbettor opens this way with the top 15% only: even converged, the subgame's
equilibrium assumes bluffs that this opponent does not have.

**Without the preflop searches** (`--no-preflop-search`: the blueprint translates every preflop
size, the searches from the flop unchanged), on the same deals 1,500..2,499 and seeds:

| hero | bb/100 raw | luck-corrected | paired with the blueprint agent, luck-corrected |
|---|---:|---:|---:|
| blueprint agent | +58.5 ± 66.3 | +58.1 ± 49.1 | |
| search agent, 0.5 s | −307.6 ± 188.9 | −311.0 ± 122.0 | −369.2 ± 123.2 |
| search agent, 0.5 s, no preflop search | +33.9 ± 102.4 | −11.4 ± 79.2 | −69.5 ± 86.1 |

Dropping the preflop searches gains +299.6 ± 104.0 bb/100 (luck-corrected, paired per deal), and
the hands with a preflop overbet go from −2,663 ± 956 to +42 ± 334 per 100 of them.  Without them
the search agent neither loses significantly to the overbettor (−11 ± 79) nor differs significantly
from the blueprint agent on these deals (−70 ± 86).  A second, smaller loss is in the answers to
postflop overbets (below).

So the mechanism is measured three ways: the loss sits in the hands with a preflop search (per-hand
attribution); on the spot itself the preflop search at the duel's budget has not converged (trash
calls or re-raises a 7.5-pot open, folds at 30,000 iterations); and without the preflop searches
the loss is gone.

**With 8 s for the preflop searches** (`--search-street-budgets preflop=8`, 0.5 s from the flop;
67,409 iterations per preflop search; the same deals 1,500..2,499; hero 1.52 s per hand):

| comparison, luck-corrected bb/100 | 8 s preflop |
|---|---:|
| against the overbettor | −80.2 ± 93.4 (raw −45.2 ± 137.8) |
| minus the 0.5 s search agent (paired) | +230.8 ± 99.9 |
| minus the search agent without preflop searches (paired) | −68.8 ± 76.9 |
| minus the blueprint agent (paired) | −138.4 ± 97.8 |

Converged preflop searches fold the trash (first preflop search: bottom half of `hole_percentile`
folded 97 of 110, re-raised 8; 50-85%: folded 29 of 61, re-raised 23, called 9), and the hands with
a preflop overbet cost −594 ± 631 per 100 of them instead of −2,663.  The agent no longer loses
significantly to the overbettor, but it stays behind the blueprint agent on these deals (−138 ± 98)
and is not better than translating preflop (−69 ± 77, not significant).  What is left, as far as
these samples show: the 50-85% hands that still re-raise a premium-only overbet (the subgame's
equilibrium expects bluffs in it), and the flop overbets (−2,692 ± 1,835 per 100 of those 43 hands).

**Where the rest of the gap sits** (`hand_log_report.py --paired ... --breakdown`, deals
1,500..2,499, paired hand by hand with the blueprint agent, luck-corrected; "per 100" = per 100
hands of the part, "share" = its part of the total per 100 hands of the run, the shares adding up
to the total).  A pair of hands is identical up to the search agent's first search (same seeds:
tested, and the 724-888 hands without a search differ by exactly 0), so the hands split by that
first search:

| first search of the search agent | R1: 0.5 s | R3: no preflop search | R4: 8 s preflop |
|---|---:|---:|---:|
| none (hands identical to the blueprint's) | 724 hands, 0 | 888, 0 | 724, 0 |
| preflop | 187: −2,556 ± 914 per 100, share −239.0 ± 91.3 | | 187: −487 ± 647, share −45.5 ± 60.7 |
| from the flop, the opponent overbet on a later street | 128: −1,243 ± 980, share −79.5 ± 63.9 | 132: −1,082 ± 873, share −71.4 ± 58.6 | 129: −841 ± 921, share −54.2 ± 59.9 |
| from the flop, no overbet at all | 961: −106 ± 113, share −50.7 ± 54.2 | 957: −25 ± 120, share −12.0 ± 57.3 | 960: −81 ± 109, share −38.6 ± 52.5 |
| from the flop, after a translated preflop overbet | | 23: +1,211 ± 1,897, share +13.9 ± 22.1 | |
| total | −369.2 ± 123.4 | −69.5 ± 84.9 | −138.4 ± 100.0 |

Besides the preflop searches, the gap sits in the answers to postflop overbets, not in ordinary
postflop play (no significant gap in the 957-961 hands without an overbet).  The answers to the
opponent's first postflop overbet in the same hands (`scripts/overbet_answers.py`; "behind" = below
1/2 equity against the opponent's actual hand at that street; the blueprint agent's reading of the
overbet replayed from its seeds):

| hands of the search agent's first postflop overbet | search agent | blueprint agent, same hands |
|---|---|---|
| R3 (0.5 s, no preflop search), 132 hands | fold 106 (all behind), call 2 (behind), raise 24 (19 behind) | fold 91 (90 behind), call 5 (3 behind), raise 0; 36 hands went another way before the overbet |
| R4 (8 s preflop), 129 hands | fold 101, call 2, raise 26 (20 behind) | fold 89, call 5, raise 0; 35 another way |
| R1 (0.5 s), 128 hands | fold 100, call 3, raise 25 (21 behind) | fold 88, call 5, raise 0; 35 another way |

Against this opponent a raise is behind even when it is right (it overbets strong hands only); the
signal is the paired result in these hands, luck-corrected: R3 −1,082 ± 873 per 100 of them (share
−71.4), R4 −841 ± 921 (−54.2), R1 −1,243 ± 980 (−79.5).  The blueprint agent read the overbet as its
all-in in 65 of the 96 hands where it faced it (R3's hands), as 4 pots in 31, and never raised: an
all-in leaves only fold or call, and against the all-in it calls narrowly.  On all of its own 339
hands with a postflop overbet it answered fold 311, call 23, raise 5 (all 5 from the 4-pot reading).
For example A♠8♦ with middle pair re-raised a 60 bb flop overbet into two pair and called the
shove; the blueprint agent folded.  By the street where the hand was
decided (R4): preflop showdowns, the all-ins after preflop searches, 19 hands, −5,838 ± 2,612 per
100, share −55.5 ± 34.6; river showdowns 290 hands, −395 ± 371, share −57.3 ± 54.0; the other
streets not significant.  Whether the re-raises behind are the 0.5 s budget (the inserted
overbet's range not converged) or the subgame's equilibrium assumption (a balanced overbet range;
this opponent has no bluffs) is what the 2 s run (R5) separates.

**At 2 s on every street (R5, the user's test budget; deals 0..1,499, the decks of the first 0.5 s
run; 15,774 iterations per preflop search, 426k per flop search, 1.12M turn, 3.34M river; the
hero 3.18 s per hand):**

| comparison on deals 0..1,499 | raw | luck-corrected |
|---|---:|---:|
| the 2 s search agent against the overbettor | −161.5 ± 121.1 | −114.5 ± 78.4 |
| the blueprint agent against it, same deals | +20.4 ± 58.4 | +26.6 ± 47.4 |
| 2 s minus the blueprint agent (paired per deal) | −181.9 ± 129.0 | −141.1 ± 86.8 |
| 2 s minus 0.5 s (paired per deal) | +126.7 ± 111.1 | +108.0 ± 84.7 |

The same breakdown, paired hand by hand with the blueprint agent, luck-corrected:

| first search of the search agent | hands | per 100 of them | share of the −141.1 |
|---|---:|---:|---:|
| none (identical) | 1,083 | 0 | 0 |
| preflop | 262 | −1,731 ± 745 | −151.2 ± 67.3 |
| from the flop, the opponent overbet on a later street | 206 | −112 ± 557 | −7.7 ± 38.2 |
| from the flop, no overbet at all | 1,449 | +37 ± 93 | +17.8 ± 45.1 |

By the street where the hand was decided: preflop showdowns (the all-ins after preflop searches), 33
hands, −8,463 ± 2,053 per 100, share −93.1 ± 38.6; flop showdowns 17 hands, share −45.0 ± 30.0;
hands folded on the turn 407, +269 ± 165 per 100, share +36.4 ± 22.6 (the only significant gain);
the rest not significant.  The answers to the first postflop overbet (206 hands): fold 176, call 6,
raise 24 (17 behind); the blueprint agent in the same hands fold 144, call 13, raise 5, 44 another
way.  The first preflop search, by the hero's hand: bottom half folded 97 of 146 (raised 37, called
12), 50-85% raised 71 of 83, top 15% raised all 33.

What this says, with the 0.5 s runs (R1, R3, R4 are on deals 1,500..2,499, so the comparison is
between parts, each paired with the blueprint agent on its own deals):

* The postflop-overbet hands no longer lose significantly at 2 s (−112 ± 557 per 100, against −841 to
  −1,243 at 0.5 s), and re-raises became rarer (24 of 206 against 24-26 of 128-132): consistent with
  under-convergence at 0.5 s; the balanced-range reading is not needed for these numbers, but with an
  interval of ±557 per 100 hands a loss there is not excluded either.
* The preflop searches still lose at 15.8k iterations (−1,731 ± 745 per 100 of those hands), as at
  3.6k (−2,556 ± 914) and more than at 67k (−487 ± 647): the preflop searches need their own, larger
  budget; the 50-85% hands still re-raise the premium-only overbet at 15.8k.
* Ordinary postflop play shows no significant difference from the blueprint agent at either budget
  (2 s: +37 ± 93 per 100 of those hands; 0.5 s: −25 to −106 ± 110-120).

So check b fails at 0.5 s and at 2 s (against the overbettor −258 ± 74 and −115 ± 78), and the whole
significant loss at 2 s is in the preflop searches.

**d. Seconds per hand** (the search agent's time in its decisions; the rest of a hand, the opponent
and the engine, takes about 0.01 s):

| budget | opponent | hands | hero s / hand | decisions / hand | searches / hand |
|---|---|---:|---:|---:|---:|
| 0.5 s | blueprint | 4,000 | 0.81 | 2.62 | 1.56 |
| 0.5 s | overbettor | 3,000 | 0.82 | 2.53 | 1.58 |
| 0.5 s | random | 300 | 0.38 | 1.51 | 0.74 |
| 2 s | blueprint | 300 | 3.22 | 2.63 | 1.60 |

A searched decision costs its budget plus 30-38 ms at flop roots (the river table) and 3-10 ms
later; with 8 s for preflop searches the agent took 1.52 s per hand against the overbettor (197
preflop searches in 2,000 hands).  Against Slumbot the blueprint agent made 1.08 preflop, 0.83 flop, 0.54 turn and 0.40 river
decisions per hand (46,128 hands of `data/slumbot/hunl200w3_pot16_s0.jsonl`): 1.77 postflop
decisions per hand, so the search agent needs about 1.77 x (budget + 0.03) s per hand, 0.94 s at 0.5 s
and 3.6 s at 2 s, plus Slumbot's own time (1.06 s per hand in the blueprint's match).

## For the Slumbot evaluation (a proposal; the plan is the user's)

* **Not with the 0.5 s defaults**: at 0.5 s the agent shows no edge over the blueprint (−1.7 ± 49.2)
  and its preflop searches are exploitable (−258 ± 74 against the overbettor).
* **Preflop**: Pluribus's rule stays (a search only for a size farther than 0.3 from the grid).
  Slumbot bets off our grid preflop rarely (120 of 17,223 of its preflop bets in the blueprint's
  match), but each such search needs a budget that converges: at 0.5 s (3.6k iterations) they lost
  −2,556 ± 914 per 100 of those hands against the overbettor, with 8 s (67k iterations) −487 ± 647,
  and the spot measured converges by 30k iterations.  Source: Modicum re-solves the preflop with the
  new size for 30 s of MCCFR and caches the solution for the next time that size comes (Depth-Limited
  Solving 2018, pp. 12-13).  The budget of preflop searches (`--search-street-budgets preflop=...`)
  and a cache are for the user to decide.
* **Budget from the flop**: 2 s (the user's test budget).  Against the overbettor it removed the
  significant loss in the postflop-overbet hands that 0.5 s had, and 2 s beat 0.5 s by +108 ± 85
  on the same deals; an edge over the blueprint agent in ordinary postflop play is not shown yet.
* **Time**: the agent needs about 1.77 x (budget + 0.03) s per hand against Slumbot's lines, plus
  Slumbot's 1.06 s: at 2 s about 4.7 s per hand, so 10,000 hands in 13 h (a 95% interval of about
  ±27 bb/100 luck-corrected, from the spread of the blueprint's match: ±13.3 at 40,000) and 40,000
  hands in 52 h (±13); at 1 s about 2.9 s per hand, 40,000 hands in 32 h.
* **Before it**: a 2 s duel against the blueprint agent long enough to see an edge (2,000 deals =
  4,000 hands, 3.6 h, about ±50 bb/100), since 300 hands at 2 s (+85 ± 93) do not show one.

## State of part 3 and how to continue (written 2026-09-25 about 20:45; R5 finished at 20:53 and is analysed above)

Done and committed on the branch `worktree-agent-ae78f3fcf39ad9442` (on top of master bcb1b68):
the agent (`negpluribus/agents/core_search.py`), the value overbettor, the duel with the card-luck
correction (`negpluribus/eval/duel.py`), `--agent search` in `scripts/play_slumbot.py` and
`scripts/eval_archetypes.py`, `scripts/duel_log.py`, `scripts/hand_log_report.py` (with
`--breakdown`), `scripts/overbet_answers.py`, the tests (`tests/test_core_search_agent.py`, 9; every
solve in `tests/test_search_core.py` runs a fixed number of iterations except the time-budget test
itself, `test_solve_respects_budgets_and_returns_distributions`), and checks a-d above.  The two
tests the coordinator named are done: `test_our_taken_actions_are_fixed_for_our_actual_hole_only`
runs 20,000 iterations (commit d328d7f, its logic unchanged); `test_our_average_is_accumulated_every_iteration`
already ran a fixed 40,000 iterations and was flaky for another reason (the fixture's blueprint is
trained on 4 threads, so it differs from process to process, and the test compared two averages
that agree only as the strategy settles); it now checks an exact one-thread identity (commit
9a2c76c).

The runs, all in the session's scratchpad
`C:\Users\AB73~1\AppData\Local\Temp\claude\C--Project-Manchatten-NegativePluriibus\9700a191-7573-420d-8867-e6a768035c26\scratchpad\p3\`
(`check_<name>.txt` the console output, `log_<name>.jsonl` one line per deal, `hands_<name>.jsonl`
one line per hand where present; all `--seed 0`):

| name | hero, opponent, budget | deals |
|---|---|---|
| a_random05 | search 0.5 s, random | 0..149 |
| c_duel05 | search 0.5 s, blueprint | 0..1,999 |
| b_overbet05 | search 0.5 s, overbettor | 0..1,499 |
| d_duel2 | search 2 s, blueprint | 0..149 |
| r2_bp (hands) | blueprint agent, overbettor | 0..2,499 |
| r1_search05 (hands) | search 0.5 s, overbettor | 1,500..2,499 |
| r3_nopre05 (hands) | search 0.5 s without preflop searches, overbettor | 1,500..2,499 |
| r4_pre8 (hands) | search 0.5 s, preflop 8 s, overbettor | 1,500..2,499 |
| r5_search2 (hands) | search 2 s everywhere, overbettor | 0..1,499 |

The R5 analysis above was made with these commands (to redo it, or to analyse another run):

```
set P=<the scratchpad p3 path above>
python scripts/duel_log.py %P%\log_r5_search2.jsonl --paired %P%\log_r2_bp.jsonl
python scripts/duel_log.py %P%\log_r5_search2.jsonl --paired %P%\log_b_overbet05.jsonl
python scripts/hand_log_report.py %P%\hands_r5_search2.jsonl --paired %P%\hands_r2_bp.jsonl --breakdown
python scripts/overbet_answers.py %P%\hands_r5_search2.jsonl --paired %P%\hands_r2_bp.jsonl
```

(`check_r5_search2.txt` ends with the iterations of the preflop searches; the data files the runs
use are in the worktree's `data/`: the binary blueprint, the buckets JSON, the bucket cache.)  No
new search runs with a time budget while other jobs load the machine: they would get less CPU.

## What comes next (proposals)

1. The budget of preflop searches (and Modicum's cache), measured with the duel and the archetypes.
2. The search on master's bucket tables (one mechanism instead of the bucket cache; river tables
   also make the rollouts of preflop searches cheaper).
3. The remaining gap: with 8 s preflop searches (and 0.5 s later) the agent still trailed the
   blueprint agent against the overbettor (−138 ± 98), in the answers to postflop overbets
   (re-raises, where the blueprint agent, reading most overbets as its all-in, only folds or calls)
   and in the 50-85% hands that re-raise a premium-only preflop overbet.  At 2 s the postflop part is
   no longer significant (above).  Two readings of what is left, not measured apart: under-convergence,
   or the search answering an overbet as the balanced size it solves it as, which is right against a
   balanced overbettor and wrong against this value-only one, while the blueprint agent's narrow
   answer to an all-in happens to be right against it.  A **balanced overbettor** would separate them:
   the blueprint whose largest raise or all-in becomes the same off-grid overbet, with the
   blueprint's own range for it (bluffs included).  There is no such archetype yet; it is a small
   variant of `ValueOverbettor` (about 20 lines and a test, half an hour), and the runs cost about 4 min
   for the blueprint agent (2,500 deals), 28 min for the search agent at 0.5 s (1,000 deals) and
   1.8-2.7 h at 2 s (1,000-1,500 deals).
