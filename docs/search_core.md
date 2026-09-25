# Real-time search in C++, part 1: the subgame core (2026-09-25)

The design is `docs/search_design.md` (Pluribus, checked against the supplement) with
`docs/search_research.md` sections 1, 2 and 7.  This part builds the subgame and its solver in
C++ (`csrc/search.h`, bindings in `csrc/bindings.cpp`); depth limits, continuation leaves and the
agent are parts 2 and 3.  The Python search (`negpluribus/cfr/search.py`, `agents/search.py`)
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

## What part 2 needs (depth limit and continuation leaves)

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
