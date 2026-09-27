# AIVAT: low-variance evaluation (2026-09-25)

## State and next steps (kept current; read this first)

Branch `worktree-agent-af77f60b1bc19ce81` (worktree `.claude/worktrees/agent-af77f60b1bc19ce81`), on master c1ece18.
Scratch outputs: `SP\aivat\` with SP = `C:\Users\AB73~1\AppData\Local\Temp\claude\C--Project-Manchatten-NegativePluriibus\9700a191-7573-420d-8867-e6a768035c26\scratchpad`
(may be cleaned by the system: the numbers that matter are copied into this file).

Done and checked:
* the estimator (Python reference + C++ twin), heuristic v1 fixed (cceb328) before any evaluation log, predictions
  written before measuring (section 5);
* tests/test_aivat.py, 9 tests, all pass (section 6): C++ = reference bit for bit per branch value; unbiasedness
  on a small HU game (self-play and a random bettor); every term mean zero, also for an arbitrary value function;
  the known-player model = the blueprint agent (keys, coins, sampling law);
* the model against the Slumbot match itself: all 60,995 hands, 173,207 decisions, 0 mismatches;
* scripts/aivat_eval.py (Slumbot logs and duel hand logs, `--known villain` for duels search vs blueprint);
* logged range strategies (section 7) in the reference and C++, tested (commit 7a428be).

Computed (section 6): Slumbot, 60,995 hands: AIVAT -16.9 +- 4.2 bb/100, sd 5.26 bb per hand (raw 16.63, -68%; card luck 13.34); the part-3 R2 log (blueprint vs overbettor), 5,000 hands: +27.2 +- 13.2, sd 4.82 (raw 18.53, -74%).

Commands (from the worktree root; P3 = the part-3 worktree's `data` folder, which holds the `.bin` blueprint, the
buckets and the warm bucket cache; bucket tables are built into `data/bucket_tables` on first use, about 1 min):

    python scripts/aivat_eval.py --slumbot data/slumbot/hunl200w3_pot16_s0.jsonl --out data/aivat/slumbot_aivat.jsonl \
        --blueprint P3/blueprint_hunl200w3_pot16_s0.bin --buckets P3/buckets_hunl200w3_pot16_s0.json \
        --cache P3/bucketcache_hunl200w3_pot16_s0.bin --root-cache data/aivat/root_v1.npz --threads 4 --first 53111
    python scripts/aivat_eval.py --duel SP/p3/hands_duel2_bp.jsonl --known villain --out data/aivat/duel2_bp_aivat.jsonl \
        --blueprint ... --buckets ... --cache ... --root-cache data/aivat/root_v1.npz --threads 1

Both resume from `--out` (per-hand lines appended), so a growing log costs only its new hands; the card-luck column
of a Slumbot log is cached in `<out>.luck.json`.

Left / how to continue without me:
1. Slumbot: the match runs until about 28.09; rerun the first command then (only new hands are scored).
2. Tonight's duel "search 2 s vs blueprint" (`SP\p3\hands_duel2_bp.jsonl`, 4,000 hands, logged with --log-hands):
   after the line `duel2_bp end` in `SP\queue_status.txt` (about 03:30), the second command, one thread (about 20
   min); the numbers are the blueprint's: the search agent's result is minus them.  Record in section 6.
3. The search agent against third parties: the part-3 agent implements the evaluation mode and the range log of
   section 7 (evaluator side ready: `AivatHand.x_rows`), and a reader from its log format to `LoggedRows`.
4. Merge: new files plus additive lines in csrc/bindings.cpp and csrc/buckettable.h (handoff_main.md, merge order 3).


Code: `negpluribus/eval/aivat.py` (Python reference; it defines the numbers), `negpluribus/eval/aivat_values.py`
(value functions, reference), `csrc/aivat.h` + `csrc/aivat_bindings.h` (the fast C++ twin),
`negpluribus/eval/aivat_fast.py` (wrapper), `scripts/aivat_eval.py` (applied to logs), tests `tests/test_aivat.py`.
Why: the user's acceptance rule "every step must beat the previous one by results", in particular "the search
agent beats the plain blueprint"; without variance reduction +-10 bb/100 cost about 97,000 hands
(docs/search_research.md, section 8).

Labels: **source** (what and where), **measured** (how), **hypothesis** (and how to check it).

The first version of this file (commit cceb328, 2026-09-25 20:25:38 +0300) was written in Russian; it fixed
heuristic v1 and the predictions of section 5 before any evaluation log was read.  This English text states the
same definition and the same predictions; nothing in them changed.

## 1. The estimator (source: Burch, Schmid, Moravcik, Morrill, Bowling, AAAI 2018, arXiv 1612.06915 v2)

It estimates the expected net of the KNOWN player x (our agent, whose strategy can be queried for any hole it
could hold) against an UNKNOWN player y (Slumbot, a test bot, a human).  Section "AIVAT Value Estimate", Eq. (1):

    AIVAT(z) = sum_{z' in W} pi_Pa(z') v(z') / sum_{z' in W} pi_Pa(z')  +  sum_{H in H} k_H(z)

* Pa = {chance, x}; pi_Pa(h) = the product of chance's and x's probabilities along h (section "Background").
* The correction term of a part H (section "AIVAT Correction Terms", the displayed formula), a_O = the observed
  action or chance outcome there:

      k_H(z) = sum_a sum_{h in H} pi_Pa(h a) u_h(a) / sum_{h in H} pi_Pa(h)
             - sum_{h in H} pi_Pa(h a_O) u_h(a_O) / sum_{h in H} pi_Pa(h a_O)

* Lemma 1 / Theorem 1: E_z[k_H(z)] = 0 for ANY functions u_h(a); the base value (section "AIVAT Base Value",
  imaginary observations of Bowling et al. 2008) is unbiased by itself.  Hence the whole estimate is unbiased,
  whatever y does and whatever the value function is.
* The partition H is the authors' HUNL choice (section "Experimental Results": "Each H has states with identical
  betting, public board cards, and private hole cards for any players in Po"): a part = all states with the same
  actions, board and y hole; x's hole free.  Conditions 1-3 hold: y's reach is the same across a part, no state of
  a part is a prefix of another, x's legal actions depend on the public state only.
* The seat is a 50/50 chance event with its own term, as in the paper ("we model this as an extended game where
  there is an initial 50/50 chance event that assigns a position to the agent, along with a AIVAT correction term
  for the position").  Our logs alternate seats exactly, so these terms sum to zero over each pair of hands.
* x's own hole deal gets no term: it is inside the imaginary observations (Figure 1 of the paper: terms for P2's
  card and the public card, none for P1's card).

Per hand (sums over the 1326 hole combos c of x; d = y's actual hole):

| part | formula |
|---|---|
| R(c) | x's reach: the product of its probabilities of its actions so far holding c, summed over its translation coins (below) |
| w(c) | 1[c shares no card with d and the board so far] * R(c): pi_Pa(h) up to a factor common to the part |
| root | C_pos - mean_{c disjoint from d} u_root(pos, c, d) (y's hole deal), plus (C_SB + C_BB)/2 - C_pos (the seat) |
| x decision | sum_a sum_c w(c) s(c,a) u(c,a) / sum_c w(c) - sum_c w(c) s(c,a_O) u(c,a_O) / sum_c w(c) s(c,a_O) |
| board deal f | sum_c w(c) E_f'[u(c,f')] / sum_c w(c) - sum_{c disjoint from f_O} w(c) u(c,f_O) / sum_{c disjoint from f_O} w(c); flop, turn and river one node each, run-outs after an all-in too |
| base | sum_{c disjoint from d and the board} R(c) net_x(c) / sum R(c) |
| y decision | no term (y's strategy is unknown) |

C_pos = the exact mean of u_root(pos, c, d) over every pair of disjoint holes.  The paper's remark on the board
card (paragraph on Figure 1) is followed: the first half of a board term includes the holes of x that contain the
card that came, the second half only holes without it.

**Base value and the last term (derived).**  When the last event of the hand is an x action or a board card whose
branch value is the exact terminal value, the base value equals the second half of that last term and the two
cancel: the chance of the last card and x's last choice are integrated out exactly.  What remains uncorrected are
y's actions (and the heuristic's error).  Our W (same actions, board, y hole) is finer than the paper's in one
case (an all-in run-out, where the paper's W may vary the last card), which gives the same total by that
cancellation; any W with condition 1 is unbiased (Bowling et al. 2008).

### The known player: the blueprint agent (negpluribus/agents/blueprint.py), exactly as it plays

* Key as the agent builds it (bucket of c on the board, history), the C++ lookup `policy(key, legal)`; an
  unknown key plays check/call (the agent's fallback).
* The sampling law: the agent takes the first action with r < the running sum, else the last one; r =
  random.random() = k/2^53.  Each action's probability is computed exactly to 2^-53: P(r < a) = ceil(a 2^53)/2^53
  (`sampling_law`).
* Bet translation (pseudo-harmonic): the agent flips one coin per bet of the hand (the opponent's, and its own
  when not exactly on the grid) and keeps it for the rest of the hand.  The coins are x's private chance,
  independent of its hole; the states of a part differ in (c, coins).  So R(c) = sum_t P(t) prod_k s(c, a_k | t)
  over the coin outcomes t, not a product of per-decision averages (one coin feeds several decisions).  A coin's
  probability is exact (`translation_outcomes`; a test checks it against BetGrid.from_concrete itself by
  bisection over the 2^53 coin values).
* **Measured** on the whole Slumbot log (60,995 hands, 173,207 of our decisions, 251 of them off the map, 585
  events with a randomized coin): at every decision the model's key history (one of its coin branches), bucket of
  our hole, legal actions, probabilities (to the logged 4 decimals), sampling row and the action agree with what
  the agent looked up during the match: 0 mismatches (`SP\aivat\check_model_slumbot.py`, 687 s, one thread).

### Unbiased with noisy values

u may be any function, Monte-Carlo estimates included, if (1) its value does not depend on the outcome observed at
the node and (2) both halves of a term use the same numbers.  So: the branch values of a node are computed once
(random streams seeded from (seed, hand number, node number, combo, rollout number), not from the outcome) and used
both in the expectation and in the taken branch.  Where the expectation over a chance node cannot be enumerated
(flop, turn), the first half is an unbiased Monte-Carlo estimate of E_f[u(c,f)] (rollouts from the state before
the deal that deal the cards themselves) independent of the dealt cards; E over the streams of (first half - E_f of
the second half) is 0 because a rollout from the state before the deal and a rollout from the state after it
continue with the same law.  If x's action closed the street, the first half of the deal takes the numbers of
that action's branch (the two terms' noise then cancels in part; both still have mean zero).  Checked by the
tests (section 6).

## 2. Heuristic v1 - FIXED BEFORE any evaluation log (Kim & Sandholm 2026, arXiv 2605.14261)

Fixed on 2026-09-25 at 20:25:38 +0300 in commit cceb328; it does not change afterwards (a new version gets a
new name, v2, and is fixed before the data it is applied to).  Before it was fixed it had seen no hand of the
Slumbot or part-3 logs, only hands I generated (the blueprint against itself), on which the speed and the rollout
noise were measured.

**Parameters (HUNL 200bb):** blueprint `blueprint_hunl200w3_pot16_s0` (what the known player plays in both logs;
the `.bin` of part 3 was written from the `.json` of the Slumbot match by scripts/export_json.py; **measured**: both
hold 3,042,766 infosets and give exactly equal `policy(key, legal)` at all 9,036 distinct lookups of the match),
buckets `buckets_hunl200w3_pot16_s0.json`, grid preflop 0.5/1/3, postflop 0.5/1/2/4 + all-in, 3 raises per street;
rollouts (preflop, flop, turn) = (4, 8, 8); 2000 boards for a preflop all-in; root table 256 rollouts, seed 0;
evaluator seed 0.

**Definition.**  u(c, state) = x's expected net in chips when BOTH players play the blueprint from the state on
(the agent's key, the bet translation deterministic - nearest in the pseudo-harmonic sense, as in the search's
rollouts; an unknown key checks/calls), x holding c, y holding its actual hole, unknown board cards uniform.  The
authors' choice for HUNL: self-play values of an equilibrium of a small abstraction (8M infosets); ours: self-play
values of our blueprint on the full state (both holes known).

**How it is computed** (the same in Python and C++):

| state | method |
|---|---|
| end of the hand (fold, showdown) | exact |
| nobody acts any more (all-in), 1-2 cards missing | exact: every run-out |
| preflop all-in (5 cards missing) | 2000 random boards per combo, stream (seed, hand, node, c, 1000000) |
| river, decisions ahead | exact: the river betting tree enumerated, V(c) = A[bucket of c] + S[bucket of c] * s(c) |
| one card before river decisions | exact: the same for every river card, averaged |
| preflop / flop / turn, decisions ahead | 4 / 8 / 8 rollouts per combo and branch; first the missing cards (index floor(u n) into the sorted rest), then one uniform per decision (the agent's rule); streams (seed 0, hand number, node number, c, k) |
| root (before any action) | a table: 256 rollouts per seat and suit-isomorphism class of the pair (c, d) (93,769 classes), seed 0; C_pos = its exact mean over all pairs |

The hand number for the streams is the log's hand / line number (known before the hand, independent of its
outcome).  Buckets come from bucket tables (`aivat_build_tables`: the river in batches per canonical board, flop and
turn through the warm cache): the same numbers as `bucket()` (**measured**: 6,000 random hands, 0 differences).

**Why this choice (measured on development data, not on evaluation logs).**  1,000 hands of the blueprint against
itself (200bb, the Slumbot grid): raw sd 17.49 bb/hand, AIVAT 1.89 (-89%); rollout noise sd 0.67 bb/hand (two
evaluations with different seeds of the same hands); 0.20 s per hand per thread.  Twice the rollouts (8/16/16) gave
noise 0.54 at 0.39 s: not worth the double cost.  The root table is cheap (81 s on 2 threads, once).

## 3. The C++ twin

`csrc/aivat.h` mirrors the reference operation for operation where a result is a per-combo number (branch
values, reach products, sampling laws, coin probabilities): those agree bit for bit (tests compare every branch
value vector with `np.array_equal`).  Sums over combos are sequential in C++ and pairwise in numpy: terms agree
to about 1e-14 relative.  Additive changes outside the new files: `BucketTables::set_table` (csrc/buckettable.h)
and two lines in csrc/bindings.cpp (the include and `register_aivat(m)`).

## 4. Cost

0.19-0.20 s per hand on one thread (**measured**, development hands and the part-3 log; section 6 for Slumbot):
per x decision about 6 branches x 1326 combos of values, most of them rollouts (about 77,000 rollouts per hand);
the river is exact.  Python reference: 1-6 s per hand at 1 rollout (tests only).

## 5. Predictions (written in cceb328, before any measurement on the logs)

* Slumbot (our blueprint against Slumbot): raw sd 16.93 bb/hand (slumbot_luck.py).  **Prediction: AIVAT -65%
  (range -55...-75%), sd about 5.9 bb/hand.**  Reasoning: the paper got -68% with a coarse heuristic (Figure 5);
  ours is sharper; but 200bb (bigger pots on the late streets, where the unknown y decides) and Slumbot plays
  differently from our blueprint, on which the heuristic is built.
* The part-3 log against the value overbettor (hands_r2_bp.jsonl): **prediction -75% (range -65...-85%)**: the
  opponent is the same blueprint except for overbets with strong hands, closer to self-play.

## 6. Checks and results

### Tests (tests/test_aivat.py, 9 tests, all pass)

**Measured on the test game** (`SP\aivat\toy_numbers.py`, `data/aivat/toy_numbers.txt`: the same game and seeds as
the tests, larger samples; 30bb, values with 1/2/2 rollouts; bb/100 +- 95%):

| x (known) against y | hands | raw | AIVAT | sd per hand raw -> AIVAT | AIVAT - raw |
|---|---|---|---|---|---|
| blueprint agent, self-play (exact value 0) | 3,000 | -18.4 +- 56.5 | **-2.9 +- 8.0** | 15.80 -> 2.24 (-86%) | +15.5 +- 56.4 |
| blueprint agent vs random bettor | 1,500 | +90.5 +- 78.5 | +106.9 +- 23.6 | 15.51 -> 4.67 (-70%) | +16.4 +- 74.7 |
| agent sampling from logged quantized rows vs random bettor, seed 52 | 1,200 | -35.7 +- 86.7 | +79.8 +- 22.0 | 15.32 -> 3.89 (-75%) | +115.5 +- 82.7 |
| the same, seeds 53 / 54 / 55 | 3 x 2,000 | +60.9 / +108.7 / +84.7 | +85.1 / +83.2 / +84.9 | about 15.4 -> 3.8-3.9 | z +0.73 / -0.75 / +0.01 |

Every term kind's mean is within its 95% CI of 0 in all three first rows.  The 2.7 standard errors of the seed-52
row came from its raw mean (-35.7 against a true value near +84: the three replications, 6,000 hands, give AIVAT
+83.2...+85.1 and raw - AIVAT within 0.75 standard errors), not from the estimator.

What the tests check:

* streams, sampling law, coin probability: Python = C++ on random inputs; the sampling law = the agent's loop
  counted on a 200,000-point grid of r;
* translation coins = `BetGrid.from_concrete` (tokens at both coin extremes, the probability by bisection over the
  2^53 coin values), 400 random raises, over 100 of them randomized;
* the model = `BlueprintAgent`: 60 hands against a random bettor, every decision's row for the actual hole under
  the coin outcome the agent drew = the sampling law of the probabilities the agent looked up;
* C++ = reference on 14 hands (random bettor and self-play): every branch value vector bit for bit, terms and
  values to 1e-9 relative, one root-table entry per seat recomputed;
* unbiasedness on a small HU game (30bb, grid 1 / 0.5-1, 8 E[HS] buckets, a 6,000-iteration blueprint as x; values
  with 1/2/2 rollouts): self-play (exact value 0, 1,500 hands) - the AIVAT mean within 4 standard errors of 0 and its
  sd below 0.35 of the raw sd; against a random bettor (1,000 hands; off-grid sizes: the coins matter) - AIVAT - net
  has mean 0; every term kind has mean 0; with an arbitrary value function (`AdditiveCardValues`, exact chance
  expectations, 250 hands) every term kind and AIVAT - net have mean 0 (Lemma 1 holds for any u);
* logged range strategies: the format round-trips and quantizes to rows of sum 65535 within 1/65535 of the input;
  C++ = reference with logged rows (8 hands, bit for bit per branch value); an agent sampling from its logged rows is
  evaluated without bias (800 hands, AIVAT - net within 4 standard errors).
* Durations (measured, 2 threads, the machine shared with a 12-thread training): the two statistical tests 358 s and
  183 s, the whole file 586 s.

### Part-3 log: the blueprint agent against the value overbettor (hands_r2_bp.jsonl)

x = the hero (blueprint agent `hunl200w3_pot16_s0`), y = `ValueOverbettor` (x2.5); 2,500 duplicate deals =
5,000 hands, 200bb; a deal is one sample of the CI; "for +/-10" = hands for a 95% half-width of 10 bb/100.

| estimate | bb/100 | 95% CI | sd per hand, bb | hands for +/-10 | hands for +/-5 |
|---|---|---|---|---|---|
| raw | +35.6 | 44.0 | 18.53 | 96,672 | 386,687 |
| card luck (chance nodes, exact equities; the log's `luck_bb`) | +39.2 | 34.6 | 14.29 | 59,757 | 239,026 |
| **AIVAT v1** | **+27.2** | **13.2** | **4.82** | **8,654** | **34,617** |

* sd per hand -74% (prediction -75%, range -65...-85%: inside); CI 3.3 times narrower than raw; the hands needed
  for +-10 bb/100 fall 11-fold (raw) and 7-fold (card luck).
* Mean terms (bb per hand, all hands): root -0.007 +- 0.030, seat 0.000 +- 0.005, x -0.277 +- 0.299, flop
  -0.035 +- 0.072, turn +0.029 +- 0.095, river +0.066 +- 0.162: all within their CI of 0; AIVAT - raw = -0.085 bb per
  hand (-8.5 bb/100, within the raw CI).
* By seat: BB raw -25.8 +- 66.2, AIVAT +28.5 +- 15.9; SB raw +97.1 +- 78.5, AIVAT +25.8 +- 21.4 (per-hand CIs).
* Evaluator: 0.193 s per hand on one thread (median 0.181, max 0.7); 484 s for 5,000 hands on 2 threads.

### Slumbot: our blueprint agent against Slumbot (data/slumbot/hunl200w3_pot16_s0.jsonl)

x = our blueprint agent `hunl200w3_pot16_s0` (the match's JSON blueprint; the model reads the equal `.bin`), y =
Slumbot, 200bb, positions alternating.  The log up to its last complete line at 20:40 on 25.09 (60,995 hand
records; snapshot `data/aivat/slumbot_snapshot_60995.jsonl`, sha256 fbe1a1597361a51d...): every one scored (a
result, Slumbot's cards, engine check "match").  The card-luck column is scripts/slumbot_luck.py's correction on the
same hands in the same order.

| estimate | bb/100 | 95% CI | sd per hand, bb | hands for +/-10 | hands for +/-5 |
|---|---|---|---|---|---|
| raw | -15.6 | 13.2 | 16.63 | 106,271 | 425,082 |
| card luck (chance nodes) | -14.8 | 10.6 | 13.34 | 68,327 | 273,308 |
| **AIVAT v1** | **-16.9** | **4.2** | **5.26** | **10,623** | **42,491** |

The first 53,111 hands (the count of docs/slumbot.md): raw -15.1 +- 14.4 (sd 16.93), card luck -14.5 +- 11.5 (sd
13.58) - both exactly the numbers slumbot_luck.py gave on them - and AIVAT -16.1 +- 4.6 (sd 5.38).

* sd per hand -68% against raw, -61% against the card-luck correction (prediction -65%, range -55...-75%, sd about
  5.9: inside; the measured sd 5.26 is a little better than predicted); hands for +-10 bb/100: 10.0 times fewer
  than raw, 6.4 times fewer than card luck.
* Mean terms (bb per hand, all hands): root -0.000 +- 0.008, seat -0.000 +- 0.001, x -0.042 +- 0.079, flop -0.026
  +- 0.027, turn +0.002 +- 0.029, river -0.021 +- 0.045 (flop at 1.9 standard errors, the others below 1.1; six
  kinds); AIVAT - raw -0.013 bb per hand (-1.3 bb/100, within the raw CI).
* By seat (all hands): BB 30,497 hands, raw -18.8 +- 17.5, AIVAT -10.5 +- 5.0; SB 30,498 hands, raw -12.4 +- 19.7,
  AIVAT -23.3 +- 6.7.  Our blueprint loses to Slumbot: -16.9 +- 4.2 bb/100 excludes 0 (raw -15.6 +- 13.2 did too,
  barely).
* Evaluator: 0.225 s per hand on one thread (median 0.210, max 1.5; the run shared the machine with a 12-thread
  training part of the time: 0.20-0.22 without it); 60,995 hands took about 70 min on 2-3 threads.  The card-luck
  column: 1,064 s on one thread.

## 7. What the search agent must log to be a known player

**Requirement (Lemma 1).**  At every decision of x the estimator needs s(c, a) for all 1326 holes c: the
probability that x, holding c at this same public history and with the same private randomness, takes each
action.  For the blueprint agent that is a lookup.  For the search agent it is the output of the search, and it is
the agent's strategy only if that output does not depend on the hole x actually holds.

**Why today's search agent cannot simply log its table** (read in csrc/search.h and
negpluribus/agents/core_search.py of the part-3 worktree, 2026-09-25).  The search reads our actual hole in three
places:

1. forcing: our actions already taken in this round are forced for our actual hole only (`SubgameSearch::traverse`,
   `ctx.actual`);
2. focus: a share `focus` = 0.5 of our traversals deal our actual hole and follow the real path;
3. the played row `r["average"]` is accumulated for our actual class only (`ctx.ours`); other classes only have the
   node table's average (`path_strategies`), accumulated differently.

And one rule of the agent reads it too: a preflop search is triggered when the blueprint has no strategy at OUR
key.  So the table's strategies for the other holes are not what the agent would have played with them (holding
another hole it would have forced and focused on that one), and using them as s(c, a) biases AIVAT (Lemma 1 needs
the true probabilities of every state of a part).  **Hypothesis**: the bias is small; the acceptance rule needs
"unbiased", not "small", so this is not an option.

**Evaluation mode the part-3 agent has to provide** (and play in, in every evaluated match):

* (a) a hand-blind search: it does not read our actual hole.  No forcing (our earlier actions of the round are
  free for our whole range, as the opponents' are), focus = 0, and the played row comes from the same range-wide
  object as the logged rows (e.g. the table average for every class, or `ours` accumulated for every class of our
  decision node);
* (b) every branch of the agent's procedure (search or blueprint, budget, depth rule) depends only on public
  information (actions, board, stacks) and on randomness independent of the hole; the preflop off-map trigger
  becomes public ("the blueprint lacks a strategy for some class of our range at this node") or goes;
* (c) the agent samples from the logged, quantized row of its actual hole: integers q_a >= 0 summing to exactly
  Q = 65535, u = rng.randrange(65535), the first a with u < q_0 + ... + q_a.  Then s(c, a) = q_a / 65535 exactly: the
  logged numbers are the probabilities it used, and the quantization loses nothing for AIVAT (it changes the
  played strategy by at most 1.5e-5 per action).

With (a)-(c) x's private randomness S (seed, the clock's iteration count, thread races) is independent of the hole:
the parts can be refined by S (states of a part differ only in c and share S), pi(S) cancels in every ratio, and S
needs no term (every term has mean zero on its own, so leaving one out keeps the estimate unbiased).  The search's
translation-free concrete actions need no coins; decisions the blueprint plays keep the blueprint model.

**Log per decision** (a field "aivat" in the decision record of hand logs and Slumbot logs):

| key | content |
|---|---|
| `actions` | `[[type, amount], ...]`: the concrete actions in the order of the table's columns (grid actions + inserted ones) |
| `q` | base64 of zlib(little-endian uint16 [rows x A]); rows = the combos (a < b, index order of csrc/search.h ComboTable) that share no card with the board: 1176 on the flop, 1128 on the turn, 1081 on the river; each row sums to 65535 |
| `played` | index of the action played |
| `by` | "search" or "blueprint" (a blueprint decision needs no `q`: the evaluator models it) |
| `rule`, `seed`, `iterations`, `seconds` | audit only: which rows (e.g. "table_average"), the search's seed and effort |

**Size (measured, `SP\aivat\range_log_size.py`, `data/aivat/range_log_size.txt`):** 12 searches of 1 s (master
SubgameSearch, focus 0, depth "pluribus", 1 thread) on flop, turn and river spots of self-generated HUNL hands; the
table average of every combo with a node at our decision, quantized as above: 509-1,090 rows of 6 actions (2 on one
river node), 2.2-13.1 KB raw, **8.4 KB per decision on average as base64(zlib), 11.5 KB at most**.  With all rows
(up to 1,176 on the flop) about 12-14 KB (**hypothesis**, from the same compression ratio).  Our agent decides about
1.5 times per hand from the flop on (Slumbot log: 2.84 decisions per hand, preflop included), so about 15 KB per
hand, 1.5 GB per 100,000 hands.

Holes whose class has no node in the table (the solver never reached it: in the spots above between 86 and 667 of
the combos off the board) still need a row: the agent must play them by a fixed rule (e.g. uniform, or the
blueprint's row) and log that row like any other, because a known player's strategy must be defined for every hole
it can hold.

**No logging at all for duels search vs blueprint (derived from Theorem 1).**  AIVAT needs ONE known player.  In a
duel of the search agent against the blueprint agent the blueprint is known exactly, and AIVAT with x = the
blueprint estimates the blueprint's result, i.e. minus the search agent's, unbiased whatever the search does (it
is y).  `scripts/aivat_eval.py --duel hands.jsonl --known villain` does it with the evaluator as it is (the duel
hand logs of scripts/eval_archetypes.py --log-hands already hold everything).  So the user's acceptance test
"search beats blueprint" can be decided now; the range log is needed for the search agent against third parties
(Slumbot, the overbettor, people) and for the both-known variant (the paper's Leduc: one known -99.8%, both -99.9%,
Figure 2).

**The evaluator side is ready.**  `LoggedRows` (negpluribus/eval/aivat.py: `from_probs` quantizes, `to_json` /
`from_json` is the `actions` + `q` format above) and `AivatHand.x_rows` (per decision of x: None = the blueprint
model, or the logged rows); the C++ evaluator reads them too (`x_rows` in the hand dict).  Tested
(tests/test_aivat.py): the format round-trips; C++ = reference with logged rows, bit for bit per branch value; an
agent that samples from its quantized logged rows (as (c) asks) is evaluated without bias.

## 8. Open questions

1. **The search agent's evaluation mode (section 7) changes the agent**: no Pluribus-style freezing of our actual
   hole, focus 0.  Whether it plays worse is not measured (**hypothesis**: little, heads-up).  A hole-independent
   alternative that keeps consistency: freeze our WHOLE range on the round's earlier actions to the previous
   search's strategy (DeepStack-like); it needs likelihood overrides for our seat on the current street in
   csrc/search.h (today only earlier streets).  The user and the part-3 agent decide.
2. **The existing search logs (R1, R3, R4, R5 against the overbettor)** cannot be scored with the search agent as
   the known player (no range log, and the forcing above).  The overbettor could be the known player instead: it is
   the blueprint except for overbets with "strong" hands, and "strong" before the river is a Monte-Carlo equity
   test (400 samples, a random seed), so its strategy for every hole is the blueprint's times the probability of
   passing the test (a trinomial tail from the exact per-sample win/tie/loss probabilities).  Not built: a new model
   whose exactness rests on the PRNG, like the coins.
3. **Duels search vs blueprint** need hand logs (`--log-hands`); the part-3 duels (c) and (d) logged deals only.
4. **AV-AIVAT** (arXiv 2608.06362, anytime-valid stopping) needs a bound on one hand's AIVAT value; here +-400 bb
   (two stacks) against an sd of about 5 bb: the confidence sequence would be wide.  **Hypothesis**; read the paper
   before building it.
5. **A sharper heuristic (v2)** would help most where v1 is weakest (section 6: the largest residual terms); by
   Kim & Sandholm it must be fixed before the data it is judged on, so it can only be evaluated on hands played
   after it is fixed (or on a held-out part fixed in advance).
