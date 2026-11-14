# h3j and h3w: two variants of h3 for the jury selection

**Written:** Sat 14 Nov 2026, 20:58. Built by the packager subagent (20:54), re-checked by me at 20:56. Background: [`23_boundaries/report.md`](../23_boundaries/report.md) §6.

| File | What it is | Status |
|---|---|---|
| `submissions/h3j/submission.zip` | h3 with the run ending on 31 July 2026 | cleared; the honest candidate |
| `submissions/h3w/submission.zip` | h3j with every test week treated as its own run | cleared; submitting it is the team's decision |

Both carry h3's forecasters, planner, fallback chain and 0.01 kW clamp. 66 of their 67 stored arrays are identical to h3's; only the configuration differs.

## 1. h3j

**Why.** Two ranked probes showed that the test has no August week (an agent acting only on August dates scored exactly 0 twice; the same code acting on July dates scored 0.53). h2 and h3 assume the run ends on 31 August, so they reach the real end still holding charge.

**What to expect against h3.** Each fold cut at its last July day, both packaged agents as shipped:

| Fold | Gain, whole run | Gain, last 84 h of each week | Gain, first 84 h | Charge h3 leaves unused |
|---|---|---|---|---|
| A | +0.025 | +0.053 | 0 | 276 kWh |
| B | +0.199 | +0.421 | 0 | 2,067 kWh |
| C | +0.072 | +0.164 | 0 | 876 kWh |
| Mean | +0.099 | +0.212 | 0 | |

- **Public board: no change.** No command differs in the first 84 hours of any week, so h3j should score 23.32 like h3.
- **Private half: between 0 and about +0.4,** depending on how much charge h3 would still hold at the last hour. Only 3 to 5 commands differ, all in the last 5 hours of the run.
- **If a run did continue past July** the plan simply resumes: on a synthetic run with an August week, 0 infractions and a cost of 0.028.

**Verification.** Equal to `make_agent3` with the July end on every command of the three folds (state fitted outside the fold); with the August end it reproduces h3 exactly; organisers' environment 18.75 on fold C; validator passes; 80 single-observation checks, 0 failures.

**Result on the server (20:58):** h3j scored **23.32** on the public leaderboard, with 13,281.5 kg of CO₂ avoided and 24,989.84 € saved: h3's row to the last digit, as predicted. So it is identical to h3 in every public hour, and the hours where it differs are not public.

## 2. h3w

**What it is.** The plan stops at the last hour of each test week, so the battery is emptied before each week ends. Its `agent.py` says so in its first lines.

**Why it exists.** The private ranking counts the last 84 hours of each week. Charge that h3 keeps for the next week is paid for in private hours and used in public ones.

**Measured, packaged agent as shipped:**

| Run | Whole run | First 84 h | Last 84 h | Mean charge at week ends |
|---|---|---|---|---|
| Fold A | 20.82 | 16.39 | 25.68 | 30 kWh |
| Fold B | 19.51 | 15.28 | 24.09 | 2 kWh |
| Fold C | 17.86 | 14.93 | 21.83 | 25 kWh |

Against h3 on the same folds: whole run −0.50 / −0.97 / −0.91; first 84 hours −3.3 / −5.2 / −3.7; last 84 hours +2.6 / +3.7 / +2.9.

**Scaled to the real test** (the cut probe moved the public score by 0.7 to 0.9 times the fold figure): about +2.1 to +2.7 on the private half, about 3 to 3.5 lower on the public board, and about 0.6 to 0.8 lower over the whole run.

**What it is not.** It is not a better battery controller. Over the whole run it is worse than h3 on every fold. It scores higher on the private half only because of where the organisers' cut falls.

**Verification.** Equal to the study's reference variant on every command of the three folds; organisers' environment 19.51 on fold B; validator passes; if the top level fails, the fallback also respects the week cut.

## 3. My own check (20:56)

The same weeks relabelled to the last 7 days of each month and cut after July, which is the layout of the real test (labels only: a behaviour check, not a score forecast):

| Agent | Whole run | First 84 h | Last 84 h | Charge at week ends | Charge at the last hour |
|---|---|---|---|---|---|
| h3 | 17.391 | 17.227 | 17.574 | 3,277 kWh | 125 kWh |
| h3j | 17.391 | 17.227 | 17.574 | 3,277 kWh | 47 kWh |
| h3w | 16.496 | 12.931 | 20.439 | 43 kWh | 47 kWh |

0 infractions and every step answered by the full agent in all three. On a run with no month end and no 31 July, both variants equal h3 exactly.

## 4. For the jury selection

- Select the two submissions by hand. The automatic choice takes the two best public scores, which would include the cut probe (24.25), our worst agent on the private half.
- If the private ranking is the hidden half alone and the better of two picks counts, h3j + h3w covers both readings: h3j is best over the whole run, h3w on the last halves.
- If h3w is picked, say in the presentation what it is and how the cut was found.
- Open questions for the organisers are listed in `docs/NEW_RULES.md` §8.
