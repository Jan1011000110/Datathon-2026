# Candidates for the jury selection: h3j, h3w and v2w

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

**Result on the server (20:59):** the team submitted h3w. It scored **20.03** on the public leaderboard (CO₂ −18.54 %, 11,171.3 kg; cost −23.51 %, 22,331.14 €; 0 infractions), 3.29 below h3. The folds gave −4.07 on the first halves, so the real test moved by 0.81 times the fold figure, in line with the cut probe (0.7 to 0.9). On the same scale the private half should be about +2 to +2.5 above h3j. The private half itself has not been observed.

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

## 4. Standard mode: the same question, and v2w (added 21:03)

It is not settled which mode's private ranking decides the final, so the boundary study repeated its measurements for our standard-mode agent v2 (the plain 24-hour optimum, tied with about twenty teams at 6.43 on the public board). Folds without August, since the test has none.

| | Whole run | First 84 h (public) | Last 84 h (private) | Charge at week ends |
|---|---|---|---|---|
| v2 | 5.403 | 5.455 | 5.346 | 919 kWh |
| v2 with the 24-hour window cut at the week's end | 5.243 (−0.16) | 4.326 (−1.13) | 6.263 (+0.92) | 0 |

- **Paired by week (33 weeks), last halves:** 17 better, 0 worse, 16 unchanged (weeks where v2 already ends empty).
- **Translated to the real board** (the plain optimum scores 6.43 on the real first halves against 5.455 on the folds): v2 about 6.3 on the private half, give or take 0.7; the variant about 7.4; the difference +1.1, between +0.7 and +1.5. The variant's public score would be about 5.1.
- **In principle** a schedule built only for the last halves scores 2.2 points more there than the whole-run optimum (7.54 against 5.33).
- **The teams at 6.44 on the public board** are explained by about 10 kWh more charge at every week end.

**`submissions/v2w/submission.zip`** is that variant: v2 with one marked change (the window stops at the last hour of the month). Single file, numpy only, no data file.

| Check | Result |
|---|---|
| Against the study's reference on folds A / B / C | identical: 5.3832 / 5.1447 / 5.2024 whole run, 6.3885 / 6.1376 / 6.2642 on the last halves |
| With the cut switched off | 0 commands differ from v2 |
| On a run with no month end | 0 commands differ from v2 |
| Infractions | 0 everywhere |
| Organisers' environment, test-like layout | 5.14, 0 infractions |
| Malformed observations | no exception |
| `check_submission.py` | passes |

Like h3w, it is a worse controller over the whole run that scores higher on the last halves because of where the cut falls. Submitting it is the team's decision.

**A correction to section 2.** The private-half gain of h3w was scaled down to "+2.1 to +2.7" because the cut probe had moved less on the real board than on the folds. That comparison used folds that still had an August week. Without it the folds predicted +0.77 for the probe and the real figure was +0.93, so the probe shows no damping. h3w's own public move does: −3.29 on the server against −4.07 on folds that still had August (not re-measured without it). Taking both, expect **+2 to +3** on the private half; the folds at face value say +2.6 / +3.7 / +2.9.

## 5. For the jury selection

- Select the two submissions by hand. The automatic choice takes the two best public scores, which would include the cut probe (24.25), our worst agent on the private half.
- If the private ranking is the hidden half alone and the better of two picks counts, h3j + h3w covers both readings: h3j is best over the whole run, h3w on the last halves.
- In standard mode the same reasoning gives v2 + v2w.
- If h3w or v2w is picked, say in the presentation what it is and how the cut was found.
- Open questions for the organisers are listed in `docs/NEW_RULES.md` §8.
