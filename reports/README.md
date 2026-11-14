# Claude's work log — battery challenge

This is the one place to see what I'm doing, what I've found and what's coming. I update it every time something lands. Branch: `claude/battery-agent`.

**Last updated:** Sat 14 Nov 2026, 16:10

---

> **Start here:** [`06_edge/report.md`](06_edge/report.md). At 16:00 the organisers said no team has reached the maximum on the *private* ranking. On clean data a plain optimiser is the maximum, so the private half most likely contains bad data (outages, a dead week). Our agent already has an outage guard; **agent v2** (`submissions/v2/agent.py`) adds every tested protection and is the recommended submission.

## 1. Where we are, in five lines

1. **A submission-ready agent exists:** [`agents/v1/agent.py`](../agents/v1/agent.py). Single file, numpy only, 0 infractions, 0.25 s for a full run, passes `check_submission.py`.
2. **It is essentially optimal.** On my three test-like validation folds it scores 5.96 / 5.78 / 5.77. The best score *possible with perfect knowledge of the future* on those same folds is 5.96 / 5.78 / 5.79.
3. **We are on the leaderboard: team Fa, 10th, 6.43** (submitted 13:57 our time; section 3). Ten teams are tied at 6.43 with the same optimum, unrounded 6.430–6.433, ordered by submission time.
4. **One team is at 6.44** ("Programi qui programi", unrounded ≈ 6.438, +0.0066 over the pack). My best explanation is that it is an effect of how the public half is cut, not a better controller (section 3a). I am not certain, and I changed my mind twice getting here; the section says what is measured and what is inferred.
5. **The hidden test is clean.** Our result matches two teams without an outage guard to the cent, so the guard never acted. The damaged weeks were dropped, not zero-filled. The guard stays (it costs nothing) but it is not an edge on this test.
6. **The jury presentation is still the other half of the result.** Two subagents are producing analysis and figures for it.

![One week of agent v1](00_overview/figs/week_v1.png)

*One week of v1 on train data. Top: the signal the score rewards, with the hours the agent charges (blue) and discharges (red). Bottom: stored energy; the agent's line sits exactly on the perfect-foresight line. Over the whole train set the two take the same action in 99.2% of hours, and the battery does about 1.4 full cycles per day.*

---

## 2. Findings so far

| # | Finding | Evidence | Why it matters |
|---|---|---|---|
| F1 | **Solar is negligible and the campus always draws more than the battery can deliver.** Mean load 1,120 kW; PV peaks at 144 kW (mean 26 kW); net load is under 500 kW in 0.5% of hours. | 5,993 hourly rows of the official table | The battery can always discharge at full power usefully. Forecasting load or solar barely matters. |
| F2 | **The score is linear in the battery schedule.** Each kWh moved is worth `w = co2 + 0.257 × price` at that hour. We see `w` for the next 24 h. | Follows from F1 and the score formula | The problem is pure arbitrage on a known signal: charge when `w` is low, discharge when it is high, subject to 500 kW, 2,000 kWh and 90% efficiency. |
| F3 | **Seeing 24 h ahead is enough.** A plain 24-hour optimiser gets 99.7–99.99% of the perfect-foresight maximum. | `experiments/00_baselines/results.csv`; `experiments/01_dp_solver/verify.py` | No headroom left for clever forecasting. Gains now are in the third decimal. |
| F4 | **The hidden test has damaged weeks.** Meter and inverters have no data at all from 1 Nov to 1 Dec 2025 (this covers the November test week). The inverter file ends on 16 Aug 2026 (the August test week probably has no PV). | Gap listing of `contador.csv` / `inversor.csv`; `experiments/06_zero_week/results.csv` | Either the organisers dropped those weeks, or they filled them with zeros. **Measured:** with one zero-load week, an optimiser without a guard loses 0.5–0.9 points; v1's guard recovers most of it but still ends 0.06–0.12 below the oracle. This is the largest improvable loss we know of, if the scenario is real. |
| F5 | **There is a real outage in the train data:** the meter reads exactly 0 for 14 h on 17–18 June. | `load_kw == 0` rows | On the fold containing it, an agent with no outage guard loses 0.04 points. |
| F6 | **The starter-kit heuristic scores ~0** on our folds: it causes 4–10 infractions per run because its clamp has no safety margin. | `experiments/00_baselines/results.csv` | Teams that only tweak the template will be far behind. |
| F7 | **The public-leaderboard baseline is ~22% lower than train data predicts** for 2,016 h (about 143,700 kg CO₂ and 237,600 € for the full test, against ~184,000 kg and ~313,000 € expected). | Derived from the "Prueba 1" row on the site | Consistent with F4: the test probably has fewer than 12 usable weeks, or weeks with zero load. Unresolved. |

| F8 | **The safety nets cost nothing; over-caution costs a lot.** v1's 0.0001 kW margin: 0.00000 points. A 1 kW margin: −0.005. Keeping the battery between 5% and 95%: **−0.48**. The load guard: 0 on clean data, +0.031 on the fold with the meter outage. | `experiments/11_safety_margin/results.csv` | v1 uses the full 0–100% range and full ±500 kW. Nothing to gain by being more aggressive: the limits are physical, and asking for more than fits earns an infraction (−0.20) and no extra energy. |
| F9 | **The meter is not read between 22:00 and 06:00.** The official night load is the 22:00 reading held flat for 8 hours, on all 257 nights; the meter's energy register shows the real night load is 31% lower (median 766 kW vs 1,015 kW). | `reports/01_eda/report.md` §1, `figs/q1_meter_night_freeze.png` | The no-battery baseline is inflated by about 7% in CO₂ and 6% in cost, so the battery's real percentage savings are higher than the score says. No effect on control. A strong data-quality point for the jury. |
| F10 | **In meter-fault hours the inverters read 0 too** (even with the sun out). No train hour has solar above load. | 15 fault hours checked | There is never free surplus solar to store, not even during faults. |
| F11 | **The two objectives partly conflict, and CO₂ dominates.** Price and CO₂ agree on the midday trough and evening peak but disagree on weekday nights (cheap but dirty). CO₂ alone gets 91% of the best score; price alone 73%. | `reports/01_eda/report.md` §6 | The battery is mostly a decarbonisation tool that also saves money. |
| F12 | **A fixed timetable gets only 64% of the optimum** (charge 13–17 h, discharge 18–22 h scores 3.66). | `reports/01_eda/report.md` §9 | The value is in reading the day-ahead signals each day, which is what the optimiser does. |

| F13 | **The whole gap to perfect foresight is accounted for, and almost all of it is meter faults.** In units of 0.0001 points on all train days: meter faults 69.7, wrong CO₂/price weighting 0.6, not seeing beyond 24 h 0.2, guard margin 0.0 (total 70.5). With the two faults patched out the total gap is 0.8. | `reports/03_gap/report.md` §1 | Nothing left to gain from forecasting, weighting or horizon. The CO₂/price ratio of 0.60 used in v1 is the most robust constant over 45 test-like runs. |
| F14 | **The guard's one real cost is the hour the meter comes back.** On 18 June at 07:00 the load was back and it was a top hour, but v1 held because the last reading was 0. That single hour is 0.0186 of fold C's 0.0188 gap. | `reports/03_gap/report.md` §1(d) | The red-team agent is tuning this trade-off. It only matters if the hidden test has a fault. |

Validation scores (all with 0 infractions):

| Agent | Fold A | Fold B | Fold C | All train days |
|---|---|---|---|---|
| Perfect foresight (upper bound) | 5.965 | 5.780 | 5.787 | 5.725 |
| **v1** | **5.961** | **5.779** | **5.768** | **5.718** |
| Starter-kit heuristic | 0.00 | 0.51 | 0.09 | 0.00 |

Folds: A = days 1–7 of every month, B = days 8–14, C = days 15–21, each concatenated into one 2,016 h run exactly like the hidden test (which is the last 7 days of every month). Missing load in November is imputed so all price/CO₂ data can be used.

---

## 3. Leaderboard

### Snapshot 2 — 14 Nov, 13:58 (after our first submission)

Raw CSV: [`05_leaderboard/leaderboard_20261114_1400.csv`](05_leaderboard/). The site's clock is one hour behind ours (UTC).

| Pos | Team | Score | CO₂ −% | Cost −% | CO₂ avoided (kg) | € saved | Unrounded (reconstructed) |
|---|---|---|---|---|---|---|---|
| 1 | **Programi qui programi** | **6.44** | 5.99 | 7.48 | 4,302.3 | 8,884.35 | ≈ 6.438 |
| 2 | Prueba 1 (organisers) | 6.43 | 5.97 | 7.50 | 4,289.4 | 8,907.87 | ≈ 6.432 |
| 3 | La Datoneta | 6.43 | 5.97 | 7.50 | 4,289.4 | 8,907.86 | ≈ 6.432 |
| 4 | Manguets | 6.43 | 5.95 | 7.55 | 4,275.5 | 8,959.68 | ≈ 6.431 |
| 5 | Los Rojos | 6.43 | 5.96 | 7.52 | 4,282.5 | 8,934.63 | ≈ 6.432 |
| 6 | hackers la l | 6.43 | 5.96 | 7.52 | 4,282.5 | 8,934.63 | ≈ 6.432 |
| 7 | SSA Gourmet | 6.43 | 5.96 | 7.54 | 4,277.6 | 8,953.44 | ≈ 6.432 |
| 8 | Viquipèdia racers | 6.43 | 5.97 | 7.50 | 4,287.6 | 8,905.61 | ≈ 6.429 |
| 9 | JAMM | 6.43 | 5.97 | 7.50 | 4,289.4 | 8,907.85 | ≈ 6.432 |
| **10** | **Fa (us, v1)** | **6.43** | 5.96 | 7.52 | 4,282.5 | 8,934.63 | ≈ 6.432 |
| 11 | Dos enginyers i mig | 6.43 | 5.95 | 7.54 | 4,276.6 | 8,957.61 | ≈ 6.431 |
| 12 | La MasIA | 4.03 | 3.72 | 4.75 | 2,670.3 | 5,639.08 | ≈ 4.027 |

Reading it:

- **v1 landed exactly where predicted** (I had written 5.96–5.97% CO₂, 7.50–7.52% cost, 4,283–4,289 kg, 8,908–8,935 €). Our row is identical to Los Rojos and hackers la l, so those teams run the same optimisation with the same CO₂-vs-price weighting, and without our guard. The guard therefore never changed an action: **no zero-load or low-load stretch in the test.**
- **Programi qui programi is doing something different.** Against the plain optimum (Prueba 1) they saved 12.9 kg more CO₂ and 23.5 € less. Moving along the plain optimum's own trade-off line costs 3.9 € per extra kg; they paid 1.8 € per kg. That is not a reweighting. It is a better policy on the public hours.
- **Why that surprises me:** on my clean validation folds, perfect foresight adds only 0.0001–0.0035 over v1. Their +0.0066 is more than that. Candidate explanations, to be tested: (a) the test weeks reward looking beyond 24 h more than my train folds do; (b) the public score only counts half the hours, so a policy that is equally good overall can score a few thousandths higher or lower on the public half by chance; (c) something about the problem I have not seen yet.
- **Each submission is a precise measurement.** With the reconstructed public baseline (CO₂ 71,797–71,832 kg; cost 118,733–118,750 €) the kg and € columns give the unrounded public score to about ±0.001. We have 9 submissions left in this shift (until 21:00), 10 tonight, 10 tomorrow.

### 3a. What explains the 6.44? (state at 14:25)

What is measured:

1. **No controller can be 0.0066 better than v1 over a whole clean run.** On 98 simulated 12-week tests, perfect foresight beats v1 by 0.0001 on average and 0.0006 at most (`experiments/09_public_noise/`).
2. **The only train situations where perfect foresight gains more are meter faults** (recorded load near zero), e.g. 6 Aug 07:00–08:00, worth 0.003 on a 2,016 h run. These cannot be predicted, and there is no free solar in them (finding F10). v1's guard never acted on the public run, so no such stretch occurred when we wanted to discharge.
3. **Hour-level sampling of the public half is ruled out.** A run made of every other hour would score 8–9, not 6.43 (`experiments/10_run_structure/`).
4. **The pack agrees too well for the public half to be cut inside weeks.** Five teams with visibly different schedules are within 0.0005 of each other. In simulation, two schedules that differ that much differ by a median of 0.013 if the public half is the first 3.5 days of each week or alternate days, but by a median of 0.0008 if it is whole weeks (`experiments/09_public_noise/run_weekmask.py`).
5. **With whole weeks as the public half, outliers still happen.** In 5% of simulated cases two equally good controllers differ by more than 0.05 on the public weeks, because one carried more energy across a week boundary (charged in a hidden week, discharged in a public one).

What I infer: the public set is most likely whole weeks (6 × 168 h = 1,008 h), and the 6.44 team's policy happens to carry a little more energy into a public week than the plain optimum does (about 70 kWh would do it). If so, their gain on the public weeks is a loss on the private weeks, because the total is fixed.

What I cannot rule out: a real effect in the hidden data that train data does not show.

What I ruled out along the way: a longer forecast horizon (worth ≤ 0.0006), free surplus solar during meter faults (the inverters read 0 too), and being "more aggressive" at the limits (finding F8).

![Public split noise](00_overview/figs/public_split_noise.png)

*Each dot is one simulated 12-week test. Top row: over the whole test, perfect foresight and v1 score the same. Other rows: scored on half of the hours only, the same two controllers differ by hundredths. (This figure shows cuts inside weeks; the whole-weeks case is in `results_weekmask.csv`.)*

**Recommendation:** do not chase the 6.44. If the final ranking uses the private half or the whole test, a policy tuned to gain on the public half will do the same or worse there. v1 is at the ceiling of what can be scored honestly.

### Snapshot 1 — 14 Nov, 13:50 (before we submitted)

Raw CSV: [`05_leaderboard/leaderboard_20261114_1350.csv`](05_leaderboard/leaderboard_20261114_1350.csv). Re-run with `experiments/05_leaderboard/scrape.py` (read-only).

| Pos | Team | Score | CO₂ −% | Cost −% | CO₂ avoided (kg) | € saved | Unrounded score (my reconstruction) |
|---|---|---|---|---|---|---|---|
| 1 | Prueba 1 (organisers' test) | 6.43 | 5.97 | 7.50 | 4,289.4 | 8,907.87 | 6.430–6.433 |
| 2 | La Datoneta | 6.43 | 5.97 | 7.50 | 4,289.4 | 8,907.86 | 6.430–6.433 |
| 3 | Manguets | 6.43 | 5.95 | 7.55 | 4,275.5 | 8,959.68 | 6.430–6.432 |
| 4 | Los Rojos | 6.43 | 5.96 | 7.52 | 4,282.5 | 8,934.63 | 6.430–6.433 |
| 5 | hackers la l | 6.43 | 5.96 | 7.52 | 4,282.5 | 8,934.63 | 6.430–6.433 |
| 6 | SSA Gourmet | 6.43 | 5.96 | 7.54 | 4,277.6 | 8,953.44 | 6.430–6.433 |
| 7 | Viquipèdia racers | 6.43 | 5.97 | 7.50 | 4,287.6 | 8,905.61 | 6.428–6.431 |
| 8 | La MasIA | 4.03 | 3.72 | 4.75 | 2,670.3 | 5,639.08 | 4.027–4.028 |

What this tells us:

- **Everyone at the top has the same solution.** Three teams match the organisers' own run to the cent. The small differences between the others are just slightly different CO₂-vs-price weightings, and they all land within 0.003 of each other. This is the flat optimum I measured in F3.
- **Public baseline, reconstructed** by intersecting the rounding intervals of all rows: CO₂ 71,797–71,832 kg and cost 118,733–118,750 €. So the public set's kg-per-€ ratio is 0.605 (I had assumed 0.60 in v1; the train set gives 0.587). The difference is worth nothing measurable.
- **The target is 6.435**, which displays as 6.44 and ranks above the whole pack regardless of submission time.
- The tab's caption says the ranking is "evaluated on the public test set (1,008 hours)". It says nothing about tie-breaks; the ordering of tied teams by submission time is what the grid shows.

---

## 3b. Why "perfect foresight" is a real ceiling

A fair question: even if you knew the whole future, isn't choosing when to charge and discharge still a hard problem? It isn't, and that is why the ceiling can be trusted.

With all the data known, the schedule is a **linear programme**:

- **Unknowns:** kW charged and kW discharged in each hour.
- **Rules:** power between 0 and 500 kW; stored energy rises by 0.949 × charge and falls by discharge ÷ 0.949; stored energy stays between 0 and 2,000 kWh.
- **Goal:** minimise the weighted sum of grid imports, where grid import = load − solar + battery power.

Every one of those relationships is a straight line. Problems of that shape have no local optima: a standard solver (HiGHS) returns the single best schedule with a mathematical guarantee that no other schedule scores higher. That schedule's score is what I call the oracle.

Checks that the number is right:

- The oracle's schedule, replayed through the official simulator, gives the same score. So my model of the physics and of the scoring matches the organisers'.
- A completely different method (the dynamic programme inside our agent) reproduces the solver's optimum to 14 decimal places on 300 random cases.

So the only thing between a real agent and the oracle is **information**: the agent sees 24 hours ahead, the oracle sees everything. v1 solves "the best plan for the 24 hours I can see" exactly, every hour. The measured cost of not seeing further is about 0.0001 points on clean data.

Caveat: the ceiling is computed on train weeks, because the hidden test is hidden. The leaderboard is consistent with it (ten teams on the same score), but I cannot compute the test's own ceiling.

---

## 3c. How v1 works

Each hour it:

1. Builds the signal `w` for the 24 known hours (CO₂ intensity plus price, weighted the way the score weighs them).
2. Solves for the best possible charge/discharge plan over those 24 hours, exactly. The solver is a small dynamic programme on "how much is one more kWh in the battery worth"; it gives the same answer as a linear programme (checked on 300 random cases, difference 1e-14) but takes 0.1 ms and needs only numpy.
3. Applies the first hour of that plan, then repeats next hour with the new information.

![How the agent decides](00_overview/figs/how_it_decides.png)

*Three hours of the same day. The black curve is what one more stored kWh will be worth later, worked out exactly from the 24 known hours; it falls as the battery fills because the best future uses get taken first. If that value is above the cost of storing a kWh now (blue), charge. If it is below what a kWh earns now (red), discharge. Otherwise wait. The gap between blue and red is the 10% round-trip loss.*

Safety layers:

- Every command is clamped with a small margin, so it can never exceed 500 kW or the 0–2,000 kWh range (no infractions).
- Discharge is capped at 95% of the last measured net load, because energy pushed to the grid earns nothing.
- If the meter reads about zero, the plan assumes the outage continues for a while instead of cycling for no benefit.
- When fewer than 24 hours of lookahead remain (the last day of the run), it empties the battery optimally, since leftover energy is worth nothing.

---

## 4. What is running right now

| Subagent | Task | Output will be in | Status |
|---|---|---|---|
| EDA analyst | Coverage map, load/PV/CO₂/price profiles, do CO₂ and price agree, predictability beyond 24 h. Presentation-grade plots. | [`reports/01_eda/report.md`](01_eda/report.md), 21 figures | **done** |
| Red-team | Fuzz the agent with malformed inputs; zero-load weeks, outages, dropped weeks, low-load weeks; tune the outage guard. | [`reports/02_robustness/report.md`](02_robustness/report.md); candidate `experiments/02_stress/candidate/agent.py` | **done** |
| Gap researcher | Break down the last 0.007 points of gap to the oracle and recover what can honestly be recovered. | [`reports/03_gap/report.md`](03_gap/report.md) | **done**: nothing honest is worth more than 0.001; keep v1 |
| What-if analyst | Yearly tCO₂ and € saved, CO₂-vs-cost Pareto frontier, battery sizing (kWh and kW), efficiency. For the jury. | [`reports/04_whatif/report.md`](04_whatif/report.md), 8 figures | **done** |
| Leaderboard scraper | Snapshot of the live leaderboard. | `reports/05_leaderboard/` | **done** (section 3) |

I report in chat when each one finishes and add its results here.

---

## 5. Ideas and open questions

- **Which hypothesis about the damaged weeks is true** (dropped or zero-filled)? Our first submission answers it. v1 has the outage guard and the 6.43 teams apparently don't (they match the organisers' plain run). If v1 scores clearly above 6.43, a zero-load or low-load stretch exists in the test and the guard is our edge. If it scores 6.43 with ~4,289 kg avoided, the test is clean and only the third decimal is left.
- **What v1 should show on the leaderboard if the test is clean:** about 5.96–5.97% CO₂, 7.50–7.52% cost, 4,283–4,289 kg, 8,908–8,935 €, 0 infractions.
- **How is the public half chosen** ("50% of each test week")? Unknown. It doesn't change what the agent should do.
- **Rounding.** Scores are rounded to 2 decimals. If we sit just below a rounding boundary, a 0.003 gain flips a digit. That is the only reason the gap research is worth doing.
- **Presentation angle.** "We proved the controller is within 0.1% of the best possible, then spent our effort on what happens when the sensors fail" is a stronger story than a leaderboard number everyone shares.
- **Not doing, on purpose:** the test weeks' prices and CO₂ are public REE data, so one could download them and hard-code the perfect schedule. That is test-set leakage, the code is audited by the jury, and I'm not going near it.

---

## 6. Log

- **13:25** Read the docs, website notes and starter kit. Created branch `claude/battery-agent` and a Python 3.12 environment in `.venv` (your `venv` is untouched).
- **13:30** First look at the data: findings F1, F4, F5. Wrote `src/data.py` (hourly tables).
- **13:35** Launched the EDA subagent. Wrote `src/sim.py` (fast simulator identical to the official one, validation folds, perfect-foresight oracle) and `src/mpc.py`.
- **13:40** First results: plain 24 h optimiser ≈ oracle (F3). Starter heuristic ≈ 0 (F6).
- **13:45** Wrote `agents/v1/agent.py` with an exact numpy-only solver. Verified against the LP, the official environment and `check_submission.py`. Committed.
- **13:48** Launched four more subagents (table in section 4).
- **13:50** Wrote this log.
- **13:51** Measured the zero-load-week scenario (F4): no guard −0.5 to −0.9 points; v1 recovers most. Sent the red-team agent an idea for the rest (use the calendar: a dead week ends at the month boundary).
- **13:52** Leaderboard scrape done: seven teams tied at 6.43, all the same optimum (section 3). Reconstructed the public baseline and the unrounded scores.
- **13:52** Added the one-week figure (`00_overview/figs/week_v1.png`).
- **13:53** Tried to infer how many usable weeks the hidden test has (`experiments/08_test_composition/`). The battery's savings per hour don't depend on campus load, so I compared the leaderboard's savings with what train data predicts. Result: 12 clean weeks over-predicts the public savings by 13–18%; 11 weeks (no November) by 6–10%; 10 weeks under-predicts CO₂ by 13%. **Inconclusive** between "November dropped" and "November present with zero load, where the plain optimisers waste energy". Only a submission of v1 can settle it.
- **13:57** You submitted v1 as team Fa: 6.43, 10th place.
- **14:00** Second leaderboard snapshot. v1's row matches the prediction; the test is clean (guard never acted); one team is at 6.44 (≈ 6.438). Told the gap researcher the new target.
- **14:08** Added section 3b (why the oracle is a true ceiling) and the "how it decides" figure, after a question from the team.
- **14:10** Investigated the 6.44. First thought it was noise from scoring half the hours; then that it was free solar during meter faults (wrong: inverters read 0 in those hours); settled on "public half = whole weeks, energy carried across a week boundary" as the best-supported reading (section 3a).
- **14:15** EDA subagent finished: 21 figures and a report (`01_eda/report.md`). New findings F9–F12. I saved its report text because subagents cannot write report files.
- **14:20** Measured what the safety nets cost, after a question from the team (finding F8). The clamp margin withholds 0.059 kWh over a whole run; the load guard changes 1 action in 2,016 on a clean fold.
- **14:30** Gap researcher finished (`03_gap/report.md`, findings F13–F14). It tried terminal values, forecasts of the next day and online estimates of the CO₂/price ratio: none gives a gain that holds across folds. Recommendation: keep v1 unchanged.
- **14:45** Leaderboard snapshot 3: 22 teams, 17 at 6.43, a second entry above the pack (Prueba_1, ≈ 6.443).
- **14:50** What-if analyst finished: `04_whatif/report.md` with 8 figures (yearly savings 42.5 t CO₂ and 91 k€; the battery is energy-limited; 70/30 costs 20.5 k€ a year against cost-only). Parked for the presentation.
- **15:00** Brute-force check of the ceiling done (`experiments/12_ceiling_check/`): exhaustive search never beats the LP.
- **15:05** Wrote `06_edge/report.md`. Prepared two diagnostic submissions in `submissions/`.
- **15:12** Simulated which explanation reproduces the leaderboard pattern (`experiments/13_fault_luck/which_mechanism.py`). A meter glitch: 0.8% of cases. Public half = first 84 h of each week, scored out of one continuous run: 77%. Measured how leaky that split is (`split_leak.py`).
- **15:15** Red-team agent finished: `02_robustness/report.md`, v1.1 candidate, reusable stress harness.
- **15:22** Rewrote `06_edge/report.md` around the split finding. Section 3a above is superseded by it.
- **15:24** Leaderboard: three more teams joined the pack after lunch; still only two entries at 6.44.
- **16:00** Organisers' message: nobody has reached the maximum on the private ranking. Reading: bad data in the private half.
- **16:06** Built and validated agent v2 (`agents/v2/agent.py`, packaged in `submissions/v2/`): v1.1 plus low-load plan caps plus surplus-solar charging. Stress harness: 0 failures.
- **16:10** Updated `06_edge/report.md` with the recommendation to submit v2.

---

## 7. Where things are

```
agents/v1/agent.py          the submission candidate
src/                        shared code: data tables, simulator, oracle
experiments/NN_*/           scripts behind every number in this file
reports/README.md           this log
reports/00_overview/figs/   figures used in this log
reports/NN_*/report.md      one report per topic, with figures
```

To re-check v1 yourself, from the repo root:

```
.venv/Scripts/python.exe experiments/01_dp_solver/verify.py
.venv/Scripts/python.exe starter_kit/check_submission.py agents/v1/agent.py
```
