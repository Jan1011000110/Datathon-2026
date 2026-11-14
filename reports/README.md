# Claude's work log — battery challenge

This is the one place to see what I'm doing, what I've found and what's coming. I update it every time something lands. Branch: `claude/battery-agent`.

**Last updated:** Sat 14 Nov 2026, 14:05

---

## 1. Where we are, in five lines

1. **A submission-ready agent exists:** [`agents/v1/agent.py`](../agents/v1/agent.py). Single file, numpy only, 0 infractions, 0.25 s for a full run, passes `check_submission.py`.
2. **It is essentially optimal.** On my three test-like validation folds it scores 5.96 / 5.78 / 5.77. The best score *possible with perfect knowledge of the future* on those same folds is 5.96 / 5.78 / 5.79.
3. **The leaderboard is a seven-way tie at 6.43** (snapshot 13:50, section 3). All seven teams found the same optimum; their unrounded scores are 6.430–6.433. Ties are ordered by earliest submission.
4. **My recommendation: submit v1 now** (run the free *Validació Ràpida* first). It should land in the 6.43 pack and lock in our tiebreak time; a later, better submission still moves us up. It also tells us things we can't learn any other way (section 6).
5. **To pass the pack we need 6.44, which means +0.002 to +0.005 points.** That is the same size as the remaining gap to perfect foresight, so it is hard but not impossible. One subagent is on exactly this.
6. **The bigger prizes are elsewhere:** robustness to bad data in the hidden test (finding F4: worth up to 0.5–0.9 points if a zero-load week exists) and the jury presentation.

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

Validation scores (all with 0 infractions):

| Agent | Fold A | Fold B | Fold C | All train days |
|---|---|---|---|---|
| Perfect foresight (upper bound) | 5.965 | 5.780 | 5.787 | 5.725 |
| **v1** | **5.961** | **5.779** | **5.768** | **5.718** |
| Starter-kit heuristic | 0.00 | 0.51 | 0.09 | 0.00 |

Folds: A = days 1–7 of every month, B = days 8–14, C = days 15–21, each concatenated into one 2,016 h run exactly like the hidden test (which is the last 7 days of every month). Missing load in November is imputed so all price/CO₂ data can be used.

---

## 3. Leaderboard snapshot (14 Nov, 13:50)

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

## 3b. How v1 works

Each hour it:

1. Builds the signal `w` for the 24 known hours (CO₂ intensity plus price, weighted the way the score weighs them).
2. Solves for the best possible charge/discharge plan over those 24 hours, exactly. The solver is a small dynamic programme on "how much is one more kWh in the battery worth"; it gives the same answer as a linear programme (checked on 300 random cases, difference 1e-14) but takes 0.1 ms and needs only numpy.
3. Applies the first hour of that plan, then repeats next hour with the new information.

Safety layers:

- Every command is clamped with a small margin, so it can never exceed 500 kW or the 0–2,000 kWh range (no infractions).
- Discharge is capped at 95% of the last measured net load, because energy pushed to the grid earns nothing.
- If the meter reads about zero, the plan assumes the outage continues for a while instead of cycling for no benefit.
- When fewer than 24 hours of lookahead remain (the last day of the run), it empties the battery optimally, since leftover energy is worth nothing.

---

## 4. What is running right now

| Subagent | Task | Output will be in | Status |
|---|---|---|---|
| EDA analyst | Coverage map, load/PV/CO₂/price profiles, do CO₂ and price agree, predictability beyond 24 h. Presentation-grade plots. | `reports/01_eda/` | running |
| Red-team | Fuzz the agent with malformed inputs; zero-load weeks, outages, dropped weeks, low-load weeks; tune the outage guard. | `reports/02_robustness/` | running |
| Gap researcher | Break down the last 0.007 points of gap to the oracle and recover what can honestly be recovered. | `reports/03_gap/` | running |
| What-if analyst | Yearly tCO₂ and € saved, CO₂-vs-cost Pareto frontier, battery sizing (kWh and kW), efficiency. For the jury. | `reports/04_whatif/` | running |
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
- **13:53** Measured the zero-load-week scenario (F4): no guard −0.5 to −0.9 points; v1 recovers most. Sent the red-team agent an idea for the rest (use the calendar: a dead week ends at the month boundary).
- **13:55** Leaderboard scrape done: seven teams tied at 6.43, all the same optimum (section 3). Reconstructed the public baseline and the unrounded scores.
- **13:58** Added the one-week figure (`00_overview/figs/week_v1.png`).
- **14:05** Tried to infer how many usable weeks the hidden test has (`experiments/08_test_composition/`). The battery's savings per hour don't depend on campus load, so I compared the leaderboard's savings with what train data predicts. Result: 12 clean weeks over-predicts the public savings by 13–18%; 11 weeks (no November) by 6–10%; 10 weeks under-predicts CO₂ by 13%. **Inconclusive** between "November dropped" and "November present with zero load, where the plain optimisers waste energy". Only a submission of v1 can settle it.

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
