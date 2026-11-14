# h0: the first hard-mode submission

**Written:** Sat 14 Nov 2026, 17:52. Built by the packager subagent (finished 17:47), re-checked by me.

**File to upload:** `submissions/h0/submission.zip` (`agent.py` + `h0_tables.json`, 29 KB).

## 1. Summary

- **h0 is the naive agent, packaged.** Average-day forecasts of the signal, the load and the solar output; a linear programme over the next 18 hours, re-solved every hour; discharge limited to 90 % of the forecast net load.
- **Validation score: 19.84 / 19.30 / 17.90** on folds A / B / C, with 0 infractions. Tables fitted on train days outside each fold.
- **It passes the local validator** and gives the same score in the organisers' `environment.py`.
- **Open risk: run time.** On this laptop, with five other jobs running, a 2,016-step run took 15 to 33 seconds. On an idle machine the estimate is 6 to 13 seconds. The server's limit is 60 seconds and its speed is unknown.

## 2. What is inside

- `agent.py` imports only `datetime`, `json`, `pathlib`, `numpy` and `scipy.optimize.linprog`.
- `h0_tables.json` holds the fitted tables: the signal by month, weekend and hour; the load by weekend and hour; the solar output by month and hour.
- Battery size and power are read from `obs` (`cap_max_kwh`, `power_max_kw`), with 6,000 / 1,500 as defaults.
- When `obs` carries fewer than 3 price/CO₂ values the run is ending, and the plan covers only the hours that remain.
- If `scipy` cannot be imported on the server, the agent stays idle (legal, scores 0). The site's quick validation will show this.

## 3. Verification

Scores of the packaged agent against the harness it was ported from (same horizon of 36 h, tables fitted outside the fold):

| Fold | Harness | Packaged | Infractions |
|---|---|---|---|
| A | 19.825 | 19.825 | 0 |
| B | 19.293 | 19.293 | 0 |
| C | 17.757 | 17.757 | 0 |

As shipped (18 h horizon, end-of-run rule on):

| Fold | Score | Infractions |
|---|---|---|
| A | 19.839 | 0 |
| B | 19.305 | 0 |
| C | 17.899 | 0 |

- **Organisers' environment, fold C:** 17.90, 0 infractions (my own run, 17:50).
- **`check_submission.py` on the ZIP:** passes.
- **Malformed observations:** 69 single cases (missing keys, NaN, empty lists, text timestamps, charge outside the range) and three corrupted full runs: no exception, no infraction.
- **The ZIP as shipped on folds A and B** (tables fitted on all train rows, so the score is not a fair estimate): 0 infractions.

## 4. Choices the packager made

- **Horizon 18 h instead of 36 h.** It halves the run time and stays within 0.02 points on every fold:

| Horizon | A | B | C |
|---|---|---|---|
| 36 h | 19.839 | 19.308 | 17.900 |
| 24 h | 19.839 | 19.309 | 17.903 |
| 18 h | 19.839 | 19.305 | 17.899 |
| 12 h | 19.814 | 19.323 | 17.888 |

- **End-of-run rule.** Emptying the battery in the last two hours is worth +0.014 / +0.015 / +0.144 on A / B / C. That is from two hours of notice only; a full battery is worth about 0.35 points, so knowing the end of the run earlier would be worth more.

## 5. Known weaknesses (being worked on)

- The forecasts are averages. Better load, solar and signal forecasts are in progress.
- The plan treats the forecast of the current hour's load as certain, apart from the 10 % haircut.
- The linear programme is slow. A faster planner is in progress.
- The calendar of future hours ignores the three-week jump between test weeks.

## 6. Files

- `agents/h0/agent.py`, `agents/h0/build.py` (fits the tables; `build.py 24` repackages with a 24 h horizon), `agents/h0/h0_tables.json`.
- `submissions/h0/submission.zip` and its two files.
