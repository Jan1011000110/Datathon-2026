# h1: the uncertainty-aware planner, packaged

**Written:** Sat 14 Nov 2026, 18:15. Built by the packager subagent (finished 18:11), re-checked by me.

**File to upload:** `submissions/h1/nov_fill/submission.zip` (`agent.py` + `h1_tables.json`, 86 KB).

## 1. Summary

- **h1 = h0's average-day forecasts + the new planner.** The planner knows the net-load forecast is uncertain and decides how much to discharge from the forecast's distribution, hour by hour.
- **Validation score: 20.58 / 19.79 / 18.35** on folds A / B / C, 0 infractions. That is +0.74 / +0.49 / +0.45 over h0.
- **It runs in under a second.** 0.4–0.7 s for 2,016 steps, against 15–33 s for h0. The 60-second limit is no longer a risk, and `scipy` is no longer needed.
- **Recommended variant: `nov_fill`.** Same agent; only the November tables differ (section 4).

## 2. Verification

The packaged agent against the harness it was ported from (tables fitted outside each fold, end-of-run rule off):

| Fold | Harness | Packaged | Commands that differ | Infractions |
|---|---|---|---|---|
| A | 20.580824 | 20.580824 | 0 of 1,844 | 0 |
| B | 19.785790 | 19.785790 | 0 of 1,848 | 0 |
| C | 18.209285 | 18.209285 | 0 of 1,725 | 0 |

As shipped (end-of-run rule on):

| Fold | Score |
|---|---|
| A | 20.581 |
| B | 19.790 |
| C | 18.353 |

- **Organisers' environment:** 18.35 on fold C and 19.79 on fold B, 0 infractions.
- **`check_submission.py`:** passes on both ZIPs (0.19 ms per step).
- **Malformed observations:** 69 single cases and three corrupted full runs, no exception and no infraction.
- **My own check (18:13):** both ZIPs extracted and run on fold B: 0 infractions, 0.42 s per 2,016 steps. The two variants have the same `agent.py`; their tables differ only for November.

## 3. Options tested and left off

Change in score per fold when each option is switched on:

| Option | A | B | C | Shipped |
|---|---|---|---|---|
| End-of-run rule | +0.001 | +0.004 | +0.144 | on |
| Calendar that knows the next test week follows a month end | −0.075 | +0.180 | −0.054 | off |
| 2.5 % haircut on top of the scenarios | −0.051 | −0.031 | −0.010 | off |
| 5 % haircut | −0.159 | −0.105 | −0.080 | off |
| 10 % haircut | −0.452 | −0.307 | −0.285 | off |
| 24 h horizon | −0.001 | 0 | 0 | 36 h kept |

- **The week calendar** helps one fold and hurts two. With only 10–11 week boundaries per fold the effect is mostly noise, so it stays off.
- **A haircut on top of the scenarios hurts at every level.** The scenarios already do that job.

## 4. The November variant

Train has no November rows, so the plain tables fall back on all-month averages for that month. That puts November's midday solar at about 1,070 kW. October and December are at 974 and 607 kW, which interpolate to about 790 kW.

The packager tested the fallback by hiding one month at a time from the fit and scoring that month's week alone:

| Fold | Month in the fit | Hidden, all-month average | Hidden, interpolated from neighbours |
|---|---|---|---|
| A | 21.49 | 20.43 | 21.21 |
| B | 20.04 | 19.11 | 19.77 |
| C | 20.06 | 18.59 | 19.34 |

- Interpolation recovers about 0.7 points on the hidden week, and wins in 23 of 32 month-fold cases.
- November is one of 12 test weeks, so the expected gain on the total is about 0.1 if the November week exists in the test. If it does not, the two variants behave the same.
- Folds A, B and C are unchanged to six decimals, because no fold row falls in November.

## 5. Files

- `agents/h1/agent.py`, `agents/h1/build.py` (`build.py neighbours` rebuilds the November variant), `agents/h1/h1_tables.json`.
- `submissions/h1/submission.zip`: plain tables.
- `submissions/h1/nov_fill/submission.zip`: recommended.
