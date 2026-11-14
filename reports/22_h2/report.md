# h2: the combined agent, packaged

**Written:** Sat 14 Nov 2026, 18:55. Built by the packager subagent (final build 18:45), re-checked by me at 18:53.

**File to upload:** `submissions/h2/submission.zip` (`agent.py` + `h2_state.npz`, 1.16 MB).

**Free server check first:** `submissions/probe_h2/submission.zip` on *Validació Ràpida* only (section 6).

## 1. Summary

- **h2 = the combined agent of [`18_combined/report.md`](../18_combined/report.md):** the uncertainty-aware planner with the load, solar and price/CO₂ forecasters, plus end-of-run handling.
- **Validation score: 21.27 / 20.37 / 18.70** on folds A / B / C (mean 20.11), 0 infractions. That is +0.54 over h1 and +1.10 over h0.
- **The packaged agent gives the same command as the test harness in every hour** of all three folds.
- **It cannot do worse than h1 for a technical reason.** If the full agent fails to build, or fails at any hour, h1's logic answers that hour.
- **Run time:** 2.5 s for 2,016 steps on a quiet laptop, 5 to 8 s with other jobs running. The limit is 60 s.

## 2. What is inside

- `agent.py` (280 KB, one file): generated mechanically from the specialists' modules, each wrapped so names cannot collide. Imports only numpy, pandas, json, math, datetime, bisect, pathlib.
- `h2_state.npz` (1.1 MB): 50 plain numeric arrays, no pickled objects, so the server's library versions cannot break loading.
- h1 and its tables are embedded as text, so h1 works even without the state file.
- Settings: 36 h horizon, 9 scenarios on 12 hours, plan only the hours that exist when `obs` shows fewer than 3 prices, nothing planned after 31 August 2026 23:00, final clamp margin 0.01 kW.

## 3. Verification

State fitted outside each fold; the calendar end is the fold's own last August day.

| Fold | Harness | Packaged, same clamp | Commands that differ | As shipped (0.01 kW margin) |
|---|---|---|---|---|
| A | 21.274461 | 21.274461 | 0 of 1,844 | 21.274420 |
| B | 20.370046 | 20.370046 | 0 of 1,848 | 20.370000 |
| C | 18.700483 | 18.700483 | 0 of 1,725 | 18.700443 |

- **Organisers' environment:** 21.27 on fold A and 18.70 on fold C, 0 infractions, 0 fallbacks.
- **`check_submission.py`:** passes.
- **State of charge rounded to 2 decimals:** 0 infractions (63 / 65 / 75 with the old 0.0001 kW margin).
- **Malformed observations:** 79 cases, each as a first observation and after two normal steps: 0 failures of 158. Corrupted full runs at 15 %, 50 % and 100 %: no exception, no infraction.
- **A run laid out like the real test** (last 7 days of each month, with a November week and an August week with zero solar): 0 infractions, 0 fallbacks.
- **My own check on the final build (18:53):** 18.70 on fold C in the organisers' environment with the state fitted outside the fold; 0 infractions and 0 fallbacks on all three folds; identical score with timestamps passed as text; 5.2–5.4 s per 2,016 steps with other jobs running.

Note on reading the scores: with the shipped setting the calendar end is 31 August 2026 at 23:00. Our folds end earlier in August, so on a fold the shipped ZIP scores as the "safe rule only" row of report 18 (21.14 / 20.30 / 18.70). The 21.27 / 20.37 / 18.70 figures use each fold's own last day as the end, which is what the real test's last day will be for the shipped agent.

## 4. The safety net

| Failure | What happens |
|---|---|
| State file missing, garbage or truncated | h1 answers every step |
| A forecaster's arrays missing, or no pandas | h1 answers every step |
| The load model's tree arrays damaged | rules-only load forecast (0.06 points lower), no fallback to h1 |
| The combined agent raises or returns NaN on some steps | h1 answers exactly those steps |

All tested on fold B with 0 infractions. `BatteryAgent.fallbacks` counts the steps h1 answered; it is 0 on every normal run.

## 5. Two engineering decisions worth knowing

- **LightGBM is not used on the server.** A model text that LightGBM cannot parse does not raise an error; it kills the Python process, which no fallback can catch. The load model's trees are stored as arrays and evaluated with numpy. The result is identical to LightGBM's on all 4,369 training rows and in closed loop.
- **No November patch.** Train has no campus data for November. Hiding one month at a time from the fit costs h2 only 0.03 to 0.18 points on that month's week (h1 with interpolated tables loses much more), so nothing was added.

## 6. The server probe

`submissions/probe_h2/submission.zip` is h2, unchanged, inside a wrapper. It plays the site's free 48-hour validation and then stops on purpose with a report in the error text:

- whether the full agent built on the server (`build_error`, `load_mode`, `fallbacks`),
- seconds to construct and milliseconds per step there, numpy and pandas versions,
- yes/no checks of our assumptions: list lengths over the run, state of charge unrounded, consecutive timestamps, night-time load frozen, solar-to-irradiance ratio, wind always zero.

It reports counts, checks and timings; no price, CO₂, load or solar value. Locally it prints, for example: `build_error=''; load_mode='model'; fallbacks=0; ms_per_step mean=1.21; list_len=3x46,2,1; soc_max_dev_kwh=0.00e+00; night_load_same_as_previous=12/12; solar/irradiance median=1.67`. Rebuild it with `experiments/25_integration/build_probe_h2.py` after any change to h2.

## 7. Files

- `agents/h2/build.py` (all settings in `CONFIG` at the top), `bundle.py`, `rt_main.py`, `rt_trees.py`, and generated runtime copies `rt_*.py`.
- `submissions/h2/submission.zip`, `agent.py`, `h2_state.npz`.
- `submissions/probe_h2/submission.zip`: validation only.
