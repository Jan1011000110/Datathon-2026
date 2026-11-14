# h3: h2 plus the two signal improvements, packaged

**Written:** Sat 14 Nov 2026, 19:14. Built by the packager subagent (19:04), re-checked by me at 19:12.

**File to upload:** `submissions/h3/submission.zip` (`agent.py` + `h3_state.npz`, 1.27 MB).

**Free server check first:** `submissions/probe_h3/submission.zip` on *Validació Ràpida* only.

## 1. Summary

- **h3 = h2 + the second-round signal forecaster + a planner that doubts the signal forecast** ([`19_signal_planning/report.md`](../19_signal_planning/report.md)).
- **Validation score: 21.33 / 20.48 / 18.77** on folds A / B / C (mean 20.19), 0 infractions. That is +0.075 over h2.
- **Expected on the public test: about 23.15.** h2 scored 23.06 there against 20.11 on the folds (ratio 1.147). The honest range for the gain is +0.05 to +0.10, since every setting was chosen on the folds.
- **Three-level safety net:** h3's own agent, then h2's exact logic, then h1's, then 0 kW. On every normal run all steps are answered by h3 itself.
- **Run time:** 3 to 4 s for 2,016 steps.

## 2. Verification

State fitted outside each fold; the calendar end is the fold's own last August day.

| Fold | Harness | Packaged, same clamp | Commands that differ | As shipped | If the new pieces fail (h2's logic) |
|---|---|---|---|---|---|
| A | 21.325552 | 21.325552 | 0 of 1,844 | 21.325512 | 21.274461 |
| B | 20.475417 | 20.475417 | 0 of 1,848 | 20.475371 | 20.370046 |
| C | 18.769109 | 18.769109 | 0 of 1,725 | 18.769069 | 18.700483 |

- **Organisers' environment:** 21.33 on fold A and 18.77 on fold C; 0 infractions, all steps answered by h3.
- **`check_submission.py`:** passes.
- **State of charge rounded to 2 decimals:** 0 infractions.
- **Malformed observations:** 79 cases, 0 failures of 158; corrupted full runs at 15 %, 50 % and 100 %: no exception, no infraction.
- **A run laid out like the real test** (with a November week and an August week with zero solar): 0 infractions, all steps answered by h3.
- **My own check (19:12):** 20.40 on fold B in the organisers' environment with the state fitted outside the fold and the shipped end-of-run date (the harness figure for that setting is 20.402); 0 infractions and all steps answered by h3 on the three folds; identical score with timestamps as text; imports are numpy, pandas and the standard library only.

## 3. The safety net

| Forced failure (fold B) | Level that answers | Check |
|---|---|---|
| New signal pieces missing or damaged; h3 raising at every step | h2, all steps | commands identical to the h2 package |
| h3 raising on some steps | h2 for exactly those steps | |
| State file missing or damaged; no pandas | h1, all steps | commands identical to the h1 package |
| Load-model trees damaged | h3 with the rules-only load forecast | |

`BatteryAgent.answered` counts the steps per level.

## 4. What changed inside, against h2

- **Signal forecaster, round 2:** a bridge to the stored train day that follows each test week, and a night-time level averaged with a per-hour-of-day model. The stored values are train rows only: none falls inside the last 7 days of any month.
- **Planner:** the value of stored energy is a half-and-half blend of the plan on the forecast and the average over 20 sampled error paths of the signal; the forecast mean is corrected from the last revealed error.
- **A small robustness fix in the embedded h1** (from the red-team): if the price lists hold no finite value at all, the horizon is not cut. The server does send the lists, so this is insurance.
- Everything else is h2: same load and solar forecasters, same net-load scenarios, same end-of-run rules, same 0.01 kW clamp margin.

## 5. Files

- `agents/h3/build.py` (all settings in `CONFIG`), `bundle.py`, `rt_main.py`, `rt_trees.py`, generated `rt_*.py`.
- `submissions/h3/submission.zip`, `agent.py`, `h3_state.npz`.
- `submissions/probe_h3/submission.zip`: validation only (built by `experiments/25_integration/build_probe_h2.py h3`).
