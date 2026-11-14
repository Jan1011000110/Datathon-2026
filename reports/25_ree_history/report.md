# Does older REE history improve the signal forecast?

**Written:** Sat 14 Nov 2026, 20:16. Code: `experiments/23_signal/fetch_ree_history.py`, `signal_fc3.py`, `round3.py`. Data: `data/external/ree_history/` (not tracked by git, like every CSV in this repo; its `README.md` is).

## 1. Summary

- **No. The history does not give a gain we can rely on.** Inside the h3 stack the best variant scores +0.017 over h3 on the mean of the folds (−0.036 / +0.031 / +0.057 on A / B / C). Week by week it is better in 15 of 33 weeks and worse in 18.
- **With net load known it is slightly negative** (−0.018).
- **In the test's own layout** (one fold) it is +0.04 to +0.05, positive in 7 to 9 of 11 weeks, still within noise.
- **Not shipped as it stands.** h3's forecaster stays as it is; section 5 says what building it as an extra candidate would take.
- This was the team's decision to try: training on public data older than the competition's own.

## 2. What was downloaded

- **Source:** Red Eléctrica's open data API (`apidatos.ree.es`, no token), the same two series the competition's files come from: the PVPC price, and the peninsular generation with and without CO₂-equivalent emissions.
- **Range:** 1 September 2023 to 31 August 2025, 17,542 hourly rows. The competition's data starts on 1 September 2025.
- **Guard:** the download script refuses any request ending after 31 August 2025 and any row dated on or after 1 September 2025, checked on every request and every table written. No value from the competition's span, and so none from a test week, was requested or stored.
- **Same recipe as the competition's files:** price = PVPC ÷ 1000; CO₂ = 0.37 × with ÷ (with + without). That formula reproduces all 6,768 rows of `data/train/co2_red.csv` exactly.
- **Continuity:** the last week of the history averages 0.136 €/kWh and 0.088 kg/kWh; the first train week 0.126 and 0.073.
- The team ran the download; a first attempt stopped on the last month because the script's guard expected hourly rows and REE had answered that month in quarter-hours. The guard was corrected to "nothing on or after 1 September 00:00".

## 3. How it was used

The forecaster is h3's (`SignalForecaster2`) with the ability to fit on extra seasons:

- the average day is kept per season-year, so each year's price level stays out of the comparison;
- the ridge models, the night-level model and the bridge weights are fitted on all seasons, with older rows optionally weighted less.

With no history supplied it reproduces h3's forecasts to 15 decimals, so any difference below comes from the history.

**A packaged agent would carry no outside values.** At run time the forecaster looks back a week at most and the test starts on 24 September 2025, so the old series are never read during a run; only the fitted coefficients depend on them.

## 4. Results

Closed loop on folds A, B, C; competition-span data fitted outside each fold.

**Inside the h3 stack** (end of run from the lists only; h3 scores 21.188 / 20.402 / 18.769 there):

| Forecaster | A | B | C | Mean | Against h3 |
|---|---|---|---|---|---|
| h3's forecaster | 21.188 | 20.402 | 18.769 | 20.120 | |
| Two years of history, full weight | 21.152 | 20.433 | 18.826 | 20.137 | +0.017 |
| Two years, weight 0.5 | 21.161 | 20.418 | 18.820 | 20.133 | +0.013 |
| Two years, weight 0.25 | 21.164 | 20.393 | 18.820 | 20.126 | +0.006 |
| One year, full weight | 21.147 | 20.421 | 18.811 | 20.126 | +0.007 |
| History in the lookup store only | 21.187 | 20.387 | 18.766 | 20.113 | −0.006 |
| Two years, a model per hour of day everywhere | 21.208 | 20.346 | 18.772 | 20.109 | −0.011 |

**With net load known** (isolates the signal; h3's forecaster scores 20.740):

| Forecaster | Mean | Against h3's |
|---|---|---|
| Two years, full weight | 20.722 | −0.018 |
| Two years, weight 0.5 | 20.724 | −0.015 |
| One year, full weight | 20.707 | −0.033 |
| Two years, a model per hour of day everywhere | 20.652 | −0.088 |

**Paired by week against h3's forecaster** (33 weeks):

| Forecaster | Harness | Weeks better / worse | Mean gain per week |
|---|---|---|---|
| Two years, full weight | h3 stack | 15 / 18 | +0.0016 |
| Two years, full weight | net load known | 18 / 15 | −0.0016 |
| Two years, weight 0.5 | h3 stack | 16 / 17 | +0.0012 |
| Two years, model per hour everywhere | net load known | 11 / 22 | −0.0080 |

**The same comparison in the test's own layout** (fold C with the competition-span data fitted on days 1–14 only, so nothing from after the week; 11 weeks; finished 20:14):

| Forecaster | Harness | Weeks better / worse | Gain on the fold | t |
|---|---|---|---|---|
| Two years, full weight | net load known | 9 / 2 | +0.05 | 1.1 |
| Two years, full weight | h3 stack | 7 / 4 | +0.05 | 0.5 |
| Two years, weight 0.5 | net load known | 8 / 3 | +0.04 | 1.8 |
| Two years, weight 0.5 | h3 stack | 7 / 4 | +0.05 | 0.5 |

So in the layout closest to the real test the history is worth about +0.04 to +0.05 on this one fold, positive in most weeks but not distinguishable from noise (t between 0.5 and 1.8).

Reading:

- **The sign depends on the harness and the fold.** In the h3 stack the two test-like folds gain (+0.03, +0.06) and the cold-start fold loses; with net load known everything is within ±0.03 except the losses. Week by week it is a coin toss.
- **Accuracy again failed to convert.** With the history, a model per hour of day cuts the forecast error for hours +3 to +11 by 6 % (12.5 against 13.3 g/kWh) and scores worse. This is the fourth time on this problem.
- **Why more data does not help (inferred):** the forecaster's weak point is the level of the next six hours on a given day, which depends on that day's wind and sun across Spain. More past years sharpen the average relationship but say nothing about today.

## 5. Decision

Not adopted as it stands. Across all the evidence the history is weakly positive where the layout resembles the test (+0.03 on fold B, +0.06 on C, +0.05 on C in the test's layout) and negative on the cold-start fold, and no comparison reaches significance. A fair summary is "perhaps +0.03 to +0.05, perhaps nothing".

It could still be built as a candidate if the team wants one: at run time the forecaster is h3's with different fitted coefficients, so the package would be h3 with part of its state file replaced. The cost is one more packaging round and a line in the presentation saying that the signal model was fitted on public data older than the competition's.

Kept in the repo as a tested idea: the download script with its guard, the forecaster class, the evaluation.
