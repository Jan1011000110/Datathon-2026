# Would a holiday calendar help? (No, not enough to ship)

**Written:** Sat 14 Nov 2026, 22:15. Study by the load specialist (finished 22:10), decision by me. Code: `experiments/21_load/cal_*.py`; open `cal_p2_log.txt` first.

## 1. Why we looked

Our validation weeks are days 1–21 of each month. The test weeks are the last seven days, and they contain things the folds never do:

- **Four test weeks start on a holiday:** 25 December and 24 June for certain; 24 September and 25 May if the campus closes on Barcelona's local holidays.
- **The December week is all Christmas closure.**
- The load forecaster has no calendar. It notices a holiday from the morning's readings, so the night before is planned as if a normal working day follows.

## 2. Verdict

**Not worth shipping.**

| Ranking | Expected gain from a calendar |
|---|---|
| Private half (last 84 h of each week) | about 0 |
| Whole run | about +0.03 |
| Public half | about +0.07 |

- All of it comes from one date, 24 June, and the estimate rests on a single comparable day in our folds (11 September).
- National holidays gain nothing, so 25 December is worth nothing.
- The calendar was built anyway, off by default. With no calendar the forecasts are bit-identical to today's.

All scores are closed loop inside the h3 stack, folds without August, 0 infractions in every run. h3 scores 19.4247 / 18.5127 / 18.5562 in that setting.

## 3. Why the gain is so small

**The feared mechanism does not happen on national holidays.** On 1 January, 1 May and 8 December (all first days of a fold week):

| Day | Grid charge 00:00–06:00 | Solar surplus of the day | Surplus lost |
|---|---|---|---|
| 1 Jan | 0 kWh | 669 kWh | 0 |
| 1 May | 0 kWh | 3,229 kWh | 0 |
| 8 Dec | 0 kWh | 716 kWh | 0 |

- **National holidays are billed like weekends, so the price signal is flat.** The morning-to-night ratio of the signal is 0.92–0.97 on those days against 1.25–1.28 on a normal weekday. With a flat signal h3 buys nothing at night, so there is nothing to get wrong.
- **The forecaster is blind until 06:00 but knows by 09:00**, before the solar surplus arrives.
- **Where it does happen:** on 11 September, a regional holiday billed as a weekday, h3 charges 1,048 kWh at night that it would not with the calendar. 24 June (Sant Joan) is the same kind of day.

## 4. Upper bounds

Gain over h3 in fold-score points if the forecaster were told the truth:

| What it is told | Fold A | Fold B |
|---|---|---|
| True holiday index on every low weekday | +0.009 | +0.025 |
| The same, plus the current hour pulled to the holiday level | +0.010 | +0.046 |
| Perfect load on the low days | +0.022 | +0.048 |
| Perfect load everywhere (for scale) | +0.195 | +0.181 |

By event, only 11 September (+0.029) and the unexplained low day of 12 February (+0.017) carry anything. Every national holiday is between 0.000 and +0.002.

## 5. The calendar as built

`CalLoadFC` in `experiments/21_load/cal_fc.py`, a subclass of `LoadFC`: each date has a kind (holiday, closure, local, Holy Week), each kind a prior fitted on the fit data only, and the day's readings keep correcting it as before.

Closed loop, gain over h3:

| Variant | A | B | C | Mean | Public half | Private half |
|---|---|---|---|---|---|---|
| Full calendar | +0.030 | +0.024 | +0.004 | +0.019 | +0.028 | +0.009 |
| Minimal: only 11 Sep acts | 0.000 | +0.031 | +0.004 | +0.012 | +0.030 | −0.009 |
| Only the first day of each stretch acts | −0.012 | +0.026 | +0.004 | +0.006 | +0.021 | −0.011 |

- **11 of 30 weeks change: 6 better, 5 worse.** The mean is +0.019 ± 0.012 per fold. That is weak evidence.
- **The prior helps on one event in seven** (11 September).
- **Holy Week is not supported:** switching that kind on made the April week worse.

Cost of a wrong entry (a normal weekday declared a holiday): about −0.008 points each at full trust, −0.002 at half trust.

## 6. What it would mean on the test

| Calendar | Whole run | Public half | Private half |
|---|---|---|---|
| 24 June only | +0.034 | +0.068 | about 0 |
| + Christmas week | adds −0.01 to +0.02 | about −0.01 | −0.01 to +0.04 |
| + 24 Sep and 25 May, both real closures | adds +0.068 | +0.136 | 0 |
| + 24 Sep and 25 May, neither real | adds −0.016 | −0.033 | 0 |

Whether the campus closes on Barcelona's local holidays cannot be checked: none falls in the training days.

## 7. Decision

Not packaged. The gain is about +0.03 over the whole run and zero on the private half, it rests on one day, and a new forecaster class in the packaged agent is a risk with nothing to pay for it.

If it were ever wanted: the packaged agent would need five methods from `cal_fc.py` in the run-time load module, the calendar as constants, and three numbers of state (priors 0.935 / 0.624 / 0.456). Reference scores for a packager are in the specialist's log.

One useful by-product: **perfect knowledge of the load is worth only about +0.19 in the h3 stack.** The rest of the 1.07 between h3 and the perfect-foresight ceiling is the price and CO₂ signal and the solar output.

## 8. Packaged after all, on the last morning (Sun 15 Nov, 10:46)

With 80 minutes left the team asked for the best general model we could ship. The calendar is the only measured, unshipped, honest gain, so it was packaged as two candidates that differ only in which dates act:

| File | Dates that act |
|---|---|
| `submissions/h4/submission.zip` | 24 June 2026 (Sant Joan), 24 September 2025 and 25 May 2026 (Barcelona's local holidays) |
| `submissions/h4a/submission.zip` | 24 June 2026 only |

Checks (packager, then repeated by me on the shipped ZIPs):

- **Same state as h3j:** 66 of 67 arrays identical; only the configuration differs.
- **Calendar off:** 0 commands differ from the shipped h3j on folds A, B, C.
- **The port matches the research class:** 19.4247 / 18.5434 / 18.5605 with only 11 September active and 19.4543 / 18.5362 / 18.5605 with the full calendar, the specialist's numbers to four decimals, 0 commands differing.
- **On a run laid out like the test:** h4 differs from h3j in 161 commands, all in the September, May and June weeks; h4a in 55, all in the June week. 0 infractions, every step answered by the full agent. Organisers' environment: 0 infractions. Validator passes.
- **If the calendar code fails,** the agent falls back as before with 0 infractions.

Expected on the public board against h3j's 23.32: h4a about +0.07; h4 about +0.14 more if the campus closes on both local holidays, about −0.03 if on neither. All three dates are in the public half, so the board measures the whole effect.

**Result on the server (Sun 15 Nov, 10:48).** h4 scored **23.55** on the public board, +0.23 over h3j's 23.32. The prediction was +0.20 if the campus closes on both local holidays and about +0.04 if on neither, so the three dates are confirmed. h4 is now the team's general pick for the jury, next to h3w.
