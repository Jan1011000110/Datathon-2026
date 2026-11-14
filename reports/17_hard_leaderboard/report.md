# Hard-mode leaderboard, 17:58, and what it implies

**Written:** Sat 14 Nov 2026, 18:06. Raw data: `hard_leaderboard_20261114_1758.csv` (the site's own CSV export). Script: `experiments/25_integration/public_ceiling.py`.

## 1. Summary

- **13 teams, 22 submissions. The top two are at 23.23 and 23.10.**
- **They are not above the ceiling.** Our 21.26 is the perfect-foresight ceiling on train folds. The public test scores higher for everyone; the ceiling there is about **23.7** (estimate, 23.5 to 24.0).
- **The top two are at about 98 % of that ceiling.** A group of four sits at 93–94 %, five more at 89–92 %.
- **We have not submitted yet.** By the same scaling, h0 would land near 21.2 and the pieces finished at 18:00 near 21.8 each, before combining them.

## 2. Best submission per team

Times on the site are UTC, one hour behind local time.

| # | Team | Score | CO₂ −% | Cost −% | CO₂ avoided, kg | € saved | Submissions | Share of estimated ceiling |
|---|---|---|---|---|---|---|---|---|
| 1 | Low BaterIA | 23.23 | 22.01 | 26.09 | 13,263.4 | 24,777.77 | 3 | 98 % |
| 2 | SSA Gourmet | 23.10 | 21.74 | 26.29 | 13,098.0 | 24,968.09 | 4 | 98 % |
| 3 | Los Rojos | 22.33 | 21.45 | 24.38 | 12,922.9 | 23,153.35 | 3 | 94 % |
| 4 | Viquipèdia racers | 22.25 | 21.34 | 24.37 | 12,858.3 | 23,146.33 | 3 | 94 % |
| 5 | PrimerMatesInfoUb | 22.23 | 20.93 | 25.27 | 12,610.2 | 23,998.89 | 1 | 94 % |
| 6 | La Datoneta | 22.02 | 20.78 | 24.92 | 12,519.3 | 23,666.72 | 1 | 93 % |
| 7 | manguets | 21.80 | 20.51 | 24.81 | 12,359.3 | 23,564.60 | 2 | 92 % |
| 8 | ASCII | 21.66 | 20.28 | 24.87 | 12,220.7 | 23,617.82 | 1 | 91 % |
| 9 | BOLETSIA.CAT | 21.08 | 20.29 | 22.94 | 12,225.8 | 21,783.66 | 1 | 89 % |
| 10 | Equip | 21.04 | 19.61 | 24.36 | 11,816.9 | 23,141.26 | 1 | 89 % |
| 11 | LosAvocados | 21.04 | 19.61 | 24.36 | 11,816.9 | 23,141.26 | 1 | 89 % |
| 12 | JAMM | 18.17 | 17.70 | 19.26 | 10,667.1 | 18,291.87 | 1 | 77 % |

"Equip" and "LosAvocados" have identical rows 24 seconds apart: the same file submitted under two names.

The no-battery baseline implied by all rows is 60,254 kg CO₂ and 94,973 €.

## 3. How the public ceiling was estimated

We cannot compute the ceiling on the hidden test. Two things we can measure give an estimate:

1. **On train folds, how much more the hard-mode optimum saves than the standard-mode optimum on the same hours:** 3.09 to 3.18 times the CO₂ and 2.82 to 2.90 times the euros.
2. **What the standard-mode optimum saves on the public test:** 4,289.4 kg and 8,907.87 €. Ten teams, including the organisers' own run, sit on that optimum.

Multiplying and dividing by the hard-mode public baseline:

| Ratio taken from | Estimated public ceiling |
|---|---|
| Fold A | 23.99 |
| Fold B | 23.49 |
| Fold C | 23.55 |
| **Mean** | **23.68** |

This assumes the test weeks respond to the larger battery and solar plant the way train weeks do. The test weeks include holidays that train does not, so the true figure could differ by a few tenths.

## 4. What the top scores would take

Shares of the ceiling that we have measured on our folds:

| Agent | Share of the ceiling |
|---|---|
| Naive forecasts, plan trusts them (h0 with haircut) | 89 % |
| Naive forecasts, uncertainty-aware planner | 92 % |
| Load forecaster, rest naive | 92 % |
| Perfect price/CO₂ knowledge, naive net load, uncertainty-aware planner | 96 % |
| Perfect price/CO₂ knowledge and perfect load for the current hour | 98 % |

So 98 % corresponds, in our tests so far, to knowing the future price and CO₂ and the current hour's load almost exactly. The combination of our four pieces has not been measured yet, so this is not a statement about what honest forecasting can reach.

Possible explanations for the top two, none confirmed:

- **Better forecasting and planning than we have built so far.**
- **Real market data.** The prices and CO₂ of the test weeks are public (REE). An agent could carry them and look them up by timestamp. I consider that test leakage and have not done it. Whether it is allowed is a question for the organisers.
- ~~The server's `obs` carries more than the site documents.~~ **Ruled out at 18:50:** our probe on the free validation showed 3 values in each list and no extra keys.
- **The ceiling estimate is a few percent low.**

## 5. What would settle it

- **A probe (done, 18:50).** The free quick validation displays error text; the server sends exactly what is documented.
- **Update, 18:46:** our own agent with a perfect signal scores 20.71 on the folds, 20.84 with the end of the run known. That is 98 % of the ceiling, about 23.2 on the public test, which is where the top two teams are. This is consistent with them knowing the true signal beyond 3 hours. It is not proof.
- **A question to the organisers:** is shipping external public data allowed?
- **Our own first submission,** which calibrates folds against the public test.
