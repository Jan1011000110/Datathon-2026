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

## 5. Are the price and CO₂ series public? (checked 18:59, train days only)

**Yes.** A subagent compared three train days (10 Sep 2025, 14 Jan 2026, 10 Jun 2026) with Red Eléctrica's open data API (REData, `apidatos.ree.es`, no token needed). No test-week value was requested or returned.

| Our column | Published by REE? | Evidence |
|---|---|---|
| `precio_eur_per_kwh` | Yes, exactly | The "PVPC" series of the real-time market prices: 72 of 72 hours identical |
| `gen_con_emisiones_mw`, `gen_sin_emisiones_mw` | Yes | The hourly peninsular series "with / without CO₂-eq emissions": 48 of 48 hours identical on the two 2026 days; 0.15 % off on the 2025 day, which REE revised three days ago |
| `co2_kg_per_kwh` | Derived by the organisers | Exactly `0.37 × with / (with + without)` on all 6,768 train rows |

What follows:

- **The test weeks' true prices and CO₂ can be reconstructed by anyone** from the same public series. So a team can give its agent the real signal for every hour of the test.
- **That would explain the top two scores.** Our own agent with a perfect signal sits at about 98 % of the ceiling, which is where they are. This is circumstantial; we have not seen their code.
- **We have not done it and the agents in this repo do not do it.** The rules describe the run as sequential "to guarantee the absence of future-information leakage", hard mode says the hours beyond +2 must be forecast with models trained on `data/train/`, and every submission is archived for the jury's code audit.
- **The decision is the team's, and the clean way to take it is to ask the organisers** whether external public data is allowed. If yes, we add it. If no, they will want to know it is possible.

One legitimate by-product: the CO₂ signal is a fixed 0.37 times the share of emitting generation. That is worth stating in the presentation, and it means the "CO₂" the battery chases is really the share of gas and other emitting plants in the mix.

## 7. Snapshot at 19:07: ten teams between 23.06 and 23.46

Raw data: `hard_leaderboard_20261114_1907.csv`. 25 teams, 66 submissions. We are not on it yet.

| # | Team | Score | CO₂ −% | Cost −% | Submissions | Share of estimated ceiling |
|---|---|---|---|---|---|---|
| 1 | SSA Gourmet | 23.46 | 22.22 | 26.37 | 7 | 99 % |
| 2 | Ommetje | 23.40 | 22.05 | 26.55 | 2 | 99 % |
| 3 | TeamName | 23.37 | 22.09 | 26.36 | 1 | 99 % |
| 4 | SanHackobos_v3 | 23.35 | 22.10 | 26.25 | 1 | 99 % |
| 5 | aniMLs | 23.30 | 22.05 | 26.23 | 3 | 98 % |
| 6 | hackers la l | 23.29 | 21.95 | 26.40 | 1 | 98 % |
| 7 | Low BaterIA | 23.23 | 22.01 | 26.09 | 4 | 98 % |
| 8 | AAA | 23.18 | 21.88 | 26.23 | 2 | 98 % |
| 9 | LosAvocados | 23.13 | 21.81 | 26.20 | 6 | 98 % |
| 10 | Fanboy & Chum Chum | 23.06 | 21.80 | 26.00 | 1 | 97 % |
| 11–15 | JAMM, EquipB, PrimerMatesInfoUb, Los Rojos, Viquipèdia Racers | 22.48–22.95 | | | | 95–97 % |
| 16–21 | La Datoneta, manguets, ASCII, Yukiyuk, BOLETSIA.CAT, Equip | 21.04–22.02 | | | | 89–93 % |

Where our agents would land if the fold-to-public scaling (×1.114) holds:

| Agent | On our folds | Scaled to the public test |
|---|---|---|
| h2 (signal forecast from train data) | 20.11 | about 22.4 |
| h3 | 20.19 | about 22.5 |
| Our agent given the true signal for every future hour | 20.84 | about 23.2 |
| Same, also given the true net load of the current hour | 21.16 | about 23.6 |
| Ceiling | 21.26 | about 23.7 |

Reading:

- **The top band is where our own agent lands when it is given the true signal.** Ten teams sit in 0.4 points, between "true signal with our net-load handling" and "true signal with the current hour's load known as well".
- **A band that narrow points to a shared level of information,** as in standard mode, where every team with the 24-hour signal converged on 6.43. Honest forecasting would spread teams out.
- **Four teams entered the band with their first submission** (TeamName, SanHackobos_v3, hackers la l, Fanboy & Chum Chum).
- **The gap from h2 to the band, about 0.9 points, equals what the signal beyond 3 hours is worth** in our own decomposition (0.72 on the folds, 0.80 scaled).
- **The signal is public** (section 5).

So the most likely source of the difference is the real price and CO₂ of the test weeks, carried inside the submission. This is inference from the scores; we have not seen anyone's code.

**The alternative we cannot yet exclude:** the public test may be easier than our folds by more than the 11 % we assume. Then the ceiling would be higher than 23.7, those teams would be at about 95 % like us, and h2 would score with them. Our first h2 submission settles this: about 22.4 means the gap is real; 23 or more means we were wrong about the ceiling.

## 6. What would settle it

- **A probe (done, 18:50).** The free quick validation displays error text; the server sends exactly what is documented.
- **Update, 18:46:** our own agent with a perfect signal scores 20.71 on the folds, 20.84 with the end of the run known. That is 98 % of the ceiling, about 23.2 on the public test, which is where the top two teams are. This is consistent with them knowing the true signal beyond 3 hours. It is not proof.
- **A question to the organisers:** is shipping external public data allowed?
- **Our own first submission,** which calibrates folds against the public test.
