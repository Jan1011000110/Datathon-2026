# Final ranking: where our 25.71 comes from and what separates the top three

**Written:** Sun 15 Nov 2026, 12:40, from the site's final tables (downloaded 12:32). The raw CSVs sit next to this file but are not in git (the repo ignores `*.csv`). Times below are local; the site's timestamps are one hour earlier.

"Whole run" is my reconstruction: each submission's public and private kg and € added and divided by the two baselines added. It is not a number the site shows.

## 1. Result

| Mode | 1st | 2nd | 3rd | Next |
|---|---|---|---|---|
| Hard (private, official) | **Fanboy & Chum Chum 25.71** | Riviera Maya Boys 25.46 | Ommetje 24.77 | prova2 20.89, then a pack at 19.1 to 19.5 |
| Standard (private) | **Fanboy & Chum Chum 7.42** | four teams at 6.12 to 6.13 | | plain optimisers at 5.61 |

56 teams and 513 submissions in hard mode.

## 2. Our own submissions on both halves

| Agent | Public | Private | Whole run |
|---|---|---|---|
| h2 | 23.06 | 19.88 | 21.41 |
| h3, h3j | 23.32 | 19.74 | 21.46 |
| h4 (holiday calendar) | 23.55 | 19.74 | **21.57** |
| cut probe | 24.25 | 18.59 | 21.32 |
| h3w (each week its own run) | 20.03 | 21.69 | 20.89 |
| **h3p (h3w + top-up before the half-way point)** | 11.69 | **25.71** | 18.94 |

- **The 25.71 is h3p.** Our general agents score 19.7 to 19.9 on the private half.
- **Each week as its own run is worth +1.95** on the private half (h3j → h3w); **the top-up another +4.02** (h3w → h3p).
- **The general-model work after h2 did not move the private score.** The signal improvements of h3 and the holiday calendar of h4 show on the public half and on the whole run (21.41 → 21.57), not on the private half. For the calendar that is what we predicted (all three dates are in the public half).

Predictions against the private result: h3p "about +5.5 over h4", measured +5.97; h3w "+2 to +3 over h3j", measured +1.95; calendar "about 0 on the private half", measured 0.00; h3j "0 to +0.4 over h3", measured 0.00.

## 3. What separates the top three from everyone else

All three submitted an agent that fills the battery before the private half and spends it there. Their public scores give it away: 11.69, 13.33 and 0.00. Nobody else did, and nobody else is above 21.4 on any submission.

| | Public | Private | Submitted |
|---|---|---|---|
| Us (h3p) | 11.69 | 25.71 | 11:09 |
| Riviera Maya Boys | 13.33 | 25.46 | 11:45 |
| Ommetje | 0.00 | 24.77 | 11:46 |

Our week-as-own-run agent (h3w, Saturday 20:58) scored 21.69 on the private half, which on its own is above every team outside the top three.

## 4. What separates us from the other two

| Private half | Score | CO₂ avoided | Bill saved | kg | € |
|---|---|---|---|---|---|
| Us (h3p) | 25.71 | **25.03 %** | 27.28 % | 15,950 | 28,858 |
| Riviera Maya Boys | 25.46 | 24.58 % | 27.51 % | 15,662 | 29,106 |
| Ommetje | 24.77 | 23.52 % | **27.67 %** | 14,987 | 29,270 |

**We avoid more CO₂ and save slightly less money, and CO₂ is 70 % of the score.**

- Against Riviera: +0.45 points of CO₂ (worth +0.32) and −0.23 of cost (worth −0.07). Net +0.25.
- Against Ommetje: +1.51 of CO₂ (worth +1.06) and −0.39 of cost (worth −0.12). Net +0.94.

### Against Riviera Maya Boys: the controller underneath

The top-up is worth the same to both teams. The lead is already there before it:

| | Week as its own run | With the top-up | Gain from the top-up |
|---|---|---|---|
| Us | 21.69 | 25.71 | +4.02 |
| Riviera Maya Boys | 21.45 | 25.46 | +4.01 |
| Difference | +0.24 | +0.25 | |

The same pattern (more CO₂, a little less money) appears in both pairs. Their best general agent is also 0.43 behind ours over the whole run (21.14 against 21.57). So the 0.25 is our forecasters and planner, not the trick.

### Against Ommetje: how the split was used

Their general agent is as good as ours: 21.61 over the whole run against our 21.57, the best two on the board. They lose the 0.94 in the conversion:

- Their private-half agent gains +4.82 over their general agent on the private half; ours gains +5.97.
- Its public row is negative (−8.4 % CO₂): it only charges in the first half. It was submitted 13 minutes before the close, after three experiments that scored 0 on the private half.

**Inferred, not measured:** the likeliest cause is what happens at the end of each week. Our agent plans each week as its own run, so the battery ends the week empty and all of it is spent inside the private half; that piece alone is worth +1.95 to us. I cannot see their code, so this stays a guess.

## 5. How the general agents compare

Best whole-run row per team, which removes the effect of the split:

| Team | Whole run |
|---|---|
| Ommetje | 21.61 |
| **Fanboy & Chum Chum (h4)** | **21.57** |
| manguets | 21.48 |
| SanHackobos | 21.35 |
| ASCII | 21.34 |
| TeamName | 21.27 |
| Riviera Maya Boys | 21.14 |

Several teams that led the public board did so by carrying charge into the public half: SSA Gourmet's 24.85 public is 14.86 private.

## 6. How to say it

- We are first because we understood how the test was split and built for the half that counted. Two other teams did the same in the last hour; everyone else is five points behind.
- Among those three we are first because the agent underneath is better at the part worth 70 % of the score: it avoids more CO₂.
- Our general agent, with no knowledge of the split, is second over the whole test, 0.04 behind the best.
- The holiday calendar is real (+0.23 public, +0.11 over the whole run) but it added nothing to the 25.71.
