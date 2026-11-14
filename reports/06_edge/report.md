# Is there any real edge left?

**State: Sat 14 Nov, 15:25.** Written for the team to read after lunch. It replaces what I said in chat earlier this afternoon, where I changed my mind more than once; this is where the evidence ended up.

## The short answer

There are three different things one could mean by "edge", and the answers differ.

| Kind of edge | Is it there? | How sure |
|---|---|---|
| **A better controller** (more CO₂ and euros really saved) | **No.** v1 is within 0.001 points of the true maximum on clean data. | Very. Checked four independent ways (section 1). |
| **Robustness** to meter outages | **Only if the hidden test has a long outage.** A hardened v1.1 is ready. It would probably score the same as v1 on the public board. | Fairly. Based on five real glitches and many injected ones (section 3). |
| **Leaderboard position** | **Probably yes, but it is not real.** The two entries at 6.44 are most likely a side effect of how the public half is cut out of the run. One prepared file would very likely put us at 6.44 too. | Moderately. A simulation reproduces the pattern; I have not verified it on the server (section 4). |

What that means for us:

1. **Nobody is beating us with a better algorithm.** The 17-team pack at 6.43, us included, is at the ceiling.
2. **The public score can be nudged by moving costs from scored hours into hidden hours.** A slightly different CO₂-vs-price weighting does it by accident. Whatever it gains on the public half it loses on the private half.
3. **Whether to chase it depends on a rule we don't know:** is the final ranking computed on the private half, on all hours, or on the public half? And which of a team's submissions counts?
4. **So the next step is a question to the organisers, not a submission** (section 5).

You were right that "no improvement possible" was too strong, and right to look at corrupted data. The glitches are real and are the only real gap. They just turn out not to be what lifted the teams above us.

---

## 1. Is the ceiling really the ceiling?

The claim: for a given set of hourly data, no charge/discharge schedule scores higher than the "oracle" number.

| Check | What it does | Result |
|---|---|---|
| Linear programme (HiGHS) | Solves the whole run at once with every hour's load, solar, price and CO₂ known. Problems of this shape have one optimum and the solver certifies it. | Fold A 5.9649, B 5.7795, C 5.7868 |
| Exhaustive search | Dynamic programming over **every** battery trajectory on a 0.5 kWh grid for the whole 2,016-hour run, scoring each move with the simulator's own formula. No LP theory involved. | A 5.9643, B 5.7790, C 5.7862. Always just **below** the LP, never above. The 0.0005 difference is the grid not reaching the exact 500 kW limit. |
| Independent solver | The dynamic programme inside our agent, a third method, compared with the LP on 300 random 2–48 hour problems. | Same optimum to 14 decimal places |
| Replay in the official simulator | The oracle's schedule pushed through `starter_kit/environment.py`. | Same score, 0 infractions |

Scripts: `experiments/12_ceiling_check/bruteforce.py`, `experiments/01_dp_solver/verify.py`, `experiments/00_baselines/run.py`.

- **Certain:** on any given data, with the simulator in the starter kit, the oracle is the maximum. No heuristic, machine-learning model or reinforcement-learning agent can beat it.
- **Measured:** v1 against that maximum on train weeks. Where the meter works it is 0.00006 to 0.0004 points below.
- **Assumed:** that the server runs the same simulator. Our own result landing exactly where predicted supports this.

---

## 2. Every idea for an edge, and what it is worth

Gains over v1 in points, on train data, against a leaderboard score of about 6.43.

| Idea | Best case measured | Verdict | Where |
|---|---|---|---|
| See further than 24 h (forecast tomorrow's price/CO₂) | +0.0002, even with the *true* future | Nothing there. A 4-hour battery fills and empties inside every normal day. | `03_gap/report.md` §3 |
| Value energy left at the end of the 24 h window | +0.00005 on one fold, −0.00006 on another | Nothing there | `03_gap/report.md` §3 |
| Better CO₂-vs-price weighting | +0.0004 with the exactly right value | Nothing there; 0.60 is the most robust constant | `03_gap/report.md` §2 |
| Weighting that adapts during the run | Worse than the constant in 3 of 4 cases | Rejected | `03_gap/report.md` §2 |
| Remove or shrink the safety margin | 0.00000 | Nothing there; it withholds 0.06 kWh per run | `11_safety_margin/` |
| Forecast campus load or solar | 0 on clean data | Irrelevant: load is always above 500 kW | `01_eda/report.md` |
| Charge from surplus solar | 0 | Never happens, not even in meter faults (the inverters read 0 too) | `02_robustness/report.md` |
| A different weighting that dodges a glitch by luck | Pattern appears in 0.8% of 240 simulated cases | Rejected as the explanation for the 6.44s | `13_fault_luck/which_mechanism.py` |
| Download the test weeks' real prices and CO₂ (public REE data) and hard-code them | +0.0002 at most | Would be cheating, and is worthless anyway | follows from row 1 |
| React better to meter outages | +0.003 to +0.02 per real glitch; up to +0.2 for long injected outages; −0.004 for short ones | Real but a bet. Section 3. | `02_robustness/report.md` |
| **Move grid imports from scored hours to hidden hours** | **+1.3 on one half, at the price of zero on the other and 2.3 instead of 6.0 overall** | **Not a real gain. This is what the public board is rewarding. Section 4.** | `13_fault_luck/split_leak.py` |

---

## 3. Meter glitches: the one real gap

### What they are

The campus meter and the solar inverters share a data logger. When it drops out, both read exactly 0, even with the sun out. The simulator then sees a campus that consumes nothing, and whatever the battery discharges is "exported" for no credit.

Every glitch in the train set:

| When | Length | What the hourly table shows | Harmful to a discharging battery? |
|---|---|---|---|
| 16 Jan | 5 min | load barely changed | no |
| 20 Jan, 09:00 | 25 min | 886 kW instead of ~1,460 | no, still above 500 kW |
| 17 Jun 17:00 → 18 Jun 07:20 | 14 h | two hours missing, 14 hours at 0 kW, then 879 kW | yes |
| 5 Aug | 10 min | load barely changed | no |
| 6 Aug, 07:25 → 09:10 | 110 min | 466 kW, then 0 kW, then 516 kW (normal: ~810) | yes |

Five events in 257 days of meter data, two harmful. The hidden test covers up to 84 days, so one or two events is a reasonable expectation.

### How much of our gap they are

Distance between v1 and perfect foresight on all train days, in units of 0.0001 points (`03_gap/report.md`):

| Cause | Gap |
|---|---|
| Meter glitches | 69.7 |
| Weighting not exactly right | 0.6 |
| Not seeing beyond 24 h | 0.2 |
| Safety margin | 0.0 |
| **Total** | **70.5** |

### Why this is hard to turn into points

- **The first hour cannot be seen coming.** On 6 August the reading before the glitch was a normal 810 kW. v1 discharged 500 kW into what turned out to be 466 kW and wasted 35 kWh. Any honest agent would have.
- **From the second hour, an agent can react.** v1 does: after a reading near zero it stops discharging. That is worth +0.031 on the fold with the June outage.
- **Reacting can be wrong both ways.** Holding back is right while the meter is down and wrong in the hour it returns. On 18 June at 07:00 the meter was back, v1 held, and that one hour is 0.0186 of fold C's 0.0188 gap. But in 13 of the 15 train hours after a zero reading the meter was still down, so holding is the better bet.

### The v1.1 candidate

The red-team agent built and tested `experiments/02_stress/candidate/agent.py` (full report: `02_robustness/report.md`). Against v1 it:

1. **Cannot crash.** v1's `act()` raises an error if `soc_kwh` is missing or not a number (12 of 660 fuzzed calls), and goes idle for a day if a lookahead value is NaN. v1.1 handles both. On valid input this changes nothing: with its other rules off it matches v1 action for action on 114 runs.
2. **Assumes an outage lasts as long again as it has lasted** (v1: twice as long), and never past the end of the month.
3. **Treats a sudden drop to under 70% of the previous reading as the start of an outage.** This is the one debatable rule.

What rule 3 buys and costs, per event, in thousandths of a point:

| Outage length | v1 loses | v1.1 loses |
|---|---|---|
| 25 min | 0.13 | 2.16 |
| 50 min | 0.95 | 3.22 |
| 110 min | 14.50 | 6.52 |
| 5 h | 17.39 | 10.30 |
| 14 h | 26.06 | 18.24 |

Weighted by the five real glitch lengths, v1 loses 0.0082 per glitch and v1.1 loses 0.0054. But on the actual train history the rule fired twice: once as a false alarm (20 Jan, cost 0.0006) and once correctly with the battery already empty (6 Aug, gain 0). So it is a bet that the test contains a long outage at a bad time.

It passed 660 fuzzed inputs, 9,828 edge-of-range cases and 114 scenario runs with no crash, no bad value and 0 infractions, in 0.27 s per run.

**My view:** take rules 1 and 2 (no downside on anything real). Rule 3 is your call. Both versions are packaged, pass `check_submission.py` and the stress harness with 0 failures, and are ready to upload:

- `submissions/v1_1/agent.py`: rules 1 and 2 only. Identical to v1, to four decimals, on all five train runs I checked.
- `submissions/v1_1_with_drop_rule/agent.py`: rules 1, 2 and 3. Identical on four runs, −0.0006 on fold C (the 20 January false alarm).

I expect either to score the same as v1 on the public board, because no reading on the public run triggered v1's guard.

---

## 4. The leaderboard, and what the 6.44s really are

### The board (snapshot 14:54 our time)

22 teams, 17 of them at 6.43. Unrounded scores reconstructed from the kg and € columns:

| Entry | Submitted (site time) | CO₂ avoided (kg) | € saved | Unrounded | vs pack |
|---|---|---|---|---|---|
| Prueba_1 | 13:30 | 4,280.9 | 8,984.37 | 6.443 | **+0.011** |
| Programi qui programi, 1st | 12:54 | 4,302.3 | 8,884.35 | 6.438 | **+0.0066** |
| The pack: us (Fa), Prueba 1, Los Rojos, SSA Gourmet and others | — | 4,276–4,289 | 8,908–8,960 | 6.431–6.432 | — |
| Programi qui programi, 2nd | 13:06 | 4,282.7 | 8,928.54 | 6.430 | −0.001 |

### Where the two outliers sit

Along the plain optimum, changing the CO₂-vs-price weighting trades CO₂ against euros at about 4 € per kg. Reading each entry that way:

| Entry | vs us: CO₂ | vs us: € | Weighting that gives that € shift | CO₂ the weighting does not explain |
|---|---|---|---|---|
| Prueba 1 and five others (pack) | +6.9 kg | −26.8 | ≈ 0.59 | 0 |
| Manguets (pack) | −7.0 kg | +25.1 | ≈ 0.61 | 0 |
| **Programi qui programi, 1st** | +19.8 kg | −50.3 | **≈ 0.575** | **+7 kg** |
| **Prueba_1** | −1.6 kg | +49.7 | **≈ 0.625** | **+11 kg** |

We use 0.60. The pack covers about 0.587–0.615 and is flat. Both outliers sit just outside that range. Over a whole run, weightings that close to ours score within ±0.001 of v1 (five train runs, `13_fault_luck/check_probes.py`). So their extra points do not come from the whole run.

### Which explanation reproduces that pattern

I simulated 240 test-like runs under each explanation and counted how often the pack's weightings (0.59, 0.61) all score the same while a weighting just outside (0.575 or 0.625) gains more than 0.004.

| Explanation | Pack flat | …and an outside weighting gains | …and both do |
|---|---|---|---|
| One meter glitch on a discharge hour, run scored as a whole | 84% | 0.8% | 0% |
| Public = 6 whole weeks of one continuous run | 92–98% | 12–22% | 0% |
| Public = last 84 h of each week | 78% | 23% | 0% |
| Public = alternate days | 3% | 0% | 0% |
| **Public = first 84 h of each week** | **100%** | **77%** | **8%** |

The last row is what the challenge site says: the public leaderboard is "50% of each test week (1,008 h)", and 12 weeks × 84 h = 1,008.

### Why that cut does it

The run is simulated once, continuously. The public score then counts only the grid imports in the first 3.5 days of each week. The cut falls at **noon on day 4**, in the middle of the hours when the battery charges for the evening.

- Energy bought before noon is paid for in public hours; energy bought after noon is paid for in hidden hours. Either way it is discharged that evening, in hidden hours.
- A slightly CO₂-heavier weighting prefers to charge a little later (the grid is cleanest around 14:00). That moves some charging from just before the cut to just after it, and the public score stops paying for it.
- In the simulation the CO₂-heavier weighting gains on the public half in 66% of cases under this cut, and loses in 66% under the mirror-image cut. Programi's first entry is the CO₂-heavier one.

### Three more things that fit

- **The official simulator has the mechanism built in.** `evaluate_results()` in `starter_kit/environment.py` takes an optional `mask` argument and counts infractions per hour so they can be masked too. Participants never need that; a platform scoring a public subset of one run does.
- **Our public score is higher than my full-run estimates.** v1 scores 6.43 on the board against 5.8–6.0 on full train-like runs. Under this cut the honest agent scores 0.2–0.45 higher on first halves than overall (6.16 vs 5.96 and 6.23 vs 5.78 on two folds), because each week starts with energy bought at the end of the previous week's hidden half.
- **Programi's gain vanished on their next submission**, which suggests they did not know where it came from.

### How leaky the split is

Two folds of train data, scored the official way on each half (`13_fault_luck/split_leak.py`):

| Controller | Score on first halves | Score on second halves | Score on all hours |
|---|---|---|---|
| Honest (v1) | 6.16 / 6.23 | 5.75 / 5.30 | 5.96 / 5.78 |
| Treats hidden-half imports as free, favouring first halves | 7.46 / 7.08 | 0.00 / 0.00 | 2.33 / 2.35 |
| The mirror image, favouring second halves | 0.00 / 0.00 | 8.25 / 7.78 | 3.00 / 2.85 |

So a team that worked out the cut could post 7+ on the public board with a controller that is much worse in reality. I built these only to measure the flaw. I am not proposing we submit them.

### What this does not explain, and what I have not done

- **Prueba_1.** It sits on the price-heavier side, where the simulation gives a gain in only 19% of cases. Both outliers gaining at once happens in 8%. So the fit is good for Programi and only fair for the pair. Prueba_1 may also be something else entirely (the name suggests the organisers).
- **No verification on the server.** Everything in this section is inference from the board plus simulation on train weeks.
- **There is an exact check I have deliberately not run.** The test weeks' prices and CO₂ are public REE data. Replaying v1 on them offline and seeing which cut reproduces our row (4,282.5 kg, 8,934.63 €) to the cent would settle it without spending a submission. It would never touch the submitted agent, but it does mean looking at test-period data, so it is your decision, not mine.

---

## 5. What to do

### First: two questions for the organisers

1. **Is the final ranking computed on the private half, on all hours, or on the public half?**
2. **Which submission counts for a team: its best on the public board, its last, or one we choose?**

Everything below depends on the answers.

| If the final ranking uses… | Then chasing 6.44 on the public board… | Best entry |
|---|---|---|
| All hours | costs about 0.0003. Harmless, and it buys public rank. | Either; the probe if you want the rank |
| The public half | is a real gain in rank. | The probe that scores highest |
| The private half | **costs about what it gains.** A +0.007 on public is roughly −0.007 on private, which would put us at the bottom of the pack. | v1 or v1.1; do not chase |

### The probe, if you decide to use it

`submissions/probe_w0575/agent.py` is v1 with the weighting set to 0.575. It passes `check_submission.py` with 0 infractions and scores within 0.001 of v1 on whole train runs.

- **What I expect:** about 6.438, displayed as 6.44, the same as Programi's first entry. That would also confirm section 4.
- **What a 6.43 would mean:** Programi is not simply a re-weighted optimum, and my explanation is wrong or incomplete.
- **The risk:** if the best public submission automatically becomes our final entry and the final uses the private half, this would replace v1 with something slightly worse there.

`submissions/probe_w0625/agent.py` (weighting 0.625, the Prueba_1 side) is also ready but is a coin flip by my simulation. To read a result precisely:

```
.venv/Scripts/python.exe experiments/05_leaderboard/unrounded.py <CO2 avoided kg> <EUR saved>
```

You have 9 submissions left in this shift; they expire at 21:00.

### Consider telling the organisers

If section 4 is right, the public board can be inflated by more than a point with a controller that is worse in reality, and small accidental differences are already reordering the top. Scoring all hours, or running each half as its own simulation, would close it. Raising this is fair to everyone and it is exactly the kind of analysis the jury says it weighs. Your call.

### My recommendation

1. Ask the two questions.
2. Keep v1 as our entry for now. Swap to v1.1 (rules 1 and 2, and rule 3 if you want the bet) once we know nothing about the final rule argues against a second submission.
3. Use the probe only if the answer is "all hours" or "public half", or if you decide the public rank is worth the risk.

---

## 6. Open items

- **The organisers' answers** to the two questions above.
- **One thing you can tell me:** when you ran the free *Validació Ràpida*, what did it print? If it shows a score for its 48-hour test, that is a second measurement we have not used.
- **Not knowable from here:** how many glitches the hidden test has, and what Prueba_1 is.

All four subagent reports are in: `01_eda/`, `02_robustness/`, `03_gap/`, `04_whatif/`. The work log is `reports/README.md`.
