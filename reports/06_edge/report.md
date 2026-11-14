# Is there any real edge left?

**State: Sat 14 Nov, 15:20.** Written for the team to read after lunch. Section 6 lists what I am waiting on and will update.

## The short answer

1. **On clean data, no.** v1 is at the true maximum to within 0.001 points. This is checked four independent ways (section 1) and I am confident in it.
2. **On dirty data, yes, and that is the only place.** When the campus meter drops out, the simulator treats the campus as consuming nothing, and whatever the battery discharges is wasted. On the train set **99% of the distance between v1 and perfect foresight comes from five such glitches** (section 3).
3. **The leaderboard says something extra exists on the public run.** Two entries now sit above the 17-team pack, by +0.007 and +0.011. Both look like our own agent with a slightly different CO₂-vs-price weighting, one a little lower than the pack's range and one a little higher (section 4). In my simulations that small a change is worth 0.001 at most, so something on the public run flips when the weighting leaves the pack's range.
4. **That is testable today with two submissions.** I have prepared and validated them: `submissions/probe_w0575/agent.py` and `submissions/probe_w0625/agent.py` (section 5). If they reproduce the 6.44, we take it and learn what the lever is. If they score 6.43, those two teams are doing something else.
5. **A glitch cannot be forecast**, so I do not expect a principled rule to capture this. A better fault guard (v1.1, section 3) protects against long outages, which could matter a lot on the private half, but will probably not move the public score.

You were right to distrust "no improvement possible". The accurate statement is: no improvement from forecasting, weighting or optimisation; some from handling corrupted data, mostly by luck on the public board.

---

## 1. Is the ceiling really the ceiling?

The claim: for a given set of hourly data, no charge/discharge schedule scores higher than the "oracle" number.

| Check | What it does | Result |
|---|---|---|
| Linear programme (HiGHS) | Solves the whole run at once with every hour's load, solar, price and CO₂ known. Problems of this shape have one optimum and the solver certifies it. | Fold A 5.9649, B 5.7795, C 5.7868 |
| Exhaustive search | Dynamic programming over **every** battery trajectory on a 0.5 kWh grid for the whole 2,016-hour run, scoring each move with the simulator's own formula. No LP theory involved. | A 5.9643, B 5.7790, C 5.7862. Always just **below** the LP, never above. The 0.0005 difference is the grid not reaching the exact 500 kW limit. |
| Independent solver | The dynamic programme inside our agent, a third method, compared with the LP on 300 random 2–48 hour problems. | Same optimum to 14 decimal places |
| Replay in the official simulator | The oracle's schedule pushed through `starter_kit/environment.py`. | Same score, 0 infractions. So my model of the rules matches the organisers' code. |

Scripts: `experiments/12_ceiling_check/bruteforce.py`, `experiments/01_dp_solver/verify.py`, `experiments/00_baselines/run.py`.

What this does and does not prove:

- **Certain:** on any given data, with the simulator in the starter kit, the oracle is the maximum. No heuristic, machine-learning model or reinforcement-learning agent can beat it.
- **Measured:** v1 against that maximum, on train weeks. Where the meter works: 0.00006 to 0.0004 points below it.
- **Assumed:** that the server runs the same simulator as the starter kit, and that the hidden test resembles the train data. The first is supported by our own result landing exactly where predicted. The second is where section 3 comes in.

---

## 2. Every idea for an edge, and what it is worth

All numbers are gains over v1 in points, on train data, against a leaderboard score of about 6.43.

| Idea | Best case measured | Verdict | Where |
|---|---|---|---|
| See further than 24 h (forecast price/CO₂ for tomorrow) | +0.0002, even with the *true* future | Nothing there. A 4-hour battery fills and empties inside every normal day. | `03_gap/report.md` §3 |
| Value energy left at the end of the 24 h window | +0.00005 on one fold, −0.00006 on another | Nothing there | `03_gap/report.md` §3 |
| Better CO₂-vs-price weighting | +0.0004 with the exactly right value | Nothing there; 0.60 is the most robust constant | `03_gap/report.md` §2 |
| Weighting that adapts during the run | Worse than the constant in 3 of 4 cases | Rejected | `03_gap/report.md` §2 |
| Remove or shrink the safety margin | 0.00000 | Nothing there; the margin withholds 0.06 kWh per run | `11_safety_margin/` |
| Forecast campus load or solar | 0 on clean data | Irrelevant: load is always above 500 kW, solar never above load | `01_eda/report.md` §2–3 |
| Charge from surplus solar | 0 | Never happens: solar ≤ 144 kW, load ≥ 459 kW; in meter faults the inverters read 0 too | this report §3 |
| A luckier weighting that dodges a glitch | 0: every weighting from 0.3 to 1.2 wastes the same 35 kWh | Rejected | `13_fault_luck/rho_sweep.py` |
| Download the test weeks' real prices and CO₂ (public REE data) | +0.0002 at most | Would be cheating, and is worthless anyway | follows from row 1 |
| **Handle meter glitches better** | **+0.003 to +0.02 per real glitch; up to +0.2 in injected long outages** | **The only real gap. Section 3.** | `02_stress/`, `03_gap/report.md` §1 |

---

## 3. The one real gap: meter glitches

### What they are

The campus meter and the solar inverters share a data logger. When it drops out, both read exactly 0 for a while, even with the sun out. The hourly table then shows a campus load of zero, or a reduced load if only part of the hour was lost.

Every glitch in the train set (5-minute samples reading zero):

| When | Length | What the hourly table shows | Harmful to a discharging battery? |
|---|---|---|---|
| 16 Jan | 5 min | load barely changed | no |
| 20 Jan, 09:00 | 25 min | 886 kW instead of ~1,460 | no, still above 500 kW |
| 17 Jun 17:00 → 18 Jun 07:20 | 14 h | two hours missing, then 14 hours at 0 kW, then 879 kW | yes |
| 5 Aug | 10 min | load barely changed | no |
| 6 Aug, 07:25 → 09:10 | 110 min | 466 kW, then 0 kW, then 516 kW (normal: ~810) | yes |

So: about five events in 257 days of meter data, two of them harmful. The hidden test covers up to 84 days, so one or two events and perhaps one harmful one is a reasonable expectation.

### Why they cost points

In a glitch hour the simulator sees less campus demand than the battery is delivering. The excess is "exported" and earns nothing. Perfect foresight simply discharges in a neighbouring hour instead.

Share of the v1-to-oracle gap on all train days, in units of 0.0001 points (from `03_gap/report.md`):

| Cause | Gap |
|---|---|
| Meter glitches | 69.7 |
| Weighting not exactly right | 0.6 |
| Not seeing beyond 24 h | 0.2 |
| Safety margin | 0.0 |
| **Total** | **70.5** |

### Why this is hard to turn into an edge

- **The first hour cannot be seen coming.** On 6 August the reading before the glitch was a normal 810 kW. v1 discharged 500 kW into what turned out to be 466 kW and wasted 35 kWh. Any other honest agent would have done the same.
- **After the first hour, an agent can react.** v1 already does: once it sees a reading near zero it stops discharging. That guard is worth +0.031 on the fold with the June outage.
- **The reaction can be wrong in both directions.** Holding back is right while the meter is down and wrong in the hour it comes back. On 18 June at 07:00 the meter was back, it was a top hour, and v1 held because the last reading was 0. That single hour is 0.0186 of fold C's 0.0188 gap. But in 13 of the 15 train hours that followed a zero reading, the meter was still down, so holding is the right bet on average.

### How much a better guard could matter (interim red-team numbers)

The red-team agent injected artificial outages into fold A and measured the loss against perfect foresight. These are interim; its final report is still to come.

| Injected outage | v1 loses | With an added "sudden drop = outage starting" rule |
|---|---|---|
| 25 min, starting 07:58 | 0.009 | 0.049 (worse: it over-reacts to a glitch that is already over) |
| 25 min, starting 19:40 | 0.004 | 0.062 (worse) |
| 110 min, starting 07:58 | 0.194 | 0.048 (better) |
| 110 min, starting 19:40 | 0.331 | 0.111 (better) |
| 5 h, starting 07:58 | 0.193 | 0.058 (better) |

Reading: against a long outage in a discharge window, v1's guard is too slow and can lose 0.2–0.3 points; a faster rule saves most of that but costs 0.04–0.06 on short glitches. On the *real* train glitches the faster rule is slightly worse than v1 (0.0072 against 0.0070 lost over all train days): over the whole train table it fires twice, once as a false alarm (20 Jan) and once correctly but with the battery already empty (6 Aug). So which rule is better depends on what kind of glitch the hidden test contains, and five events are not enough to know.

The red-team agent's candidate is at `experiments/02_stress/candidate/agent.py`. Besides that rule it also:

- makes `act()` impossible to crash. v1 raises an error if `soc_kwh` is missing or garbage (12 of 660 fuzzed calls). On the real server that would end the run, so this part is worth taking regardless;
- stops assuming an outage continues past the end of the month, where the next test week starts;
- returns exactly v1's actions whenever no reading falls below 70% of the previous one or below 50 kW (checked on 114 runs).

It passes 660 fuzzed inputs and 9,828 edge-of-range cases with no crash, no NaN and 0 infractions.

**Why the guard may still matter even though it never acted on the public run:** the November test week has no meter data at all. If the organisers filled it with zeros and it sits in the private half, every team without a guard loses 0.5–0.9 points there and we do not.

---

## 4. What the leaderboard says (snapshot 14:54 our time)

22 teams. 17 are at 6.43. Reconstructing unrounded scores from the kg and € columns:

| Entry | Submitted (site time) | CO₂ avoided (kg) | € saved | Unrounded score | vs pack |
|---|---|---|---|---|---|
| Prueba_1 | 13:30 | 4,280.9 | 8,984.37 | 6.443 | **+0.011** |
| Programi qui programi, 1st | 12:54 | 4,302.3 | 8,884.35 | 6.438 | **+0.0066** |
| The pack: us (Fa), Prueba 1, Los Rojos, SSA Gourmet and others | — | 4,276–4,289 | 8,908–8,960 | 6.431–6.432 | — |
| Programi qui programi, 2nd | 13:06 | 4,282.7 | 8,928.54 | 6.430 | −0.001 |
| Viquipèdia racers, Psycode | — | — | — | 6.429–6.431 | −0.001 to −0.002 |

What I read from it:

- **The pack is the plain optimum.** Six or more distinct schedules, differing only in CO₂-vs-price weighting, all within 0.0005 of each other. That flatness is exactly what my weighting sweep predicts.
- **Two entries are above it, none are equally far below.** So on the public run there is value a plain optimiser misses. In energy terms the two gains are about 30 kWh and 50 kWh of discharge not wasted.
- **That is the size of one short glitch.** A 25-minute dropout on an hour when the battery discharges costs a plain optimiser about 0.006 on a 2,016-hour run, so about 0.012 on a 1,008-hour public run.
- **Prueba_1 is probably the organisers.** The first leaderboard entry was their "Prueba 1". If "Prueba_1" is their own run with full knowledge of the hidden data, then **6.443 is the true ceiling of the public set**, and the total headroom above us is 0.011. This is a guess from the name.
- **Programi's gain did not survive their next submission**, which landed back in the pack. That suggests they did not know why the first one scored higher either.

### Where the two outliers sit

Each team's kg and € can be compared with ours. Along the plain optimum, shifting the weighting trades CO₂ against euros at about 4 € per kg (measured in my sweep and visible in the pack itself):

| Entry | vs us: CO₂ | vs us: € | Weighting that would give that € shift | CO₂ beyond what that weighting explains |
|---|---|---|---|---|
| Prueba 1 and five others (pack) | +6.9 kg | −26.8 | ≈ 0.59 | 0 |
| Manguets (pack) | −7.0 kg | +25.1 | ≈ 0.61 | 0 |
| **Programi qui programi, 1st** | +19.8 kg | −50.3 | **≈ 0.575** | **+7 kg** |
| **Prueba_1** | −1.6 kg | +49.7 | **≈ 0.625** | **+11 kg** |

We use 0.60. So the pack covers roughly 0.587–0.615 and is flat. The two outliers sit just outside that range, on opposite sides, and each has 7–11 kg of CO₂ savings that the weighting shift does not account for.

On train data the same two weightings score within ±0.001 of v1 on five different runs (`experiments/13_fault_luck/check_probes.py`). So something specific to the public run flips at those weightings. Two readings fit:

- **A glitch hour.** The balanced weighting happens to discharge in an hour where the meter had dropped out; a slightly different weighting moves that discharge to a neighbouring hour. In this case the gain is real.
- **A boundary effect.** If the public score counts only some weeks of one long run, a slightly different weighting can move a charge into a hidden week and its discharge into a public one. In this case the gain on the public half is a loss on the private half.

I cannot tell these apart from outside. Either way the public score should respond to the weighting, which is what the probes in section 5 test.

Earlier in the day I thought the 6.44 was an artefact of scoring only half the hours. I no longer think that is the best explanation: the pack is too tight for the public half to be a noisy slice, and two outliers on the same side fit the glitch reading better. The split-noise experiments are still in `experiments/09_public_noise/` and the reasoning in `reports/README.md` §3a.

---

## 5. What we could do next (your call)

**Option A — submit an agent with a better fault guard (v1.1).** The red-team agent is finishing one. Honest expectation: the same 6.43 on the public board, because v1's guard never acted there. Its value is on the private half and in the jury story ("we handle sensor failures"). I will only recommend it if it is no worse than v1 on every real train glitch.

**Option B — two diagnostic submissions (recommended first).** Each is v1 with one number changed, the CO₂-vs-price weighting. Both pass `check_submission.py` with 0 infractions on five train runs.

| File to upload | Weighting | Mimics | On train data it scores, vs v1 | If the public run is like train | If it reproduces the outlier |
|---|---|---|---|---|---|
| `submissions/probe_w0575/agent.py` | 0.575 | Programi's 1st entry | −0.0002 to +0.0010 | 6.43 (about 6.431–6.433) | 6.44 (about 6.438) |
| `submissions/probe_w0625/agent.py` | 0.625 | Prueba_1 | −0.0010 to 0.0000 | 6.43 (about 6.431) | 6.44 (about 6.443) |

After each submission, run this with the two numbers from our new row to get the unrounded score:

```
.venv/Scripts/python.exe experiments/05_leaderboard/unrounded.py <CO2 avoided kg> <EUR saved>
```

Or just tell me the row and I will do it. Two further probes are ready if the first two show an effect and we want to see how far it goes: `submissions/probe_w0550/` and `submissions/probe_w0650/`.

What each outcome means:

- **Both score 6.43:** the public run responds to weighting the way train data does. The two teams above us are doing something I have not identified, and v1 stays our entry.
- **One or both score 6.44:** the lever is the weighting, and we are immediately third on the public board (ties go to the earlier submission). Then the question becomes whether to tune to the public board, knowing it may be a boundary effect that reverses on the private half.

You have 9 submissions left in this shift and they expire at 21:00.

**Option C — fish for it.** Submit many timing variants and keep whichever happens to dodge the glitch. It could put 6.44 next to our name. It would not improve the private-half score, and could lower it slightly. I don't recommend it, but it is your leaderboard.

**My recommendation:** B first. It costs two submissions that would otherwise expire and answers the question you asked. Then decide on A once the red-team report is in.

---

## 6. Open items

- **Red-team final report** (fault guard tuning, interface fuzzing, timing). Running. I will add its numbers to section 3 and say whether v1.1 is worth adopting.
- **One thing you can tell me:** when you ran the free *Validació Ràpida* before submitting, what did it show? If it prints a score for its 48-hour test, that is a second, unlimited measurement we have not used.
- **Not knowable from here:** the number and length of glitches in the hidden test, and how the final ranking is computed (public half, private half or both).
