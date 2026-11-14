# New Rules: Hard Mode ("🔥 Mode Difícil")

The organisers added a sixth tab to the challenge site, **🔥 Mode Difícil**. It is a second, harder version of the challenge with its own leaderboard, and it is the one we are aiming for now.

- **Source:** <https://repteupc.datathon.cat/>, tab *🔥 Mode Difícil*, plus the participant ZIP downloaded from the site.
- **Checked on:** 2026-11-14 at about 16:45, and again at 16:57 (rules and ZIP unchanged between the two).
- **Standard mode** is still open and unchanged. Its rules are in [`WEBSITE_NOTES.md`](WEBSITE_NOTES.md) and [`ENUNCIAT.md`](ENUNCIAT.md). Anything this page does not mention works as described there.

The site's own summary: after seeing the effect of the first battery, this mode evaluates an *expansion* of the campus's energy infrastructure, and it is a mode "on la predicció de la demanda del campus, la generació solar i el mercat elèctric és decisiva".

---

## 1. What changes

| Topic | Standard mode | Hard mode |
|---|---|---|
| Battery capacity | 2,000 kWh | **6,000 kWh** (×3) |
| Max power (charge and discharge) | ±500 kW | **±1,500 kW** (×3) |
| Initial SOC | 50 % = 1,000 kWh | 50 % = **3,000 kWh** |
| Round-trip efficiency | 90 % | 90 % (same) |
| Solar plant | As measured | **×12** (`pv_kw × 12`), "up to ~1,650 kW" |
| Price and CO₂ lookahead in `obs` | 24 values | **3 values**: the current hour and the next 2 |
| `obs["noves_dades_5min"]` | Last hour's load and solar | Adds **4 weather readings** from the last hour. `ultima_solar_kw` is already ×12. |
| New `obs` keys | — | `cap_max_kwh` = 6000.0, `power_max_kw` = 1500.0 |
| Leaderboard | *Classificació General* tab | **Separate ranking** inside the hard-mode tab |
| Submission quota | 10 per shift | **Its own quota**; hard-mode submissions do not use standard-mode attempts |

The three changes the site spells out:

1. **More solar and a bigger battery.**
   - On sunny middays, solar output exceeds campus consumption. The surplus (the site says up to 1,000 kW) is free: 0 € and 0 kg CO₂. Use it to charge.
   - Discharge can now reach −1,500 kW, but the campus's net consumption "usually moves between 600 kW and 1,300 kW". Discharging more than the net load sends the rest to the grid for 0 € and 0 kg CO₂ saved. So the agent must predict the real load and solar before discharging hard.
2. **Only 3 hours of market data.**
   - `obs["preus_properes_24h"]` and `obs["co2_properes_24h"]` keep their names but hold only 3 values.
   - Planning a full daily cycle needs our own forecast of prices and CO₂ for hours +3 to +24, trained on `data/train/`.
3. **Live weather station in `obs`.** See §2.1.

---

## 2. Details

### 2.1 The `obs` dictionary in hard mode

```python
obs = {
    "timestamp": Timestamp,
    "hour": int,
    "day_of_week": int,
    "soc_kwh": float,                 # 0 to 6000
    "soc_pct": float,                 # 0.0 to 1.0
    "soc_percent": float,             # 0 to 100
    "cap_max_kwh": 6000.0,            # NEW
    "power_max_kw": 1500.0,           # NEW
    "preu_actual": float,
    "co2_actual": float,
    "preus_properes_24h": [p_t, p_t+1, p_t+2],   # only 3 values
    "co2_properes_24h":   [c_t, c_t+1, c_t+2],   # only 3 values
    "noves_dades_5min": {
        "ultim_consum_kw": float,     # load of the last hour
        "ultima_solar_kw": float,     # solar of the last hour, already ×12
        "irradiancia_wm2": float,     # NEW, W/m²
        "temp_ambient_c": float,      # NEW, °C
        "temp_panel_c": float,        # NEW, °C
        "vent_ms": float,             # NEW, m/s
    },
}
```

- The weather values are "the readings of the last hour of `meteo.csv`". Like the load and solar values, they describe the hour that just ended, not the hour being decided.
- Index 0 of the two lookahead vectors is the current hour, so it equals `preu_actual` / `co2_actual`.
- The site does not say which `meteo.csv` columns feed each key. See §2.3 for our mapping.

### 2.2 Safety clamp (the site's code for hard mode)

```python
cap_max = float(obs.get("cap_max_kwh", 6000.0))      # 6,000 kWh in hard mode
p_max = float(obs.get("power_max_kw", 1500.0))       # ±1,500 kW in hard mode
soc_kwh = float(obs["soc_kwh"])
eta = 0.90 ** 0.5
eps = 1e-4

max_carrega = max(0.0, min(p_max, (cap_max - soc_kwh) / eta - eps))
max_descarrega = max(0.0, min(p_max, soc_kwh * eta - eps))
consigna_segura = float(max(-max_descarrega, min(max_carrega, consigna_desitjada)))
return consigna_segura
```

> ⚠️ The site calls this "compatible with both modes", but the fallback values are the hard-mode ones. If the standard-mode server does not send `cap_max_kwh` and `power_max_kw`, this snippet would allow 1,500 kW there and cause infractions. We don't know whether it sends them. Keep separate agents per mode, or use 2,000 / 500 as fallbacks in a standard-mode agent.

### 2.3 Simulating hard mode locally (the site's code)

```python
from starter_kit.environment import CampusBatteryEnv

df_hard = df_campus_hourly.copy()
df_hard["pv_kw"] = df_hard["pv_kw"] * 12.0

spec_hard = {
    "capacity_kwh": 6000.0,
    "power_kw": 1500.0,
    "efficiency": 0.90,
    "soc_init": 0.50,
    "soc_min": 0.0,
    "soc_max": 1.0,
    "dt_hours": 1.0,
    "forecast_horizon": 3
}

env_hard = CampusBatteryEnv(df_hard, spec=spec_hard)
```

> ⚠️ **This needs the `environment.py` the site published on 2026-11-14**, which is now the one in this repo (§5). The earlier version ignores `forecast_horizon` and silently keeps giving 24 values, so check any other copy before trusting a hard-mode run.

Checked locally with the new file: this spec gives 3 lookahead values, a starting SOC of 3,000 kWh, and `cap_max_kwh` / `power_max_kw` of 6000 / 1500.

**Weather columns.** The new `environment.py` adds the four weather keys only if the hourly DataFrame has columns with exactly those names. It reads them from the previous row, like load and solar. Our mapping from `meteo.csv` (an assumption; the site does not state it), as hourly means:

| `obs` key | `meteo.csv` column |
|---|---|
| `irradiancia_wm2` | `Atersa - IRRADIANCIA (W/m2)` |
| `temp_ambient_c` | `Atersa - TEMPERATURA AMBIENTE (SENSOR 1) (ºC)` |
| `temp_panel_c` | `Atersa - TEMPERATURA PANEL (SENSOR 1) (ºC)` |
| `vent_ms` | `Atersa - VELOCIDAD VIENTO (SENSOR 1) (m/s)` |

### 2.4 Submitting to hard mode

The form is inside the *🔥 Mode Difícil* tab (section 2 of that tab), not the usual *Lliurament* tab.

- **Fields:** team name, team PIN (the placeholder says "the same PIN as your team"), members' names (required the first time the team registers there), and the file (`agent.py` or `submission.zip`).
- **Two buttons:**
  - **🧪 Validació Ràpida Mode Difícil:** does not use quota.
  - **🔥 Trametre al Rànquing Mode Difícil:** uses one hard-mode attempt.
- **Quota:** "pròpia quota d'intents independent". The site does not give the number; we assume 10 per shift, as in standard mode.

### 2.5 Hard-mode leaderboard

- Scored on the **public test set, 1,008 hours**, the same size as the standard public set.
- Same CSV columns as the standard leaderboard: `Data, Equip, Puntuacio, Reduccio CO2 (%), Estalvi Cost (%), CO2 Evitat (kg), Estalvi Factura (€), Infraccions`.
- **Snapshot at 16:57:** two teams, one submission each.

| # | Team | Score | CO₂ reduction | Cost saving | CO₂ avoided | Bill saving | Infractions | Time on the site |
|---|---|---|---|---|---|---|---|---|
| 1 | SSA Gourmet | **21.27** | 20.17 % | 23.85 % | 12,151.5 kg | 22,655.67 € | 0 | 15:54:12 |
| 2 | Low BaterIA | 4.50 | 4.04 % | 5.56 % | 2,434.0 kg | 5,282.47 € | 0 | 15:41:44 |

What we can derive from those two rows:

- **Scores run much higher than in standard mode.** One submission already reached 21.27. The standard-mode record at the same moment was 6.44.
- **The score formula is the same.** 0.70 × 20.17 + 0.30 × 23.85 = 21.27, and 0.70 × 4.04 + 0.30 × 5.56 = 4.50.
- **The baseline already includes the ×12 solar.** Both rows give a public hard-mode baseline of about **60,200 kg CO₂** and **95,000 €** (for example 12,151.5 / 0.2017 and 22,655.67 / 0.2385). The standard public baseline is about 71,800 kg and 118,800 €. So the score measures only what the battery adds on top of the bigger solar plant.
- **Leaderboard times look like UTC, one hour behind local time.** The row stamped 15:54:12 was not there at 16:49 local and was there at 16:57.

---

## 3. What stays the same

**Stated by the site or shown by its code:**

- Efficiency 90 % round trip (√0.90 each way), so the 11.1 % arbitrage rule still holds.
- SOC bounds 0 % to 100 %, hourly steps.
- Exports earn 0 € and no CO₂ credit.
- Score = 0.70 × ΔCO₂ % + 0.30 × ΔCost %, against the no-battery baseline.
- `BatteryAgent.act(obs)` contract and file format (`agent.py` or `submission.zip`).

**Not restated for hard mode; we assume they carry over:**

- −0.20 points per infraction.
- The same 12 test weeks and the same public/private split.
- SOC set once at the start and never reset.
- 60 s timeout, 25 MB limit, the AST import scanner, shift times.

---

## 4. Another new rule: pick 2 submissions for the jury

This is not specific to hard mode. It appeared on the site today (*Bases* tab, FAQ, and the updated `ENUNCIAT.md`).

- Before the Datathon ends, each team must choose its **2 official submissions**.
- Those two are **the only ones the jury considers** on the private ranking that decides the podium.
- Where: the *Lliurament de Solucions* tab, after entering team name and PIN, in the section "🎯 Selecció Oficial per al Jurat".
- The choice can be changed as often as we like until the close.
- If a team chooses nothing, the system takes its two best submissions on the public ranking.

The site does not say whether hard mode has its own selection of 2.

---

## 5. Starter kit update (synced into this repo)

The site's participant ZIP changed on 2026-11-14. Two files differed from what we had:

| File | What changed on the site |
|---|---|
| `starter_kit/environment.py` | New `forecast_horizon` (constructor argument or spec key, default 24). `obs` now has `cap_max_kwh` and `power_max_kw`, in standard mode too (2000 / 500). `noves_dades_5min` includes the four weather keys when the DataFrame has those columns. Scoring code is unchanged. |
| `docs/ENUNCIAT.md` | One added bullet: the "2 submissions for the jury" rule from §4. |

- **Both are now copied into this repo** (2026-11-14, about 17:00), with no other edits.
- **The data did not change.** All six CSVs in `data/train/` are identical to the site's. So are `agent.py`, `check_submission.py`, `battery_spec.json`, the READMEs, `GUIA_VARIABLES.md` and the tutorial notebook.
- **Standard-mode results are unaffected.** The starter agent and our v1 and v2 agents give exactly the same results under the old and the new `environment.py` (fold A of the train set).
- `battery_spec.json` still describes the standard battery. Hard mode has no spec file; use the `spec_hard` dictionary from §2.3.
- Our own fast simulator, `src/sim.py`, now sends the two new `obs` keys as well, but it is still standard mode only (2,000 kWh, 500 kW, 24 values).

---

## 6. What this means for our agent

These are our own numbers, from the official hourly train table (5,993 hours) with solar ×12.

- **Discharge is limited by the load, not the inverter.** Net load (load − solar) is below 1,500 kW in 98 % of hours. Its median is 927 kW and half of all hours fall between 668 and 1,104 kW. A full −1,500 kW discharge would almost always spill.
- **The agent has to guess the current hour's net load.** `obs` only gives the last hour's load and solar. Any overestimate wastes stored energy; any underestimate leaves grid imports uncovered.
- **Surplus solar is new.** In standard mode there is no hour of surplus in the whole train set. With ×12 there is surplus in 9 % of hours, on 108 of 250 days. On those days the median surplus is about 1,200 kWh and the largest is about 5,900 kWh. The peak surplus in one hour is about 1,120 kW.
- **Solar is now a real share of the campus.** It goes from 2.4 % of consumption to 28 %. Train peak with ×12 is about 1,730 kW.
- **Prices and CO₂ beyond +2 h must be forecast.** Our standard-mode agents plan over the 24 known values, so they need a forecast model in front of the planner.
- **Battery timing is unchanged.** Capacity and power both tripled, so a full charge still takes about 4.2 h and a full discharge delivers about 5,690 kWh over at least 3.8 h.
- **Standard-mode agents and tools will not work as they are.** Anything that indexes 24 lookahead values or hard-codes 2,000 kWh / 500 kW must change. That includes `src/sim.py` and its oracle.

---

## 7. Confirmed on the server (probe, 2026-11-14 18:50)

We ran a probe agent on the site's free quick validation; the page shows the error text an agent raises. In hard mode the server's `obs` has:

- **Keys:** `timestamp, hour, day_of_week, soc_kwh, soc_pct, soc_percent, cap_max_kwh, power_max_kw, preu_actual, co2_actual, preus_properes_24h, co2_properes_24h, noves_dades_5min`. Nothing else.
- **Lookahead:** both lists have 3 values, and the first equals the current hour's value.
- **Readings:** `ultim_consum_kw, ultima_solar_kw, irradiancia_wm2, temp_ambient_c, temp_panel_c, vent_ms`, all finite floats.
- **Battery:** `cap_max_kwh` = 6000.0, `power_max_kw` = 1500.0, starting `soc_kwh` = 3000.0.
- **Timestamp:** a pandas Timestamp. The first hour is 2025-09-24 00:00 (a Wednesday), so the first test week is 24–30 September 2025.
- The quick validation runs 48 hours and displays the text of any exception.
- **The test has no August week; the run ends on 2026-07-31 23:00.** An agent acting only on August 2026 dates scored exactly 0 twice; the same code acting on July dates scored 0.53 (two ranked submissions, 2026-11-14 about 20:40 and 20:50).
- **The public half is the first 84 hours of each test week; the private half is the last 84.** Established with a ranked probe (see `reports/23_boundaries/report.md` §6).

**Second probe, 19:00: h2 itself on the 48 validation hours.** The server's report:

- **h2 built and ran as designed:** no build error, the load model in use, 0 steps handed to the fallback.
- **Speed:** 1.4 ms per step on average, 1.9 ms at most; construction 0.04 s. About 3 s for a full run, against the 60 s limit.
- **Libraries:** numpy 2.5.4 and pandas 3.1.1, the same as on our machine.
- **List lengths over the run:** 3 for 46 hours, then 2, then 1. The lists shrink only in the last two hours of a run, as in the published environment.
- **State of charge:** follows the published dynamics exactly (deviation 0 over 47 steps). It is not rounded, and none of h2's commands was clipped.
- **Night-time load:** equal to the previous reading in 12 of 12 night hours, so the meter freeze holds in the test data.
- **Solar ÷ irradiance:** median 1.67 over 17 sunlit hours (train: 1.64), so solar is ×12 and our mapping of the weather columns is right.
- **Wind** is 0 in every hour; the panel is warmer than the air at midday in 10 of 10 hours; no unusable reading.

## 8. Open questions for the organisers

1. Which ranking decides the prizes: hard mode, standard mode, or both? The site only calls the hard-mode ranking "independent".
2. How many hard-mode submissions per shift?
3. Does hard mode have its own "2 submissions for the jury" selection?
4. Is the penalty per infraction still −0.20 in hard mode?
5. Which `meteo.csv` columns and what aggregation feed the four weather keys? (The keys exist and are numeric: confirmed in §7.)
6. Does the standard-mode server now send `cap_max_kwh` and `power_max_kw` too?
7. Is shipping external public data (for example REE's published prices and CO₂) inside a submission allowed?
8. Does the final score use the standard-mode ranking, the hard-mode ranking, or both, and how are they combined? (Question 1 above, restated: it is still unanswered.)
9. Is the private ranking computed on the hidden 50 % of each week only, or on the whole test?
10. Of the two submissions selected for the jury, does the better one count?
11. Is "70 % private ranking + 30 % presentation" (said on Discord, 2026-11-14) official?
