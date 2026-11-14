# Notes from the Challenge Website

What the official platform says beyond `docs/`, plus the values we treat as true wherever the sources disagree.

- **Source:** <https://repteupc.datathon.cat/>. It is a Streamlit app with 6 tabs: *Bases i Enunciat Tècnic*, *Dades del Campus i Starter Kit*, *Lliurament de Solucions*, *Classificació General (Leaderboard)*, *🔥 Mode Difícil* (added on the afternoon of 2026-11-14) and *Preguntes Freqüents*.
- **Checked on:** 2026-11-14 (Saturday of the Datathon), at midday. Re-checked at 16:57.
- **Organisers:** AED × UPC Sostenible.

> **Update, 2026-11-14 16:57:** the site added the *🔥 Mode Difícil* tab and a rule about choosing 2 submissions for the jury (§2.1). Hard mode is described in [`NEW_RULES.md`](NEW_RULES.md). The rest of this page describes standard mode; its tabs had no other changes since midday.

The site's "Paquet per als Participants" ZIP (13 MB) matches this repo exactly: `README.md`, `docs/`, `starter_kit/` and the 6 CSVs in `data/train/`. Only the line endings differ. So everything below comes from the web page itself, not from any extra file.

The ZIP changed once during the day: `starter_kit/environment.py` and `docs/ENUNCIAT.md` were updated on the site and synced here at about 17:00 (details in `NEW_RULES.md` §5). The data never changed.

---

## 1. Conflicts and the value we take as true

**Rule:** when sources disagree, the starter kit code wins (`starter_kit/environment.py`, `check_submission.py`, `battery_spec.json`). It is the only executable definition of the scoring we have. If the code says nothing, we go by the website and the actual data.

> ⚠️ The server runs its own hidden evaluator. `environment.py` is our best proxy for it, not a guarantee. If the leaderboard ever disagrees with a local run, check this section again.

| # | Topic | `docs/` says | Website says | Code / data says | **We take as true** |
|---|---|---|---|---|---|
| 1 | Penalty per infraction | −1.00 pt (`ENUNCIAT.md`, LaTeX) | −0.20 pt (formula box and "Consell d'Enginyeria"), but the FAQ title says −1.00 | `penalties = 0.002 * infraccions`, then score ×100, so **−0.20 pt** (`environment.py:184`) | **−0.20 pt per infraction** |
| 2 | Price of energy exported to the grid | 0 € (not paid) | Section 2: compensated at **0.04 €/kWh**. FAQ: `export_price = 0.0` | `export_price: float = 0.0` (`environment.py:14`), and no caller overrides it | **0.00 €/kWh**. Exports also give no CO₂ credit. |
| 3 | Forbidden imports | Only specific calls are banned (`os.system`, `os.popen`, `sys.exit`, `shutil.rmtree`…) | FAQ: whole libraries are banned (`os`, `sys`, `inspect`, …) | **Whole modules** are banned (`check_submission.py:28`). See §1.1. | **Never import any module in the list below**, not even `os` or `sys` |
| 4 | Submission quota | `ENUNCIAT.md`: 10 per shift, 30 total. LaTeX: "10 valid attempts per team per day" | 10 per shift, 3 shifts (30 total) | — | **10 per shift, 30 total** (the LaTeX wording is outdated) |
| 5 | I/O signals file name | `entrades_salides.csv` | `entradas_salidas.csv` | The file is `data/train/entradas_salidas.csv` | **`entradas_salidas.csv`** |
| 6 | `contador.csv` sampling rate | 5 min | 15 min (data tab table) | Timestamps are 5 min apart (00:05, 00:10, …) | **5 min** |
| 7 | Keys in `obs` | No `soc_percent` | Lists `soc_percent` (0–100) | `obs` includes `soc_percent` (`environment.py:82`) | `soc_percent` **exists** (it equals `soc_pct × 100`). Since the 2026-11-14 update, the local `obs` also has `cap_max_kwh` (2000.0) and `power_max_kw` (500.0); whether the standard-mode server sends them is unconfirmed. |
| 8 | When the SOC starts at 50% | "At the start of each evaluation period" (ambiguous) | Only at hour 0 of the whole simulation. **No daily or weekly reset.** | `reset()` is called once per run | **Once.** The SOC carries over between days and across the gaps between test weeks. |
| 9 | What counts as an infraction | "Each setpoint that violates the limits" | "Each hour with physical clipping or a limit violation" | Counts **each event**. One hour can add up to 2: power clip (>±500 kW) plus capacity clip, or an SOC bound violation. Any overshoot counts, even a tiny one. | Count per event. Use the safety clamp with `eps` (§2.4). |
| 10 | PIN | "Minimum 4 characters" | "Minimum 4 characters" (form) / "4-character key" (rules) | — | **At least 4 characters** |

### 1.1 Forbidden names according to the local validator (`check_submission.py`)

- **Modules** (any `import`): `os`, `sys`, `shutil`, `subprocess`, `socket`, `urllib`, `requests`, `http`, `ftplib`, `paramiko`, `telnetlib`, `pty`, `posix`, `ctypes`, `importlib`, `inspect`, `gc`, `builtins`, `fcntl`, `signal`, `threading`, `multiprocessing`
- **Calls** (as attributes): `system`, `popen`, `spawn*`, `rmtree`, `get_objects`, `currentframe`, `stack`
- **Built-ins:** `eval`, `exec`, `__import__`, `compile`, `breakpoint`, `globals`, `locals`, `vars`
- **Attributes:** `f_back`, `f_locals`, `f_globals`, `__subclasses__`, `__builtins__`, `__globals__`, `__code__`

To load files, use `pathlib.Path(__file__).parent`. Never use `os.path`.

---

## 2. Information found only on the website

### 2.1 Evaluation, leaderboard and jury

- **The jury has the final say.** The website's wording: "El Leaderboard és una guia quantitativa fonamental, però no ho decideix tot. Per proclamar els guanyadors, el jurat valorarà de forma decisiva la presentació final, la claredat del raonament tècnic, l'anàlisi de les dades i la justificació de l'estratègia." So the final presentation, the clarity of the technical reasoning, the data analysis and the justification of the strategy count heavily, not just the score.
- **Public vs private leaderboard:**
  - The live leaderboard uses the **public test set: 50% of each test week, 1,008 h**.
  - The other 50% (private) stays hidden until the final verdict, to stop teams overfitting to the public ranking.
  - The full test set is 12 weeks × 7 days = 2,016 h.
- **Tie-break order:**
  1. Higher Score.
  2. Fewer physical infractions.
  3. Earlier submission time.
- **Pick 2 submissions for the jury** (rule added on 2026-11-14):
  - Before the close, each team chooses its 2 official submissions in the *Lliurament* tab, section "🎯 Selecció Oficial per al Jurat" (shown after entering team name and PIN).
  - Only those two are considered on the private ranking that decides the podium.
  - The choice can be changed until the close. With no choice, the system takes the team's two best on the public ranking.
- **Code audit:** every submission is archived automatically for review by the UPC jury.
- **Leaderboard columns** (from the CSV backup you can download on the Leaderboard tab): `Data, Equip, Puntuacio, Reduccio CO2 (%), Estalvi Cost (%), CO2 Evitat (kg), Estalvi Factura (€), Infraccions`.
- **Snapshot on 2026-11-14 ~12:50:** 1 team ("Prueba 1", which looks like an organiser test), 2 submissions, best score 6.43 pts, CO₂ −5.97 %, cost −7.5 %, 4,289.4 kg CO₂ avoided, 8,907.87 € saved, 0 infractions.
  - *Derived by us, approximate:* on the public 1,008 h, the no-battery baseline is ≈ 4,289.4 / 0.0597 ≈ **71,800 kg CO₂** and ≈ 8,907.87 / 0.075 ≈ **118,800 €**.
- **Score simulator:** the site has an interactive slider widget for the formula. It holds no extra information. Example: 5 % CO₂ + 6 % cost + 0 infractions = 5.30 pts.

### 2.2 Battery physics (from the FAQ)

- **Efficiency example:** charging 100 kWh from the grid stores 94.9 kWh, and discharging that delivers 90 kWh to the campus.
- **Rule of thumb for arbitrage:** a charge→discharge cycle only pays off if the expensive or dirty hour is at least **11.1 % worse** (1/0.90) than the cheap or clean hour you charged in.
- **No selling to the grid:** the battery serves only Campus Nord's own consumption. If you discharge more than the campus's net demand, the excess goes to the grid at 0 € and earns no CO₂ saving.
- **No degradation** (`degradation = false`, also in `battery_spec.json`). There is no cycling penalty.
- **Free surplus solar:** when solar output exceeds demand, charging from that surplus costs nothing. The website explicitly encourages using it.

### 2.3 Data and the `obs` dictionary

- **How the test set is cut:** the test set is whole 7-day blocks (the **last week of each month**). Train holds the first ~3 weeks of each month.
  - **Tip:** detect jumps with `(obs["timestamp"] - self.prev_ts) > 2 hours`. Reset short rolling windows and lags when a jump happens.
- **What `obs` gives you about the future:**
  - It **does** include the next 24 h of PVPC prices and CO₂ intensities (as if day-ahead market prices were known).
  - It does **not** include future campus load or future solar output. You must forecast those, for example from the hour, weekday, month and `noves_dades_5min`.
- **Agent lifetime:** the evaluator creates **one** `BatteryAgent` instance and calls `act(obs)` once per hour, in order. `self.historial` therefore persists for the whole run.
- **Return type:** returning `np.float64` is fine; it gets converted to `float`.
- **The meter sign:** the sensor is literally named `Contador - POTENCIA ACTIVA EXPORTADA (kW)`, and consumption appears as negative values (e.g. −1,100 kW means 1,100 kW consumed). Load = `(-col).clip(lower=0, upper=5000)`.
- **Digital inputs:** `entradas_salidas.csv` holds the digital input channels **DI_00 to DI_07**, binary on/off states of compressors, pumps and ventilation.
- **Variable dictionary:** the data tab has an interactive table of 20 variables. Its content matches `GUIA_VARIABLES.md`, with a few extra ranges: power factor nominal ~0.95–1.0, inverter DC bus voltage ~600–800 V.
- **Data pipeline snippet published on the website**, for loading everything hourly:

```python
import pandas as pd

df_contador = pd.read_csv("data/train/contador.csv", parse_dates=["Fecha"]).set_index("Fecha")
demanda_raw = (-df_contador["Contador - POTENCIA ACTIVA EXPORTADA (kW)"]).clip(lower=0, upper=5000)
load_hourly = demanda_raw.resample("1h").mean().dropna()

df_inversor = pd.read_csv("data/train/inversor.csv", parse_dates=["Fecha"]).set_index("Fecha")
inv_cols = [c for c in df_inversor.columns if "POTENCIA ACTIVA" in c and "FASE" not in c]
pv_raw = df_inversor[inv_cols].sum(axis=1).clip(lower=0)
pv_hourly = pv_raw.resample("1h").mean().dropna()

df_pvpc = pd.read_csv("data/train/precio_pvpc.csv", parse_dates=["Fecha"]).set_index("Fecha")
df_co2 = pd.read_csv("data/train/co2_red.csv", parse_dates=["Fecha"]).set_index("Fecha")

df_campus_hourly = pd.DataFrame({
    "load_kw": load_hourly,
    "pv_kw": pv_hourly,
    "precio_eur_per_kwh": df_pvpc["precio_eur_per_kwh"],
    "co2_kg_per_kwh": df_co2["co2_kg_per_kwh"],
}).dropna()
```

  Hourly mean power in kW equals the energy in kWh for that hour.

### 2.4 Safety clamp (the website's recommended code)

```python
eta = 0.9 ** 0.5
eps = 1e-4
espai_disponible_kwh = max(0.0, 2000.0 - obs["soc_kwh"])
energia_disponible_kwh = max(0.0, obs["soc_kwh"])

max_carrega = max(0.0, min(500.0, espai_disponible_kwh / eta - eps))
max_descarrega = max(0.0, min(500.0, energia_disponible_kwh * eta - eps))

consigna_segura = float(np.clip(consigna_desitjada, -max_descarrega, max_carrega))
```

Note: `starter_kit/agent.py` uses `eta = 0.9487` and no `eps`. The version above, with `eps`, is the safer one.

### 2.5 Submission process (*Lliurament* tab)

- **Form fields:**
  - Team name.
  - Team PIN (at least 4 characters).
  - Members' full names (required on the first submission).
  - Solution file: `.py` or `.zip`. The upload widget says "200MB per file", but **the rule is 25 MB**.
- **Two buttons:**
  - **🧪 Validació Ràpida:** free and **unlimited**. It runs a **48 h** test to check imports, format and `.pkl` loading. It does not use quota.
  - **🚀 Trametre al Leaderboard:** uses 1 attempt from the current shift.
- **Shifts:**
  1. Saturday until 21:00.
  2. Saturday 21:00 to Sunday 08:00.
  3. Sunday 08:00 until the challenge closes.

  10 submissions per shift.
- **Other limits:** 1 min timeout in an isolated subprocess. Failed runs (syntax errors, missing class) do not use quota (per `ENUNCIAT.md`).
- **ZIP layout:** put `agent.py` and the `.pkl` files **at the root of the ZIP**, with no enclosing folder. The last cell of `starter_kit/tutorial.ipynb` builds a correct `submission.zip`.
- **Model files:** load them with a path relative to `Path(__file__).parent`. Absolute paths (`C:\Users\...`, `/home/...`) will fail on the server.
- **Local check:** `python starter_kit/check_submission.py agent.py` (a `submission.zip` path also works). It runs the AST scan plus 24 test steps.

### 2.6 Quick-start guide from the website

1. Download the participant ZIP from the *Dades del Campus* tab.
2. Open `starter_kit/tutorial.ipynb`.
3. On the *Lliurament* tab, run **Validació Ràpida** with the template `agent.py`.

## Site re-read, Sat 14 Nov 2026, 21:41

Every tab was read again and compared with the earlier dumps.

- **No rule text changed** in the rules, hard-mode or FAQ tabs.
- **Submission tab:** the line on the quota now reads "màxim 20 lliuraments per equip en aquest torn" (it said 10 at 21:01). The rules tab still says 10 per shift. Not tested.
- **FAQ, unchanged, the two sentences that bear on the final:** the ranking seen during the Datathon "correspon al 50% públic de cada setmana de test", the other 50 % stays hidden until the verdict; and the two selected submissions "seran les úniques considerades pel jurat per al rànquing privat secret que decidirà el podi final".
- **Rules tab, unchanged:** "el jurat valorarà de forma decisiva la presentació final, la claredat del raonament tècnic, l'anàlisi de les dades i la justificació".
- Nothing on the site says which mode's private ranking counts.
