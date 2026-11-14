"""
src/data.py
Shared data access for the UPC Sostenible battery challenge.

Two hourly tables, both indexed by `Fecha` (hourly timestamps):

* load_hourly_official()  -> exactly the organisers' pipeline (tutorial.ipynb cell 4 /
  website snippet): columns load_kw, pv_kw, precio_eur_per_kwh, co2_kg_per_kwh, with
  .dropna() applied. 5,993 rows. This is what the hidden test set is (presumably) built like.

* load_hourly_full()      -> every hour for which price/CO2 exist (6,768 rows = 282 train
  days). load_kw / pv_kw are NaN where the sensors have no data (all of Nov 2025, PV after
  2026-08-16, a few gaps). Also carries extra columns (REE generation mix, hourly meteo).

Both are cached as parquet under data/processed/ (ignored by git). Delete the folder to rebuild.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "train"
PROC_DIR = ROOT / "data" / "processed"

LOAD_COL = "Contador - POTENCIA ACTIVA EXPORTADA (kW)"
CORE_COLS = ["load_kw", "pv_kw", "precio_eur_per_kwh", "co2_kg_per_kwh"]

# The hidden test set = last 7 days of each calendar month, Sep 2025 .. Aug 2026.
TEST_MONTHS = [(2025, 9), (2025, 10), (2025, 11), (2025, 12), (2026, 1), (2026, 2),
               (2026, 3), (2026, 4), (2026, 5), (2026, 6), (2026, 7), (2026, 8)]


def _read(name: str, usecols=None) -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / name, parse_dates=["Fecha"], usecols=usecols).set_index("Fecha")


def _build_full() -> pd.DataFrame:
    # 1. Campus load (kW): sign-flipped exported power, clipped to [0, 5000]  (official recipe)
    cont = _read("contador.csv", usecols=["Fecha", LOAD_COL])
    load_5min = (-cont[LOAD_COL]).clip(lower=0, upper=5000)
    load_hourly = load_5min.resample("1h").mean().dropna()

    # 2. PV (kW): sum of the three inverters' AC active power, NaN -> 0, clipped at 0
    inv_cols = [f"Inversor {i} - POTENCIA ACTIVA (kW)" for i in (1, 2, 3)]
    inv = _read("inversor.csv", usecols=["Fecha"] + inv_cols)
    pv_5min = sum(inv[c].fillna(0) for c in inv_cols).clip(lower=0)
    pv_hourly = pv_5min.resample("1h").mean().dropna()

    # 3. Market signals (hourly, complete)
    pvpc = _read("precio_pvpc.csv")
    co2 = _read("co2_red.csv")

    # 4. Meteo (hourly means) -- extras, not part of the official table
    met = _read("meteo.csv")
    met = met.rename(columns={
        "Atersa - TEMPERATURA AMBIENTE (SENSOR 1) (ºC)": "temp_amb_c",
        "Atersa - TEMPERATURA PANEL (SENSOR 1) (ºC)": "temp_panel_c",
        "Atersa - VELOCIDAD VIENTO (SENSOR 1) (m/s)": "wind_ms",
        "Atersa - IRRADIANCIA (W/m2)": "irradiance_wm2",
    })
    met_hourly = met[["temp_amb_c", "temp_panel_c", "wind_ms", "irradiance_wm2"]].resample("1h").mean()

    df = pd.DataFrame({
        "load_kw": load_hourly,
        "pv_kw": pv_hourly,
        "precio_eur_per_kwh": pvpc["precio_eur_per_kwh"],
        "co2_kg_per_kwh": co2["co2_kg_per_kwh"],
        "pct_emitting": co2["porcentaje_con_emisiones"],
        "gen_emitting_mw": co2["gen_con_emisiones_mw"],
        "gen_clean_mw": co2["gen_sin_emisiones_mw"],
    })
    df = df.join(met_hourly, how="left")
    # keep only hours where the market signals exist (= the train calendar)
    df = df.dropna(subset=["precio_eur_per_kwh", "co2_kg_per_kwh"])
    df.index.name = "Fecha"
    return df


def load_hourly_full(rebuild: bool = False) -> pd.DataFrame:
    path = PROC_DIR / "hourly_full.parquet"
    if path.exists() and not rebuild:
        return pd.read_parquet(path)
    df = _build_full()
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def load_hourly_official(rebuild: bool = False) -> pd.DataFrame:
    """Organisers' table: 4 core columns, rows with any NaN dropped."""
    return load_hourly_full(rebuild)[CORE_COLS].dropna()


def load_hourly_filled(rebuild: bool = False) -> pd.DataFrame:
    """Every train hour (6,768). Missing load/pv are imputed so that all the price/CO2 data
    can be used for evaluating control policies (control barely depends on load, see reports).

    Imputation: mean by (month, weekday-type, hour) with fallbacks; for Nov 2025 (no meter data
    at all) the Oct/Dec average profile is used. Column `imputed` flags those rows.
    """
    full = load_hourly_full(rebuild)
    df = full[CORE_COLS].copy()
    df["imputed"] = df["load_kw"].isna() | df["pv_kw"].isna()
    idx = df.index
    key_m, key_h = idx.month, idx.hour
    key_we = (idx.day_of_week >= 5).astype(int)
    for col in ("load_kw", "pv_kw"):
        s = df[col]
        g1 = s.groupby([key_m, key_we, key_h]).transform("mean")
        # Nov has no data at all for load: borrow the neighbours' profile (Oct & Dec)
        prof = s.groupby([key_m, key_we, key_h]).mean()
        nov = pd.Series(
            [np.nanmean([prof.get((10, w, h), np.nan), prof.get((12, w, h), np.nan)])
             for w, h in zip(key_we, key_h)], index=idx)
        g2 = s.groupby([key_we, key_h]).transform("mean")
        df[col] = s.fillna(g1).fillna(nov.where(key_m == 11)).fillna(g2)
    return df


if __name__ == "__main__":
    full = load_hourly_full(rebuild=True)
    off = load_hourly_official()
    fil = load_hourly_filled()
    print("full    :", full.shape, full.index.min(), "->", full.index.max())
    print("official:", off.shape)
    print("filled  :", fil.shape, " imputed rows:", int(fil["imputed"].sum()),
          " NaNs left:", int(fil[CORE_COLS].isna().sum().sum()))
    print(off.describe().round(4))
