# Usage: .venv/Scripts/python.exe experiments/23_signal/fetch_ree_history.py [START_MONTH [END_MONTH]]
#        (months as YYYY-MM; default 2023-09 to 2025-08)
"""Download OLDER public history of the two market signals from Red Electrica's open API (no token):
PVPC price and the generation with / without CO2-equivalent emissions of the peninsular system, hourly.

THE BOUNDARY. The competition's data starts on 2025-09-01. Nothing dated after 2025-08-31 may be
requested, stored or looked at: the hidden test weeks lie inside the competition span and their values
are public. This script therefore
  * refuses any request whose range ends after 2025-08-31 23:59 (assert in `guard_request`, called on
    every request),
  * refuses to write any row dated on or after 2025-09-01 00:00 (assert in `guard_rows`, called on every
    table written, raw or processed),
  * never builds a range from "today" or from an open end: ranges are whole months up to 2025-08.
Everything goes to data/external/ree_history/ (raw JSON per month, hourly CSVs, README.md).

Endpoints (https://apidatos.ree.es/es/datos/...):
  mercados/precios-mercados-tiempo-real                      series "PVPC" (id 1001), EUR/MWh
  generacion/evolucion-estructura-generacion-emisiones-asociadas  (geo: peninsular, id 8741)
        series "Con emisiones de CO2 eq." and "Sin emisiones de CO2 eq.", MW
Recipe of the competition's own files: precio_eur_per_kwh = PVPC / 1000;
co2_kg_per_kwh = 0.37 * con / (con + sin). Timestamps: local time, hour beginning.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "external" / "ree_history"
LAST_ALLOWED = pd.Timestamp("2025-08-31 23:59")          # nothing after this instant, ever
LAST_ALLOWED_ROW = pd.Timestamp("2025-08-31 23:59:59")   # REE answers in quarter-hours for part of 2025: the last
                                                         # row of August is 23:45. Still nothing on or after 2025-09-01 00:00.
BASE = "https://apidatos.ree.es/es/datos/"
EP = {
    "pvpc": "mercados/precios-mercados-tiempo-real?start_date={a}&end_date={b}&time_trunc=hour",
    "co2": ("generacion/evolucion-estructura-generacion-emisiones-asociadas?start_date={a}&end_date={b}"
            "&time_trunc=hour&geo_trunc=electric_system&geo_limit=peninsular&geo_ids=8741"),
}
HEADERS = {"Accept": "application/json", "User-Agent": "python-urllib"}      # generic client, no identity


def guard_request(start: pd.Timestamp, end: pd.Timestamp):
    assert start <= end, (start, end)
    assert end <= LAST_ALLOWED, f"REFUSED: request range ends {end}, after the boundary {LAST_ALLOWED}"
    assert start <= LAST_ALLOWED, f"REFUSED: request range starts {start}, after the boundary"


def guard_rows(ts):
    ts = pd.DatetimeIndex(ts)
    assert len(ts) == 0 or ts.max() <= LAST_ALLOWED_ROW, f"REFUSED: a row is dated {ts.max()}, after the boundary"


def fetch_month(kind: str, month: pd.Period, force=False) -> dict:
    start = month.start_time.normalize()
    end = month.end_time.normalize() + pd.Timedelta(hours=23, minutes=59)
    guard_request(start, end)
    raw = OUT / "raw" / f"{kind}_{month}.json"
    if raw.exists() and not force:
        return json.loads(raw.read_text(encoding="utf-8"))
    url = BASE + EP[kind].format(a=start.strftime("%Y-%m-%dT%H:%M"), b=end.strftime("%Y-%m-%dT%H:%M"))
    assert "2025-09" not in url and "2025-1" not in url and "2026" not in url, "REFUSED: url mentions a forbidden month"
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                js = json.loads(r.read().decode("utf-8"))
            break
        except Exception as e:                                  # retry on transient errors
            if attempt == 3:
                raise
            time.sleep(2.0 * (attempt + 1))
    # check the payload itself before anything is written
    for inc in js.get("included", []):
        vals = inc["attributes"]["values"]
        if vals:
            guard_rows([pd.Timestamp(v["datetime"][:19]) for v in (vals[0], vals[-1])])
            guard_rows([max(pd.Timestamp(v["datetime"][:19]) for v in vals)])
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps(js), encoding="utf-8")
    time.sleep(0.4)
    return js


def series(js: dict, title: str) -> pd.DataFrame:
    for inc in js["included"]:
        if inc["attributes"]["title"] == title or inc["type"] == title:
            v = inc["attributes"]["values"]
            return pd.DataFrame({"datetime_local": [pd.Timestamp(x["datetime"][:19]) for x in v],
                                 "utc_offset": [x["datetime"][-6:] for x in v],
                                 "value": [float(x["value"]) for x in v]})
    raise KeyError(f"series {title!r} not in the answer: {[i['attributes']['title'] for i in js['included']]}")


def build(months):
    pv, co = [], []
    for m in months:
        a = series(fetch_month("pvpc", m), "PVPC").rename(columns={"value": "pvpc_eur_per_mwh"})
        js = fetch_month("co2", m)
        b = series(js, "Con emisiones de CO2 eq.").rename(columns={"value": "gen_con_emisiones_mw"})
        c = series(js, "Sin emisiones de CO2 eq.").rename(columns={"value": "gen_sin_emisiones_mw"})
        b = b.merge(c, on=["datetime_local", "utc_offset"], how="outer")
        pv.append(a); co.append(b)
        print(f"{m}: pvpc {len(a)} rows, co2 {len(b)} rows", flush=True)
    pv = pd.concat(pv, ignore_index=True)
    co = pd.concat(co, ignore_index=True)
    for t in (pv, co):
        guard_rows(t.datetime_local)
    pv.to_csv(OUT / "pvpc_hourly_raw.csv", index=False)
    co.to_csv(OUT / "co2_hourly_raw.csv", index=False)
    # hourly table in the convention of data/train: naive local time, hour beginning, one row per clock hour.
    # The repeated hour of the autumn clock change is averaged; the skipped hour of the spring change is absent.
    # Where REE answers in quarter-hours, the hour's value is the mean of its quarters.
    p = pv.groupby(pv.datetime_local.dt.floor("h"))["pvpc_eur_per_mwh"].mean()
    g = co.groupby(co.datetime_local.dt.floor("h"))[["gen_con_emisiones_mw", "gen_sin_emisiones_mw"]].mean()
    p.index.name = g.index.name = "datetime_local"
    t = pd.concat([p, g], axis=1).sort_index()
    t["precio_eur_per_kwh"] = t.pvpc_eur_per_mwh / 1000.0
    t["co2_kg_per_kwh"] = 0.37 * t.gen_con_emisiones_mw / (t.gen_con_emisiones_mw + t.gen_sin_emisiones_mw)
    t.index.name = "Fecha"
    guard_rows(t.index)
    t.to_csv(OUT / "ree_history_hourly.csv")
    return t


def load_history() -> pd.DataFrame:
    """The stored hourly history (never touches the network). Columns as data/train: precio_eur_per_kwh,
    co2_kg_per_kwh. Asserts the boundary once more on read."""
    t = pd.read_csv(OUT / "ree_history_hourly.csv", index_col="Fecha", parse_dates=True)
    guard_rows(t.index)
    return t


if __name__ == "__main__":
    a = pd.Period(sys.argv[1] if len(sys.argv) > 1 else "2023-09", freq="M")
    b = pd.Period(sys.argv[2] if len(sys.argv) > 2 else "2025-08", freq="M")
    assert b <= pd.Period("2025-08", freq="M"), "REFUSED: the last month allowed is 2025-08"
    OUT.mkdir(parents=True, exist_ok=True)
    have = sorted({pd.Period(p.stem.split("_")[1], freq="M") for p in (OUT / "raw").glob("pvpc_*.json")}) if (OUT / "raw").exists() else []
    months = sorted(set(pd.period_range(a, b, freq="M")) | set(have))
    t0 = time.strftime("%Y-%m-%d %H:%M:%S")
    t = build(months)
    full = pd.date_range(t.index.min(), t.index.max(), freq="h")
    print(f"rows {len(t)}, from {t.index.min()} to {t.index.max()}, clock hours missing {len(full.difference(t.index))}, "
          f"NaN price {int(t.precio_eur_per_kwh.isna().sum())}, NaN co2 {int(t.co2_kg_per_kwh.isna().sum())}")
    (OUT / "README.md").write_text(f"""# REE history (external data, older than the competition)

Hourly PVPC price and peninsular generation with / without CO2-equivalent emissions, downloaded from
Red Electrica de Espana's open data API. **Contains no value dated after 2025-08-31.**

- **Date range:** {t.index.min()} to {t.index.max()} (local time, hour beginning), {len(t)} hourly rows.
- **Downloaded:** {t0} (local time of the machine), by `experiments/23_signal/fetch_ree_history.py`.
- **Why the range stops there:** the competition's data runs from 2025-09-01 to 2026-09-01 and its hidden test
  weeks lie inside that span. The script refuses any request ending after 2025-08-31 23:59 and any row dated
  after 2025-08-31 23:00 (assertions on every request and on every table written).
- **Endpoints** (`https://apidatos.ree.es/es/datos/`, no token, one calendar month per request):
  - `mercados/precios-mercados-tiempo-real?start_date=YYYY-MM-01T00:00&end_date=YYYY-MM-DDT23:59&time_trunc=hour`,
    series "PVPC" (id 1001), EUR/MWh;
  - `generacion/evolucion-estructura-generacion-emisiones-asociadas?start_date=...&end_date=...&time_trunc=hour&geo_trunc=electric_system&geo_limit=peninsular&geo_ids=8741`,
    series "Con emisiones de CO2 eq." and "Sin emisiones de CO2 eq.", MW.
- **Files**
  - `raw/pvpc_YYYY-MM.json`, `raw/co2_YYYY-MM.json`: the API answers, unmodified.
  - `pvpc_hourly_raw.csv`, `co2_hourly_raw.csv`: the series as returned, with the UTC offset of every hour.
  - `ree_history_hourly.csv`: one row per clock hour, same convention as `data/train/`:
    `precio_eur_per_kwh = PVPC / 1000`, `co2_kg_per_kwh = 0.37 * con / (con + sin)`. The repeated hour of the
    autumn clock change is averaged; the skipped hour of the spring change is absent.
- **Use:** only to fit forecasting models of price and CO2 (`experiments/23_signal/signal_fc3.py`).
""", encoding="utf-8")
    print("written to", OUT)
