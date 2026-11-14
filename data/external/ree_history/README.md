# REE history (external data, older than the competition)

Hourly PVPC price and peninsular generation with / without CO2-equivalent emissions, downloaded from
Red Electrica de Espana's open data API. **Contains no value dated after 2025-08-31.**

- **Date range:** 2023-09-01 00:00:00 to 2025-08-31 23:00:00 (local time, hour beginning), 17542 hourly rows.
- **Downloaded:** 2026-11-14 20:11:34 (local time of the machine), by `experiments/23_signal/fetch_ree_history.py`.
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
