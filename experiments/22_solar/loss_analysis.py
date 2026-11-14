# Usage: .venv/Scripts/python.exe experiments/22_solar/loss_analysis.py
"""Where the remaining solar error costs score (perfect load and price/CO2). Uses traces/ and offline_final.csv."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).parent
rd = lambda i, f: pd.read_csv(HERE / "traces" / f"{i}_{f}.csv", index_col=0, parse_dates=True)   # noqa: E731
o = pd.read_csv(HERE / "offline_final.csv", parse_dates=["ts"])
h0 = o[o.h == 0].set_index("ts")
rows = []
for f in "ABC":
    perf, fc, base = rd(0, f), rd(1, f), rd(3, f)
    for name, tr in (("SolarFC", fc), ("profile", base)):
        pts = 100 * tr.w * (tr.grid_kw - perf.grid_kw)
        morning = pts[(pts.index.hour >= 7) & (pts.index.hour <= 13)].sum()
        rows.append({"fold": f, "fc": name, "loss_pts": pts.sum(), "loss 07-13h": morning, "spill_MWh": tr.spill_kw.sum() / 1000,
                     "spill 07-13h MWh": tr.spill_kw[(tr.index.hour >= 7) & (tr.index.hour <= 13)].sum() / 1000,
                     "value of spilled energy, pts": 100 * (tr.w * tr.spill_kw).sum()})
    if f in "BC":
        d = pd.DataFrame({"loss": 100 * fc.w * (fc.grid_kw - perf.grid_kw), "spill": fc.spill_kw}).join(h0[h0.fold == f][["y", "fc", "q10", "q90"]], how="inner")
        d["err"] = d.fc - d.y
        day = d.groupby(d.index.normalize()).agg(loss=("loss", "sum"), spill=("spill", "sum"), abs_err=("err", lambda e: e.abs().sum()),
                                                under=("err", lambda e: (-e.clip(upper=0)).sum()), over=("err", lambda e: e.clip(lower=0).sum()),
                                                band=("q90", "sum"), band_lo=("q10", "sum"), y=("y", "sum"))
        day["width"] = (day.band - day.band_lo) / day.band.clip(lower=1)
        print(f"fold {f}: {len(day)} days; corr(daily loss, sum|err h0|) = {day.loss.corr(day.abs_err):.2f}; "
              f"with under-forecast kWh = {day.loss.corr(day.under):.2f}; with over-forecast kWh = {day.loss.corr(day.over):.2f}; "
              f"corr(daily spill, under-forecast) = {day.spill.corr(day.under):.2f}")
        top = day.loss.sort_values(ascending=False)
        n10 = max(1, len(day) // 5)
        print(f"   worst 20% of days carry {100 * top.iloc[:n10].sum() / day.loss.sum():.0f}% of the loss; "
              f"loss on the third of days with the widest band: {100 * day.loss[day.width >= day.width.quantile(2 / 3)].sum() / day.loss.sum():.0f}%")
print(pd.DataFrame(rows).groupby("fc").mean(numeric_only=True).round(3).to_string())
