# Usage: .venv/Scripts/python.exe experiments/21_load/cal_check.py
"""Checks of CalLoadFC: (1) no calendar = LoadFC bit for bit (fresh LoadFC against CalLoadFC, fold B run);
(2) to_state / from_state round trip with the calendar; (3) time per call with and without the calendar."""
import time

import numpy as np

from cal_lib import CALENDAR, CalLoadFC, LoadFC, cb, fit_table_for, fold_cut, record_load

f = "B"
df = fold_cut(f)
fit = fit_table_for(f)
a = LoadFC(fit)
b = CalLoadFC(fit)
c = CalLoadFC(fit, calendar=CALENDAR, cal_blend=1.0)
Fa, Qa = record_load(a, df)
Fb, Qb = record_load(b, df)
print("LoadFC vs CalLoadFC(no calendar), fold B without August:", Fa.shape, Qa.shape,
      "point forecasts identical:", bool(np.array_equal(Fa, Fb)), "| quantiles identical:", bool(np.array_equal(Qa, Qb)))
print("model strings identical (LoadFC, CalLoadFC no calendar, CalLoadFC with calendar):",
      a.model.model_to_string() == b.model.model_to_string() == c.model.model_to_string())
Fc, Qc = record_load(c, df)
c2 = CalLoadFC.from_state(c.to_state())
Fd, Qd = record_load(c2, df)
print("from_state round trip with calendar: point forecasts identical", bool(np.array_equal(Fc, Fd)),
      "| quantiles max abs diff (float32 residual tables in to_state, as LoadFC)", float(np.abs(Qc.astype(float) - Qd.astype(float)).max()),
      "| rows that differ from LoadFC:", int((np.abs(Fc - Fa).max(axis=1) > 0).sum()), "of", len(df))
for name, fc in (("LoadFC", a), ("CalLoadFC no calendar", b), ("CalLoadFC calendar", c)):
    out, sec = cb.record(df, {"load_fc": fc}, quantiles=True)
    print(f"{name:24s} predict_load(36) {1e3 * sec['load_fc'] / len(df):.3f} ms/call, 45 quantiles {1e3 * sec['load_fc_q'] / len(df):.3f} ms/call")
