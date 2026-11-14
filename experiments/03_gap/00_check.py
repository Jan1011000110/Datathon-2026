"""Sanity: GapAgent defaults == agents/v1 (same actions), fold facts, baseline gaps."""
import time
from lib import *

folds = get_folds("filled")
for f in FOLDS:
    df = folds[f]
    t0 = time.perf_counter(); a = run(v1.BatteryAgent(), df); t1 = time.perf_counter()
    b = run(GapAgent(), df)
    o = oracle(df, return_trace=True)
    same = np.allclose(a["trace"]["p_kw"], b["trace"]["p_kw"], atol=1e-12)
    net = (df.load_kw - df.pv_kw)
    print("fold %-3s n=%d blocks=%d weeks=%d rho=%.4f  v1 %.4f  gap-agent %.4f (same actions: %s)  oracle %.4f  gap %.4f  infr %d  %.2fs" % (
        f, len(df), block_ids(df.index).max() + 1, week_ids(df).max() + 1, true_rho(df), a["score"], b["score"], same,
        o["score"], o["score"] - a["score"], a["infractions"], t1 - t0))
    print("     sum(contrib) = %.6f ; net<526: %d h, net<500: %d h, net<50: %d h, imputed: %d h" % (
        a["contrib"].sum(), int((net < 526.3).sum()), int((net < 500).sum()),
        int((net < 50).sum()), int(df["imputed"].sum()) if "imputed" in df else -1))
