"""How far can the RHO of the full test be from the RHO of its public half (0.6048)?

Emulated on the train folds: split each fold's hours in two halves in several ways and compare
the RHO of one half with the RHO of the whole fold. No simulation, just baseline sums."""
from lib import *

folds = get_folds("filled")
rng = np.random.default_rng(0)
rows = []
for f in ("A", "B", "C"):
    df = folds[f]
    base = np.maximum(df.load_kw - df.pv_kw, 0.0).to_numpy()
    c, e = base * df.co2_kg_per_kwh.to_numpy(), base * df.precio_eur_per_kwh.to_numpy()
    full = c.sum() / e.sum()
    blk = block_ids(df.index)
    pos = np.concatenate([np.arange((blk == b).sum()) for b in np.unique(blk)])     # hour within the week
    masks = {"first 84 h of each week": pos < 84, "last 84 h of each week": pos >= 84,
             "odd days": (pos // 24) % 2 == 0, "even days": (pos // 24) % 2 == 1,
             "odd hours": pos % 2 == 0}
    for name, m in masks.items():
        rows.append({"fold": f, "half": name, "rho_half": c[m].sum() / e[m].sum(), "rho_full": full})
    # random halves: whole days, and single hours
    for name, unit in (("random days (500 draws)", pos // 24 + 7 * blk), ("random hours (500 draws)", np.arange(len(df)))):
        d = []
        u = np.unique(unit)
        for _ in range(500):
            pick = rng.permutation(u)[:len(u) // 2]
            m = np.isin(unit, pick)
            d.append(c[m].sum() / e[m].sum() - full)
        rows.append({"fold": f, "half": name, "rho_half": np.nan, "rho_full": full, "sd(half-full)": float(np.std(d)), "max|half-full|": float(np.max(np.abs(d)))})
t = pd.DataFrame(rows)
t["half-full"] = t["rho_half"] - t["rho_full"]
pd.set_option("display.width", 200)
print(t.round(4).to_string(index=False))
print("\nlargest |half - full| among the structured splits: %.4f" % t["half-full"].abs().max())
