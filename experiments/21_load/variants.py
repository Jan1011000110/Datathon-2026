"""Named forecaster variants shared by eval_offline.py and closed_loop.py."""
from load_fc import COLS_V1, LoadFC, LoadQ


def make(variant, fit):
    """v2 = recommended; v1 = first version (15 features, l1 loss, no holiday index);
    nomodel = regime logic + reference profile only; v2_q0.35 = v2 with hour 0 at the 35 % quantile;
    v2_phi0.95_rho0.8 etc. = parameter variants."""
    if variant == "v1":
        return LoadFC(fit, cols=COLS_V1, mix=False, gbm_params=dict(objective="l1", feature_fraction=0.8))
    if variant == "nomodel":
        return LoadFC(fit, use_model=False, mix=False)
    if variant == "nomodel_mix":
        return LoadFC(fit, use_model=False)
    if variant.startswith("nm"):                  # no LightGBM at all (pure-Python fallback), optional _q0.2
        fc = LoadFC(fit, use_model=False)
        return LoadQ(fc, float(variant.split("_q")[1])) if "_q" in variant else fc
    kw = {}
    q = None
    for part in variant.split("_")[1:]:
        if part.startswith("q"):
            q = float(part[1:])
        elif part.startswith("phi"):
            kw["phi"] = float(part[3:])
        elif part.startswith("rho"):
            kw["rho"] = float(part[3:])
        elif part.startswith("kap"):
            kw["kappa"] = float(part[3:])
        elif part.startswith("hb"):
            kw["hol_blend"] = float(part[2:])
        elif part == "nomix":
            kw["mix"] = False
        else:
            raise ValueError(variant)
    if not variant.startswith("v2"):
        raise ValueError(variant)
    fc = LoadFC(fit, **kw)
    return LoadQ(fc, q) if q is not None else fc
