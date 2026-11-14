"""Unrounded public score from a leaderboard row.
Usage:  .venv/Scripts/python.exe experiments/05_leaderboard/unrounded.py <CO2 avoided kg> <EUR saved>
The public baseline (CO2 71,797-71,832 kg; cost 118,733-118,750 EUR) was reconstructed by intersecting
the rounding intervals of every row on the board (see reports/README.md section 3)."""
import sys
CO2_LO, CO2_HI, EUR_LO, EUR_HI = 71797.0, 71832.0, 118733.0, 118750.0
V1 = (4282.5, 8934.63)                                      # our v1 row (team Fa)

def score(kg, eur):
    lo = 100 * (0.7 * kg / CO2_HI + 0.3 * eur / EUR_HI); hi = 100 * (0.7 * kg / CO2_LO + 0.3 * eur / EUR_LO)
    return lo, hi, (lo + hi) / 2

if __name__ == "__main__":
    kg, eur = float(sys.argv[1]), float(sys.argv[2])
    lo, hi, mid = score(kg, eur); _, _, v1 = score(*V1)
    print(f"unrounded score {mid:.4f}  (between {lo:.4f} and {hi:.4f});  v1 = {v1:.4f};  difference vs v1 = {mid - v1:+.4f}")
    print(f"vs v1: {kg - V1[0]:+.1f} kg CO2, {eur - V1[1]:+.2f} EUR")
