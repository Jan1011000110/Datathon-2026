"""
experiments/30_boundaries/rules3.py
"Each week is its own run", as a measurement variant (follow-up of 20:42):

    WeekCut(fc)     signal forecaster that returns nothing beyond the last hour of the current 7-day block, the way
                    combined3.EndCut does for the end of the run. The plan then stops at the week's end and energy
                    left there is worth nothing. This is the well-defined form of "signal beyond the week end x 0":
                    the multiplicative form is not exactly zero in h3's planner (the signal is floored at 1e-9, the
                    error paths add their positive errors to it and the last-error correction is added on top).
"""
import numpy as np

from rules import Calendar


class WeekCut:
    def __init__(self, fc):
        self.fc, self.cal, self._seen = fc, Calendar(), None

    def predict(self, obs, hist, H):
        w = np.asarray(self.fc.predict(obs, hist, H), float)
        i = len(hist) - 1
        if self._seen != i:
            self._r, _ = self.cal.step(obs["timestamp"])
            self._seen = i
        return w[:self._r] if self._r >= 1 else w
