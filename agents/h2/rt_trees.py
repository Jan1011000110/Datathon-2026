"""
agents/h2/rt_trees.py  --  LightGBM regression trees evaluated with numpy (no LightGBM at run time).

build.py reads the trees out of the fitted model's text and stores them as arrays; this class walks them
with the same decision rule as LightGBM (Tree::NumericalDecision) and adds the leaf values in tree order,
so it returns what Booster.predict returns (checked on every training row at build time and again on a
few stored rows each time the state is loaded). Why not load the model text with LightGBM on the server:
its version there is unknown, the library may be missing, and a model text it cannot parse aborts the
whole Python process instead of raising.
"""
import numpy as np

ZERO = 1e-35                 # LightGBM's kZeroThreshold


class TreeModel:
    def __init__(self, feat, thr, dtype, left, right, leaf, n_int):
        """All arrays are padded to the same number of nodes per tree: feat / thr / dtype / left / right
        (T, M) describe the internal nodes, leaf (T, L) the leaf values, n_int (T,) the number of internal
        nodes of each tree. A child >= 0 is an internal node, a child < 0 is leaf number ~child."""
        self.feat = np.asarray(feat, np.int64)
        self.thr = np.asarray(thr, float)
        dt = np.asarray(dtype, np.int64)
        self.left = np.asarray(left, np.int64)
        self.right = np.asarray(right, np.int64)
        self.leaf = np.asarray(leaf, float)
        n_int = np.asarray(n_int, np.int64)
        T, M = self.feat.shape
        if not (self.thr.shape == dt.shape == self.left.shape == self.right.shape == (T, M)
                and self.leaf.shape[0] == T and n_int.shape == (T,) and T > 0):
            raise ValueError("tree arrays do not fit together")
        if (dt & 1).any():
            raise ValueError("categorical splits are not supported")
        if np.isnan(self.thr).any() or not np.isfinite(self.leaf).all():     # a threshold may be +inf (split on "is missing")
            raise ValueError("non-finite tree values")
        if (self.left >= M).any() or (self.right >= M).any() or (self.left < -self.leaf.shape[1]).any() \
                or (self.right < -self.leaf.shape[1]).any() or (self.feat < 0).any():
            raise ValueError("tree pointers out of range")
        self.dleft = (dt & 2) > 0                    # where a missing value goes
        self.mtype = (dt >> 2) & 3                   # 0 = none, 1 = zero, 2 = NaN
        self.start = np.where(n_int > 0, 0, -1)      # a tree without splits is its leaf 0
        self.rows = np.arange(T)
        self.n_features = int(self.feat.max()) + 1
        self.max_steps = M + 1

    def _one(self, x):
        rows = self.rows
        node = self.start.copy()
        for _ in range(self.max_steps):
            act = node >= 0
            if not act.any():
                break
            n = np.where(act, node, 0)
            f = x[self.feat[rows, n]]
            mt = self.mtype[rows, n]
            nan = f != f
            f = np.where(nan & (mt != 2), 0.0, f)
            default = ((mt == 1) & (f >= -ZERO) & (f <= ZERO)) | ((mt == 2) & nan)
            go_left = np.where(default, self.dleft[rows, n], f <= self.thr[rows, n])
            node = np.where(act, np.where(go_left, self.left[rows, n], self.right[rows, n]), node)
        if (node >= 0).any():
            raise ValueError("a tree did not end in a leaf")
        return float(np.cumsum(self.leaf[rows, ~node])[-1])      # cumsum: added one by one, in tree order

    def predict(self, X, num_threads=1):
        X = np.atleast_2d(np.asarray(X, float))
        if X.shape[1] < self.n_features:
            raise ValueError("too few features")
        return np.array([self._one(X[r]) for r in range(len(X))])
