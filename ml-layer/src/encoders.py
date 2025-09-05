import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Leak-safe, smoothed target encoding for high-cardinality categoricals.
    Fits on the training fold (inside Pipeline/CV).
    Encodes each column to a smoothed mean of the target (e.g., ppm²).
    """
    def __init__(self, cols, alpha=10.0):
        self.cols = cols
        self.alpha = alpha
        self.prior_ = None
        self.maps_ = {}

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("SmoothedTargetEncoder requires y during fit.")
        X = X.copy()
        self.prior_ = float(np.mean(y))
        self.maps_ = {}
        for c in self.cols:
            s = pd.Series(X[c].astype(str).values, name=c)
            dfc = pd.DataFrame({c: s, "y": y})
            grp = dfc.groupby(c)["y"].agg(["sum", "count"])
            enc = (grp["sum"] + self.prior_ * self.alpha) / (grp["count"] + self.alpha)
            self.maps_[c] = enc.to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        out = []
        for c in self.cols:
            m = self.maps_.get(c, {})
            out.append(X[c].astype(str).map(m).fillna(self.prior_).to_numpy().reshape(-1, 1))
        return np.concatenate(out, axis=1)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"te_{c}" for c in self.cols])
