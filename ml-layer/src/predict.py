import os, sys, types
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from utils import normalize_and_engineer
import encoders  # ensure module is importable

# ---- Legacy unpickle shim (for old models saved under '__mp_main__') ----
# After you retrain once (model will point to 'encoders.SmoothedTargetEncoder'),
# this shim is no longer needed—but it's safe to keep.
if "__mp_main__" not in sys.modules:
    legacy = types.ModuleType("__mp_main__")
    legacy.SmoothedTargetEncoder = encoders.SmoothedTargetEncoder
    sys.modules["__mp_main__"] = legacy

# ---- Robust artifact resolution ----
HERE = Path(__file__).resolve()             # .../ml-layer/src/predict.py
ML_LAYER_DIR = HERE.parents[1]              # .../ml-layer
DEFAULT_ARTIFACT_DIR = ML_LAYER_DIR / "artifacts"
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", str(DEFAULT_ARTIFACT_DIR))).resolve()

MODEL_PATH     = ARTIFACT_DIR / "ppm2_model.joblib"   # the trained pipeline (ppm²)
STATS_DIR      = ARTIFACT_DIR / "baseline_stats"      # median ppm² fallback tables
CITY_BIAS_CSV  = ARTIFACT_DIR / "city_bias.csv"
DEFAULT_ALPHA  = float(os.getenv("BLEND_ALPHA", "0.8"))

# Path validation and logging
print(f"Using artifact directory: {ARTIFACT_DIR}")
print(f"Model path: {MODEL_PATH}")

def _load_baseline_stats():
    if not STATS_DIR.exists():
        print(f"Warning: Stats directory not found at {STATS_DIR}")
    def _csv(name):
        p = STATS_DIR / name
        return pd.read_csv(p) if p.exists() else None
    m_crt = _csv("m_crt.csv")
    m_cr  = _csv("m_cr.csv")
    m_c   = _csv("m_c.csv")
    m_g_p = STATS_DIR / "m_g.txt"
    m_g   = float(m_g_p.read_text()) if m_g_p.exists() else 1.0
    if m_crt is None: m_crt = pd.DataFrame(columns=["city","region","type","m_crt"])
    if m_cr  is None: m_cr  = pd.DataFrame(columns=["city","region","m_cr"])
    if m_c   is None: m_c   = pd.DataFrame(columns=["city","m_c"])
    return {"m_crt": m_crt, "m_cr": m_cr, "m_c": m_c, "m_g": m_g}

def _predict_ppm2_baseline(X: pd.DataFrame, stats: dict) -> np.ndarray:
    s = X[["city","region","type"]].astype(str).copy()
    out = s.merge(stats["m_crt"], on=["city","region","type"], how="left")
    out = out.merge(stats["m_cr"], on=["city","region"], how="left")
    out["ppm2"] = out["m_crt"].fillna(out["m_cr"])
    out = out.merge(stats["m_c"], on=["city"], how="left")
    out["ppm2"] = out["ppm2"].fillna(out["m_c"]).fillna(stats["m_g"])
    return out["ppm2"].to_numpy()

def _load_city_bias():
    p = CITY_BIAS_CSV
    if p.exists():
        s = pd.read_csv(p, index_col=0)["bias"].to_dict()
        return {str(k): float(v) for k, v in s.items()}
    return {}

def _round_price(x: float, step: int = 10000) -> float:
    return float(int((x + step/2) // step) * step)

class PriceEstimator:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        print(f"Loading model from: {MODEL_PATH}")
        self.model = joblib.load(MODEL_PATH)
        self.stats = _load_baseline_stats()
        self.city_bias = _load_city_bias()
        self.alpha = DEFAULT_ALPHA
        print(f"PriceEstimator initialized with blend_alpha={self.alpha}")

    def predict_payload(self, payload: dict):
        df = pd.DataFrame([payload])
        df = normalize_and_engineer(df)

        area = df["area"].clip(lower=1).astype(float).values
        city = df["city"].astype(str).values

        ppm2_model = self.model.predict(df)                  # model ppm²
        ppm2_base  = _predict_ppm2_baseline(df, self.stats)  # baseline ppm²
        ppm2_blend = self.alpha * ppm2_model + (1 - self.alpha) * ppm2_base

        price = ppm2_blend * area
        bias  = np.array([self.city_bias.get(c, 1.0) for c in city], dtype=float)
        price_adj = price * bias

        out_price = _round_price(float(price_adj[0]))
        return {
            "price": out_price,
            "currency": "EGP",
            "details": {
                "ppm2_model": float(ppm2_model[0]),
                "ppm2_baseline": float(ppm2_base[0]),
                "ppm2_blend": float(ppm2_blend[0]),
                "blend_alpha": float(self.alpha),
                "city_bias": float(bias[0]),
                "artifact_dir": str(ARTIFACT_DIR),
            }
        }

if __name__ == "__main__":
    est = PriceEstimator()
    example = {
        "type": "apartment",
        "area": 170,
        "bedrooms": 3,
        "bathrooms": 2,
        "level": "9",
        "furnished": "no",
        "city": "cairo",
        "region": "zahraa al maadi"
    }
    print(est.predict_payload(example))
