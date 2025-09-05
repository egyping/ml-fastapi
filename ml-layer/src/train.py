import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from utils import normalize_and_engineer
from encoders import SmoothedTargetEncoder

# ------------ Paths & Config ------------
HERE = Path(__file__).resolve()                  # .../ml-layer/src/train.py
ML_LAYER = HERE.parents[1]                       # .../ml-layer
DEFAULT_DATA = ML_LAYER / "data" / "Eg_RealState_Data_Cleaned.csv"

DATA_PATH = os.getenv("DATA_PATH", str(DEFAULT_DATA))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", str(ML_LAYER / "artifacts")))
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Path validation
if not Path(DATA_PATH).exists():
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
print(f"Using data file: {DATA_PATH}")
print(f"Using artifact directory: {ARTIFACT_DIR}")

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "realestate-eg-price")

# cleaning thresholds
MIN_REGION_FREQ = int(os.getenv("MIN_REGION_FREQ", "30"))
OUTLIER_Q_LOW  = float(os.getenv("OUTLIER_Q_LOW",  "0.01"))
OUTLIER_Q_HIGH = float(os.getenv("OUTLIER_Q_HIGH", "0.99"))

# per-city robust trim
CITY_Q_LOW      = float(os.getenv("CITY_Q_LOW",  "0.05"))
CITY_Q_HIGH     = float(os.getenv("CITY_Q_HIGH", "0.95"))
CITY_MIN_COUNT  = int(os.getenv("CITY_MIN_COUNT", "50"))

# blend grid for model vs baseline
ALPHA_GRID = [0.5, 0.6, 0.7, 0.8]


# --------- Baseline ppm² medians with fallbacks ----------
def _fit_ppm2_baseline(X: pd.DataFrame, yppm2: pd.Series):
    df = pd.DataFrame({
        "city": X["city"].astype(str),
        "region": X["region"].astype(str),
        "type": X["type"].astype(str),
        "ppm2": yppm2.astype(float)
    })
    m_crt = df.groupby(["city", "region", "type"])["ppm2"].median().rename("m_crt").reset_index()
    m_cr  = df.groupby(["city", "region"])["ppm2"].median().rename("m_cr").reset_index()
    m_c   = df.groupby(["city"])["ppm2"].median().rename("m_c").reset_index()
    m_g   = float(df["ppm2"].median())
    return {"m_crt": m_crt, "m_cr": m_cr, "m_c": m_c, "m_g": m_g}

def _predict_ppm2_baseline(X: pd.DataFrame, stats: dict) -> np.ndarray:
    s = X[["city", "region", "type"]].astype(str).copy()
    out = s.merge(stats["m_crt"], on=["city", "region", "type"], how="left")
    out = out.merge(stats["m_cr"], on=["city", "region"], how="left")
    out["ppm2"] = out["m_crt"].fillna(out["m_cr"])
    out = out.merge(stats["m_c"], on=["city"], how="left")
    out["ppm2"] = out["ppm2"].fillna(out["m_c"]).fillna(stats["m_g"])
    return out["ppm2"].to_numpy()


def load_and_clean(path: str) -> pd.DataFrame:
    """Load CSV, keep sale listings, normalize, engineer, remove outliers, clean categories."""
    df = pd.read_csv(path)

    # Keep sale listings only if "rent" exists
    if "rent" in df.columns:
        df["rent"] = df["rent"].astype(str).str.lower()
        df = df[df["rent"] == "no"]

    # Ensure required columns are present
    required = ["price", "area", "bedrooms", "bathrooms", "type", "city", "region"]
    df = df.dropna(subset=[c for c in required if c in df.columns])

    # Normalize & engineer
    df = normalize_and_engineer(df)

    # Global extreme outliers (ppm2)
    df["ppm2_tmp"] = df["price"] / df["area"].clip(lower=1)
    lo, hi = df["ppm2_tmp"].quantile([OUTLIER_Q_LOW, OUTLIER_Q_HIGH])
    df = df[(df["ppm2_tmp"] >= lo) & (df["ppm2_tmp"] <= hi)]

    # Drop obviously bad city tokens
    bad_cities = {"v", "", "nan", "none", "null"}
    df = df[~df["city"].isin(bad_cities)]

    # Bucket rare regions
    vc = df["region"].value_counts()
    keep_regions = set(vc[vc >= MIN_REGION_FREQ].index)
    df["region"] = df["region"].where(df["region"].isin(keep_regions), "__other__")

    # Per-city robust trimming where city has enough data
    city_counts = df.groupby("city")["ppm2_tmp"].transform("count")
    city_lo = df.groupby("city")["ppm2_tmp"].transform(lambda s: s.quantile(CITY_Q_LOW))
    city_hi = df.groupby("city")["ppm2_tmp"].transform(lambda s: s.quantile(CITY_Q_HIGH))
    use_city = city_counts >= CITY_MIN_COUNT
    df = df[(~use_city & (df["ppm2_tmp"] >= lo) & (df["ppm2_tmp"] <= hi)) |
            (use_city  & (df["ppm2_tmp"] >= city_lo) & (df["ppm2_tmp"] <= city_hi))]

    df = df.drop(columns=["ppm2_tmp"])
    return df


def build_pipeline(num_feats_num, cat_ohe_cols, cat_te_cols) -> Pipeline:
    """
    Preprocess:
      - numeric passthrough
      - OneHot for low-cardinality categorical ('type')
      - Smoothed target encoding for high-cardinality ('city','region')
    Regressor:
      - HGB on log1p(ppm2) via TransformedTargetRegressor
    """
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_feats_num),
            ("type_ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_ohe_cols),
            ("cat_te", SmoothedTargetEncoder(cols=cat_te_cols, alpha=10.0), cat_te_cols),
        ]
    )

    reg = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_depth=None,
        learning_rate=0.06,
        max_iter=900,
        l2_regularization=0.0,
        early_stopping=True,
        random_state=42,
    )

    model = Pipeline(
        steps=[
            ("pre", pre),
            ("reg", TransformedTargetRegressor(
                regressor=reg,
                func=np.log1p,
                inverse_func=np.expm1
            )),
        ]
    )
    return model


def eval_metrics(y_true_price, pred_price):
    rmse = root_mean_squared_error(y_true_price, pred_price)
    mae  = mean_absolute_error(y_true_price, pred_price)
    mape = (np.abs((y_true_price - pred_price) / np.maximum(y_true_price, 1.0))).mean() * 100
    return rmse, mae, mape


def print_slice_metrics(X, y_price, pred_price, key: str, top=10):
    """Quick diagnostic: worst slices by MAPE."""
    dfm = pd.DataFrame({key: X[key].values, "y": y_price.values, "p": pred_price})
    def agg(g):
        e = g["y"] - g["p"]
        mae = np.mean(np.abs(e))
        rmse = np.sqrt(np.mean(e ** 2))
        mape = np.mean(np.abs(e) / np.maximum(g["y"], 1.0)) * 100
        return pd.Series({"n": len(g), "MAE": mae, "RMSE": rmse, "MAPE": mape})
    out = (
        dfm.groupby(key)
           .apply(agg, include_groups=False)
           .sort_values("MAPE", ascending=False)
           .round(1)
    )
    print(f"\nWorst {top} slices by {key} (by MAPE):")
    print(out.head(top))


def main():
    print(f"Starting training with experiment: {EXPERIMENT_NAME}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_and_clean(DATA_PATH)

    num_feats_num = [
        "area", "bedrooms", "bathrooms",
        "level_num", "is_ground", "furnished_bin",
        "area_per_bedroom", "bathrooms_per_bedroom", "rooms_total",
        "log_area", "sqrt_area", "bed_bath_ratio", "area_per_room",
    ]
    cat_ohe_cols = ["type"]                # small cardinality
    cat_te_cols  = ["city", "region"]      # high cardinality -> target encoded

    cols_needed = ["price"] + num_feats_num + cat_ohe_cols + cat_te_cols
    df = df[cols_needed].dropna()

    # Base arrays
    X = df.drop(columns=["price"])
    y_price = df["price"].astype(float)
    area_clip = X["area"].clip(lower=1).astype(float)
    y_ppm2 = (y_price / area_clip).astype(float)  # training target

    X_train, X_test, yppm2_train, yppm2_test, yprice_train, yprice_test, area_train, area_test = \
        train_test_split(
            X, y_ppm2, y_price, area_clip,
            test_size=0.15, random_state=42, stratify=X["city"]
        )

    model = build_pipeline(num_feats_num, cat_ohe_cols, cat_te_cols)

    # ---- Phase 1: pick blend alpha via CV (model vs baseline) ----
    alpha_rmse = {a: [] for a in ALPHA_GRID}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, va_idx in kf.split(X_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        yppm2_tr, yppm2_va = yppm2_train.iloc[tr_idx], yppm2_train.iloc[va_idx]
        yprice_va = yprice_train.iloc[va_idx]
        area_va = area_train.iloc[va_idx]

        model.fit(X_tr, yppm2_tr)
        pred_ppm2_va_model = model.predict(X_va)

        stats = _fit_ppm2_baseline(X_tr, yppm2_tr)
        pred_ppm2_va_base = _predict_ppm2_baseline(X_va, stats)

        for a in ALPHA_GRID:
            pred_price_va = (a * pred_ppm2_va_model + (1 - a) * pred_ppm2_va_base) * area_va.values
            rmse, _, _ = eval_metrics(yprice_va, pred_price_va)
            alpha_rmse[a].append(rmse)

    best_alpha = min(ALPHA_GRID, key=lambda a: np.mean(alpha_rmse[a]))
    cv_rmse_mean = float(np.mean(alpha_rmse[best_alpha]))
    cv_rmse_std  = float(np.std(alpha_rmse[best_alpha]))

    # ---- Phase 2: OOF city bias factors on blended predictions ----
    city_ratios = {}
    for tr_idx, va_idx in kf.split(X_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        yppm2_tr, yppm2_va = yppm2_train.iloc[tr_idx], yppm2_train.iloc[va_idx]
        yprice_va = yprice_train.iloc[va_idx]
        area_va = area_train.iloc[va_idx]

        model.fit(X_tr, yppm2_tr)
        pred_ppm2_va_model = model.predict(X_va)
        stats = _fit_ppm2_baseline(X_tr, yppm2_tr)
        pred_ppm2_va_base = _predict_ppm2_baseline(X_va, stats)
        pred_price_va = (best_alpha * pred_ppm2_va_model + (1 - best_alpha) * pred_ppm2_va_base) * area_va.values

        ratio = (yprice_va / np.maximum(pred_price_va, 1.0)).to_numpy()
        for city, r in zip(X_va["city"].astype(str).to_numpy(), ratio):
            city_ratios.setdefault(city, []).append(r)

    city_bias = {c: float(np.clip(np.median(rs), 0.6, 1.6)) for c, rs in city_ratios.items()}

    # ---- Final fit & test with blend + city bias ----
    model.fit(X_train, yppm2_train)
    pred_ppm2_test_model = model.predict(X_test)
    stats_full = _fit_ppm2_baseline(X_train, yppm2_train)
    pred_ppm2_test_base = _predict_ppm2_baseline(X_test, stats_full)

    pred_price_test = (best_alpha * pred_ppm2_test_model + (1 - best_alpha) * pred_ppm2_test_base) * area_test.values
    bias_vec = np.array([city_bias.get(c, 1.0) for c in X_test["city"].astype(str).to_numpy()])
    pred_price_test_adj = pred_price_test * bias_vec

    rmse_raw, mae_raw, mape_raw = eval_metrics(yprice_test, pred_price_test)
    rmse, mae, mape = eval_metrics(yprice_test, pred_price_test_adj)

    with mlflow.start_run():
        # ---- Log params/metrics ----
        mlflow.log_param("model", "HGB + log1p(ppm2) + TE(city,region) + blended baseline + city-bias")
        mlflow.log_param("num_features", len(num_feats_num))
        mlflow.log_param("ohe_features", len(cat_ohe_cols))
        mlflow.log_param("te_features", len(cat_te_cols))
        mlflow.log_param("min_region_freq", MIN_REGION_FREQ)
        mlflow.log_param("outlier_q", f"{OUTLIER_Q_LOW}-{OUTLIER_Q_HIGH}")
        mlflow.log_param("city_trim_q", f"{CITY_Q_LOW}-{CITY_Q_HIGH}")
        mlflow.log_param("city_min_count", CITY_MIN_COUNT)
        mlflow.log_param("blend_alpha", best_alpha)

        mlflow.log_metric("cv_rmse_mean", cv_rmse_mean)
        mlflow.log_metric("cv_rmse_std",  cv_rmse_std)
        mlflow.log_metric("test_rmse_pre_bias", rmse_raw)
        mlflow.log_metric("test_mae_pre_bias",  mae_raw)
        mlflow.log_metric("test_mape_pre_bias", mape_raw)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mae",  mae)
        mlflow.log_metric("test_mape", mape)

        # ---- Persist model + artifacts ----
        out_path = ARTIFACT_DIR / "ppm2_model.joblib"
        joblib.dump(model, out_path)

        stats_dir = ARTIFACT_DIR / "baseline_stats"
        stats_dir.mkdir(exist_ok=True, parents=True)
        stats_full["m_crt"].to_csv(stats_dir / "m_crt.csv", index=False)
        stats_full["m_cr"].to_csv(stats_dir / "m_cr.csv", index=False)
        stats_full["m_c"].to_csv(stats_dir / "m_c.csv", index=False)
        (stats_dir / "m_g.txt").write_text(str(stats_full["m_g"]))
        pd.Series(city_bias, name="bias").to_csv(ARTIFACT_DIR / "city_bias.csv")

        mlflow.log_artifact(str(out_path))
        mlflow.log_artifact(str(stats_dir / "m_crt.csv"))
        mlflow.log_artifact(str(stats_dir / "m_cr.csv"))
        mlflow.log_artifact(str(stats_dir / "m_c.csv"))
        mlflow.log_artifact(str(stats_dir / "m_g.txt"))
        mlflow.log_artifact(str(ARTIFACT_DIR / "city_bias.csv"))

        # ---- MLflow model with signature (casts numerics to float64) ----
        X_for_sig = X_train.copy()
        num_cols = num_feats_num
        X_for_sig[num_cols] = X_for_sig[num_cols].astype("float64")
        sig = infer_signature(X_for_sig.head(100), model.predict(X_for_sig.head(100)))
        mlflow.sklearn.log_model(
            model,
            name="ppm2_sklearn_model",
            signature=sig,
            input_example=X_for_sig.iloc[[0]],
        )

        print(f"Saved: {out_path}")
        print(f"Best blend alpha={best_alpha}")
        print(f"Pre-bias Test RMSE={rmse_raw:,.0f} | MAE={mae_raw:,.0f} | MAPE={mape_raw:.2f}%")
        print(f"Post-bias Test RMSE={rmse:,.0f} | MAE={mae:,.0f} | MAPE={mape:.2f}%")

        print_slice_metrics(X_test, yprice_test, pred_price_test_adj, key="city", top=8)
        print_slice_metrics(X_test, yprice_test, pred_price_test_adj, key="region", top=8)


if __name__ == "__main__":
    main()
