import numpy as np
import pandas as pd

def _to_level_num(v):
    s = str(v).strip().lower()
    if s in ("ground", "g", "0"):
        return 0
    try:
        return int(float(s))
    except Exception:
        return np.nan

def normalize_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize raw columns into consistent types and add engineered features.
    Safe for both training and inference.
    """
    df = df.copy()

    # --- normalize text columns (lowercase, strip)
    for col in ["type", "city", "region", "furnished", "rent", "level"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    # --- coerce numerics (prices/areas can come as strings like "1,500")
    for col in ["price", "area", "bedrooms", "bathrooms"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"[^\d\.]", "", regex=True)  # drop commas/garbage
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- level features
    df["level_num"] = df.get("level", pd.Series([np.nan] * len(df))).apply(_to_level_num)
    df["is_ground"] = (df["level_num"] == 0).astype(int)

    # --- furnished -> binary
    if "furnished" in df.columns:
        df["furnished_bin"] = df["furnished"].map({"yes": 1, "no": 0}).fillna(0)
    else:
        df["furnished_bin"] = 0

    # --- engineered numeric features
    safe_bedrooms = df["bedrooms"].fillna(1).clip(lower=1)
    df["area_per_bedroom"] = df["area"] / safe_bedrooms
    df["bathrooms_per_bedroom"] = df["bathrooms"] / safe_bedrooms
    df["rooms_total"] = df["bedrooms"] + df["bathrooms"]

    # diminishing returns / scale features
    df["log_area"] = np.log1p(df["area"])
    df["sqrt_area"] = np.sqrt(df["area"])

    # interactions (lightweight)
    df["bed_bath_ratio"] = df["bathrooms"] / safe_bedrooms
    df["area_per_room"] = df["area"] / df["rooms_total"].clip(lower=1)

    return df
