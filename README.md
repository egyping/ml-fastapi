# EG Property Estimator (ML + FastAPI + React)

---

## Architecture and Flow

This project is a simple, end-to-end pipeline that turns raw listing data into a live property-price estimator website.

1. **Collect data → CSV**
   Scrape/export property listings (type, area, bedrooms, bathrooms, level, furnished, city, region, price). Do a **quick visual clean** in Excel/Sheets (fix obvious typos, remove duplicates/empty prices). Save as **`Eg_RealState_Data_Cleaned.csv`** and put it in: `ml-layer/data/Eg_RealState_Data_Cleaned.csv`.

2. **Train (Python)**
   Run `train.py`. It **cleans** data, performs light **feature engineering** (e.g., level → number, “is ground”, area/rooms ratios, log/sqrt of area), builds a **pipeline** (numeric passthrough, One-Hot for `type`, Smoothed Target Encoding for `city`/`region`), trains a gradient-boosting model on **price-per-m²**, **blends** with a median baseline, applies a small **city-bias** calibration, and reports RMSE/MAE/MAPE.

3. **Save the model**
   The trained pipeline is saved to **`ml-layer/artifacts/ppm2_model.joblib`**.
   *What is `joblib`?* It’s the standard way to serialize scikit-learn models/pipelines (Pickle under the hood, efficient with NumPy arrays).

4. **Serve (FastAPI)**
   FastAPI loads `ppm2_model.joblib` + baseline stats once at startup and exposes **`POST /predict`**. The API reuses the same feature engineering, predicts ppm², blends with baseline, applies city bias, and returns a final **EGP price** and debug details.

5. **Frontend (React + Vite)**
   A small React app (Vite) shows a **centered blue/white form**. On submit, it calls the FastAPI endpoint (via Vite proxy) and displays the estimate.

That’s it: **CSV → Train & save model → API → React form**.

---

## Prerequisites

* **Python** 3.13
* **Node.js** ≥ **20.19** or **22.12+** (Vite requirement)
  Recommended (macOS) via `nvm`:

  ```bash
  brew install nvm && mkdir -p ~/.nvm
  echo 'export NVM_DIR="$HOME/.nvm"' >> ~/.zshrc
  echo '. "/opt/homebrew/opt/nvm/nvm.sh"' >> ~/.zshrc
  source ~/.zshrc
  nvm install 22.12.0
  nvm use 22.12.0
  ```

> **Dataset**: place your CSV at
> `ml-layer/data/Eg_RealState_Data_Cleaned.csv`

---

## Project Structure

```
ml-fastapi/ml-fastapi/
├─ ml-layer/
│  ├─ src/
│  │  ├─ train.py         # training script (logs to MLflow + saves artifacts)
│  │  ├─ predict.py       # inference helper used by FastAPI
│  │  ├─ utils.py         # feature normalization & engineering
│  │  └─ encoders.py      # SmoothedTargetEncoder (safe unpickling)
│  ├─ data/
│  │  └─ Eg_RealState_Data_Cleaned.csv   # <— put CSV here
│  ├─ artifacts/          # created by training (saved model + stats)
│  │  ├─ ppm2_model.joblib
│  │  ├─ city_bias.csv
│  │  └─ baseline_stats/
│  │     ├─ m_crt.csv     # median ppm² by (city,region,type)
│  │     ├─ m_cr.csv      # median ppm² by (city,region)
│  │     ├─ m_c.csv       # median ppm² by (city)
│  │     └─ m_g.txt       # global median ppm²
│  └─ mlruns/             # MLflow tracking store (created on first run)
│     └─ ...              # experiments/runs metadata
├─ fastapi-layer/
│  └─ main.py             # FastAPI app (POST /predict, /health)
└─ frontend-layer/        # Vite + React app (blue/white UI)
```

---

## 1) Python Env & Dependencies

From the **project root**:

```bash
cd /Users/pythonarabia/Desktop/code_local/ml-fastapi/ml-fastapi

# Create & activate venv
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python deps
pip install scikit-learn==1.7.1 pandas==2.2.2 numpy==1.26.4 \
            mlflow==3.3.2 joblib==1.4.2 \
            fastapi==0.115.0 uvicorn==0.30.6 pydantic==2.9.2

# Verify dataset
ls -lh ml-layer/data/Eg_RealState_Data_Cleaned.csv
```

---

## 2) Training with MLflow

### What training does

* **Cleaning**

  * Keep sale listings, remove bad city tokens, bucket rare regions
  * Outlier control in **price-per-m² (ppm²)**:

    * Global trim at **1–99th percentiles**
    * **Per-city** robust trim at **5–95th percentiles** for cities with ≥50 rows
* **Feature engineering (numeric)**

  * `level_num`, `is_ground`, `furnished_bin`
  * `area_per_bedroom`, `bathrooms_per_bedroom`, `rooms_total`
  * `log_area`, `sqrt_area`, `bed_bath_ratio`, `area_per_room`
* **Categoricals**

  * `type` → **One-Hot Encoding**
  * `city`, `region` → **Smoothed Target Encoding** (leak-safe inside CV)
* **Target & model**

  * Train on **ppm²** with **log1p** transform (via `TransformedTargetRegressor`)
  * Regressor: **HistGradientBoostingRegressor** (`loss="absolute_error"`)
* **Blending & calibration**

  * Blend model ppm² with a **median baseline** by location/type (alpha tuned by CV)
  * **City bias** calibration from out-of-fold predictions (clipped to 0.6–1.6)
* **Metrics (price space, EGP)**

  * **RMSE**, **MAE**, **MAPE** + **per-city/region diagnostics**

### Train command

```bash
# From project root (venv active)
python ml-layer/src/train.py
```

### Example output & evaluation

```
Saved: ml-layer/artifacts/ppm2_model.joblib
Best blend alpha=0.8
Pre-bias Test RMSE=943,439 | MAE=634,289 | MAPE=55.04%
Post-bias Test RMSE=943,693 | MAE=634,629 | MAPE=55.12%
```

* **Interpretation:** With listing prices and limited features (no text/geo), **MAPE \~55%** is expected.
* **Next gains:** add **geo (lat/lon)**, **text features** (title/description), or **segment** models.

Artifacts created:

```
ml-layer/artifacts/
├─ ppm2_model.joblib        # sklearn pipeline (preprocess + model) that predicts ppm²
├─ baseline_stats/
│  ├─ m_crt.csv             # median ppm² by (city,region,type)
│  ├─ m_cr.csv              # median ppm² by (city,region)
│  ├─ m_c.csv               # median ppm² by (city)
│  └─ m_g.txt               # global median ppm²
└─ city_bias.csv            # city-level calibration factors
```

### MLflow UI

```bash
# From project root
mlflow ui --port 5000 --backend-store-uri ./ml-layer/mlruns
# Open http://127.0.0.1:5000 → experiment: realestate-eg-price
```

---

## 3) Start the FastAPI Backend

What it does:

* Loads `ppm2_model.joblib`, baseline medians & city bias once.
* **POST `/predict`**:

  * Normalizes request → predicts ppm² (model) → blends with baseline → applies city bias → returns **price (EGP)**.

Run:

```bash
# From project root (venv active)
uvicorn --app-dir fastapi-layer main:app --reload --port 8000
```

### Health & Predict examples

```bash
curl http://127.0.0.1:8000/health

curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"type":"apartment","area":170,"bedrooms":3,"bathrooms":2,
       "level":"9","furnished":"no","city":"cairo","region":"zahraa al maadi"}'
```

**Sample response**

```json
{
  "price": 1580000.0,
  "currency": "EGP",
  "details": {
    "ppm2_model": 9102.57,
    "ppm2_baseline": 9956.71,
    "ppm2_blend": 9273.40,
    "blend_alpha": 0.8,
    "city_bias": 1.0000
  }
}
```

---

## 4) Frontend (Vite + React)

Dev server uses a proxy so `/api/*` goes to `http://127.0.0.1:8000/*`.

```bash
# Ensure Node is 20.19+ or 22.12+
node -v

# Install & run
cd /Users/pythonarabia/Desktop/code_local/ml-fastapi/ml-fastapi/frontend-layer
npm install
npm run dev
```

Open: **[http://127.0.0.1:5173](http://127.0.0.1:5173)**
Submit the form → see the estimated price card.

---

## 5) Environment Variables (optional)

```bash
# ML (train.py)
export MLFLOW_EXPERIMENT="realestate-eg-price"
export DATA_PATH="ml-layer/data/Eg_RealState_Data_Cleaned.csv"
export ARTIFACT_DIR="ml-layer/artifacts"

# API (main.py / predict.py)
export ARTIFACT_DIR="$(pwd)/ml-layer/artifacts"   # absolute path recommended
export BLEND_ALPHA="0.8"                           # try 0.7..0.9 without retraining
```

---

## 6) Troubleshooting

* **CSV not found** → place it at `ml-layer/data/Eg_RealState_Data_Cleaned.csv` or set `DATA_PATH`.
* **Pickle error: `SmoothedTargetEncoder`** → retrain once after encoder was moved to `encoders.py`.
* **Vite / Node “crypto.hash”** → `nvm use 22.12.0`, then reinstall.
* **macOS reload quirks** → run Uvicorn without `--reload`.

---

## 7) Roadmap to Improve Accuracy

* **Geo features:** geocode regions/compounds → lat/lon; distances to Ring Rd, Nile, POIs
* **Text features:** TF-IDF (1–2 grams) + SVD(64) on title/description
* **Segmentation:** separate models (Cairo/New Cairo vs Giza/Zayed/October); fallback for low-data cities
* **Price range:** quantile models (low/median/high)
* **Comps:** nearest comparable listings (location + area + rooms)

---

## One-shot Script (cheat sheet)

From the **project root**:

```bash
# ---- Python setup ----
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install scikit-learn==1.7.1 pandas==2.2.2 numpy==1.26.4 \
            mlflow==3.3.2 joblib==1.4.2 \
            fastapi==0.115.0 uvicorn==0.30.6 pydantic==2.9.2

# ---- Verify dataset ----
ls -lh ml-layer/data/Eg_RealState_Data_Cleaned.csv

# ---- Train ----
python ml-layer/src/train.py

# ---- MLflow UI (optional) ----
mlflow ui --port 5000 --backend-store-uri ./ml-layer/mlruns

# ---- Start API ----
uvicorn --app-dir fastapi-layer main:app --reload --port 8000
curl http://127.0.0.1:8000/health
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" \
  -d '{"type":"apartment","area":170,"bedrooms":3,"bathrooms":2,"level":"9","furnished":"no","city":"cairo","region":"zahraa al maadi"}'

# ---- Frontend ----
cd frontend-layer
npm install
npm run dev
# open http://127.0.0.1:5173
```

---

**License**: MIT (or your preferred license).
