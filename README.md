# EG Property Estimator (ML + FastAPI + React)

End-to-end project to estimate Egyptian property prices from listing features.

* **ML layer:** scikit-learn pipeline trained on 54k listings
* **Serving:** FastAPI `/predict` returns estimated **price (EGP)**
* **Frontend:** Vite + React form

---

## 0) Prerequisites

* **Python** 3.13 (use `python3 --version`)
* **Node.js** ≥ **20.19** or **22.12+** (Vite requirement)

  * Recommended: `nvm`

    ```bash
    brew install nvm && mkdir -p ~/.nvm
    echo 'export NVM_DIR="$HOME/.nvm"' >> ~/.zshrc
    echo '. "/opt/homebrew/opt/nvm/nvm.sh"' >> ~/.zshrc
    source ~/.zshrc
    nvm install 22.12.0
    nvm use 22.12.0
    ```
* macOS shell: `zsh` (default on macOS)

Repo layout (relative to **project root**):

```
ml-fastapi/ml-fastapi/
├─ ml-layer/
│  ├─ src/          # train.py, predict.py, utils.py, encoders.py
│  ├─ data/         # Eg_RealState_Data_Cleaned.csv   <-- put CSV here
│  └─ artifacts/    # created after training (model + stats)
├─ fastapi-layer/   # main.py (API)
└─ frontend-layer/  # React app (Vite)
```

> **Important:** Place your dataset at:
> `ml-layer/data/Eg_RealState_Data_Cleaned.csv`

---

## 1) Python environment & dependencies

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
```

Verify dataset exists:

```bash
ls -lh ml-layer/data/Eg_RealState_Data_Cleaned.csv
```

---

## 2) Train the model (with MLflow tracking)

**What this training does (high level):**

* **Cleaning:** keep sale listings, remove obvious bad cities, bucket rare regions
* **Outliers:** global **ppm²** trim (1–99th pct) + **per-city** robust trim (5–95th pct for cities with ≥50 rows)
* **Feature engineering (numeric):**
  `level_num, is_ground, furnished_bin, area_per_bedroom, bathrooms_per_bedroom, rooms_total, log_area, sqrt_area, bed_bath_ratio, area_per_room`
* **Categoricals:**

  * `type` → **One-Hot Encoding**
  * `city, region` → **Smoothed Target Encoding** (leak-safe inside CV)
* **Target:** **price-per-m² (ppm²)** with **log1p** transform (via `TransformedTargetRegressor`)
* **Regressor:** `HistGradientBoostingRegressor(loss="absolute_error")`
* **Blending:** CV-tuned blend of model ppm² with **median-based baseline**
  (medians computed on `(city, region, type)` → `(city, region)` → `(city)` → global)
* **Calibration:** Out-of-fold **city bias** multiplier (capped to 0.6–1.6)
* **Metrics reported:** RMSE, MAE, **MAPE** in **EGP** price space

Train:

```bash
# From project root (venv active)
python ml-layer/src/train.py
```

**Example output (yours will vary):**

```
Saved: ml-layer/artifacts/ppm2_model.joblib
Best blend alpha=0.8
Pre-bias Test RMSE=943,439 | MAE=634,289 | MAPE=55.04%
Post-bias Test RMSE=943,693 | MAE=634,629 | MAPE=55.12%

Worst 8 slices by city (by MAPE):
... (diagnostics)
```

**Evaluation:**

* Current **MAPE \~55%** (listing prices + limited features).
* Biggest remaining errors are **intra-city** (compound/finishing/payment terms).
* Accuracy will improve after adding **geo (lat/lon)** and **text** features (title/description) or segmenting models.

Artifacts created:

```
ml-layer/artifacts/
├─ ppm2_model.joblib        # sklearn pipeline (preprocess + model) that predicts ppm²
├─ baseline_stats/          # median ppm² fallback tables (+ global)
└─ city_bias.csv            # city-level calibration factors
```

### 2.1 MLflow UI (optional but recommended)

```bash
# Point UI to the correct store (ml-layer/mlruns)
mlflow ui --port 5000 --backend-store-uri ./ml-layer/mlruns
# Open http://127.0.0.1:5000 → experiment: realestate-eg-price
```

You will see params (feature counts, trim settings, blend alpha) & metrics (cv/test RMSE/MAE/MAPE).

---

## 3) Start the FastAPI backend

What it does:

* Loads `ppm2_model.joblib`, baseline medians & city bias
* Endpoint **POST `/predict`** → returns **price (EGP)** and details (ppm² model/baseline, blend α, city bias)

Run:

```bash
# From project root (venv active)
uvicorn --app-dir fastapi-layer main:app --reload --port 8000
```

Quick health/predict checks:

```bash
curl http://127.0.0.1:8000/health

curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"type":"apartment","area":170,"bedrooms":3,"bathrooms":2,
       "level":"9","furnished":"no","city":"cairo","region":"zahraa al maadi"}'
```

**Example response:**

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

**Evaluation:**

* Output includes breakdown to aid debugging.
* Rounding to nearest **10k EGP** for UX; adjust in `predict.py` if needed.

---

## 4) Frontend (Vite + React) — dev server

What you’ll get:

* Centered, clean **blue/white** form with clear labels (required fields marked)
* Submits to FastAPI via **Vite proxy** (`/api → http://127.0.0.1:8000`)

Install & run:

```bash
# one-time (in case node_modules not installed yet)
cd /Users/pythonarabia/Desktop/code_local/ml-fastapi/ml-fastapi/frontend-layer
npm install

# start dev server (requires Node 20.19+ or 22.12+)
npm run dev
```

Open: **[http://127.0.0.1:5173](http://127.0.0.1:5173)**

**Example result card after submit:**

```
Estimated Price: 1,580,000 EGP
ppm² (model): 9,103 · ppm² (baseline): 9,957 · blend α: 0.8
Note: treat as a starting point; will improve with geo/text features.
```

---

## 5) Common environment variables (optional)

You generally don’t need to set these; sensible defaults are hard-coded.

```bash
# ML (train.py)
export MLFLOW_EXPERIMENT="realestate-eg-price"
export DATA_PATH="ml-layer/data/Eg_RealState_Data_Cleaned.csv"
export ARTIFACT_DIR="ml-layer/artifacts"

# API (main.py / predict.py)
export ARTIFACT_DIR="$(pwd)/ml-layer/artifacts"    # absolute path preferred
export BLEND_ALPHA="0.8"                            # try 0.7..0.9 without retraining
```

---

## 6) Troubleshooting

* **`FileNotFoundError: data/...csv`**
  Make sure the CSV is at `ml-layer/data/Eg_RealState_Data_Cleaned.csv`, or set `DATA_PATH`.

* **`Can't get attribute 'SmoothedTargetEncoder'` when serving**
  We extracted `SmoothedTargetEncoder` into `ml-layer/src/encoders.py` and added a legacy shim in `predict.py`. If you trained before this refactor, **retrain once** to pickle under the correct module.

* **Vite error `crypto.hash is not a function` or “requires Node 20.19+”**
  Use `nvm use 22.12.0`, then:

  ```bash
  cd frontend-layer
  rm -rf node_modules package-lock.json
  npm install
  npm run dev
  ```

* **macOS reloader oddities**
  Run `uvicorn` without `--reload` for serving:

  ```bash
  uvicorn --app-dir fastapi-layer main:app --port 8000
  ```

---

## 7) Roadmap to improve accuracy

* **Geo features:** geocode regions/compounds → lat/lon, distances to Ring Rd, Nile, POIs
* **Text features:** TF-IDF(1–2) + SVD(64) on title/description (captures “Palm Hills”, “fully finished”, “installments”, “sea view”)
* **Segmentation:** separate models for Cairo/New Cairo vs Giza/Zayed/October; fallback for low-data cities
* **Price range:** add quantile models and return low/median/high
* **Comps:** nearest-neighbor comps by location + area + rooms for transparency

---

## One-shot script (cheat sheet)

From project root:

```bash
# ---- Setup Python ----
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install scikit-learn==1.7.1 pandas==2.2.2 numpy==1.26.4 \
            mlflow==3.3.2 joblib==1.4.2 \
            fastapi==0.115.0 uvicorn==0.30.6 pydantic==2.9.2

# ---- Verify dataset ----
ls -lh ml-layer/data/Eg_RealState_Data_Cleaned.csv

# ---- Train (MLflow logs to ml-layer/mlruns) ----
python ml-layer/src/train.py

# ---- (Optional) MLflow UI ----
mlflow ui --port 5000 --backend-store-uri ./ml-layer/mlruns

# ---- Start API ----
uvicorn --app-dir fastapi-layer main:app --reload --port 8000
# test
curl http://127.0.0.1:8000/health
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" \
  -d '{"type":"apartment","area":170,"bedrooms":3,"bathrooms":2,"level":"9","furnished":"no","city":"cairo","region":"zahraa al maadi"}'

# ---- Frontend (Node 22.12+) ----
cd frontend-layer
npm install
npm run dev
# open http://127.0.0.1:5173
```

---

