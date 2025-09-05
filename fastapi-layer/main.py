# /Users/pythonarabia/Desktop/code_local/ml-fastapi/ml-fastapi/fastapi-layer/main.py
import os, sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Resolve project paths (works no matter where you launch uvicorn) ---
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]                       # .../ml-fastapi/ml-fastapi
ML_SRC = PROJECT_ROOT / "ml-layer" / "src"
sys.path.insert(0, str(ML_SRC))                      # import ml-layer/src

# --- Force absolute artifacts dir so reload/subprocesses see it ---
artifact_dir = (PROJECT_ROOT / "ml-layer" / "artifacts").resolve()
os.environ["ARTIFACT_DIR"] = str(artifact_dir)
os.environ.setdefault("BLEND_ALPHA", "0.8")

print(f"FastAPI starting with project root: {PROJECT_ROOT}")
print(f"ML source directory: {ML_SRC}")
print(f"Artifact directory: {artifact_dir}")

from predict import PriceEstimator  # noqa: E402

app = FastAPI(title="EG Property Estimator (beta)", version="0.1.0")

# Open CORS for dev (tighten later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy load on startup (avoid import-time failures)
estimator = None

class PredictIn(BaseModel):
    type: str = Field(..., examples=["apartment","duplex","villa"])
    area: float = Field(..., gt=0)
    bedrooms: float
    bathrooms: float
    level: str
    furnished: str  # "yes" | "no"
    city: str
    region: str

class PredictOut(BaseModel):
    price: float
    currency: str = "EGP"
    details: dict

@app.on_event("startup")
def _load():
    global estimator
    estimator = PriceEstimator()

@app.get("/health")
def health():
    return {"ok": True, "artifacts": os.environ.get("ARTIFACT_DIR")}

@app.post("/predict", response_model=PredictOut)
def predict_price(payload: PredictIn):
    return estimator.predict_payload(payload.dict())
