import pickle
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODELS_DIR = Path(__file__).parent.parent / "models"

with open(MODELS_DIR / "model.pkl",        "rb") as f: model    = pickle.load(f)
with open(MODELS_DIR / "scaler.pkl",       "rb") as f: scaler   = pickle.load(f)
with open(MODELS_DIR / "le_tectonic.pkl",  "rb") as f: le_tect  = pickle.load(f)
with open(MODELS_DIR / "le_plate.pkl",     "rb") as f: le_plate = pickle.load(f)

app = FastAPI(title="Seismic Risk API", version="1.0.0")


class EarthquakeInput(BaseModel):
    magnitude:           float
    depth_km:            float
    latitude:            float
    longitude:           float
    tsunami_flag:        int
    risk_tier:           int
    tectonic_setting:    str
    plate_boundary_type: str


class PredictionOutput(BaseModel):
    prediction:         int
    risk_label:         str
    damage_probability: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: EarthquakeInput):
    try:
        tectonic_enc = int(le_tect.transform([data.tectonic_setting])[0])
        plate_enc    = int(le_plate.transform([data.plate_boundary_type])[0])
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Unknown category value: {e}")

    is_shallow      = int(data.depth_km < 70)
    is_intermediate = int(70 <= data.depth_km < 300)
    mag_class       = int(np.digitize(data.magnitude, [5.4, 5.9, 6.4, 6.9, 99]))

    features = np.array([[
        data.magnitude, data.depth_km,
        is_shallow, is_intermediate, mag_class,
        data.risk_tier, tectonic_enc, plate_enc,
        data.latitude, data.longitude, data.tsunami_flag,
    ]])

    scaled = scaler.transform(features)
    pred   = int(model.predict(scaled)[0])
    prob   = round(float(model.predict_proba(scaled)[0][1]), 4)

    return PredictionOutput(
        prediction=pred,
        risk_label="HIGH" if pred == 1 else "LOW",
        damage_probability=prob,
    )