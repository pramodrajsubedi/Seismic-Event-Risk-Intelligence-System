from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_returns_valid_schema():
    payload = {
        "magnitude": 6.5,
        "depth_km": 35.0,
        "latitude": 27.7,
        "longitude": 85.3,
        "tsunami_flag": 0,
        "risk_tier": 2,
        "tectonic_setting": "Subduction Zone",
        "plate_boundary_type": "Convergent"
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["risk_label"] in ("HIGH", "LOW")
    assert 0.0 <= body["damage_probability"] <= 1.0