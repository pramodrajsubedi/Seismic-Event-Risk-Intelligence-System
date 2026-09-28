# Seismic Event Risk Intelligence System

An end-to-end data science pipeline that ingests live earthquake data from the USGS API, validates physical laws through EDA, and predicts damaging aftershock probability using an XGBoost classifier — deployed as an interactive Streamlit dashboard and a live REST API.

| | |
|---|---|
| **Streamlit Dashboard** | https://seismic-event-risk-intelligence-system.streamlit.app/ |
| **REST API** | https://seismic-event-risk-intelligence-system.onrender.com/docs |

---

## What it does

| Layer | Description |
|---|---|
| Data ingestion | Fetches M4+ earthquakes worldwide from the USGS REST API into a normalized 3-table SQLite database |
| SQL analytics | 5 queries covering JOINs, window functions, and physics formulas embedded directly in SQL |
| EDA | 8-chart notebook validating Gutenberg-Richter and Omori-Utsu laws in the data |
| ML model | XGBoost binary classifier predicting whether a mainshock will produce a damaging aftershock within 48 hours |
| REST API | FastAPI endpoint containerized with Docker, deployed on Render with CI/CD via GitHub Actions |
| Dashboard | 3-tab Streamlit app with a live USGS-fed world map and real-time risk predictor |

---

## Model Performance

5-fold stratified cross-validation on 186 USGS events (M5+).

| Metric | Mean | Std |
|---|---|---|
| ROC-AUC | 0.84 | ±0.04 |
| F1 Score | 0.79 | ±0.02 |
| Accuracy | 0.77 | ±0.02 |
| Precision | 0.79 | ±0.02 |
| Recall | 0.80 | ±0.06 |

---

## REST API

Built with FastAPI, containerized with Docker, deployed on Render.

```bash
curl -X POST https://seismic-event-risk-intelligence-system.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "magnitude": 6.5,
    "depth_km": 35.0,
    "latitude": 27.7,
    "longitude": 85.3,
    "tsunami_flag": 0,
    "risk_tier": 2,
    "tectonic_setting": "Subduction zone",
    "plate_boundary_type": "Convergent"
  }'
```

```json
{
  "prediction": 1,
  "risk_label": "HIGH",
  "damage_probability": 0.8785
}
```

---

## Key Findings

- b-value ≈ 1.0 in the Gutenberg-Richter analysis, confirming the dataset is physically consistent with the global average
- Magnitude (r = 0.61) is the strongest predictor of damaging aftershocks, consistent with Bath's Law
- Subduction zones generated a 100% damaging aftershock rate across 72 M5+ mainshocks
- Shallow events (depth < 70 km) account for all tsunami-flagged earthquakes, confirmed by SQL depth analysis
- M7.8 Philippines (June 2026): 232 aftershocks within 200 km, Omori-Utsu decay curve visible in the data

---

## Architecture

```
USGS Earthquake API
        │
        ▼
   ingest.py              ← REST API fetch, GeoJSON parsing
        │
        ▼
  SQLite database         ← 3 tables: earthquakes, regions, aftershock_sequences
        │
   ┌────┴────┐
   ▼         ▼
eda.ipynb  aftershocks.py ← Physics law validation | Spatial sequence builder
   │         │
   └────┬────┘
        ▼
  train_model.py          ← XGBoost classifier + SHAP interpretability
        │
   ┌────┴────┐
   ▼         ▼
app.py   api/main.py      ← Streamlit dashboard | FastAPI REST endpoint
             │
         Dockerfile       ← Containerized for portable deployment
             │
           Render         ← Live public API
```

---

## Database Schema

```sql
earthquakes          regions                  aftershock_sequences
─────────────        ──────────────────       ────────────────────
quake_id   PK        region_id   PK           seq_id        PK
event_time           name                     mainshock_id  FK
latitude             tectonic_setting         aftershock_id FK
longitude            plate_boundary_type      delta_hours
depth_km             risk_tier                mag_ratio
magnitude            avg_depth_km             had_damaging_after
region_id  FK        historical_max_mag
tsunami_flag
```

---

## SQL Highlights

**Gutenberg-Richter law:**
```sql
SELECT ROUND(magnitude, 1)       AS mag_bin,
       COUNT(*)                  AS event_count,
       ROUND(LOG10(COUNT(*)), 4) AS log10_count
FROM   earthquakes
GROUP  BY mag_bin
ORDER  BY mag_bin DESC;
```

**Monthly seismic energy release:**
```sql
SELECT strftime('%Y-%m', event_time) AS month,
       ROUND(SUM(POWER(10.0, 1.5 * magnitude + 4.8)) / 1e15, 4) AS energy_PJ,
       SUM(SUM(POWER(10.0, 1.5 * magnitude + 4.8)) / 1e15)
           OVER (ORDER BY strftime('%Y-%m', event_time)
                 ROWS UNBOUNDED PRECEDING) AS cumulative_PJ
FROM   earthquakes
GROUP  BY month;
```

---

## Physics Laws Validated

**Gutenberg-Richter Law** — `log₁₀(N) = a − b·M`
The number of earthquakes drops log-linearly with magnitude. The b-value in this dataset is ~1.0, matching the global average.

**Omori-Utsu Law** — `n(t) = K / (t + c)^p`
Aftershock rate decays as a power law after a mainshock. The M7.8 Philippines sequence shows a steep initial decay in the first 24 hours, consistent with a p-value of ~1.1.

---

## Stack

| Tool | Role |
|---|---|
| Python 3.10+ | Core language |
| pandas · numpy · scikit-learn | Data processing and ML utilities |
| XGBoost · SHAP | Classifier and explainability |
| SQLite | 3-table normalized database · 5 analytical queries |
| FastAPI | REST API with Pydantic validation |
| Docker | Containerized deployment |
| GitHub Actions | CI/CD — automated tests and Docker build on every push |
| Render | Live cloud deployment |
| Plotly | Interactive charts and globe map |
| Streamlit | 3-tab dashboard · live USGS API feed |
| USGS API | Real-time earthquake catalog |

---

## Author

**Pramod Raj Subedi**
[LinkedIn](https://www.linkedin.com/in/pramodrajsubedi/)
