# Seismic Event Risk Intelligence System

An end-to-end data science pipeline that ingests live earthquake data from the USGS API,
validates physical laws through EDA, and predicts damaging aftershock probability
using an XGBoost classifier — deployed as an interactive Streamlit dashboard.

**[Live Demo →](https://your-app-name.streamlit.app)**

---

## What it does

| Layer | Description |
|-------|-------------|
| **Data ingestion** | Fetches M4+ earthquakes worldwide from the USGS REST API into a normalized 3-table SQLite database |
| **SQL analytics** | 5 queries covering JOINs, window functions, and physics formulas (Richter energy, Gutenberg-Richter) embedded directly in SQL |
| **EDA** | 8-chart notebook validating Gutenberg-Richter and Omori-Utsu laws in the data |
| **ML model** | XGBoost binary classifier predicting whether a mainshock will produce a damaging aftershock (M4+) within 48 hours |
| **Deployment** | 3-tab Streamlit app with a live USGS-fed world map and real-time risk predictor |

---

## Live app

The app has three tabs:

**Tab 1 — Live Seismic Map**
Pulls M4+ earthquakes from the USGS API in real time. Magnitude-scaled markers on an interactive globe. Colour-coded by severity. M5.5+ events listed in a table below.

**Tab 2 — Aftershock Risk Predictor**
Set magnitude, depth, tectonic setting, and geographic location. The XGBoost model returns a risk level (High / Moderate / Low), a probability percentage, and a gauge chart.

**Tab 3 — Historical Analysis**
Four interactive charts from the SQLite database: Gutenberg-Richter validation, most active tectonic regions, monthly seismic energy release, and aftershock damage rate by magnitude class.

---

## Key findings

- **b-value ≈ 1.0** in the Gutenberg-Richter analysis — confirming the dataset is physically consistent with the global average
- **Magnitude (r = 0.61)** is the strongest predictor of damaging aftershocks, matching Bath's Law which states the largest aftershock is ~1.2 magnitude units below the mainshock
- **Subduction zones** generated 100% damaging aftershock rate across 72 M5+ mainshocks
- **Shallow events** (depth < 70 km) account for all tsunami-flagged earthquakes — confirmed by SQL depth analysis
- **M7.8 Philippines** (June 2026): 232 aftershocks within 200 km, 179 damaging — Omori-Utsu decay curve visible in the data

---

## Architecture

```
USGS Earthquake API
        │
        ▼
   ingest.py          ← REST API fetch, GeoJSON parsing
        │
        ▼
  SQLite database     ← 3 tables: earthquakes, regions, aftershock_sequences
        │
   ┌────┴────┐
   │         │
   ▼         ▼
eda.ipynb  aftershocks.py   ← Gutenberg-Richter, Omori decay | Spatial sequence builder
   │         │
   └────┬────┘
        │
        ▼
  train_model.py      ← XGBoost classifier + SHAP interpretability
        │
        ▼
     app.py           ← Streamlit: live map + predictor + analytics
```

---

## Database schema

```sql
earthquakes          regions                aftershock_sequences
──────────────       ──────────────────     ────────────────────
quake_id  PK         region_id  PK          seq_id     PK
event_time           name                   mainshock_id  FK
latitude             tectonic_setting       aftershock_id FK
longitude            plate_boundary_type    delta_hours
depth_km             risk_tier              mag_ratio
magnitude            avg_depth_km           had_damaging_after
region_id  FK        historical_max_mag
tsunami_flag
```

---

## SQL highlights

**Gutenberg-Richter law** — verifies the log-linear magnitude-frequency relationship:
```sql
SELECT ROUND(magnitude, 1) AS mag_bin,
       COUNT(*)             AS event_count,
       ROUND(LOG10(COUNT(*)), 4) AS log10_count
FROM   earthquakes
GROUP  BY mag_bin
ORDER  BY mag_bin DESC;
```

**Monthly seismic energy release** — Richter-Gutenberg formula embedded in SQL:
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

## ML model

| Detail | Value |
|--------|-------|
| Algorithm | XGBoost binary classifier |
| Target | `had_damaging_after` (M4+ within 48h of mainshock) |
| Features | magnitude, depth_km, is_shallow, tectonic_setting, plate_boundary_type, risk_tier, latitude, longitude, tsunami_flag |
| Validation | Stratified 5-fold cross-validation |
| Imbalance handling | `scale_pos_weight` |
| Interpretability | SHAP beeswarm plot |

**Feature importance (SHAP):**
Magnitude is the dominant feature — consistent with Bath's Law. Depth and tectonic setting provide additional signal, particularly for distinguishing subduction zone sequences from intraplate events.

---

## Physics laws validated

**Gutenberg-Richter Law** `log₁₀(N) = a − b·M`
The number of earthquakes drops log-linearly with magnitude. The b-value in this dataset is ~1.0, matching the global average. A straight line on a log-scale plot confirms data quality before any model runs.

**Omori-Utsu Law** `n(t) = K / (t + c)^p`
Aftershock rate decays as a power law after a mainshock. The M7.8 Philippines sequence shows a steep initial decay in the first 24 hours, consistent with a p-value of ~1.1.

---

## Stack

```
Python 3.10+    pandas · numpy · scikit-learn
SQLite          3 normalized tables · 5 analytical queries
XGBoost         binary classifier · SHAP interpretability
Plotly          interactive charts and globe map
Streamlit       3-tab dashboard · live USGS API feed
USGS API        real-time earthquake catalog
```

---



## Author

**[Pramod Raj Subedi]**


[LinkedIn](https://www.linkedin.com/in/pramodrajsubedi/) · 