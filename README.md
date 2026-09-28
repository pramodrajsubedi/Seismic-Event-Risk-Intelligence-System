
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

**Gutenberg-Richter law** (verifies the log-linear magnitude-frequency relationship):
```sql
SELECT ROUND(magnitude, 1) AS mag_bin,
       COUNT(*)             AS event_count,
       ROUND(LOG10(COUNT(*)), 4) AS log10_count
FROM   earthquakes
GROUP  BY mag_bin
ORDER  BY mag_bin DESC;
```

**Monthly seismic energy release** (Richter-Gutenberg formula embedded in SQL):
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

## Physics laws validated

**Gutenberg-Richter Law** `log₁₀(N) = a − b·M`
The number of earthquakes drops log-linearly with magnitude. The b-value in this dataset is ~1.0, matching the global average. A straight line on a log-scale plot confirms data quality before any model runs.

**Omori-Utsu Law** `n(t) = K / (t + c)^p`
Aftershock rate decays as a power law after a mainshock. The M7.8 Philippines sequence shows a steep initial decay in the first 24 hours, consistent with a p-value of ~1.1.

---

## Stack

```
| Tool | Role |
|------|------|
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
```

---



## Author

**[Pramod Raj Subedi]**


[LinkedIn](https://www.linkedin.com/in/pramodrajsubedi/) · 