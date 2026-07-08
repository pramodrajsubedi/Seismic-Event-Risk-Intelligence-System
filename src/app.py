"""
app.py — Seismic Risk Intelligence Dashboard
3 tabs: Live Map | Aftershock Risk Predictor | Historical Analysis

Run from project root:
    streamlit run src/app.py
"""

import sys, pickle, warnings
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import plotly.express     as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

from database   import get_connection
from setup_data import needs_setup, run_setup

# Auto-setup (runs on Streamlit Cloud when database is missing) 
if needs_setup():
    st.title("🌍 Seismic Risk Intelligence System")
    st.info("First run detected — building database from live USGS data. This takes ~2 minutes.")
    status = st.empty()
    with st.spinner("Setting up pipeline..."):
        run_setup(status_container=status)
    st.success("Setup complete! Reloading...")
    st.rerun()

# Paths 
ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
USGS_URL   = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# Page config 
st.set_page_config(
    page_title = "Seismic Risk Intelligence",
    page_icon  = "🌍",
    layout     = "wide",
    initial_sidebar_state = "collapsed",
)

st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .risk-high     { background:#fee2e2; border-left:4px solid #ef4444;
                     padding:12px 16px; border-radius:8px; }
    .risk-moderate { background:#fef3c7; border-left:4px solid #f59e0b;
                     padding:12px 16px; border-radius:8px; }
    .risk-low      { background:#dcfce7; border-left:4px solid #22c55e;
                     padding:12px 16px; border-radius:8px; }
    .metric-card   { background:#f8fafc; border:1px solid #e2e8f0;
                     border-radius:10px; padding:16px; text-align:center; }
</style>
""", unsafe_allow_html=True)

st.title("🌍 Seismic Event Risk Intelligence System")
st.caption("Live USGS data · XGBoost aftershock classifier · SQLite analytics")

tab1, tab2, tab3 = st.tabs(["🗺️ Live Seismic Map", "⚡ Aftershock Risk Predictor", "📊 Historical Analysis"])


# TAB 1 — LIVE SEISMIC MAP

with tab1:
    st.subheader("Real-time Global Seismic Activity")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])
    with col_ctrl1:
        days = st.selectbox("Time window", [1, 3, 7, 14, 30], index=2)
    with col_ctrl2:
        min_mag = st.selectbox("Min magnitude", [4.0, 5.0, 6.0], index=0)
    with col_ctrl3:
        st.write("")
        refresh = st.button("🔄 Refresh live data", use_container_width=True)

    @st.cache_data(ttl=1800)   # cache 30 min
    def fetch_live(days_: int, min_mag_: float):
        end   = datetime.utcnow()
        start = end - timedelta(days=days_)
        try:
            resp = requests.get(USGS_URL, params={
                "format": "geojson",
                "starttime": start.strftime("%Y-%m-%d"),
                "endtime":   end.strftime("%Y-%m-%d"),
                "minmagnitude": min_mag_,
                "limit": 20000,
                "orderby": "time-asc",
            }, timeout=30)
            resp.raise_for_status()
            features = resp.json().get("features", [])
            rows = []
            for f in features:
                p = f["properties"]
                c = f["geometry"]["coordinates"]
                rows.append({
                    "id":        f["id"],
                    "time":      datetime.utcfromtimestamp(p["time"]/1000).strftime("%Y-%m-%d %H:%M UTC"),
                    "place":     p.get("place", "Unknown"),
                    "magnitude": p.get("mag", 0),
                    "depth_km":  c[2],
                    "latitude":  c[1],
                    "longitude": c[0],
                    "tsunami":   p.get("tsunami", 0),
                    "alert":     p.get("alert") or "none",
                })
            return pd.DataFrame(rows)
        except Exception as e:
            return pd.DataFrame()

    if refresh:
        st.cache_data.clear()

    with st.spinner("Fetching live USGS data..."):
        live_df = fetch_live(days, min_mag)

    if live_df.empty:
        st.warning("Could not reach USGS API — check your internet connection.")
    else:
        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total events",   f"{len(live_df):,}")
        m2.metric("Max magnitude",  f"M{live_df['magnitude'].max():.1f}")
        m3.metric("Avg depth",      f"{live_df['depth_km'].mean():.0f} km")
        m4.metric("Tsunami alerts", f"{int(live_df['tsunami'].sum())}")

        st.write("")

        # Colour by magnitude
        live_df["size"]  = live_df["magnitude"] ** 2.8
        live_df["color"] = live_df["magnitude"].apply(
            lambda m: "#ef4444" if m >= 6 else ("#f59e0b" if m >= 5 else "#6366f1")
        )

        fig_map = px.scatter_geo(
            live_df,
            lat="latitude", lon="longitude",
            size="size", color="magnitude",
            color_continuous_scale="Reds",
            range_color=[min_mag, max(7.0, live_df["magnitude"].max())],
            hover_name="place",
            hover_data={
                "magnitude": True, "depth_km": True,
                "time": True, "tsunami": True,
                "size": False, "latitude": False, "longitude": False,
            },
            projection="natural earth",
            height=540,
        )
        fig_map.update_layout(
            coloraxis_colorbar=dict(title="Magnitude", thickness=12),
            geo=dict(showland=True, landcolor="#f1f5f9",
                     showocean=True, oceancolor="#dbeafe",
                     showcountries=True, countrycolor="#cbd5e1",
                     showcoastlines=True, coastlinecolor="#94a3b8"),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # Recent significant events table
        sig = live_df[live_df["magnitude"] >= 5.5].sort_values(
            "magnitude", ascending=False).head(10)
        if not sig.empty:
            st.write("**M5.5+ events in this window**")
            st.dataframe(
                sig[["time","place","magnitude","depth_km","tsunami","alert"]].reset_index(drop=True),
                use_container_width=True, hide_index=True,
            )



# TAB 2 — AFTERSHOCK RISK PREDICTOR

with tab2:
    st.subheader("Aftershock Damage Risk Predictor")
    st.caption("Predicts probability of a damaging aftershock (M4+) within 48 hours.")

    # Check model exists
    model_ok = (MODELS_DIR / "model.pkl").exists()
    if not model_ok:
        st.error("Model not found. Run `python src/train_model.py` first.")
        st.stop()

    @st.cache_resource
    def load_model():
        with open(MODELS_DIR / "model.pkl",         "rb") as f: model     = pickle.load(f)
        with open(MODELS_DIR / "scaler.pkl",        "rb") as f: scaler    = pickle.load(f)
        with open(MODELS_DIR / "feature_names.pkl", "rb") as f: feat_cols = pickle.load(f)
        return model, scaler, feat_cols

    model, scaler, feat_cols = load_model()

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.write("**Enter earthquake parameters**")
        magnitude  = st.slider("Magnitude",               4.5, 9.5, 6.5, 0.1)
        depth_km   = st.slider("Depth (km)",              1.0, 700.0, 35.0, 1.0)
        latitude   = st.slider("Latitude",               -90.0, 90.0, 5.6, 0.1)
        longitude  = st.slider("Longitude",             -180.0, 180.0, 125.0, 0.1)
        tectonic   = st.selectbox("Tectonic setting", [
            "Subduction zone", "Collision zone", "Spreading ridge",
            "Transform fault", "Continental rift", "Intraplate"
        ])
        plate_type = st.selectbox("Plate boundary type", [
            "Convergent", "Divergent", "Transform", "Intraplate"
        ])
        risk_tier  = st.selectbox("Region risk tier", [1, 2, 3, 4], index=2)
        tsunami_f  = st.checkbox("Tsunami potential (depth<100km, near ocean)")

        predict_btn = st.button("⚡ Predict Aftershock Risk", use_container_width=True,
                                type="primary")

    with col_out:
        st.write("**Risk assessment**")

        if predict_btn:
            # Feature engineering (mirrors train_model.py)
            is_shallow      = int(depth_km < 70)
            is_intermediate = int(70 <= depth_km < 300)
            mag_class       = int(min(4, max(0, int((magnitude - 4.9) / 0.5))))

            tect_map  = {"Subduction zone":0,"Collision zone":1,"Continental rift":2,
                         "Intraplate":3,"Spreading ridge":4,"Transform fault":5}
            plate_map = {"Convergent":0,"Divergent":1,"Intraplate":2,"Transform":3}

            row = {
                "magnitude":       magnitude,
                "depth_km":        depth_km,
                "is_shallow":      is_shallow,
                "is_intermediate": is_intermediate,
                "mag_class":       mag_class,
                "risk_tier":       risk_tier,
                "tectonic_enc":    tect_map.get(tectonic, 3),
                "plate_enc":       plate_map.get(plate_type, 2),
                "latitude":        latitude,
                "longitude":       longitude,
                "tsunami_flag":    int(tsunami_f),
            }

            X = np.array([[row[f] for f in feat_cols]])
            X_sc = scaler.transform(X)
            prob = model.predict_proba(X_sc)[0][1]

            if prob >= 0.7:
                label, css, emoji = "HIGH RISK",     "risk-high",     "🔴"
            elif prob >= 0.4:
                label, css, emoji = "MODERATE RISK", "risk-moderate", "🟡"
            else:
                label, css, emoji = "LOW RISK",      "risk-low",      "🟢"

            st.markdown(f"""
            <div class="{css}">
                <h2>{emoji} {label}</h2>
                <p style="font-size:2rem; font-weight:600; margin:4px 0">
                    {prob*100:.1f}% probability
                </p>
                <p style="color:#666; margin:0">
                    Damaging aftershock (M4+) within 48 hours
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.write("")

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = round(prob * 100, 1),
                title = {"text": "Aftershock damage probability (%)"},
                gauge = {
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#ef4444" if prob>=0.7 else
                                      "#f59e0b" if prob>=0.4 else "#22c55e"},
                    "steps": [
                        {"range": [0,  40], "color": "#dcfce7"},
                        {"range": [40, 70], "color": "#fef3c7"},
                        {"range": [70,100], "color": "#fee2e2"},
                    ],
                    "threshold": {"line": {"color":"#1e293b","width":4},
                                  "thickness":0.75, "value": prob*100},
                },
                number={"suffix": "%", "font": {"size": 36}},
            ))
            fig_gauge.update_layout(height=280, margin=dict(t=30,b=0,l=20,r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Input summary
            st.write("**Input summary**")
            summary = pd.DataFrame({
                "Parameter": ["Magnitude","Depth","Tectonic setting",
                               "Plate boundary","Risk tier","Shallow?"],
                "Value":     [f"M{magnitude}",f"{depth_km:.0f} km",
                               tectonic, plate_type, risk_tier,
                               "Yes" if is_shallow else "No"],
            })
            st.dataframe(summary, hide_index=True, use_container_width=True)
        else:
            st.info("Set the earthquake parameters on the left and click **Predict**.")

            # Show SHAP plot if it exists
            shap_path = MODELS_DIR / "shap_beeswarm.png"
            if shap_path.exists():
                st.write("**Feature importance (SHAP)**")
                st.image(str(shap_path), caption="Mean |SHAP value| — higher = more impact on prediction")



# TAB 3 — HISTORICAL ANALYSIS

with tab3:
    st.subheader("Historical Seismic Analysis")

    @st.cache_data(ttl=3600)
    def load_analytics():
        conn = get_connection()

        gr = pd.read_sql_query("""
            SELECT ROUND(magnitude,1) AS mag_bin, COUNT(*) AS event_count,
                   ROUND(LOG10(COUNT(*)),4) AS log10_count
            FROM earthquakes WHERE magnitude IS NOT NULL
            GROUP BY mag_bin ORDER BY mag_bin
        """, conn)

        regions = pd.read_sql_query("""
            SELECT r.name AS region, r.tectonic_setting,
                   COUNT(e.quake_id) AS quake_count,
                   ROUND(AVG(e.magnitude),2) AS avg_mag,
                   MAX(e.magnitude) AS max_mag
            FROM   earthquakes e JOIN regions r ON e.region_id=r.region_id
            GROUP  BY r.name ORDER BY quake_count DESC LIMIT 12
        """, conn)

        energy = pd.read_sql_query("""
            SELECT strftime('%Y-%m',event_time) AS month, COUNT(*) AS events,
                   ROUND(SUM(POWER(10.0,1.5*magnitude+4.8))/1e15,4) AS energy_PJ
            FROM   earthquakes WHERE magnitude IS NOT NULL
            GROUP  BY month ORDER BY month
        """, conn)

        depth = pd.read_sql_query("""
            SELECT e.depth_km, e.magnitude,
                   CASE WHEN e.depth_km<70 THEN 'Shallow'
                        WHEN e.depth_km<300 THEN 'Intermediate'
                        ELSE 'Deep' END AS depth_class,
                   r.tectonic_setting
            FROM   earthquakes e JOIN regions r ON e.region_id=r.region_id
            WHERE  e.depth_km IS NOT NULL
        """, conn)

        aftershock = pd.read_sql_query("""
            SELECT CASE WHEN e.magnitude>=7 THEN 'M7+'
                        WHEN e.magnitude>=6 THEN 'M6-6.9'
                        WHEN e.magnitude>=5.5 THEN 'M5.5-5.9'
                        ELSE 'M5-5.4' END AS mag_class,
                   COUNT(DISTINCT a.mainshock_id) AS mainshocks,
                   ROUND(AVG(a.had_damaging_after)*100,1) AS pct_damaging
            FROM   aftershock_sequences a
            JOIN   earthquakes e ON a.mainshock_id=e.quake_id
            GROUP  BY mag_class ORDER BY MIN(e.magnitude) DESC
        """, conn)

        conn.close()
        return gr, regions, energy, depth, aftershock

    gr, regions, energy, depth_df, aftershock = load_analytics()

    row1_c1, row1_c2 = st.columns(2)

    # Chart 1: Gutenberg-Richter
    with row1_c1:
        valid = gr[gr["log10_count"] > 0]
        if len(valid) >= 2:
            b = abs(round(np.polyfit(valid["mag_bin"], valid["log10_count"], 1)[0], 3))
        else:
            b = "N/A"
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=gr["mag_bin"], y=gr["log10_count"],
                                   mode="markers+lines", marker=dict(size=7, color="#6366f1")))
        fig1.update_layout(title=f"Gutenberg-Richter  (b = {b})",
                           xaxis_title="Magnitude", yaxis_title="log₁₀(count)",
                           height=340, template="plotly_white",
                           margin=dict(t=40,b=40,l=40,r=20))
        st.plotly_chart(fig1, use_container_width=True)

    # Chart 2: Top regions bar
    with row1_c2:
        fig2 = px.bar(regions, x="quake_count", y="region", orientation="h",
                      color="tectonic_setting", height=340,
                      title="Most Active Regions",
                      labels={"quake_count":"Event count","region":"Region"},
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig2.update_layout(template="plotly_white", margin=dict(t=40,b=40,l=10,r=20),
                           legend=dict(orientation="h", y=-0.25, font=dict(size=10)),
                           yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=True)

    row2_c1, row2_c2 = st.columns(2)

    # Chart 3: Monthly energy
    with row2_c1:
        fig3 = go.Figure(go.Bar(x=energy["month"], y=energy["energy_PJ"],
                                 marker_color="#6366f1", opacity=0.85))
        fig3.update_layout(title="Monthly Seismic Energy Release (PJ)",
                           xaxis_title="Month", yaxis_title="Energy (PJ)",
                           height=320, template="plotly_white",
                           margin=dict(t=40,b=60,l=50,r=20))
        st.plotly_chart(fig3, use_container_width=True)

    # Chart 4: Aftershock damage probability
    with row2_c2:
        if not aftershock.empty:
            colors = {"M7+":"#ef4444","M6-6.9":"#f59e0b",
                      "M5.5-5.9":"#6366f1","M5-5.4":"#22c55e"}
            fig4 = px.bar(aftershock, x="mag_class", y="pct_damaging",
                          color="mag_class", color_discrete_map=colors,
                          title="Aftershock Damage Rate by Mainshock Magnitude",
                          labels={"mag_class":"Magnitude class",
                                  "pct_damaging":"% with M4+ aftershock in 48h"},
                          height=320, text="pct_damaging")
            fig4.update_traces(texttemplate="%{text}%", textposition="outside")
            fig4.update_layout(template="plotly_white", showlegend=False,
                               yaxis=dict(range=[0,115]),
                               margin=dict(t=40,b=40,l=50,r=20))
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Run aftershocks.py to populate this chart.")

    # Depth scatter
    st.write("**Earthquake Depth vs Magnitude by Tectonic Setting**")
    color_map = {"Subduction zone":"#ef4444","Collision zone":"#f59e0b",
                 "Spreading ridge":"#6366f1","Transform fault":"#22c55e",
                 "Intraplate":"#6f87a8","Continental rift":"#0ea5e9"}
    fig5 = px.scatter(depth_df, x="depth_km", y="magnitude",
                      color="tectonic_setting", opacity=0.5,
                      color_discrete_map=color_map,
                      labels={"depth_km":"Depth (km)","magnitude":"Magnitude"},
                      height=380)
    fig5.update_traces(marker=dict(size=5))
    fig5.update_layout(template="plotly_white",
                       legend=dict(orientation="h", y=1.08, font=dict(size=11)),
                       margin=dict(t=10,b=40,l=50,r=20))
    st.plotly_chart(fig5, use_container_width=True)