"""
app.py — Seismic Risk Intelligence Dashboard
Tabs: Live Map | Risk Predictor | Historical Analysis | Batch Scorer
+ Always-visible sidebar with live global stats

Run: streamlit run src/app.py
"""

import sys, pickle, json, warnings
from pathlib import Path
from datetime import datetime, timedelta
import io

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import plotly.express       as px
import plotly.graph_objects as go
import requests, shap
import streamlit as st

from database   import get_connection
from setup_data import needs_setup, run_setup

# ── Page config ───────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Seismic Risk Intelligence",
                   page_icon="🌍", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
  .main{padding-top:1rem}
  .risk-high{background:#fee2e2;border-left:4px solid #ef4444;
             padding:14px 18px;border-radius:8px;margin-bottom:1rem}
  .risk-moderate{background:#fef3c7;border-left:4px solid #f59e0b;
                 padding:14px 18px;border-radius:8px;margin-bottom:1rem}
  .risk-low{background:#dcfce7;border-left:4px solid #22c55e;
            padding:14px 18px;border-radius:8px;margin-bottom:1rem}
  .banner{background:#1e293b;color:#f8fafc;border-radius:10px;
          padding:16px 22px;margin-bottom:1.2rem;font-size:14px;line-height:1.7}
</style>
""", unsafe_allow_html=True)

# ── Auto-setup ────────────────────────────────────────────────────────────────────
if needs_setup():
    st.title("🌍 Seismic Risk Intelligence System")
    st.info("First run — building database from live USGS data (~2 min).")
    box = st.empty()
    with st.spinner("Setting up..."):
        run_setup(status_container=box)
    st.success("Done — loading dashboard...")
    st.rerun()

# ── Constants ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
USGS_URL   = "https://earthquake.usgs.gov/fdsnws/event/1/query"
TECT_LIST  = ["Subduction zone","Collision zone","Spreading ridge",
              "Transform fault","Continental rift","Intraplate"]
PLATE_LIST = ["Convergent","Divergent","Transform","Intraplate"]
TECT_MAP   = {t:i for i,t in enumerate(TECT_LIST)}
PLATE_MAP  = {p:i for i,p in enumerate(PLATE_LIST)}

# ── Model ─────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODELS_DIR/"model.pkl",         "rb") as f: m  = pickle.load(f)
    with open(MODELS_DIR/"scaler.pkl",        "rb") as f: sc = pickle.load(f)
    with open(MODELS_DIR/"feature_names.pkl", "rb") as f: fc = pickle.load(f)
    return m, sc, fc, shap.TreeExplainer(m)

model_ok = (MODELS_DIR/"model.pkl").exists()
if model_ok:
    model, scaler, feat_cols, explainer = load_model()

def build_row(mag, dep, lat, lon, tect, plt_, rt, tsun):
    row = {"magnitude":mag, "depth_km":dep,
           "is_shallow":int(dep<70), "is_intermediate":int(70<=dep<300),
           "mag_class":int(min(4,max(0,int((mag-4.9)/0.5)))),
           "risk_tier":rt,
           "tectonic_enc":TECT_MAP.get(tect,3),
           "plate_enc":PLATE_MAP.get(plt_,0),
           "latitude":lat, "longitude":lon, "tsunami_flag":int(tsun)}
    return np.array([[row[f] for f in feat_cols]])

def predict_prob(mag, dep, lat, lon, tect, plt_, rt, tsun):
    X = build_row(mag,dep,lat,lon,tect,plt_,rt,tsun)
    return float(model.predict_proba(scaler.transform(X))[0][1])

# ── Shared USGS fetch ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=900)
def fetch_live(days_, min_mag_):
    end = datetime.utcnow()
    start = end - timedelta(days=days_)
    try:
        r = requests.get(USGS_URL, params={
            "format":"geojson",
            "starttime":start.strftime("%Y-%m-%d"),
            "endtime":end.strftime("%Y-%m-%d"),
            "minmagnitude":min_mag_,
            "limit":20000, "orderby":"time-asc"}, timeout=30)
        r.raise_for_status()
        rows = []
        for f in r.json().get("features",[]):
            p,c = f["properties"], f["geometry"]["coordinates"]
            rows.append({"id":f["id"],
                "time":datetime.utcfromtimestamp(p["time"]/1000).strftime("%Y-%m-%d %H:%M UTC"),
                "place":p.get("place","Unknown"),
                "magnitude":p.get("mag",0),
                "depth_km":c[2], "latitude":c[1], "longitude":c[0],
                "tsunami":p.get("tsunami",0),
                "alert":p.get("alert") or "none"})
        return pd.DataFrame(rows)
    except:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════════
# SIDEBAR — LIVE GLOBAL STATS
# ══════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌐 Live Global Stats")
    st.caption("Updates every 15 minutes · Source: USGS")

    sidebar_df = fetch_live(1, 4.0)   # last 24h, M4+

    if not sidebar_df.empty:
        st.metric("M4+ events (24h)", f"{len(sidebar_df):,}")

        max_row = sidebar_df.loc[sidebar_df["magnitude"].idxmax()]
        st.metric("Largest today", f"M{max_row['magnitude']:.1f}")
        st.caption(f"📍 {str(max_row['place'])[:35]}")

        tsun_count = int(sidebar_df["tsunami"].sum())
        if tsun_count > 0:
            st.error(f"⚠️ {tsun_count} tsunami alert(s) active")
        else:
            st.success("✅ No active tsunami alerts")

        # Most active region from DB
        try:
            conn_sb = get_connection()
            active = pd.read_sql_query("""
                SELECT r.name, COUNT(*) AS n
                FROM   earthquakes e
                JOIN   regions r ON e.region_id = r.region_id
                WHERE  e.event_time >= datetime('now','-1 day')
                GROUP  BY r.name ORDER BY n DESC LIMIT 1
            """, conn_sb)
            conn_sb.close()
            if not active.empty:
                st.metric("Most active region (24h)",
                          str(active.iloc[0]["name"])[:28],
                          f"{int(active.iloc[0]['n'])} events")
        except:
            pass

        # Hourly spark-line
        sidebar_df["hour"] = pd.to_datetime(sidebar_df["time"]).dt.hour
        hourly = sidebar_df.groupby("hour").size().reset_index(name="count")
        fig_spark = go.Figure(go.Scatter(
            x=hourly["hour"], y=hourly["count"], mode="lines",
            fill="tozeroy", line=dict(color="#6366f1",width=1.5),
            fillcolor="rgba(99,102,241,0.15)"))
        fig_spark.update_layout(height=90, margin=dict(l=0,r=0,t=4,b=0),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.caption("Hourly M4+ activity (last 24h)")
        st.plotly_chart(fig_spark, width = 'stretch')

        st.divider()
        if st.button("🔄 Refresh all data", width = 'stretch'):
            st.cache_data.clear()
            st.rerun()

        # M6+ events today
        big = sidebar_df[sidebar_df["magnitude"] >= 6.0]
        if not big.empty:
            st.write("**M6+ in last 24h**")
            for _, row in big.sort_values("magnitude",ascending=False).iterrows():
                st.markdown(f"🔴 **M{row['magnitude']:.1f}** — {str(row['place'])[:28]}")
    else:
        st.warning("USGS API unreachable")
        if st.button("🔄 Retry", width = 'stretch'):
            st.cache_data.clear()
            st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────────
st.title("🌍 Seismic Event Risk Intelligence System")
st.markdown("""<div class="banner">
Satellite operators, power grid managers, 
and insurers need up to 48 hours of advance 
warning before potentially damaging aftershocks. 
This system delivers those forecasts by combining 
live USGS earthquake data, physics-informed SQL analytics, 
and an XGBoost classifier with SHAP-based interpretability.


</div>""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Live Seismic Map",
    "⚡ Aftershock Risk Predictor",
    "📊 Historical Analysis",
    "📋 Batch Risk Scorer",
])


# ══════════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE MAP
# ══════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Real-time Global Seismic Activity")
    c1, c2, _ = st.columns([1,1,3])
    with c1: days    = st.selectbox("Time window",[1,3,7,14,30],index=2)
    with c2: min_mag = st.selectbox("Min magnitude",[4.0,5.0,6.0],index=0)

    with st.spinner("Fetching live USGS data..."):
        live_df = fetch_live(days, min_mag)

    if live_df.empty:
        st.warning("Cannot reach USGS API — check internet connection.")
    else:
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Total events",   f"{len(live_df):,}")
        m2.metric("Max magnitude",  f"M{live_df['magnitude'].max():.1f}")
        m3.metric("Avg depth",      f"{live_df['depth_km'].mean():.0f} km")
        m4.metric("Tsunami alerts", f"{int(live_df['tsunami'].sum())}")

        live_df["size"] = live_df["magnitude"] ** 2.8
        fig_map = px.scatter_geo(live_df,
            lat="latitude", lon="longitude",
            size="size", color="magnitude",
            color_continuous_scale="Reds",
            range_color=[min_mag, max(7.0, live_df["magnitude"].max())],
            hover_name="place",
            hover_data={"magnitude":True,"depth_km":True,"time":True,
                        "tsunami":True,"size":False,"latitude":False,"longitude":False},
            projection="natural earth", height=520)
        fig_map.update_layout(
            coloraxis_colorbar=dict(title="Magnitude",thickness=12),
            geo=dict(showland=True,landcolor="#f1f5f9",
                     showocean=True,oceancolor="#dbeafe",
                     showcountries=True,countrycolor="#cbd5e1"),
            margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_map, width = 'stretch')

        sig = live_df[live_df["magnitude"]>=5.5].sort_values(
            "magnitude",ascending=False).head(10)
        if not sig.empty:
            st.write("**M5.5+ events**")
            st.dataframe(sig[["time","place","magnitude","depth_km","tsunami","alert"]
                            ].reset_index(drop=True),
                         width = 'stretch', hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════════
# TAB 2 — RISK PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Aftershock Damage Risk Predictor")
    st.caption("Predicts probability of a damaging aftershock (M4+) within 48 hours.")

    if not model_ok:
        st.error("Run `python src/train_model.py` first.")
        st.stop()

    @st.cache_data(ttl=3600)
    def fetch_recent():
        end = datetime.utcnow(); start = end - timedelta(days=30)
        try:
            r = requests.get(USGS_URL, params={"format":"geojson",
                "starttime":start.strftime("%Y-%m-%d"),
                "endtime":end.strftime("%Y-%m-%d"),
                "minmagnitude":5.0,"limit":20,"orderby":"magnitude"}, timeout=20)
            r.raise_for_status()
            out = []
            for f in r.json().get("features",[]):
                p,c = f["properties"], f["geometry"]["coordinates"]
                out.append({"label":f"M{p.get('mag',0):.1f} — {p.get('place','')[:45]}",
                            "magnitude":p.get("mag",6.0),"depth_km":c[2],
                            "latitude":c[1],"longitude":c[0]})
            return out
        except: 
            return []

    recent = fetch_recent()

    DEFS = {"magnitude":6.5,"depth_km":35.0,"latitude":5.6,"longitude":125.0,
            "tectonic":"Subduction zone","plate":"Convergent","risk_tier":4,"tsunami":False}
    for k,v in DEFS.items():
        if k not in st.session_state: 
            st.session_state[k] = v

    if recent:
        opts = ["— manual entry —"] + [e["label"] for e in recent]
        chosen = st.selectbox("📡 Load a real recent M5+ earthquake", opts, index=0)
        if chosen != "— manual entry —":
            ev = next(e for e in recent if e["label"]==chosen)
            st.session_state.update({
                "magnitude":float(ev["magnitude"]),
                "depth_km":float(ev["depth_km"]),
                "latitude":float(ev["latitude"]),
                "longitude":float(ev["longitude"]),
                "tectonic":"Subduction zone","plate":"Convergent",
                "risk_tier":4,
                "tsunami":bool(ev["depth_km"]<100 and ev["magnitude"]>=6.5)})

    st.write("")
    
    # ── ROW 1: Two columns for Inputs (a) and Risk/SHAP (b) ──
    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.write("**Earthquake parameters**")
        mag  = st.slider("Magnitude",   4.5, 9.5,   float(st.session_state["magnitude"]), 0.1)
        dep  = st.slider("Depth (km)",  1.0, 700.0,  float(st.session_state["depth_km"]),  1.0)
        lat  = st.slider("Latitude",  -90.0,  90.0,  float(st.session_state["latitude"]),  0.1)
        lon  = st.slider("Longitude",-180.0, 180.0,  float(st.session_state["longitude"]), 0.1)
        tect = st.selectbox("Tectonic setting", TECT_LIST,
                             index=TECT_LIST.index(st.session_state["tectonic"]))
        plt_ = st.selectbox("Plate boundary",   PLATE_LIST,
                             index=PLATE_LIST.index(st.session_state["plate"]))
        rt   = st.selectbox("Region risk tier (1=low 4=high)", [1,2,3,4],
                             index=[1,2,3,4].index(st.session_state["risk_tier"]))
        tsun = st.checkbox("Tsunami potential", value=bool(st.session_state["tsunami"]))
        
        predict_btn = st.button("⚡ Predict Aftershock Risk",
                                width = 'stretch', type="primary")
        
        if predict_btn:
            st.session_state["prediction_made"] = True

    with col_out:
        st.write("**Risk assessment**")
        if st.session_state.get("prediction_made", False):
            X    = build_row(mag, dep, lat, lon, tect, plt_, rt, tsun)
            X_sc = scaler.transform(X)
            prob = float(model.predict_proba(X_sc)[0][1])

            if   prob>=0.7: label,css,emoji = "HIGH RISK",    "risk-high",    "🔴"
            elif prob>=0.4: label,css,emoji = "MODERATE RISK","risk-moderate","🟡"
            else:           label,css,emoji = "LOW RISK",     "risk-low",     "🟢"

            st.markdown(f"""<div class="{css}">
              <h2 style="margin:0 0 4px">{emoji} {label}</h2>
              <p style="font-size:2rem;font-weight:700;margin:0">{prob*100:.1f}%</p>
              <p style="color:#555;margin:4px 0 0;font-size:13px">
                Probability of M4+ aftershock within 48 hours</p>
            </div>""", unsafe_allow_html=True)

            # Gauge
            fig_g = go.Figure(go.Indicator(mode="gauge+number",value=round(prob*100,1),
                title={"text":"Aftershock damage probability (%)","font":{"size":13}},
                gauge={"axis":{"range":[0,100]},
                    "bar":{"color":"#ef4444" if prob>=0.7 else
                                   "#f59e0b" if prob>=0.4 else "#22c55e"},
                    "steps":[{"range":[0,40],"color":"#dcfce7"},
                             {"range":[40,70],"color":"#fef3c7"},
                             {"range":[70,100],"color":"#fee2e2"}],
                    "threshold":{"line":{"color":"#1e293b","width":3},
                                 "thickness":0.75,"value":prob*100}},
                number={"suffix":"%","font":{"size":34}}))
            fig_g.update_layout(height=240,margin=dict(t=40,b=0,l=20,r=20))
            st.plotly_chart(fig_g, width = 'stretch')

            # Live SHAP
            sv = explainer.shap_values(X_sc)[0]
            label_map = {
                "magnitude":"Magnitude","depth_km":"Depth (km)",
                "is_shallow":"Is shallow (<70km)","is_intermediate":"Is intermediate depth",
                "mag_class":"Magnitude class","risk_tier":"Region risk tier",
                "tectonic_enc":"Tectonic setting","plate_enc":"Plate boundary",
                "latitude":"Latitude","longitude":"Longitude",
                "tsunami_flag":"Tsunami potential"}
            shap_df = pd.DataFrame({"feature":feat_cols,"shap":sv,"value":X[0]})
            shap_df["label"] = shap_df["feature"].map(label_map).fillna(shap_df["feature"])
            shap_df = shap_df.sort_values("shap",key=abs,ascending=True)
            shap_df["color"] = shap_df["shap"].apply(
                lambda v: "#ef4444" if v>0 else "#6366f1")
            shap_df["ann"] = shap_df["value"].apply(lambda v: f"{v:.2f}")

            fig_s = go.Figure(go.Bar(
                x=shap_df["shap"], y=shap_df["label"],
                orientation="h", marker_color=shap_df["color"],
                text=shap_df["ann"], textposition="outside", textfont=dict(size=10)))
            fig_s.add_vline(x=0,line_width=1,line_color="#94a3b8")
            fig_s.update_layout(
                title="SHAP — why this score?  (red=raises risk · blue=lowers risk)",
                xaxis_title="SHAP value", height=360, template="plotly_white",
                margin=dict(t=50,b=30,l=10,r=90), font=dict(size=11))
            st.plotly_chart(fig_s, width = 'stretch')
        else:
            st.info("Set parameters and click **Predict**.")
            sp = MODELS_DIR/"shap_beeswarm.png"
            if sp.exists():
                st.write("**Overall feature importance (training data)**")
                st.image(str(sp), caption="Mean |SHAP| across all training mainshocks")

    # ── ROW 2: Full Width Section (c) ──
    if st.session_state.get("prediction_made", False):
        st.write("**Similar historical events — what actually happened?**")
        try:
            conn_sim = get_connection()
            sim = pd.read_sql_query(f"""
                SELECT e.magnitude, e.depth_km, e.place, e.event_time,
                       COALESCE(seq.had_damaging,0)      AS actual_outcome,
                       COALESCE(seq.aftershock_count,0)  AS aftershock_count,
                       ABS(e.magnitude - {mag}) +
                       ABS(e.depth_km  - {dep}) / 100.0  AS similarity_score
                FROM   earthquakes e
                LEFT JOIN (
                    SELECT mainshock_id,
                           MAX(had_damaging_after) AS had_damaging,
                           COUNT(*)                AS aftershock_count
                    FROM   aftershock_sequences
                    GROUP  BY mainshock_id
                ) seq ON seq.mainshock_id = e.quake_id
                WHERE  e.magnitude >= 5.0
                ORDER  BY similarity_score ASC
                LIMIT  5
            """, conn_sim)
            conn_sim.close()

            if not sim.empty:
                sim["outcome"] = sim["actual_outcome"].apply(
                    lambda v: "🔴 Damaging aftershock" if v==1 else "🟢 No damage")
                sim["event_time"] = sim["event_time"].astype(str).str[:10]
                st.dataframe(
                    sim[["magnitude","depth_km","place","event_time",
                         "aftershock_count","outcome"]].rename(columns={
                        "magnitude":"M","depth_km":"Depth km",
                        "place":"Location","event_time":"Date",
                        "aftershock_count":"Aftershocks",
                        "outcome":"Actual outcome"}),
                    width = 'stretch', hide_index=True)
                pct = sim["actual_outcome"].mean()*100
                st.caption(f"Among the 5 most similar events, "
                           f"{pct:.0f}% produced a damaging aftershock — "
                           f"model predicted {prob*100:.0f}%.")
        except Exception as e:
            st.caption(f"Could not load similar events: {e}")

        # ── ROW 3: Full Width Section (d) ──
        st.divider()
        st.write("### What-if Sensitivity Analysis")
        sweep_choice = st.selectbox("Sweep parameter",
            ["Magnitude (M5→M9)","Depth (0→700 km)","Risk tier (1→4)"],
            key="sweep_sel")

        if sweep_choice == "Magnitude (M5→M9)":
            xs    = np.arange(5.0, 9.1, 0.1)
            ys    = [predict_prob(m, dep, lat, lon, tect, plt_, rt, tsun) for m in xs]
            cur_x, xlabel, cur_lbl = mag, "Magnitude", f"Current M{mag}"
        elif sweep_choice == "Depth (0→700 km)":
            xs    = np.arange(0, 701, 10)
            ys    = [predict_prob(mag, d, lat, lon, tect, plt_, rt, tsun) for d in xs]
            cur_x, xlabel, cur_lbl = dep, "Depth (km)", f"Current {dep:.0f} km"
        else:
            xs    = [1, 2, 3, 4]
            ys    = [predict_prob(mag, dep, lat, lon, tect, plt_, t, tsun) for t in xs]
            cur_x, xlabel, cur_lbl = rt, "Risk tier", f"Current tier {rt}"

        fig_sw = go.Figure()
        fig_sw.add_trace(go.Scatter(
            x=xs, y=[p*100 for p in ys], mode="lines",
            line=dict(color="#6366f1",width=2.5),
            fill="tozeroy", fillcolor="rgba(99,102,241,0.1)"))
        fig_sw.add_vline(x=cur_x, line_dash="dash", line_color="#ef4444",
            annotation_text=cur_lbl, annotation_position="top right")
        fig_sw.add_hline(y=70, line_dash="dot", line_color="#ef4444",
            annotation_text="High risk (70%)")
        fig_sw.add_hline(y=40, line_dash="dot", line_color="#f59e0b",
            annotation_text="Moderate (40%)")
        fig_sw.update_layout(
            title=f"Risk probability as {xlabel} varies "
                  f"(all other parameters fixed)",
            xaxis_title=xlabel, yaxis_title="Damage probability (%)",
            yaxis=dict(range=[0,105]), height=340,
            template="plotly_white",
            margin=dict(t=45,b=40,l=55,r=20))
        st.plotly_chart(fig_sw, width = 'stretch')

# ══════════════════════════════════════════════════════════════════════════════════
# TAB 3 — HISTORICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Historical Seismic Analysis")

    rc1, _ = st.columns([1,4])
    with rc1:
        if st.button("🔄 Refresh analytics", width = 'stretch'):
            st.cache_data.clear()
            st.rerun()

    @st.cache_data(ttl=600)
    def load_analytics():
        conn = get_connection()

        gr = pd.read_sql_query("""
            SELECT ROUND(magnitude,1) AS mag_bin,
                   COUNT(*) AS event_count,
                   ROUND(LOG10(COUNT(*)),4) AS log10_count
            FROM   earthquakes WHERE magnitude IS NOT NULL
            GROUP  BY mag_bin ORDER BY mag_bin""", conn)

        regions = pd.read_sql_query("""
            SELECT r.name AS region, r.tectonic_setting,
                   COUNT(e.quake_id) AS quake_count,
                   ROUND(AVG(e.magnitude),2) AS avg_mag,
                   MAX(e.magnitude) AS max_mag
            FROM   earthquakes e JOIN regions r ON e.region_id=r.region_id
            GROUP  BY r.name ORDER BY quake_count DESC LIMIT 12""", conn)

        energy = pd.read_sql_query("""
            SELECT strftime('%Y-%m',event_time) AS month,
                   COUNT(*) AS events,
                   ROUND(SUM(POWER(10.0,1.5*magnitude+4.8))/1e15,4) AS energy_PJ
            FROM   earthquakes WHERE magnitude IS NOT NULL
            GROUP  BY month ORDER BY month""", conn)

        depth_df = pd.read_sql_query("""
            SELECT e.depth_km, e.magnitude,
                   CASE WHEN e.depth_km<70  THEN 'Shallow (<70 km)'
                        WHEN e.depth_km<300 THEN 'Intermediate (70-300 km)'
                        ELSE 'Deep (>300 km)' END AS depth_class,
                   r.tectonic_setting
            FROM   earthquakes e JOIN regions r ON e.region_id=r.region_id
            WHERE  e.depth_km IS NOT NULL""", conn)

        seq_count = conn.execute(
            "SELECT COUNT(*) FROM aftershock_sequences").fetchone()[0]

        aftershock = pd.read_sql_query("""
            SELECT CASE WHEN e.magnitude>=7   THEN 'M7+'
                        WHEN e.magnitude>=6   THEN 'M6-6.9'
                        WHEN e.magnitude>=5.5 THEN 'M5.5-5.9'
                        ELSE 'M5-5.4' END AS mag_class,
                   COUNT(DISTINCT a.mainshock_id) AS mainshocks,
                   ROUND(AVG(a.had_damaging_after)*100,1) AS pct_damaging
            FROM   aftershock_sequences a
            JOIN   earthquakes e ON a.mainshock_id=e.quake_id
            GROUP  BY mag_class ORDER BY MIN(e.magnitude) DESC""", conn)

        omori_opts = pd.read_sql_query("""
            SELECT DISTINCT e.quake_id, e.magnitude, e.place,
                   e.event_time, e.latitude, e.longitude
            FROM   aftershock_sequences a
            JOIN   earthquakes e ON a.mainshock_id=e.quake_id
            WHERE  e.magnitude>=6.0
            ORDER  BY e.magnitude DESC LIMIT 15""", conn)

        daily = pd.read_sql_query("""
            SELECT DATE(event_time) AS day, COUNT(*) AS events
            FROM   earthquakes WHERE magnitude>=4.0
            GROUP  BY day ORDER BY day""", conn)

        conn.close()
        return gr,regions,energy,depth_df,aftershock,omori_opts,seq_count,daily

    gr,regions,energy,depth_df,aftershock,omori_opts,seq_count,daily = load_analytics()

    # Row 1
    r1a, r1b = st.columns(2)

    with r1a:
        valid = gr[gr["log10_count"]>0]
        b = abs(round(np.polyfit(valid["mag_bin"],valid["log10_count"],1)[0],3)) \
            if len(valid)>=2 else "N/A"
        f1 = go.Figure()
        f1.add_trace(go.Scatter(x=gr["mag_bin"],y=gr["log10_count"],
            mode="markers+lines",marker=dict(size=7,color="#6366f1"),name="Observed"))
        if len(valid)>=2:
            fit = np.poly1d(np.polyfit(
                valid["mag_bin"],valid["log10_count"],1))(valid["mag_bin"])
            f1.add_trace(go.Scatter(x=valid["mag_bin"],y=fit,mode="lines",
                name=f"G-R fit (b={b})",
                line=dict(dash="dash",color="#ef4444",width=2)))
        f1.update_layout(title=f"Gutenberg-Richter Law  ",
            xaxis_title="Magnitude",yaxis_title="log₁₀(count)",height=340,
            template="plotly_white",margin=dict(t=45,b=40,l=45,r=20),
            legend=dict(orientation="h",y=1.12))
        st.plotly_chart(f1, width = 'stretch')
        st.caption("A straight line confirms data quality and G-R consistency.")

    with r1b:
        f2 = px.bar(regions, x="quake_count", y="region",
            orientation="h", color="tectonic_setting", height=340,
            title="Most Active Tectonic Regions",
            labels={"quake_count":"Event count","region":""},
            color_discrete_sequence=px.colors.qualitative.Set2)
        f2.update_layout(template="plotly_white",
            margin=dict(t=45,b=40,l=10,r=20),
            legend=dict(orientation="h",y=-0.28,font=dict(size=10)),
            yaxis=dict(autorange="reversed"))
        st.plotly_chart(f2, width = 'stretch')
        st.caption("Indonesia and the Philippines show the highest activity, consistent with their location along the Ring of Fire.")

    # Row 2
    r2a, r2b = st.columns(2)

    with r2a:
        f3 = go.Figure(go.Bar(x=energy["month"],y=energy["energy_PJ"],
            marker_color="#6366f1",opacity=0.85))
        f3.update_layout(title="Monthly Seismic Energy Release  [E=10^(1.5M+4.8)]",
            xaxis_title="Month",yaxis_title="Energy (PJ)",height=320,
            template="plotly_white",margin=dict(t=45,b=60,l=55,r=20))
        st.plotly_chart(f3, width = 'stretch')
        st.caption("One M7+ event can exceed the cumulative energy of all other months.")

    with r2b:
        if not aftershock.empty:
            cmap = {"M7+":"#ef4444","M6-6.9":"#f59e0b",
                    "M5.5-5.9":"#6366f1","M5-5.4":"#22c55e"}
            f4 = px.bar(aftershock, x="mag_class", y="pct_damaging",
                color="mag_class", color_discrete_map=cmap, height=320,
                text="pct_damaging",
                title="% of Mainshocks → Damaging Aftershock within 48h",
                labels={"mag_class":"Magnitude class",
                        "pct_damaging":"% with M4+ aftershock"})
            f4.update_traces(texttemplate="%{text}%",textposition="outside")
            f4.update_layout(template="plotly_white",showlegend=False,
                yaxis=dict(range=[0,115]),margin=dict(t=45,b=40,l=55,r=20))
            st.plotly_chart(f4, width = 'stretch')
            st.caption("Near-certain probabilities for M7+ events align with Bath's Law.")
        else:
            msg = ("Click **🔄 Refresh analytics** above."
                   if seq_count > 0 else "Run `python src/aftershocks.py` first.")
            st.info(f"Database contains {seq_count:,} sequence records. {msg}")

    # Daily seismicity trend
    st.divider()
    st.write("### Daily Seismicity Trend (M4+)")
    if not daily.empty:
        daily["day"] = pd.to_datetime(daily["day"])
        daily["rolling7"] = daily["events"].rolling(7,min_periods=1).mean().round(1)
        fig_d = go.Figure()
        fig_d.add_trace(go.Bar(x=daily["day"],y=daily["events"],
            name="Daily M4+ count",marker_color="#cbd5e1",opacity=0.7))
        fig_d.add_trace(go.Scatter(x=daily["day"],y=daily["rolling7"],
            mode="lines",name="7-day rolling avg",
            line=dict(color="#6366f1",width=2.5)))
        fig_d.update_layout(
            title="Daily M4+ Event Count with 7-day Rolling Average",
            xaxis_title="Date",yaxis_title="Events",height=320,
            template="plotly_white",
            legend=dict(orientation="h",y=1.1),
            margin=dict(t=50,b=40,l=55,r=20))
        st.plotly_chart(fig_d, width = 'stretch')
        st.caption("Spikes above the rolling average indicate aftershock sequences "
                   "or swarm activity following a significant mainshock.")

    # Omori-Utsu + Sequence Map
    st.divider()
    st.write("### Omori-Utsu Aftershock Decay & Spatial Distribution")

    if omori_opts.empty:
        msg = ("Click **🔄 Refresh analytics** above."
               if seq_count > 0 else "Run `python src/aftershocks.py` first.")
        st.info(f"Database contains {seq_count:,} sequence records. {msg}")
    else:
        omori_opts["lbl"] = omori_opts.apply(
            lambda r: f"M{r['magnitude']:.1f} | {str(r['place'])[:40]} | "
                      f"{str(r['event_time'])[:10]}", axis=1)
        chosen_lbl = st.selectbox("Select mainshock", omori_opts["lbl"].tolist())
        chosen_row = omori_opts[omori_opts["lbl"]==chosen_lbl].iloc[0]
        chosen_id  = chosen_row["quake_id"]
        chosen_mag = chosen_row["magnitude"]
        main_lat   = float(chosen_row["latitude"])
        main_lon   = float(chosen_row["longitude"])

        @st.cache_data(ttl=600)
        def load_sequence(qid):
            conn = get_connection()
            decay = pd.read_sql_query(f"""
                SELECT ROUND(a.delta_hours/6)*6 AS hour_bin,
                       COUNT(*) AS aftershock_count
                FROM   aftershock_sequences a
                WHERE  a.mainshock_id='{qid}'
                GROUP  BY hour_bin ORDER BY hour_bin""", conn)
            seq_map = pd.read_sql_query(f"""
                SELECT e.latitude, e.longitude, e.magnitude, a.delta_hours,
                       CASE WHEN a.delta_hours<=24  THEN '0-24h'
                            WHEN a.delta_hours<=72  THEN '24-72h'
                            WHEN a.delta_hours<=120 THEN '72-120h'
                            ELSE '120h+' END AS time_bin
                FROM   aftershock_sequences a
                JOIN   earthquakes e ON a.aftershock_id=e.quake_id
                WHERE  a.mainshock_id='{qid}'""", conn)
            conn.close()
            return decay, seq_map

        decay_df, seq_map_df = load_sequence(chosen_id)
        decay_df = decay_df[decay_df["hour_bin"]<=168]

        om_col, map_col = st.columns(2)

        with om_col:
            if decay_df.empty:
                st.warning("No decay data for this event.")
            else:
                t   = decay_df["hour_bin"].replace(0,1).values
                fit = decay_df["aftershock_count"].iloc[0]*2.0 / (t+2)**1.1
                fo  = go.Figure()
                fo.add_trace(go.Bar(
                    x=decay_df["hour_bin"], y=decay_df["aftershock_count"],
                    name="Observed / 6h", marker_color="#6366f1", opacity=0.75))
                fo.add_trace(go.Scatter(
                    x=decay_df["hour_bin"], y=fit, mode="lines",
                    name="Omori-Utsu K/(t+c)^p  (p≈1.1)",
                    line=dict(color="#ef4444",width=2.5,dash="dash")))
                fo.update_layout(
                    title=f"Aftershock Decay — M{chosen_mag:.1f}",
                    xaxis_title="Hours after mainshock",
                    yaxis_title="Aftershocks per 6h", height=380,
                    template="plotly_white",
                    legend=dict(orientation="h",y=1.12),
                    margin=dict(t=50,b=50,l=55,r=20))
                st.plotly_chart(fo, width = 'stretch')
                st.caption("A steep initial decay followed by gradual flattening, reflecting the characteristic Omori-Utsu pattern.")

        with map_col:
            if seq_map_df.empty:
                st.warning("No spatial data for this sequence.")
            else:
                time_colors = {"0-24h":"#ef4444","24-72h":"#f59e0b",
                               "72-120h":"#6366f1","120h+":"#94a3b8"}
                seq_map_df["size"] = seq_map_df["magnitude"]**2.2

                fig_seq = go.Figure()
                for tbin, grp in seq_map_df.groupby("time_bin"):
                    fig_seq.add_trace(go.Scattergeo(
                        lat=grp["latitude"], lon=grp["longitude"],
                        mode="markers", name=tbin,
                        marker=dict(size=grp["size"],
                                    color=time_colors.get(tbin,"#94a3b8"),
                                    opacity=0.7),
                        hovertext=grp.apply(
                            lambda r: f"M{r['magnitude']:.1f} | "
                                      f"{r['delta_hours']:.0f}h after", axis=1)))

                fig_seq.add_trace(go.Scattergeo(
                    lat=[main_lat], lon=[main_lon],
                    mode="markers", name="Mainshock",
                    marker=dict(size=18, color="#1e293b", symbol="star",
                                line=dict(color="white",width=1.5))))

                fig_seq.update_layout(
                    title=f"Spatial Distribution — M{chosen_mag:.1f} Sequence",
                    height=380,
                    geo=dict(showland=True, landcolor="#f1f5f9",
                             showocean=True, oceancolor="#dbeafe",
                             showcountries=True, countrycolor="#cbd5e1",
                             center=dict(lat=main_lat, lon=main_lon),
                             projection_scale=8),
                    legend=dict(orientation="h",y=-0.12,font=dict(size=10)),
                    margin=dict(l=0,r=0,t=45,b=0))
                st.plotly_chart(fig_seq, width = 'stretch')
                st.caption("Colour shows time elapsed since mainshock. "
                           "Aftershocks cluster within 1–2 fault lengths of the rupture zone.")

    # Depth scatter
    st.divider()
    st.write("### Depth vs Magnitude by Tectonic Setting")
    cmap2 = {"Subduction zone":"#ef4444","Collision zone":"#f59e0b",
             "Spreading ridge":"#6366f1","Transform fault":"#22c55e",
             "Intraplate":"#94a3b8","Continental rift":"#0ea5e9"}
    f5 = px.scatter(depth_df, x="depth_km", y="magnitude",
        color="tectonic_setting", opacity=0.5,
        color_discrete_map=cmap2, height=380,
        labels={"depth_km":"Depth (km)","magnitude":"Magnitude",
                "tectonic_setting":"Setting"})
    f5.update_traces(marker=dict(size=5))
    f5.update_layout(template="plotly_white",
        legend=dict(orientation="h",y=1.08,font=dict(size=11)),
        margin=dict(t=10,b=50,l=55,r=20))
    st.plotly_chart(f5, width = 'stretch')


# ══════════════════════════════════════════════════════════════════════════════════
# TAB 4 — BATCH RISK SCORER
# ══════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Batch Aftershock Risk Scorer")
    st.caption("Score multiple earthquakes at once. Upload a CSV or paste data below.")

    if not model_ok:
        st.error("Run `python src/train_model.py` first.")
        st.stop()
    st.write("**CSV format required:**")
    st.code("magnitude,depth_km,latitude,longitude,tectonic_setting,"
            "plate_boundary_type,risk_tier,tsunami_flag")

    EXAMPLE_CSV = (
        "magnitude,depth_km,latitude,longitude,"
        "tectonic_setting,plate_boundary_type,risk_tier,tsunami_flag\n"
        "7.8,35.0,5.6,125.1,Subduction zone,Convergent,4,1\n"
        "6.5,120.0,35.5,140.2,Subduction zone,Convergent,4,0\n"
        "5.2,10.0,38.0,30.5,Collision zone,Convergent,3,0\n"
        "5.8,55.0,-33.0,-71.0,Subduction zone,Convergent,4,0\n"
        "6.1,280.0,-7.5,110.0,Subduction zone,Convergent,4,0"
    )

    mode = st.radio("Input method",
        ["📋 Paste CSV","📁 Upload CSV","⚡ Use example data"], horizontal=True)

    raw_df = None
    if mode == "📋 Paste CSV":
        text = st.text_area("Paste CSV here", height=160)
        if text.strip():
            try:    raw_df = pd.read_csv(io.StringIO(text))
            except: st.error("Could not parse — check format.")
    elif mode == "📁 Upload CSV":
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up:
            try:    raw_df = pd.read_csv(up)
            except: st.error("Could not read file.")
    else:
        raw_df = pd.read_csv(io.StringIO(EXAMPLE_CSV))
        st.info("Using 5 example earthquakes of varying magnitude and depth.")

    if raw_df is not None and not raw_df.empty:
        st.write(f"**{len(raw_df)} earthquakes loaded**")
        st.dataframe(raw_df, width = 'stretch', hide_index=True)

        if st.button("⚡ Score all", type="primary", width = 'stretch'):
            required = ["magnitude","depth_km","latitude","longitude",
                        "tectonic_setting","plate_boundary_type",
                        "risk_tier","tsunami_flag"]
            missing = [c for c in required if c not in raw_df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                feat_label = {
                    "magnitude":"Magnitude","depth_km":"Depth",
                    "is_shallow":"Shallow depth","is_intermediate":"Interm. depth",
                    "mag_class":"Mag class","risk_tier":"Risk tier",
                    "tectonic_enc":"Tectonic setting","plate_enc":"Plate boundary",
                    "latitude":"Latitude","longitude":"Longitude",
                    "tsunami_flag":"Tsunami potential"}

                results = []
                for _, row in raw_df.iterrows():
                    try:
                        X    = build_row(float(row["magnitude"]),float(row["depth_km"]),
                                         float(row["latitude"]),float(row["longitude"]),
                                         str(row["tectonic_setting"]),
                                         str(row["plate_boundary_type"]),
                                         int(row["risk_tier"]),int(row["tsunami_flag"]))
                        X_sc = scaler.transform(X)
                        prob = float(model.predict_proba(X_sc)[0][1])

                        sv       = explainer.shap_values(X_sc)[0]
                        top_feat = feat_cols[int(np.argmax(np.abs(sv)))]
                        top_lbl  = feat_label.get(top_feat, top_feat)

                        label = ("HIGH RISK"     if prob>=0.7 else
                                 "MODERATE RISK" if prob>=0.4 else "LOW RISK")
                        results.append({"probability_%":round(prob*100,1),
                                        "risk_level":label,
                                        "top_driver":top_lbl})
                    except Exception as e:
                        results.append({"probability_%":None,
                                        "risk_level":f"Error: {e}",
                                        "top_driver":"—"})

                out_df = pd.concat([raw_df.reset_index(drop=True),
                                    pd.DataFrame(results)], axis=1)

                def colour_risk(val):
                    if "HIGH"     in str(val): return "background-color:#fee2e2"
                    if "MODERATE" in str(val): return "background-color:#fef3c7"
                    if "LOW"      in str(val): return "background-color:#dcfce7"
                    return ""

                st.write("**Results** — `top_driver` shows the key factor per row")
                st.dataframe(
                    out_df.style.map(colour_risk, subset=["risk_level"]),
                    width = 'stretch', hide_index=True)

                # Distribution chart
                rc = (pd.DataFrame(results)["risk_level"]
                      .value_counts().reset_index())
                rc.columns = ["Risk Level","Count"]
                rc["Risk Level"] = pd.Categorical(
                    rc["Risk Level"],
                    categories=["HIGH RISK","MODERATE RISK","LOW RISK"],
                    ordered=True)
                rc = rc.sort_values("Risk Level")
                fb = px.bar(rc, x="Risk Level", y="Count",
                    color="Risk Level",
                    color_discrete_map={"HIGH RISK":"#ef4444",
                                        "MODERATE RISK":"#f59e0b",
                                        "LOW RISK":"#22c55e"},
                    title="Batch Risk Distribution",
                    height=280, text="Count")
                fb.update_traces(textposition="outside")
                fb.update_layout(showlegend=False, template="plotly_white",
                    yaxis=dict(range=[0,len(raw_df)+1]),
                    margin=dict(t=45,b=40,l=40,r=20))
                st.plotly_chart(fb, width = 'stretch')

                st.download_button("⬇️ Download results CSV",
                    data=out_df.to_csv(index=False),
                    file_name="seismic_risk_scores.csv",
                    mime="text/csv", width = 'stretch')
    else:
        if mode == "📋 Paste CSV":
            st.write("**Expected format:**")
            st.code(EXAMPLE_CSV)