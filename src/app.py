"""
app.py — Seismic Risk Intelligence Dashboard (full version)
Tabs: Live Map | Risk Predictor | Historical Analysis | Batch Scorer

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
import plotly.express      as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests, shap
import streamlit as st

from database   import get_connection
from setup_data import needs_setup, run_setup

# ── Page config ───────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Seismic Risk Intelligence",
                   page_icon="🌍", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
  .main{padding-top:1rem}
  .risk-high    {background:#fee2e2;border-left:4px solid #ef4444;padding:14px 18px;border-radius:8px;margin-bottom:1rem}
  .risk-moderate{background:#fef3c7;border-left:4px solid #f59e0b;padding:14px 18px;border-radius:8px;margin-bottom:1rem}
  .risk-low     {background:#dcfce7;border-left:4px solid #22c55e;padding:14px 18px;border-radius:8px;margin-bottom:1rem}
  .metric-pill  {background:#f1f5f9;border:1px solid #e2e8f0;border-radius:8px;padding:12px 16px;text-align:center}
  .banner       {background:#1e293b;color:#f8fafc;border-radius:10px;padding:16px 22px;margin-bottom:1.2rem;font-size:14px;line-height:1.7}
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

# ── Load model ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODELS_DIR/"model.pkl",         "rb") as f: m  = pickle.load(f)
    with open(MODELS_DIR/"scaler.pkl",        "rb") as f: sc = pickle.load(f)
    with open(MODELS_DIR/"feature_names.pkl", "rb") as f: fc = pickle.load(f)
    return m, sc, fc, shap.TreeExplainer(m)

model_ok = (MODELS_DIR/"model.pkl").exists()
if model_ok:
    model, scaler, feat_cols, explainer = load_model()

def load_metrics():
    p = MODELS_DIR/"metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}

def build_feature_row(mag, dep, lat, lon, tect, plt_, rt, tsun):
    row = {
        "magnitude":       mag,
        "depth_km":        dep,
        "is_shallow":      int(dep<70),
        "is_intermediate": int(70<=dep<300),
        "mag_class":       int(min(4,max(0,int((mag-4.9)/0.5)))),
        "risk_tier":       rt,
        "tectonic_enc":    TECT_MAP.get(tect,3),
        "plate_enc":       PLATE_MAP.get(plt_,0),
        "latitude":        lat,
        "longitude":       lon,
        "tsunami_flag":    int(tsun),
    }
    return np.array([[row[f] for f in feat_cols]])

# ── Header ────────────────────────────────────────────────────────────────────────
st.title("🌍 Seismic Event Risk Intelligence System")
st.markdown("""<div class="banner">
Satellite operators, power grid managers, 
and insurers need up to 48 hours of advance warning 
before potentially damaging aftershocks. 
This system delivers those forecasts by combining live USGS earthquake data, 
physics-informed SQL analytics, and an XGBoost classifier with SHAP-based interpretability.
</div>""", unsafe_allow_html=True)

tab1,tab2,tab3,tab4 = st.tabs([
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
    c1,c2,c3 = st.columns([1,1,2])
    with c1: days    = st.selectbox("Time window",[1,3,7,14,30],index=2)
    with c2: min_mag = st.selectbox("Min magnitude",[4.0,5.0,6.0],index=0)
    with c3:
        st.write("")
        if st.button("🔄 Refresh",use_container_width=True):
            st.cache_data.clear()

    @st.cache_data(ttl=1800)
    def fetch_live(d,m):
        end=datetime.utcnow(); start=end-timedelta(days=d)
        try:
            r=requests.get(USGS_URL,params={"format":"geojson",
                "starttime":start.strftime("%Y-%m-%d"),"endtime":end.strftime("%Y-%m-%d"),
                "minmagnitude":m,"limit":20000,"orderby":"time-asc"},timeout=30)
            r.raise_for_status()
            rows=[]
            for f in r.json().get("features",[]):
                p,c=f["properties"],f["geometry"]["coordinates"]
                rows.append({"id":f["id"],
                    "time":datetime.utcfromtimestamp(p["time"]/1000).strftime("%Y-%m-%d %H:%M UTC"),
                    "place":p.get("place","Unknown"),"magnitude":p.get("mag",0),
                    "depth_km":c[2],"latitude":c[1],"longitude":c[0],
                    "tsunami":p.get("tsunami",0),"alert":p.get("alert") or "none"})
            return pd.DataFrame(rows)
        except: return pd.DataFrame()

    with st.spinner("Fetching live USGS data..."):
        live_df=fetch_live(days,min_mag)

    if live_df.empty:
        st.warning("Cannot reach USGS API — check internet connection.")
    else:
        m1,m2,m3,m4=st.columns(4)
        m1.metric("Total events",f"{len(live_df):,}")
        m2.metric("Max magnitude",f"M{live_df['magnitude'].max():.1f}")
        m3.metric("Avg depth",f"{live_df['depth_km'].mean():.0f} km")
        m4.metric("Tsunami alerts",f"{int(live_df['tsunami'].sum())}")
        live_df["size"]=live_df["magnitude"]**2.8
        fig=px.scatter_geo(live_df,lat="latitude",lon="longitude",size="size",
            color="magnitude",color_continuous_scale="Reds",
            range_color=[min_mag,max(7.0,live_df["magnitude"].max())],
            hover_name="place",
            hover_data={"magnitude":True,"depth_km":True,"time":True,
                        "tsunami":True,"size":False,"latitude":False,"longitude":False},
            projection="natural earth",height=520)
        fig.update_layout(
            coloraxis_colorbar=dict(title="Magnitude",thickness=12),
            geo=dict(showland=True,landcolor="#f1f5f9",showocean=True,
                     oceancolor="#dbeafe",showcountries=True,countrycolor="#cbd5e1"),
            margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig,use_container_width=True)
        sig=live_df[live_df["magnitude"]>=5.5].sort_values("magnitude",ascending=False).head(10)
        if not sig.empty:
            st.write("**M5.5+ events**")
            st.dataframe(sig[["time","place","magnitude","depth_km","tsunami","alert"]
                            ].reset_index(drop=True),use_container_width=True,hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════════
# TAB 2 — RISK PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Aftershock Damage Risk Predictor")
    st.caption("Predicts probability of a damaging aftershock (M4+) within 48 hours.")

    if not model_ok:
        st.error("Run `python src/train_model.py` first.")
        st.stop()

    # Load recent events for dropdown
    @st.cache_data(ttl=3600)
    def fetch_recent():
        end=datetime.utcnow(); start=end-timedelta(days=30)
        try:
            r=requests.get(USGS_URL,params={"format":"geojson",
                "starttime":start.strftime("%Y-%m-%d"),"endtime":end.strftime("%Y-%m-%d"),
                "minmagnitude":5.0,"limit":20,"orderby":"magnitude"},timeout=20)
            r.raise_for_status()
            out=[]
            for f in r.json().get("features",[]):
                p,c=f["properties"],f["geometry"]["coordinates"]
                out.append({"label":f"M{p.get('mag',0):.1f} — {p.get('place','')[:45]}",
                            "magnitude":p.get("mag",6.0),"depth_km":c[2],
                            "latitude":c[1],"longitude":c[0]})
            return out
        except: return []

    recent=fetch_recent()

    # Session state
    DEFS={"magnitude":6.5,"depth_km":35.0,"latitude":5.6,"longitude":125.0,
          "tectonic":"Subduction zone","plate":"Convergent","risk_tier":4,"tsunami":False}
    for k,v in DEFS.items():
        if k not in st.session_state: st.session_state[k]=v

    if recent:
        opts=["— manual entry —"]+[e["label"] for e in recent]
        chosen=st.selectbox("📡 Load a real recent M5+ earthquake",opts,index=0)
        if chosen!="— manual entry —":
            ev=next(e for e in recent if e["label"]==chosen)
            st.session_state.update({"magnitude":float(ev["magnitude"]),
                "depth_km":float(ev["depth_km"]),"latitude":float(ev["latitude"]),
                "longitude":float(ev["longitude"]),"tectonic":"Subduction zone",
                "plate":"Convergent","risk_tier":4,
                "tsunami":bool(ev["depth_km"]<100 and ev["magnitude"]>=6.5)})

    st.write("")
    col_in,col_out=st.columns([1,1],gap="large")

    with col_in:
        st.write("**Earthquake parameters**")
        mag  = st.slider("Magnitude",    4.5,9.5, float(st.session_state["magnitude"]),0.1)
        dep  = st.slider("Depth (km)",   1.0,700.0,float(st.session_state["depth_km"]),1.0)
        lat  = st.slider("Latitude",   -90.0,90.0, float(st.session_state["latitude"]),0.1)
        lon  = st.slider("Longitude", -180.0,180.0,float(st.session_state["longitude"]),0.1)
        tect = st.selectbox("Tectonic setting",  TECT_LIST,
                             index=TECT_LIST.index(st.session_state["tectonic"]))
        plt_ = st.selectbox("Plate boundary",    PLATE_LIST,
                             index=PLATE_LIST.index(st.session_state["plate"]))
        rt   = st.selectbox("Region risk tier (1=low 4=high)",[1,2,3,4],
                             index=[1,2,3,4].index(st.session_state["risk_tier"]))
        tsun = st.checkbox("Tsunami potential",value=bool(st.session_state["tsunami"]))
        predict_btn=st.button("⚡ Predict Aftershock Risk",
                               use_container_width=True,type="primary")

    with col_out:
        st.write("**Risk assessment**")
        if predict_btn:
            X    = build_feature_row(mag,dep,lat,lon,tect,plt_,rt,tsun)
            X_sc = scaler.transform(X)
            prob = model.predict_proba(X_sc)[0][1]

            if prob>=0.7:   label,css,emoji="HIGH RISK",    "risk-high",    "🔴"
            elif prob>=0.4: label,css,emoji="MODERATE RISK","risk-moderate","🟡"
            else:           label,css,emoji="LOW RISK",     "risk-low",     "🟢"

            st.markdown(f"""<div class="{css}">
              <h2 style="margin:0 0 4px">{emoji} {label}</h2>
              <p style="font-size:2rem;font-weight:700;margin:0">{prob*100:.1f}%</p>
              <p style="color:#555;margin:4px 0 0;font-size:13px">
                Probability of M4+ aftershock within 48 hours</p>
            </div>""",unsafe_allow_html=True)

            # Gauge
            fig_g=go.Figure(go.Indicator(mode="gauge+number",value=round(prob*100,1),
                title={"text":"Aftershock damage probability (%)","font":{"size":13}},
                gauge={"axis":{"range":[0,100]},
                    "bar":{"color":"#ef4444" if prob>=0.7 else "#f59e0b" if prob>=0.4 else "#22c55e"},
                    "steps":[{"range":[0,40],"color":"#dcfce7"},
                             {"range":[40,70],"color":"#fef3c7"},
                             {"range":[70,100],"color":"#fee2e2"}],
                    "threshold":{"line":{"color":"#1e293b","width":3},
                                 "thickness":0.75,"value":prob*100}},
                number={"suffix":"%","font":{"size":34}}))
            fig_g.update_layout(height=240,margin=dict(t=40,b=0,l=20,r=20))
            st.plotly_chart(fig_g,use_container_width=True)

            # Live SHAP bar chart
            sv   = explainer.shap_values(X_sc)[0]
            label_map={"magnitude":"Magnitude","depth_km":"Depth (km)",
                "is_shallow":"Is shallow (<70km)","is_intermediate":"Is intermediate depth",
                "mag_class":"Magnitude class","risk_tier":"Region risk tier",
                "tectonic_enc":"Tectonic setting","plate_enc":"Plate boundary",
                "latitude":"Latitude","longitude":"Longitude","tsunami_flag":"Tsunami potential"}
            shap_df=pd.DataFrame({"feature":feat_cols,"shap":sv,"value":X[0]})
            shap_df["label"]=shap_df["feature"].map(label_map).fillna(shap_df["feature"])
            shap_df=shap_df.sort_values("shap",key=abs,ascending=True)
            shap_df["color"]=shap_df["shap"].apply(lambda v:"#ef4444" if v>0 else "#6366f1")
            shap_df["ann"]=shap_df["value"].apply(lambda v:f"{v:.2f}")

            fig_s=go.Figure(go.Bar(x=shap_df["shap"],y=shap_df["label"],
                orientation="h",marker_color=shap_df["color"],
                text=shap_df["ann"],textposition="outside",textfont=dict(size=10)))
            fig_s.add_vline(x=0,line_width=1,line_color="#94a3b8")
            fig_s.update_layout(
                title="SHAP — why this risk score?  (red=raises risk, blue=lowers risk)",
                xaxis_title="SHAP value",height=360,template="plotly_white",
                margin=dict(t=50,b=30,l=10,r=90),font=dict(size=11))
            st.plotly_chart(fig_s,use_container_width=True)

            # Sensitivity sweep: magnitude M5→M9, all else fixed
            st.write("**Sensitivity — how does risk change with magnitude?**")
            mags_sweep = np.arange(5.0,9.1,0.1)
            probs_sweep = []
            for ms in mags_sweep:
                Xs=build_feature_row(ms,dep,lat,lon,tect,plt_,rt,tsun)
                probs_sweep.append(model.predict_proba(scaler.transform(Xs))[0][1])
            fig_sw=go.Figure()
            fig_sw.add_trace(go.Scatter(x=mags_sweep,y=[p*100 for p in probs_sweep],
                mode="lines",line=dict(color="#6366f1",width=2.5),
                fill="tozeroy",fillcolor="rgba(99,102,241,0.1)"))
            fig_sw.add_vline(x=mag,line_dash="dash",line_color="#ef4444",
                annotation_text=f"Current M{mag}",annotation_position="top right")
            fig_sw.add_hline(y=70,line_dash="dot",line_color="#ef4444",
                annotation_text="High risk threshold")
            fig_sw.add_hline(y=40,line_dash="dot",line_color="#f59e0b",
                annotation_text="Moderate threshold")
            fig_sw.update_layout(title="Risk probability vs Magnitude (all other params fixed)",
                xaxis_title="Magnitude",yaxis_title="Damage probability (%)",
                yaxis=dict(range=[0,105]),height=320,template="plotly_white",
                margin=dict(t=45,b=40,l=55,r=20))
            st.plotly_chart(fig_sw,use_container_width=True)

        else:
            st.info("Set parameters and click **Predict**.")
            sp=MODELS_DIR/"shap_beeswarm.png"
            if sp.exists():
                st.write("**Overall feature importance (training data)**")
                st.image(str(sp),caption="Mean |SHAP| across all training mainshocks")


# ══════════════════════════════════════════════════════════════════════════════════
# TAB 3 — HISTORICAL ANALYSIS + MODEL METRICS
# ══════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Historical Seismic Analysis")

    @st.cache_data(ttl=3600)
    def load_analytics():
        conn=get_connection()
        gr=pd.read_sql_query("""
            SELECT ROUND(magnitude,1) AS mag_bin, COUNT(*) AS event_count,
                   ROUND(LOG10(COUNT(*)),4) AS log10_count
            FROM earthquakes WHERE magnitude IS NOT NULL
            GROUP BY mag_bin ORDER BY mag_bin""",conn)
        regions=pd.read_sql_query("""
            SELECT r.name AS region, r.tectonic_setting,
                   COUNT(e.quake_id) AS quake_count,
                   ROUND(AVG(e.magnitude),2) AS avg_mag, MAX(e.magnitude) AS max_mag
            FROM earthquakes e JOIN regions r ON e.region_id=r.region_id
            GROUP BY r.name ORDER BY quake_count DESC LIMIT 12""",conn)
        energy=pd.read_sql_query("""
            SELECT strftime('%Y-%m',event_time) AS month, COUNT(*) AS events,
                   ROUND(SUM(POWER(10.0,1.5*magnitude+4.8))/1e15,4) AS energy_PJ
            FROM earthquakes WHERE magnitude IS NOT NULL
            GROUP BY month ORDER BY month""",conn)
        depth_df=pd.read_sql_query("""
            SELECT e.depth_km, e.magnitude,
                   CASE WHEN e.depth_km<70 THEN 'Shallow (<70 km)'
                        WHEN e.depth_km<300 THEN 'Intermediate (70-300 km)'
                        ELSE 'Deep (>300 km)' END AS depth_class,
                   r.tectonic_setting
            FROM earthquakes e JOIN regions r ON e.region_id=r.region_id
            WHERE e.depth_km IS NOT NULL""",conn)
        aftershock=pd.read_sql_query("""
            SELECT CASE WHEN e.magnitude>=7 THEN 'M7+'
                        WHEN e.magnitude>=6 THEN 'M6-6.9'
                        WHEN e.magnitude>=5.5 THEN 'M5.5-5.9'
                        ELSE 'M5-5.4' END AS mag_class,
                   COUNT(DISTINCT a.mainshock_id) AS mainshocks,
                   ROUND(AVG(a.had_damaging_after)*100,1) AS pct_damaging
            FROM aftershock_sequences a
            JOIN earthquakes e ON a.mainshock_id=e.quake_id
            GROUP BY mag_class ORDER BY MIN(e.magnitude) DESC""",conn)
        omori_opts=pd.read_sql_query("""
            SELECT DISTINCT e.quake_id, e.magnitude, e.place, e.event_time
            FROM aftershock_sequences a
            JOIN earthquakes e ON a.mainshock_id=e.quake_id
            WHERE e.magnitude>=6.0 ORDER BY e.magnitude DESC LIMIT 15""",conn)
        conn.close()
        return gr,regions,energy,depth_df,aftershock,omori_opts

    gr,regions,energy,depth_df,aftershock,omori_opts=load_analytics()

    # Row 1
    r1a,r1b=st.columns(2)
    with r1a:
        valid=gr[gr["log10_count"]>0]
        b=abs(round(np.polyfit(valid["mag_bin"],valid["log10_count"],1)[0],3)) if len(valid)>=2 else "N/A"
        f1=go.Figure()
        f1.add_trace(go.Scatter(x=gr["mag_bin"],y=gr["log10_count"],mode="markers+lines",
            marker=dict(size=7,color="#6366f1"),name="Observed"))
        if len(valid)>=2:
            fit=np.poly1d(np.polyfit(valid["mag_bin"],valid["log10_count"],1))(valid["mag_bin"])
            f1.add_trace(go.Scatter(x=valid["mag_bin"],y=fit,mode="lines",
                name=f"G-R fit (b={b})",line=dict(dash="dash",color="#ef4444",width=2)))
        f1.update_layout(title=f"Gutenberg-Richter Law ",
            xaxis_title="Magnitude",yaxis_title="log₁₀(count)",height=340,
            template="plotly_white",margin=dict(t=45,b=40,l=45,r=20),
            legend=dict(orientation="h",y=1.12))
        st.plotly_chart(f1,use_container_width=True)
        st.caption("A straight line confirms data quality and physical G-R consistency.")

    with r1b:
        f2=px.bar(regions,x="quake_count",y="region",orientation="h",
            color="tectonic_setting",height=340,title="Most Active Tectonic Regions",
            labels={"quake_count":"Event count","region":""},
            color_discrete_sequence=px.colors.qualitative.Set2)
        f2.update_layout(template="plotly_white",margin=dict(t=45,b=40,l=10,r=20),
            legend=dict(orientation="h",y=-0.28,font=dict(size=10)),
            yaxis=dict(autorange="reversed"))
        st.plotly_chart(f2,use_container_width=True)
        st.caption("Indonesia and the Philippines show the highest activity, consistent with their location along the Ring of Fire.")

    # Row 2
    r2a,r2b=st.columns(2)
    with r2a:
        f3=go.Figure(go.Bar(x=energy["month"],y=energy["energy_PJ"],
            marker_color="#6366f1",opacity=0.85))
        f3.update_layout(title="Monthly Seismic Energy Release  [E=10^(1.5M+4.8)]",
            xaxis_title="Month",yaxis_title="Energy (PJ)",height=320,
            template="plotly_white",margin=dict(t=45,b=60,l=55,r=20))
        st.plotly_chart(f3,use_container_width=True)
        st.caption("One M7+ event can exceed cumulative energy of all other months.")

    with r2b:
        if not aftershock.empty:
            cmap={"M7+":"#ef4444","M6-6.9":"#f59e0b","M5.5-5.9":"#6366f1","M5-5.4":"#22c55e"}
            f4=px.bar(aftershock,x="mag_class",y="pct_damaging",color="mag_class",
                color_discrete_map=cmap,height=320,text="pct_damaging",
                title="% of Mainshocks → Damaging Aftershock within 48h",
                labels={"mag_class":"Magnitude class","pct_damaging":"% with M4+ aftershock"})
            f4.update_traces(texttemplate="%{text}%",textposition="outside")
            f4.update_layout(template="plotly_white",showlegend=False,
                yaxis=dict(range=[0,115]),margin=dict(t=45,b=40,l=55,r=20))
            st.plotly_chart(f4,use_container_width=True)
            st.caption("Near-certain probabilities for M7+ events align with Bath's Law.")
        else:
            st.info("Run aftershocks.py to populate this chart.")

    # Omori-Utsu
    st.divider()
    st.write("### Omori-Utsu Aftershock Decay")
    if omori_opts.empty:
        st.info("No M6+ sequences in database.")
    else:
        omori_opts["lbl"]=omori_opts.apply(
            lambda r:f"M{r['magnitude']:.1f} | {str(r['place'])[:45]} | {str(r['event_time'])[:10]}",axis=1)
        chosen_lbl=st.selectbox("Select mainshock",omori_opts["lbl"].tolist())
        chosen_id =omori_opts.loc[omori_opts["lbl"]==chosen_lbl,"quake_id"].iloc[0]
        chosen_mag=omori_opts.loc[omori_opts["lbl"]==chosen_lbl,"magnitude"].iloc[0]

        @st.cache_data(ttl=3600)
        def load_omori(qid):
            conn=get_connection()
            df=pd.read_sql_query(f"""
                SELECT ROUND(delta_hours/6)*6 AS hour_bin, COUNT(*) AS aftershock_count
                FROM aftershock_sequences WHERE mainshock_id='{qid}'
                GROUP BY hour_bin ORDER BY hour_bin""",conn)
            conn.close(); return df

        om=load_omori(chosen_id)
        om=om[om["hour_bin"]<=168]
        if om.empty:
            st.warning("No sequence data for this event.")
        else:
            t=om["hour_bin"].replace(0,1).values
            fit=om["aftershock_count"].iloc[0]*2.0/(t+2)**1.1
            fo=go.Figure()
            fo.add_trace(go.Bar(x=om["hour_bin"],y=om["aftershock_count"],
                name="Observed / 6h",marker_color="#6366f1",opacity=0.75))
            fo.add_trace(go.Scatter(x=om["hour_bin"],y=fit,mode="lines",
                name="Omori-Utsu K/(t+c)^p  (p≈1.1)",
                line=dict(color="#ef4444",width=2.5,dash="dash")))
            fo.update_layout(title=f"Aftershock Decay for an M{chosen_mag:.1f} Mainshock",
                xaxis_title="Hours after mainshock",yaxis_title="Aftershocks per 6h",
                height=380,template="plotly_white",
                legend=dict(orientation="h",y=1.1),margin=dict(t=50,b=50,l=55,r=20))
            st.plotly_chart(fo,use_container_width=True)
            st.caption("A steep initial decay followed by gradual flattening, reflecting the characteristic Omori-Utsu pattern.")

    # Depth scatter
    st.divider()
    st.write("### Depth vs Magnitude by Tectonic Setting")
    cmap2={"Subduction zone":"#ef4444","Collision zone":"#f59e0b",
           "Spreading ridge":"#6366f1","Transform fault":"#22c55e",
           "Intraplate":"#94a3b8","Continental rift":"#0ea5e9"}
    f5=px.scatter(depth_df,x="depth_km",y="magnitude",color="tectonic_setting",
        opacity=0.5,color_discrete_map=cmap2,height=380,
        labels={"depth_km":"Depth (km)","magnitude":"Magnitude","tectonic_setting":"Setting"})
    f5.update_traces(marker=dict(size=5))
    f5.update_layout(template="plotly_white",
        legend=dict(orientation="h",y=1.08,font=dict(size=11)),
        margin=dict(t=10,b=50,l=55,r=20))
    st.plotly_chart(f5,use_container_width=True)




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
    st.code("magnitude,depth_km,latitude,longitude,tectonic_setting,plate_boundary_type,risk_tier,tsunami_flag")

    EXAMPLE_CSV="""magnitude,depth_km,latitude,longitude,tectonic_setting,plate_boundary_type,risk_tier,tsunami_flag
7.8,35.0,5.6,125.1,Subduction zone,Convergent,4,1
6.5,120.0,35.5,140.2,Subduction zone,Convergent,4,0
5.2,10.0,38.0,30.5,Collision zone,Convergent,3,0
5.8,55.0,-33.0,-71.0,Subduction zone,Convergent,4,0
6.1,280.0,-7.5,110.0,Subduction zone,Convergent,4,0"""

    mode=st.radio("Input method",["📋 Paste CSV","📁 Upload CSV file","⚡ Use example data"],
                  horizontal=True)

    raw_df=None
    if mode=="📋 Paste CSV":
        text=st.text_area("Paste your CSV data here",height=180)
        if text.strip():
            try:   raw_df=pd.read_csv(io.StringIO(text))
            except: st.error("Could not parse CSV — check format.")
    elif mode=="📁 Upload CSV file":
        uploaded=st.file_uploader("Upload CSV",type=["csv"])
        if uploaded:
            try:   raw_df=pd.read_csv(uploaded)
            except: st.error("Could not read file.")
    else:
        raw_df=pd.read_csv(io.StringIO(EXAMPLE_CSV))
        st.info("Using example data featuring five earthquakes of varying magnitudes and depths.")

    if raw_df is not None and not raw_df.empty:
        st.write(f"**{len(raw_df)} earthquakes loaded**")
        st.dataframe(raw_df,use_container_width=True,hide_index=True)

        if st.button("⚡ Score all earthquakes",type="primary",use_container_width=True):
            required=["magnitude","depth_km","latitude","longitude",
                      "tectonic_setting","plate_boundary_type","risk_tier","tsunami_flag"]
            missing=[c for c in required if c not in raw_df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                results=[]
                for _,row in raw_df.iterrows():
                    try:
                        X=build_feature_row(
                            float(row["magnitude"]),float(row["depth_km"]),
                            float(row["latitude"]),float(row["longitude"]),
                            str(row["tectonic_setting"]),str(row["plate_boundary_type"]),
                            int(row["risk_tier"]),int(row["tsunami_flag"]))
                        prob=model.predict_proba(scaler.transform(X))[0][1]
                        label=("HIGH RISK" if prob>=0.7 else
                               "MODERATE RISK" if prob>=0.4 else "LOW RISK")
                        results.append({"probability_%":round(prob*100,1),"risk_level":label})
                    except Exception as e:
                        results.append({"probability_%":None,"risk_level":f"Error: {e}"})

                out_df=pd.concat([raw_df.reset_index(drop=True),
                                  pd.DataFrame(results)],axis=1)

                # Colour-code risk column
                def colour_risk(val):
                    if "HIGH"     in str(val): return "background-color:#fee2e2"
                    if "MODERATE" in str(val): return "background-color:#fef3c7"
                    if "LOW"      in str(val): return "background-color:#dcfce7"
                    return ""

                st.write("**Results**")
                st.dataframe(out_df.style.applymap(colour_risk,subset=["risk_level"]),
                             use_container_width=True,hide_index=True)

                # Risk distribution chart
                risk_counts=pd.DataFrame(results)["risk_level"].value_counts().reset_index()
                risk_counts.columns=["Risk Level","Count"]
                order=["HIGH RISK","MODERATE RISK","LOW RISK"]
                risk_counts["Risk Level"]=pd.Categorical(risk_counts["Risk Level"],
                                                          categories=order,ordered=True)
                risk_counts=risk_counts.sort_values("Risk Level")
                cmap3={"HIGH RISK":"#ef4444","MODERATE RISK":"#f59e0b","LOW RISK":"#22c55e"}
                fb=px.bar(risk_counts,x="Risk Level",y="Count",color="Risk Level",
                    color_discrete_map=cmap3,title="Batch Risk Distribution",
                    height=300,text="Count")
                fb.update_traces(textposition="outside")
                fb.update_layout(showlegend=False,template="plotly_white",
                    yaxis=dict(range=[0,len(raw_df)+1]),margin=dict(t=45,b=40,l=40,r=20))
                st.plotly_chart(fb,use_container_width=True)

                # Download button
                csv_out=out_df.to_csv(index=False)
                st.download_button("⬇️ Download results CSV",data=csv_out,
                    file_name="seismic_risk_scores.csv",mime="text/csv",
                    use_container_width=True)
    else:
        if mode!="📋 Paste CSV":
            pass
        else:
            st.write("**Example data format:**")
            st.code(EXAMPLE_CSV)