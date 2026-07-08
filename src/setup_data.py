"""
setup_data.py — Builds the full database pipeline from scratch.
Called automatically by app.py when data/seismic.db is missing or empty.
This is what allows the app to self-deploy on Streamlit Cloud.

Flow:
    init_db → fetch USGS (last 60 days) → seed_regions
    → assign regions → build aftershocks → train model
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from database     import init_db, get_connection
from ingest       import fetch_usgs_window, insert_events
from seed_region import seed_regions, assign_regions, update_region_stats
from aftershock  import build_sequences

from datetime import datetime, timedelta
import pickle, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")


def needs_setup() -> bool:
    """Return True if database is missing or has fewer than 100 earthquakes."""
    db_path = Path(__file__).parent.parent / "data" / "seismic.db"
    if not db_path.exists():
        return True
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM earthquakes").fetchone()[0]
    conn.close()
    return count < 100


def run_setup(status_container=None) -> None:
    """
    Full pipeline: ingest → regions → aftershocks → model.
    status_container: a Streamlit container for progress messages (optional).
    """
    def log(msg: str):
        if status_container:
            status_container.write(msg)
        else:
            print(msg)

    log("🗄️  Initialising database...")
    init_db(verbose=False)

    log(" Fetching earthquakes from USGS API (last 60 days, M4.0+)...")
    end   = datetime.utcnow()
    start = end - timedelta(days=60)
    rows  = fetch_usgs_window(
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        min_mag=4.0,
    )
    conn = get_connection()
    insert_events(rows, conn)
    count = conn.execute("SELECT COUNT(*) FROM earthquakes").fetchone()[0]
    log(f"   {count:,} events loaded")

    log(" Seeding tectonic regions...")
    seed_regions(conn)
    updated = assign_regions(conn)
    update_region_stats(conn)
    log(f"   ✓ {updated:,} earthquakes assigned to regions")
    conn.close()

    log(" Building aftershock sequences...")
    build_sequences(verbose=False)
    conn = get_connection()
    seq_count = conn.execute("SELECT COUNT(*) FROM aftershock_sequences").fetchone()[0]
    conn.close()
    log(f"  {seq_count:,} sequence records built")

    # Backfill tsunami_flag
    conn = get_connection()
    conn.execute("""
        UPDATE earthquakes SET tsunami_flag = 1
        WHERE  depth_km < 100 AND magnitude >= 6.5
        AND    region_id IN (
            SELECT region_id FROM regions
            WHERE  tectonic_setting IN ('Subduction zone','Spreading ridge')
            OR     plate_boundary_type = 'Convergent'
        ) AND tsunami_flag = 0
    """)
    conn.commit()
    conn.close()

    log(" Training XGBoost model...")
    _train_model()
    log(" Model trained and saved")

    log(" Setup complete — loading dashboard...")


def _train_model():
    """Inline model training (mirrors train_model.py, no file I/O dependency)."""
    import xgboost as xgb
    from sklearn.preprocessing  import StandardScaler, LabelEncoder
    from sklearn.model_selection import StratifiedKFold, cross_validate

    MODELS_DIR = Path(__file__).parent.parent / "models"
    MODELS_DIR.mkdir(exist_ok=True)

    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT e.magnitude, e.depth_km, e.latitude, e.longitude,
               e.tsunami_flag, r.tectonic_setting, r.plate_boundary_type,
               r.risk_tier, COALESCE(seq.had_damaging,0) AS had_damaging
        FROM   earthquakes e
        LEFT JOIN regions r ON e.region_id = r.region_id
        LEFT JOIN (
            SELECT mainshock_id, MAX(had_damaging_after) AS had_damaging
            FROM   aftershock_sequences GROUP BY mainshock_id
        ) seq ON seq.mainshock_id = e.quake_id
        WHERE  e.magnitude >= 5.0 AND e.depth_km IS NOT NULL
          AND  r.tectonic_setting IS NOT NULL
    """, conn)
    conn.close()

    if len(df) < 10:
        return

    df["is_shallow"]      = (df["depth_km"] < 70).astype(int)
    df["is_intermediate"] = ((df["depth_km"] >= 70) & (df["depth_km"] < 300)).astype(int)
    df["mag_class"]       = pd.cut(df["magnitude"],
                                    bins=[4.9,5.4,5.9,6.4,6.9,99],
                                    labels=[0,1,2,3,4]).astype(int)

    le_tect  = LabelEncoder()
    le_plate = LabelEncoder()
    df["tectonic_enc"] = le_tect.fit_transform(df["tectonic_setting"].fillna("Unknown"))
    df["plate_enc"]    = le_plate.fit_transform(df["plate_boundary_type"].fillna("Unknown"))

    feat_cols = ["magnitude","depth_km","is_shallow","is_intermediate","mag_class",
                 "risk_tier","tectonic_enc","plate_enc","latitude","longitude","tsunami_flag"]

    X = df[feat_cols].values
    y = df["had_damaging"].values

    n_pos = y.sum(); n_neg = len(y) - n_pos
    spw   = round(n_neg / n_pos, 2) if n_pos > 0 else 1.0

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric="logloss",
        use_label_encoder=False, random_state=42, verbosity=0,
    )
    model.fit(X_scaled, y)

    with open(MODELS_DIR / "model.pkl",         "wb") as f: pickle.dump(model,     f)
    with open(MODELS_DIR / "scaler.pkl",        "wb") as f: pickle.dump(scaler,    f)
    with open(MODELS_DIR / "feature_names.pkl", "wb") as f: pickle.dump(feat_cols, f)