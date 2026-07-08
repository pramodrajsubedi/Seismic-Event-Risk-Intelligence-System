"""
train_model.py — XGBoost binary classifier for aftershock damage risk.

Target  : had_damaging_after (1 = an M4+ aftershock occurred within 48 h)
Features: magnitude, depth_km, is_shallow, risk_tier,
          tectonic_setting (encoded), plate_boundary_type (encoded),
          latitude, longitude
Level   : one row per M5+ mainshock

Outputs (saved to models/):
    model.pkl          XGBoost classifier
    scaler.pkl         StandardScaler for numeric features
    feature_names.pkl  Ordered list of feature columns
    shap_beeswarm.png  SHAP feature importance plot

Run from project root:
    python src/train_model.py
"""

import sys, pickle, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection   import StratifiedKFold, cross_validate
from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.metrics           import (classification_report,
                                       confusion_matrix,
                                       roc_auc_score,
                                       ConfusionMatrixDisplay)
import xgboost as xgb
import shap

from database import get_connection

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ── 1. Load data ────────────────────────────────────────────────────────────────

def load_training_data() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT
            e.quake_id,
            e.magnitude,
            e.depth_km,
            e.latitude,
            e.longitude,
            e.tsunami_flag,
            r.tectonic_setting,
            r.plate_boundary_type,
            r.risk_tier,
            COALESCE(seq.had_damaging, 0) AS had_damaging
        FROM   earthquakes e
        LEFT JOIN regions r ON e.region_id = r.region_id
        LEFT JOIN (
            SELECT mainshock_id, MAX(had_damaging_after) AS had_damaging
            FROM   aftershock_sequences
            GROUP  BY mainshock_id
        ) seq ON seq.mainshock_id = e.quake_id
        WHERE  e.magnitude      >= 5.0
          AND  e.depth_km        IS NOT NULL
          AND  r.tectonic_setting IS NOT NULL
    """, conn)
    conn.close()
    return df


# 2. Feature engineering 

def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add derived features and encode categoricals. Returns (df, feature_cols)."""
    df = df.copy()

    # Depth class flags (from physics: shallow = most destructive)
    df["is_shallow"]      = (df["depth_km"] < 70).astype(int)
    df["is_intermediate"] = ((df["depth_km"] >= 70) & (df["depth_km"] < 300)).astype(int)

    # Magnitude bins (non-linear bucketing)
    df["mag_class"] = pd.cut(
        df["magnitude"],
        bins=[4.9, 5.4, 5.9, 6.4, 6.9, 99],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    # Encode tectonic_setting
    le_tect = LabelEncoder()
    df["tectonic_enc"] = le_tect.fit_transform(
        df["tectonic_setting"].fillna("Unknown")
    )

    # Encode plate_boundary_type
    le_plate = LabelEncoder()
    df["plate_enc"] = le_plate.fit_transform(
        df["plate_boundary_type"].fillna("Unknown")
    )

    feature_cols = [
        "magnitude",        # strongest predictor (corr 0.61)
        "depth_km",         # physical depth signal
        "is_shallow",       # binary depth flag
        "is_intermediate",  # binary depth flag
        "mag_class",        # non-linear magnitude bucket
        "risk_tier",        # region risk level 1–4
        "tectonic_enc",     # encoded tectonic setting
        "plate_enc",        # encoded boundary type
        "latitude",         # geographic position
        "longitude",
        "tsunami_flag",     # coastal / seafloor indicator
    ]

    return df, feature_cols


# 3. Train 

def train(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    X = df[feature_cols].values
    y = df["had_damaging"].values

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    spw   = round(n_neg / n_pos, 2) if n_pos > 0 else 1.0

    print(f"Samples        : {len(df)}")
    print(f"Class balance  : 0={n_neg}  1={n_pos}  (scale_pos_weight={spw})")
    print(f"Features       : {feature_cols}\n")

    if n_pos < 5:
        print("⚠  Fewer than 5 positive samples — run on real USGS data for meaningful results.")
        print("   Proceeding anyway so you can verify the pipeline structure.\n")

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # XGBoost model
    model = xgb.XGBClassifier(
        n_estimators      = 300,
        max_depth         = 4,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = spw,
        eval_metric       = "logloss",
        use_label_encoder = False,
        random_state      = 42,
        verbosity         = 0,
    )

    # 5-fold stratified cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model, X_scaled, y, cv=cv,
        scoring=["accuracy", "f1", "roc_auc"],
        return_train_score=True
    )

    print(" Cross-validation (5-fold) ")
    for metric in ["accuracy", "f1", "roc_auc"]:
        test_scores  = cv_results[f"test_{metric}"]
        train_scores = cv_results[f"train_{metric}"]
        print(f"  {metric:<12}  CV={test_scores.mean():.3f} ± {test_scores.std():.3f}"
              f"   Train={train_scores.mean():.3f}")

    # Final fit on full data
    model.fit(X_scaled, y)

    # Full-data metrics
    y_pred  = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    print("\n Full-data classification report ")
    print(classification_report(y, y_pred, target_names=["No damage (0)", "Damage (1)"]))

    auc = roc_auc_score(y, y_proba) if n_pos >= 2 else float("nan")
    print(f"ROC-AUC (full data): {auc:.4f}")

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y, y_pred),
                           display_labels=["No damage", "Damage"]).plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix (full training data)")
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved: models/confusion_matrix.png")

    return {"model": model, "scaler": scaler,
            "X_scaled": X_scaled, "y": y,
            "feature_cols": feature_cols, "auc": auc}


# 4. SHAP 

def compute_shap(model, X_scaled: np.ndarray,
                 feature_cols: list[str]) -> None:
    print("\n SHAP feature importance ")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    # Summary (beeswarm)
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, X_scaled,
                      feature_names=feature_cols,
                      plot_type="dot", show=False, max_display=11)
    plt.title("SHAP Beeswarm — Feature Impact on Aftershock Damage Risk",
              fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: models/shap_beeswarm.png")

    # Bar summary (mean |SHAP|)
    mean_shap = np.abs(shap_values).mean(axis=0)
    ranked    = sorted(zip(feature_cols, mean_shap),
                       key=lambda x: x[1], reverse=True)
    print("\n  Mean |SHAP| per feature:")
    for feat, val in ranked:
        bar = "█" * int(val / mean_shap.max() * 20)
        print(f"  {feat:<22} {val:.4f}  {bar}")


# 5. Save 

def save_artifacts(results: dict) -> None:
    model_path   = MODELS_DIR / "model.pkl"
    scaler_path  = MODELS_DIR / "scaler.pkl"
    feats_path   = MODELS_DIR / "feature_names.pkl"

    with open(model_path,  "wb") as f: pickle.dump(results["model"],        f)
    with open(scaler_path, "wb") as f: pickle.dump(results["scaler"],       f)
    with open(feats_path,  "wb") as f: pickle.dump(results["feature_cols"], f)

    print(f"\nSaved: {model_path}")
    print(f"Saved: {scaler_path}")
    print(f"Saved: {feats_path}")


# 6. predict() — used by Streamlit 

def predict(magnitude: float, depth_km: float, tectonic_setting: str,
            plate_boundary_type: str, risk_tier: int,
            latitude: float, longitude: float,
            tsunami_flag: int = 0) -> dict:
    """
    Load saved model and return risk prediction for a new event.
    Called by Streamlit app on each user query.

    Returns:
        { "label": "High Risk" | "Moderate Risk" | "Low Risk",
          "probability": float,
          "confidence": str }
    """
    with open(MODELS_DIR / "model.pkl",        "rb") as f: model    = pickle.load(f)
    with open(MODELS_DIR / "scaler.pkl",       "rb") as f: scaler   = pickle.load(f)
    with open(MODELS_DIR / "feature_names.pkl","rb") as f: feat_cols = pickle.load(f)

    # Replicate feature engineering for a single row
    is_shallow      = int(depth_km < 70)
    is_intermediate = int(70 <= depth_km < 300)
    mag_class       = int(pd.cut([magnitude],
                                  bins=[4.9,5.4,5.9,6.4,6.9,99],
                                  labels=[0,1,2,3,4])[0])

    tect_map  = {"Subduction zone":0, "Collision zone":1,
                 "Continental rift":2, "Intraplate":3, "Spreading ridge":4,
                 "Transform fault":5}
    plate_map = {"Convergent":0, "Divergent":1,
                 "Intraplate":2, "Transform":3}

    row = {
        "magnitude":       magnitude,
        "depth_km":        depth_km,
        "is_shallow":      is_shallow,
        "is_intermediate": is_intermediate,
        "mag_class":       mag_class,
        "risk_tier":       risk_tier,
        "tectonic_enc":    tect_map.get(tectonic_setting, 3),
        "plate_enc":       plate_map.get(plate_boundary_type, 2),
        "latitude":        latitude,
        "longitude":       longitude,
        "tsunami_flag":    tsunami_flag,
    }

    X = np.array([[row[f] for f in feat_cols]])
    X_scaled = scaler.transform(X)

    prob   = model.predict_proba(X_scaled)[0][1]
    label  = "High Risk" if prob >= 0.7 else ("Moderate Risk" if prob >= 0.4 else "Low Risk")
    color  = "red"       if prob >= 0.7 else ("orange"        if prob >= 0.4 else "green")

    return {"label": label, "probability": round(prob, 4),
            "confidence": f"{prob*100:.1f}%", "color": color}


# Entry-point 

if __name__ == "__main__":
    print("Loading training data...")
    df = load_training_data()
    df, feature_cols = engineer_features(df)

    results = train(df, feature_cols)
    compute_shap(results["model"], results["X_scaled"], feature_cols)
    save_artifacts(results)

    print("\n Model training complete.")
    print("  Next step: python src/app.py  (Streamlit dashboard)")