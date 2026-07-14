"""
train_model.py — XGBoost aftershock risk classifier with full metrics.

Outputs saved to models/:
    model.pkl            XGBoost classifier
    scaler.pkl           StandardScaler
    feature_names.pkl    Ordered feature list
    metrics.json         All scores: CV, ROC-AUC, F1, PR curve, ROC curve data
    shap_beeswarm.png    SHAP feature importance
    confusion_matrix.png Confusion matrix

Run from project root:
    python src/train_model.py
"""

import sys, pickle, json, warnings
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection    import StratifiedKFold, cross_validate
from sklearn.preprocessing      import StandardScaler, LabelEncoder
from sklearn.metrics            import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score, accuracy_score, precision_score, recall_score,
)
import xgboost as xgb
import shap

from database import get_connection

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ── 1. Load data ─────────────────────────────────────────────────────────────────
def load_training_data() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT
            e.quake_id, e.magnitude, e.depth_km,
            e.latitude, e.longitude, e.tsunami_flag,
            r.tectonic_setting, r.plate_boundary_type, r.risk_tier,
            COALESCE(seq.had_damaging, 0) AS had_damaging
        FROM   earthquakes e
        LEFT JOIN regions r ON e.region_id = r.region_id
        LEFT JOIN (
            SELECT mainshock_id, MAX(had_damaging_after) AS had_damaging
            FROM   aftershock_sequences GROUP BY mainshock_id
        ) seq ON seq.mainshock_id = e.quake_id
        WHERE  e.magnitude >= 5.0
          AND  e.depth_km IS NOT NULL
          AND  r.tectonic_setting IS NOT NULL
    """, conn)
    conn.close()
    return df


# ── 2. Feature engineering ────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame):
    df = df.copy()
    df["is_shallow"]      = (df["depth_km"] < 70).astype(int)
    df["is_intermediate"] = ((df["depth_km"] >= 70) & (df["depth_km"] < 300)).astype(int)
    df["mag_class"]       = pd.cut(
        df["magnitude"], bins=[4.9,5.4,5.9,6.4,6.9,99], labels=[0,1,2,3,4]
    ).astype(int)

    le_tect  = LabelEncoder()
    le_plate = LabelEncoder()
    df["tectonic_enc"] = le_tect.fit_transform(df["tectonic_setting"].fillna("Unknown"))
    df["plate_enc"]    = le_plate.fit_transform(df["plate_boundary_type"].fillna("Unknown"))

    feat_cols = [
        "magnitude","depth_km","is_shallow","is_intermediate","mag_class",
        "risk_tier","tectonic_enc","plate_enc","latitude","longitude","tsunami_flag",
    ]
    return df, feat_cols


# ── 3. Train + full metrics ───────────────────────────────────────────────────────
def train(df: pd.DataFrame, feat_cols: list[str]) -> dict:
    X = df[feat_cols].values
    y = df["had_damaging"].values

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    spw   = round(n_neg / n_pos, 2) if n_pos > 0 else 1.0

    print(f"\nSamples       : {len(df)}")
    print(f"Positive (1)  : {n_pos}  |  Negative (0): {n_neg}")
    print(f"scale_pos_weight: {spw}")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric="logloss",
        use_label_encoder=False, random_state=42, verbosity=0,
    )

    # ── 5-fold cross-validation ───────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_res = cross_validate(
        model, X_scaled, y, cv=cv,
        scoring=["accuracy","f1","roc_auc","precision","recall"],
        return_train_score=True,
    )

    print("\n── Cross-validation (5-fold stratified) ──────────────────────────")
    cv_summary = {}
    for metric in ["accuracy","f1","roc_auc","precision","recall"]:
        scores = cv_res[f"test_{metric}"]
        cv_summary[metric] = {"mean": round(float(scores.mean()),4),
                               "std":  round(float(scores.std()), 4),
                               "all":  [round(float(s),4) for s in scores]}
        print(f"  {metric:<12}  {scores.mean():.4f} ± {scores.std():.4f}")

    # ── Final fit ─────────────────────────────────────────────────────────────────
    model.fit(X_scaled, y)
    y_pred  = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:,1]

    # ── Full-data metrics ─────────────────────────────────────────────────────────
    acc   = round(float(accuracy_score(y, y_pred)), 4)
    f1_w  = round(float(f1_score(y, y_pred, average="weighted", zero_division=0)), 4)
    f1_m  = round(float(f1_score(y, y_pred, average="macro",    zero_division=0)), 4)
    prec  = round(float(precision_score(y, y_pred, zero_division=0)), 4)
    rec   = round(float(recall_score(y, y_pred, zero_division=0)), 4)
    auc   = round(float(roc_auc_score(y, y_proba)) if n_pos>=2 else 0.0, 4)
    ap    = round(float(average_precision_score(y, y_proba)) if n_pos>=2 else 0.0, 4)

    print(f"\n── Full-data metrics ─────────────────────────────────────────────")
    print(f"  Accuracy      : {acc}")
    print(f"  ROC-AUC       : {auc}")
    print(f"  Avg Precision : {ap}")
    print(f"  F1 (weighted) : {f1_w}")
    print(f"  F1 (macro)    : {f1_m}")
    print(f"  Precision     : {prec}")
    print(f"  Recall        : {rec}")

    print("\n── Classification report ─────────────────────────────────────────")
    report = classification_report(
        y, y_pred, target_names=["No damage (0)","Damage (1)"],
        output_dict=True, zero_division=0,
    )
    print(classification_report(y, y_pred,
          target_names=["No damage (0)","Damage (1)"], zero_division=0))

    # ── ROC curve data ────────────────────────────────────────────────────────────
    roc_fpr, roc_tpr, _ = roc_curve(y, y_proba) if n_pos>=2 else ([0,1],[0,1],[])
    pr_prec, pr_rec, _  = precision_recall_curve(y, y_proba) if n_pos>=2 else ([1,0],[0,1],[])

    # ── Confusion matrix plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5,4))
    ConfusionMatrixDisplay(confusion_matrix(y, y_pred),
                           display_labels=["No damage","Damage"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix  |  Accuracy={acc:.3f}", fontsize=11)
    plt.tight_layout()
    plt.savefig(MODELS_DIR/"confusion_matrix.png", dpi=150)
    plt.close()

    # ── ROC curve plot ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(roc_fpr, roc_tpr, color="#6366f1", lw=2, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.5)
    ax.fill_between(roc_fpr, roc_tpr, alpha=0.1, color="#6366f1")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Aftershock Damage Classifier")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(MODELS_DIR/"roc_curve.png", dpi=150)
    plt.close()

    # ── Precision-Recall curve plot ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(pr_rec, pr_prec, color="#ef4444", lw=2, label=f"PR (AP={ap:.3f})")
    ax.axhline(n_pos/len(y), color="gray", linestyle="--", lw=1,
               label=f"Baseline (prevalence={n_pos/len(y):.2f})")
    ax.fill_between(pr_rec, pr_prec, alpha=0.1, color="#ef4444")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Aftershock Damage Classifier")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(MODELS_DIR/"pr_curve.png", dpi=150)
    plt.close()

    # ── Compile full metrics dict ─────────────────────────────────────────────────
    metrics = {
        "trained_at":        datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        "n_samples":         int(len(df)),
        "n_positive":        n_pos,
        "n_negative":        n_neg,
        "positive_rate_pct": round(n_pos/len(df)*100, 1),
        "accuracy":          acc,
        "roc_auc":           auc,
        "avg_precision":     ap,
        "f1_weighted":       f1_w,
        "f1_macro":          f1_m,
        "precision":         prec,
        "recall":            rec,
        "cv":                cv_summary,
        "class_report":      {k: {m: round(float(v),4) for m,v in vd.items()}
                              for k,vd in report.items() if isinstance(vd, dict)},
        "roc_fpr":           [round(float(v),4) for v in roc_fpr],
        "roc_tpr":           [round(float(v),4) for v in roc_tpr],
        "pr_precision":      [round(float(v),4) for v in pr_prec],
        "pr_recall":         [round(float(v),4) for v in pr_rec],
    }

    return {"model":model,"scaler":scaler,"X_scaled":X_scaled,
            "y":y,"feat_cols":feat_cols,"metrics":metrics}


# ── 4. SHAP ───────────────────────────────────────────────────────────────────────
def compute_shap(model, X_scaled, feat_cols):
    print("\n── SHAP feature importance ───────────────────────────────────────")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    fig, ax = plt.subplots(figsize=(8,5))
    shap.summary_plot(shap_values, X_scaled, feature_names=feat_cols,
                      plot_type="dot", show=False, max_display=11)
    plt.title("SHAP Beeswarm — Feature Impact on Aftershock Damage Risk",
              fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(MODELS_DIR/"shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: models/shap_beeswarm.png")

    mean_shap = np.abs(shap_values).mean(axis=0)
    ranked    = sorted(zip(feat_cols, mean_shap), key=lambda x: x[1], reverse=True)
    for feat, val in ranked:
        print(f"  {feat:<22} {val:.4f}  {'█'*int(val/mean_shap.max()*20)}")


# ── 5. Save ───────────────────────────────────────────────────────────────────────
def save_artifacts(results):
    with open(MODELS_DIR/"model.pkl",         "wb") as f: pickle.dump(results["model"],     f)
    with open(MODELS_DIR/"scaler.pkl",        "wb") as f: pickle.dump(results["scaler"],    f)
    with open(MODELS_DIR/"feature_names.pkl", "wb") as f: pickle.dump(results["feat_cols"], f)
    with open(MODELS_DIR/"metrics.json",      "w")  as f: json.dump(results["metrics"], f, indent=2)
    print(f"\n✓ Artifacts saved to: {MODELS_DIR}")
    print(f"  model.pkl  |  scaler.pkl  |  feature_names.pkl  |  metrics.json")
    print(f"  confusion_matrix.png  |  roc_curve.png  |  pr_curve.png  |  shap_beeswarm.png")


# ── Entry-point ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading training data...")
    df = load_training_data()
    df, feat_cols = engineer_features(df)
    results = train(df, feat_cols)
    compute_shap(results["model"], results["X_scaled"], feat_cols)
    save_artifacts(results)
    print("\n✓ Done. Run: streamlit run src/app.py")