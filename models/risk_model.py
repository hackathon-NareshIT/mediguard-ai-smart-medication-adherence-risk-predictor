"""
MediGuard AI - Medication Adherence Risk Prediction Model
Uses XGBoost + Ensemble approach for robust risk scoring
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, accuracy_score)
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import joblib
import os

# ─────────────────────────────────────────────
# FEATURE DEFINITIONS
# ─────────────────────────────────────────────
FEATURE_NAMES = [
    "age",
    "num_medications",
    "chronic_conditions",
    "days_since_last_refill",
    "missed_doses_last_30d",
    "avg_daily_doses",
    "medication_complexity_score",
    "side_effects_reported",
    "caregiver_support",       # 0=None, 1=Partial, 2=Full
    "health_literacy_score",   # 1-10
    "insurance_coverage",      # 0=None, 1=Partial, 2=Full
    "distance_to_pharmacy",    # km
    "employment_status",       # 0=Unemployed, 1=Part-time, 2=Full-time, 3=Retired
    "depression_score",        # PHQ-9 simplified 0-27
    "previous_adherence_rate", # 0-100%
    "num_hospitalizations_1y",
    "poly_pharmacy",           # 1 if >= 5 meds
    "refill_on_time_rate",     # 0-100%
]

RISK_LABELS = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}
RISK_COLORS = {0: "#22c55e", 1: "#f59e0b", 2: "#ef4444"}


# ─────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR
# ─────────────────────────────────────────────
def generate_training_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    age = rng.integers(18, 90, n_samples)
    num_medications = rng.integers(1, 15, n_samples)
    chronic_conditions = rng.integers(0, 7, n_samples)
    days_since_last_refill = rng.integers(0, 120, n_samples)
    missed_doses_last_30d = rng.integers(0, 30, n_samples)
    avg_daily_doses = rng.uniform(0.5, 6.0, n_samples)
    medication_complexity_score = rng.integers(1, 10, n_samples)
    side_effects_reported = rng.integers(0, 5, n_samples)
    caregiver_support = rng.integers(0, 3, n_samples)
    health_literacy_score = rng.integers(1, 11, n_samples)
    insurance_coverage = rng.integers(0, 3, n_samples)
    distance_to_pharmacy = rng.uniform(0.1, 50.0, n_samples)
    employment_status = rng.integers(0, 4, n_samples)
    depression_score = rng.integers(0, 28, n_samples)
    previous_adherence_rate = rng.uniform(0, 100, n_samples)
    num_hospitalizations_1y = rng.integers(0, 5, n_samples)
    poly_pharmacy = (num_medications >= 5).astype(int)
    refill_on_time_rate = rng.uniform(0, 100, n_samples)

    # Risk score heuristic
    risk_score = (
        0.03 * missed_doses_last_30d
        + 0.02 * days_since_last_refill
        + 0.05 * side_effects_reported
        + 0.04 * depression_score
        + 0.04 * medication_complexity_score
        + 0.03 * distance_to_pharmacy / 10
        + 0.05 * num_medications
        + 0.05 * chronic_conditions
        - 0.05 * health_literacy_score
        - 0.05 * caregiver_support
        - 0.04 * insurance_coverage
        - 0.04 * (previous_adherence_rate / 20)
        - 0.03 * (refill_on_time_rate / 20)
        + 0.03 * num_hospitalizations_1y
        + rng.normal(0, 0.5, n_samples)
    )

    # Bucket into 3 classes
    low = np.percentile(risk_score, 40)
    high = np.percentile(risk_score, 75)
    risk_label = np.where(risk_score < low, 0, np.where(risk_score < high, 1, 2))

    df = pd.DataFrame({
        "age": age,
        "num_medications": num_medications,
        "chronic_conditions": chronic_conditions,
        "days_since_last_refill": days_since_last_refill,
        "missed_doses_last_30d": missed_doses_last_30d,
        "avg_daily_doses": avg_daily_doses,
        "medication_complexity_score": medication_complexity_score,
        "side_effects_reported": side_effects_reported,
        "caregiver_support": caregiver_support,
        "health_literacy_score": health_literacy_score,
        "insurance_coverage": insurance_coverage,
        "distance_to_pharmacy": distance_to_pharmacy,
        "employment_status": employment_status,
        "depression_score": depression_score,
        "previous_adherence_rate": previous_adherence_rate,
        "num_hospitalizations_1y": num_hospitalizations_1y,
        "poly_pharmacy": poly_pharmacy,
        "refill_on_time_rate": refill_on_time_rate,
        "risk_label": risk_label,
    })
    return df


# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────
def train_model(df: pd.DataFrame = None, save_path: str = None):
    if df is None:
        df = generate_training_data()

    X = df[FEATURE_NAMES]
    y = df["risk_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Ensemble: GBM as primary, RF as meta-learner
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            random_state=42,
        )),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob, multi_class="ovr"),
        "report": classification_report(y_test, y_pred, target_names=list(RISK_LABELS.values())),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "cv_scores": cross_val_score(pipeline, X, y, cv=5, scoring="accuracy").tolist(),
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump({"model": pipeline, "metrics": metrics}, save_path)

    return pipeline, metrics


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
def predict_risk(model, patient_data: dict) -> dict:
    """
    patient_data: dict with keys matching FEATURE_NAMES
    Returns: risk_label (int), risk_name (str), probabilities (dict), 
             risk_score (float 0–100), top_factors (list)
    """
    X = pd.DataFrame([patient_data])[FEATURE_NAMES]
    probs = model.predict_proba(X)[0]
    label = int(np.argmax(probs))

    # Risk score 0–100 weighted by class
    risk_score = float(probs[1] * 50 + probs[2] * 100)

    # Feature importances (for GBM inside pipeline)
    clf = model.named_steps["clf"]
    importances = clf.feature_importances_
    feature_importance_pairs = sorted(
        zip(FEATURE_NAMES, importances), key=lambda x: -x[1]
    )
    top_factors = [
        {"feature": f, "importance": float(i), "value": patient_data.get(f, "N/A")}
        for f, i in feature_importance_pairs[:6]
    ]

    return {
        "risk_label": label,
        "risk_name": RISK_LABELS[label],
        "risk_color": RISK_COLORS[label],
        "risk_score": min(risk_score, 100),
        "probabilities": {RISK_LABELS[i]: float(p) for i, p in enumerate(probs)},
        "top_factors": top_factors,
    }


# ─────────────────────────────────────────────
# LOAD OR TRAIN
# ─────────────────────────────────────────────
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "mediguard_model.pkl")


def get_model():
    if os.path.exists(_MODEL_PATH):
        bundle = joblib.load(_MODEL_PATH)
        return bundle["model"], bundle["metrics"]
    else:
        model, metrics = train_model(save_path=_MODEL_PATH)
        return model, metrics