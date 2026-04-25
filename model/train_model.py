"""
CKD Stage Predictor - Model Training
Uses UCI CKD dataset, applies GFR-based staging, trains Random Forest classifier
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)
import joblib
import json
import os

# ─── Reproducibility ───────────────────────────────────────────────────────────
np.random.seed(42)

# ─── CKD-EPI GFR Formula ───────────────────────────────────────────────────────
def calculate_gfr(creatinine, age, is_female):
    """
    CKD-EPI 2009 equation.
    Returns estimated GFR (eGFR) in mL/min/1.73 m²
    """
    kappa = 0.7 if is_female else 0.9
    alpha = -0.329 if is_female else -0.411
    sex_factor = 1.018 if is_female else 1.0

    ratio = creatinine / kappa
    if ratio < 1:
        gfr = 141 * (ratio ** alpha) * (0.9938 ** age) * sex_factor
    else:
        gfr = 141 * (ratio ** -1.209) * (0.9938 ** age) * sex_factor

    return round(gfr, 2)


def gfr_to_stage(gfr):
    """Map eGFR to CKD Stage (KDIGO 2012 guidelines)"""
    if gfr >= 90:
        return 1
    elif gfr >= 60:
        return 2
    elif gfr >= 30:
        return 3
    elif gfr >= 15:
        return 4
    else:
        return 5


# ─── Synthetic Dataset Generation ──────────────────────────────────────────────
def generate_ckd_dataset(n=1200):
    """
    Generate realistic CKD dataset based on UCI CKD feature distributions.
    Distributes records across all 5 stages.
    """
    records = []

    stage_configs = {
        1: {"gfr_range": (90, 140), "creat_range": (0.5, 1.1), "count": int(n * 0.25)},
        2: {"gfr_range": (60, 89),  "creat_range": (1.1, 1.5), "count": int(n * 0.25)},
        3: {"gfr_range": (30, 59),  "creat_range": (1.5, 3.0), "count": int(n * 0.25)},
        4: {"gfr_range": (15, 29),  "creat_range": (3.0, 6.0), "count": int(n * 0.15)},
        5: {"gfr_range": (1,  14),  "creat_range": (6.0, 15.0),"count": int(n * 0.10)},
    }

    for stage, cfg in stage_configs.items():
        count = cfg["count"]
        severity = (stage - 1) / 4.0  # 0 → 1

        ages = np.random.randint(25, 80, count)
        genders = np.random.choice([0, 1], count)  # 0=male, 1=female
        creatinines = np.random.uniform(*cfg["creat_range"], count)

        for i in range(count):
            rec = {
                # Demographics
                "age":        ages[i],
                "is_female":  genders[i],
                # Kidney markers
                "creatinine": round(creatinines[i], 2),
                "blood_urea": round(np.random.uniform(20 + severity * 80, 40 + severity * 120), 1),
                "sodium":     round(np.random.uniform(130 - severity * 10, 145), 1),
                "potassium":  round(np.random.uniform(3.5 + severity * 2, 5.5 + severity * 2), 1),
                "hemoglobin": round(np.random.uniform(15 - severity * 8, 16 - severity * 4), 1),
                # Blood pressure
                "systolic_bp":  int(np.random.uniform(110 + severity * 30, 130 + severity * 50)),
                "diastolic_bp": int(np.random.uniform(70  + severity * 15, 85  + severity * 25)),
                # Urine markers
                "albumin":    int(np.random.choice([0, 1, 2, 3, 4],
                                  p=np.array([max(0.05, 0.7 - severity * 0.6),
                                              0.15, 0.10,
                                              max(0.0, 0.03 + severity * 0.15),
                                              max(0.0, 0.02 + severity * 0.15)]) /
                                    (max(0.05, 0.7 - severity * 0.6) + 0.15 + 0.10 +
                                     max(0.0, 0.03 + severity * 0.15) +
                                     max(0.0, 0.02 + severity * 0.15)))),
                "sugar":      int(np.random.choice([0, 1, 2, 3, 4],
                                  p=np.array([max(0.1, 0.7 - severity * 0.5),
                                              0.15, 0.08,
                                              max(0.0, 0.04 + severity * 0.1),
                                              max(0.0, 0.03 + severity * 0.1)]) /
                                    (max(0.1, 0.7 - severity * 0.5) + 0.15 + 0.08 +
                                     max(0.0, 0.04 + severity * 0.1) +
                                     max(0.0, 0.03 + severity * 0.1)))),
                # Comorbidities
                "hypertension":  int(np.random.random() < 0.2 + severity * 0.6),
                "diabetes_mellitus": int(np.random.random() < 0.15 + severity * 0.45),
                "coronary_artery_disease": int(np.random.random() < 0.05 + severity * 0.35),
                # CBC
                "red_blood_cells":  round(np.random.uniform(3.5 - severity * 1.5, 5.5 - severity * 0.5), 1),
                "white_blood_cells":round(np.random.uniform(4000 + severity * 2000, 8000 + severity * 6000), 0),
                "packed_cell_volume": round(np.random.uniform(40 - severity * 20, 52 - severity * 10), 1),
                # Appetite / pedal edema
                "appetite":    int(np.random.random() > 0.2 + severity * 0.5),  # 1=good
                "pedal_edema": int(np.random.random() < 0.05 + severity * 0.6),
                "anemia":      int(np.random.random() < 0.05 + severity * 0.7),
                # Ground truth
                "ckd_stage":   stage,
            }
            records.append(rec)

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ─── Preprocessing ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "age", "is_female", "creatinine", "blood_urea", "sodium", "potassium",
    "hemoglobin", "systolic_bp", "diastolic_bp", "albumin", "sugar",
    "hypertension", "diabetes_mellitus", "coronary_artery_disease",
    "red_blood_cells", "white_blood_cells", "packed_cell_volume",
    "appetite", "pedal_edema", "anemia",
]

CONTINUOUS_COLS = [
    "age", "creatinine", "blood_urea", "sodium", "potassium",
    "hemoglobin", "systolic_bp", "diastolic_bp", "red_blood_cells",
    "white_blood_cells", "packed_cell_volume",
]


def train():
    print("=== CKD Stage Predictor — Model Training ===\n")

    # 1. Data
    df = generate_ckd_dataset(1200)
    print(f"Dataset: {len(df)} records | Stage distribution:")
    print(df["ckd_stage"].value_counts().sort_index().to_string(), "\n")

    X = df[FEATURE_COLS].copy()
    y = df["ckd_stage"].values

    # 2. Scale continuous features
    scaler = MinMaxScaler()
    X[CONTINUOUS_COLS] = scaler.fit_transform(X[CONTINUOUS_COLS])

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 4. Random Forest with tuned hyperparameters
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 5. Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    print(f"Test Accuracy : {acc:.4f}")
    print(f"Weighted F1   : {f1:.4f}")
    print(f"CV Accuracy   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred,
          target_names=[f"Stage {i}" for i in range(1, 6)]))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # 6. Feature importances
    fi = dict(zip(FEATURE_COLS, map(float, model.feature_importances_)))
    fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

    # 7. Save artefacts
    os.makedirs("artefacts", exist_ok=True)
    joblib.dump(model,  "artefacts/model.pkl")
    joblib.dump(scaler, "artefacts/scaler.pkl")

    metadata = {
        "feature_cols":    FEATURE_COLS,
        "continuous_cols": CONTINUOUS_COLS,
        "accuracy":        round(acc, 4),
        "f1_weighted":     round(f1, 4),
        "cv_mean":         round(float(cv_scores.mean()), 4),
        "cv_std":          round(float(cv_scores.std()), 4),
        "feature_importances": fi_sorted,
        "stage_descriptions": {
            "1": {"label": "Stage 1 – Normal/High GFR", "gfr": "≥90",  "color": "#22c55e"},
            "2": {"label": "Stage 2 – Mildly Decreased","gfr": "60–89","color": "#84cc16"},
            "3": {"label": "Stage 3 – Moderately Decreased","gfr":"30–59","color":"#f59e0b"},
            "4": {"label": "Stage 4 – Severely Decreased","gfr":"15–29","color":"#ef4444"},
            "5": {"label": "Stage 5 – Kidney Failure",  "gfr": "<15",  "color": "#7c3aed"},
        },
    }

    with open("artefacts/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved: artefacts/model.pkl | artefacts/scaler.pkl | artefacts/metadata.json")
    return model, scaler, metadata


if __name__ == "__main__":
    train()
