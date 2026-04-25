"""
CKD Stage Predictor — Flask REST API Backend
Endpoints: /predict, /health, /metadata
"""

import os
import json
import math
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ─── Load artefacts ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(BASE_DIR, "..", "artefacts")

model    = joblib.load(os.path.join(ART_DIR, "model.pkl"))
scaler   = joblib.load(os.path.join(ART_DIR, "scaler.pkl"))
with open(os.path.join(ART_DIR, "metadata.json")) as f:
    META = json.load(f)

FEATURE_COLS    = META["feature_cols"]
CONTINUOUS_COLS = META["continuous_cols"]
STAGE_INFO      = META["stage_descriptions"]

# ─── Helpers ───────────────────────────────────────────────────────────────────
def calculate_gfr(creatinine: float, age: int, is_female: bool) -> float:
    kappa      = 0.7  if is_female else 0.9
    alpha      = -0.329 if is_female else -0.411
    sex_factor = 1.018  if is_female else 1.0
    ratio = creatinine / kappa
    if ratio < 1:
        gfr = 141 * (ratio ** alpha) * (0.9938 ** age) * sex_factor
    else:
        gfr = 141 * (ratio ** -1.209) * (0.9938 ** age) * sex_factor
    return round(gfr, 2)


def interpret_gfr(gfr: float) -> str:
    if gfr >= 90: return "Normal or High"
    if gfr >= 60: return "Mildly Decreased"
    if gfr >= 45: return "Mildly to Moderately Decreased"
    if gfr >= 30: return "Moderately to Severely Decreased"
    if gfr >= 15: return "Severely Decreased"
    return "Kidney Failure"


def clinical_recommendations(stage: int) -> list:
    recs = {
        1: ["Annual kidney function monitoring", "Blood pressure control (<130/80 mmHg)",
            "Lifestyle changes: low-sodium diet, regular exercise", "Avoid NSAIDs and nephrotoxic drugs"],
        2: ["Bi-annual nephrology follow-up", "Strict BP control (<130/80 mmHg)",
            "Manage diabetes if present (HbA1c <7%)", "Protein-restricted diet (0.8g/kg/day)",
            "ACE inhibitors or ARBs if proteinuria present"],
        3: ["Quarterly nephrology visits", "Refer to nephrologist immediately",
            "Erythropoietin for anaemia (Hb <10g/dL)", "Phosphate binders if needed",
            "Dietary protein restriction (0.6g/kg/day)", "Monitor bone mineral density"],
        4: ["Monthly nephrology follow-up", "Begin dialysis preparation education",
            "Arteriovenous fistula planning for dialysis", "Strict fluid and potassium restriction",
            "Manage metabolic acidosis", "Evaluate transplant eligibility"],
        5: ["URGENT: Immediate nephrology referral", "Initiate renal replacement therapy (dialysis)",
            "Kidney transplant evaluation", "Intensive fluid and electrolyte management",
            "Uremia management", "Palliative care consultation if appropriate"],
    }
    return recs.get(stage, [])


def build_feature_vector(data: dict) -> np.ndarray:
    """Extract and order features from request JSON."""
    vec = [
        float(data.get("age", 50)),
        float(data.get("is_female", 0)),
        float(data.get("creatinine", 1.0)),
        float(data.get("blood_urea", 30)),
        float(data.get("sodium", 140)),
        float(data.get("potassium", 4.0)),
        float(data.get("hemoglobin", 13.5)),
        float(data.get("systolic_bp", 120)),
        float(data.get("diastolic_bp", 80)),
        float(data.get("albumin", 0)),
        float(data.get("sugar", 0)),
        float(data.get("hypertension", 0)),
        float(data.get("diabetes_mellitus", 0)),
        float(data.get("coronary_artery_disease", 0)),
        float(data.get("red_blood_cells", 4.5)),
        float(data.get("white_blood_cells", 6000)),
        float(data.get("packed_cell_volume", 42)),
        float(data.get("appetite", 1)),
        float(data.get("pedal_edema", 0)),
        float(data.get("anemia", 0)),
    ]
    return np.array(vec)


# ─── Routes ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "CKD Stage Predictor v1.0"})


@app.route("/metadata", methods=["GET"])
def metadata():
    return jsonify({
        "accuracy":          META["accuracy"],
        "f1_weighted":       META["f1_weighted"],
        "cv_mean":           META["cv_mean"],
        "cv_std":            META["cv_std"],
        "feature_importances": META["feature_importances"],
        "stage_descriptions":  META["stage_descriptions"],
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    # Validate required fields
    required = ["age", "creatinine"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 422

    try:
        # Build raw feature vector
        raw_vec = build_feature_vector(data)
        feat_df = dict(zip(FEATURE_COLS, raw_vec))

        # Scale continuous features
        import pandas as pd
        X = pd.DataFrame([feat_df])
        X[CONTINUOUS_COLS] = scaler.transform(X[CONTINUOUS_COLS])
        X_arr = X[FEATURE_COLS].values

        # Predict
        stage      = int(model.predict(X_arr)[0])
        proba      = model.predict_proba(X_arr)[0]
        confidence = float(np.max(proba))

        stage_probs = {
            f"stage_{i+1}": round(float(p), 4)
            for i, p in enumerate(proba)
        }

        # GFR
        gfr = calculate_gfr(
            float(data.get("creatinine", 1.0)),
            int(data.get("age", 50)),
            bool(data.get("is_female", False)),
        )

        info = STAGE_INFO[str(stage)]

        return jsonify({
            "predicted_stage":   stage,
            "stage_label":       info["label"],
            "stage_color":       info["color"],
            "confidence":        round(confidence, 4),
            "stage_probabilities": stage_probs,
            "estimated_gfr":     gfr,
            "gfr_interpretation": interpret_gfr(gfr),
            "recommendations":   clinical_recommendations(stage),
            "disclaimer": (
                "This tool is for educational/screening purposes only. "
                "It does not replace clinical diagnosis or medical advice."
            ),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("CKD Stage Predictor API running on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
