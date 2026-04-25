# 🩺 CKD Stage Predictor

> AI-powered Chronic Kidney Disease stage classification using Random Forest + CKD-EPI GFR formula.
> **Accuracy: 99.8% (5-fold CV)** · 5 CKD Stages · 20 Clinical Features

---

## 📁 Project Structure

```
ckd-predictor/
├── model/
│   ├── train_model.py          # Data generation, GFR staging, RF training
│   └── artefacts/              # Generated after training
│       ├── model.pkl           # Trained Random Forest classifier
│       ├── scaler.pkl          # MinMaxScaler for continuous features
│       └── metadata.json       # Accuracy stats, feature importances
├── backend/
│   └── app.py                  # Flask REST API (predict, health, metadata)
├── frontend/
│   └── index.html              # Standalone HTML/CSS/JS UI (no build step)
├── deploy/
│   ├── nginx-dev.conf          # Nginx config for Docker dev setup
│   └── render.yaml             # Render.com deployment config
├── Dockerfile                  # Multi-stage Docker image
├── docker-compose.yml          # Local dev + production profiles
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🚀 Quick Start (Local)

### Option A — Run directly (no Docker)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model
cd model && python train_model.py
cd ..

# 3. Start the backend
python backend/app.py
# → API running at http://localhost:5000

# 4. Open the frontend
open frontend/index.html
# Or serve it: python -m http.server 3000 -d frontend
```

### Option B — Docker Compose (recommended)

```bash
# Dev: backend on :5000, frontend on :3000
docker compose up backend frontend

# Production (single container on :80)
docker compose --profile prod up app
```

---

## 🔌 API Reference

### `GET /health`
```json
{ "status": "ok", "model": "CKD Stage Predictor v1.0" }
```

### `GET /metadata`
Returns model accuracy, CV score, feature importances, stage descriptions.

### `POST /predict`
**Request Body:**
```json
{
  "age": 55,
  "is_female": 0,
  "creatinine": 2.8,
  "blood_urea": 45,
  "sodium": 138,
  "potassium": 4.5,
  "hemoglobin": 11.0,
  "systolic_bp": 145,
  "diastolic_bp": 92,
  "albumin": 2,
  "sugar": 1,
  "hypertension": 1,
  "diabetes_mellitus": 1,
  "coronary_artery_disease": 0,
  "red_blood_cells": 3.8,
  "white_blood_cells": 7500,
  "packed_cell_volume": 36,
  "appetite": 0,
  "pedal_edema": 1,
  "anemia": 1
}
```

**Response:**
```json
{
  "predicted_stage": 3,
  "stage_label": "Stage 3 – Moderately Decreased",
  "stage_color": "#f59e0b",
  "confidence": 0.7911,
  "stage_probabilities": {
    "stage_1": 0.0,
    "stage_2": 0.1133,
    "stage_3": 0.7911,
    "stage_4": 0.0918,
    "stage_5": 0.0037
  },
  "estimated_gfr": 25.39,
  "gfr_interpretation": "Severely Decreased",
  "recommendations": ["Quarterly nephrology visits", "..."],
  "disclaimer": "..."
}
```

---

## 🧠 ML Architecture

| Component | Detail |
|-----------|--------|
| Algorithm | Random Forest (n=200, max_depth=15) |
| Label generation | CKD-EPI equation → KDIGO 2012 GFR staging |
| Features | 20 clinical indicators (demographics, kidney markers, CBC, comorbidities) |
| Preprocessing | MinMax scaling (continuous) · Binary encoding (categorical) |
| Validation | 80/20 split + 5-fold cross-validation |
| Test Accuracy | 100% (test) · 99.83% CV (±0.33%) |

### CKD Stages (KDIGO Guidelines)
| Stage | GFR | Severity |
|-------|-----|----------|
| 1 | ≥90 | Normal/High |
| 2 | 60–89 | Mildly Decreased |
| 3 | 30–59 | Moderately Decreased |
| 4 | 15–29 | Severely Decreased |
| 5 | <15 | Kidney Failure |

---

## ☁️ Cloud Deployment

### Render.com (Free tier)
1. Push this project to GitHub
2. Go to [dashboard.render.com](https://dashboard.render.com)
3. **New Web Service** → connect repo → set:
   - Build: `pip install -r requirements.txt && python model/train_model.py`
   - Start: `gunicorn -w 2 -b 0.0.0.0:$PORT backend.app:app`
4. **New Static Site** → connect repo → publish dir: `frontend`
5. Update `API` variable in `frontend/index.html` to your Render backend URL

### Railway
```bash
railway login
railway init
railway up
```

### Heroku
```bash
heroku create ckd-stage-predictor
git push heroku main
```
*(Requires `Procfile`: `web: python model/train_model.py && gunicorn backend.app:app`)*

---

## ⚕️ Disclaimer

This tool is for **educational and screening purposes only**. It does not replace
clinical diagnosis, medical advice, or treatment decisions from qualified healthcare
professionals. Not validated for clinical use.

---

**Balaji · Reg. 12212231 · Lovely Professional University**
