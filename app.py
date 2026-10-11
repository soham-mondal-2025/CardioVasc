import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load trained models and metadata
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'ml_pipeline', 'saved_models')

print("Loading ML models and SHAP explainers...")
cad_model = joblib.load(os.path.join(MODELS_DIR, 'cath_model.joblib'))
lad_model = joblib.load(os.path.join(MODELS_DIR, 'lad_model.joblib'))
lcx_model = joblib.load(os.path.join(MODELS_DIR, 'lcx_model.joblib'))
rca_model = joblib.load(os.path.join(MODELS_DIR, 'rca_model.joblib'))

shap_explainer = joblib.load(os.path.join(MODELS_DIR, 'shap_explainer.joblib'))
feature_subsets = joblib.load(os.path.join(MODELS_DIR, 'feature_subsets.joblib'))
thresholds = joblib.load(os.path.join(MODELS_DIR, 'thresholds.joblib'))
all_features = joblib.load(os.path.join(MODELS_DIR, 'all_features.joblib'))

with open(os.path.join(MODELS_DIR, 'metrics.json'), 'r') as f:
    model_metrics = json.load(f)

print("Loaded all models successfully!")

# Preset clinical sample patients for demo
SAMPLE_PATIENTS = [
    {
        "id": "patient-1",
        "name": "Patient #1 (Severe Multi-Vessel CAD)",
        "description": "64yo male with typical exertional chest pain, hypertension, diabetes, anterior RWMA, ST depression and T-inversion.",
        "data": {
            "Age": 64,
            "Weight": 82,
            "Length": 172,
            "Sex": 1,
            "BMI": 27.7,
            "DM": 1,
            "HTN": 1,
            "Current Smoker": 1,
            "EX-Smoker": 0,
            "FH": 1,
            "Obesity": 1,
            "CRF": 0,
            "CVA": 0,
            "Airway disease": 0,
            "Thyroid Disease": 0,
            "CHF": 0,
            "DLP": 1,
            "BP": 150,
            "PR": 88,
            "Edema": 0,
            "Weak Peripheral Pulse": 0,
            "Lung rales": 0,
            "Systolic Murmur": 0,
            "Diastolic Murmur": 0,
            "Typical Chest Pain": 1,
            "Dyspnea": 1,
            "Function Class": 2,
            "Atypical": 0,
            "Nonanginal": 0,
            "Exertional CP": 0,
            "LowTH Ang": 0,
            "Q Wave": 1,
            "St Elevation": 0,
            "St Depression": 1,
            "Tinversion": 1,
            "LVH": 1,
            "Poor R Progression": 1,
            "BBB_LBBB": 0,
            "BBB_RBBB": 0,
            "FBS": 165,
            "CR": 1.2,
            "TG": 240,
            "LDL": 145,
            "HDL": 34,
            "BUN": 22,
            "ESR": 28,
            "HB": 14.2,
            "K": 4.3,
            "Na": 140,
            "WBC": 8900,
            "Lymph": 24,
            "Neut": 68,
            "PLT": 260,
            "EF-TTE": 40,
            "Region RWMA": 3,
            "VHD": 1
        }
    },
    {
        "id": "patient-2",
        "name": "Patient #2 (Isolated LAD Stenosis)",
        "description": "55yo female with atypical chest pain, regional anterior wall hypokinesia, elevated LDL/HDL ratio.",
        "data": {
            "Age": 55,
            "Weight": 68,
            "Length": 160,
            "Sex": 0,
            "BMI": 26.5,
            "DM": 0,
            "HTN": 1,
            "Current Smoker": 0,
            "EX-Smoker": 1,
            "FH": 1,
            "Obesity": 0,
            "CRF": 0,
            "CVA": 0,
            "Airway disease": 0,
            "Thyroid Disease": 0,
            "CHF": 0,
            "DLP": 1,
            "BP": 135,
            "PR": 74,
            "Edema": 0,
            "Weak Peripheral Pulse": 0,
            "Lung rales": 0,
            "Systolic Murmur": 0,
            "Diastolic Murmur": 0,
            "Typical Chest Pain": 1,
            "Dyspnea": 0,
            "Function Class": 1,
            "Atypical": 1,
            "Nonanginal": 0,
            "Exertional CP": 0,
            "LowTH Ang": 0,
            "Q Wave": 0,
            "St Elevation": 1,
            "St Depression": 0,
            "Tinversion": 1,
            "LVH": 0,
            "Poor R Progression": 1,
            "BBB_LBBB": 0,
            "BBB_RBBB": 0,
            "FBS": 110,
            "CR": 0.9,
            "TG": 180,
            "LDL": 130,
            "HDL": 42,
            "BUN": 16,
            "ESR": 18,
            "HB": 13.1,
            "K": 4.1,
            "Na": 139,
            "WBC": 7200,
            "Lymph": 30,
            "Neut": 60,
            "PLT": 230,
            "EF-TTE": 48,
            "Region RWMA": 2,
            "VHD": 0
        }
    },
    {
        "id": "patient-3",
        "name": "Patient #3 (Healthy / Low Risk Control)",
        "description": "42yo male asymptomatic, normal ECG, EF 55%, normal lipid panel and no regional wall motion abnormalities.",
        "data": {
            "Age": 42,
            "Weight": 70,
            "Length": 175,
            "Sex": 1,
            "BMI": 22.8,
            "DM": 0,
            "HTN": 0,
            "Current Smoker": 0,
            "EX-Smoker": 0,
            "FH": 0,
            "Obesity": 0,
            "CRF": 0,
            "CVA": 0,
            "Airway disease": 0,
            "Thyroid Disease": 0,
            "CHF": 0,
            "DLP": 0,
            "BP": 115,
            "PR": 68,
            "Edema": 0,
            "Weak Peripheral Pulse": 0,
            "Lung rales": 0,
            "Systolic Murmur": 0,
            "Diastolic Murmur": 0,
            "Typical Chest Pain": 0,
            "Dyspnea": 0,
            "Function Class": 0,
            "Atypical": 0,
            "Nonanginal": 0,
            "Exertional CP": 0,
            "LowTH Ang": 0,
            "Q Wave": 0,
            "St Elevation": 0,
            "St Depression": 0,
            "Tinversion": 0,
            "LVH": 0,
            "Poor R Progression": 0,
            "BBB_LBBB": 0,
            "BBB_RBBB": 0,
            "FBS": 88,
            "CR": 0.8,
            "TG": 95,
            "LDL": 85,
            "HDL": 55,
            "BUN": 12,
            "ESR": 6,
            "HB": 15.0,
            "K": 4.2,
            "Na": 141,
            "WBC": 5800,
            "Lymph": 36,
            "Neut": 54,
            "PLT": 210,
            "EF-TTE": 58,
            "Region RWMA": 0,
            "VHD": 0
        }
    }
]

def preprocess_input(patient_data):
    """Calculates clinical biomarkers and formats into complete feature dictionary."""
    row = patient_data.copy()
    
    # Biomarkers
    hdl = float(row.get('HDL', 40))
    tg = float(row.get('TG', 150))
    ldl = float(row.get('LDL', 100))
    age = float(row.get('Age', 50))
    tcp = float(row.get('Typical Chest Pain', 0))
    
    row['TG_HDL_ratio'] = round(tg / (hdl + 1e-5), 2)
    row['LDL_HDL_ratio'] = round(ldl / (hdl + 1e-5), 2)
    row['Non_HDL'] = round(ldl + 0.2 * tg, 1)
    row['Typical_CP_Age'] = float(tcp * (1 if age > 50 else 0))
    
    ecg_burden = (
        float(row.get('Q Wave', 0)) +
        float(row.get('St Elevation', 0)) +
        float(row.get('St Depression', 0)) +
        float(row.get('Tinversion', 0)) +
        float(row.get('Poor R Progression', 0))
    )
    row['ECG_burden'] = ecg_burden
    
    rwma = float(row.get('Region RWMA', 0))
    ef = float(row.get('EF-TTE', 50))
    row['Echo_score'] = float((1 if rwma > 0 else 0) * 2 + (1 if ef < 50 else 0))
    
    # Ensure all features exist with float type
    feat_df = pd.DataFrame([row])
    for col in all_features:
        if col not in feat_df.columns:
            feat_df[col] = 0.0
            
    return feat_df[all_features].astype(float)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/sample-patients', methods=['GET'])
def get_sample_patients():
    return jsonify(SAMPLE_PATIENTS)


@app.route('/api/model-metrics', methods=['GET'])
def get_model_metrics():
    return jsonify(model_metrics)


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json or {}
        patient_data = data.get('patient', {})
        
        X_input = preprocess_input(patient_data)
        
        # 1. Overall CAD Prediction
        cad_cols = feature_subsets['Cath']
        cad_X = X_input[cad_cols]
        cad_prob = float(cad_model.predict_proba(cad_X)[0, 1])
        cad_th = thresholds['Cath']
        cad_pred = bool(cad_prob >= cad_th)
        
        # 2. LAD Stenosis
        lad_cols = feature_subsets['LAD']
        lad_prob = float(lad_model.predict_proba(X_input[lad_cols])[0, 1])
        lad_th = thresholds['LAD']
        lad_pred = bool(lad_prob >= lad_th)
        
        # 3. LCX Stenosis
        lcx_cols = feature_subsets['LCX']
        lcx_prob = float(lcx_model.predict_proba(X_input[lcx_cols])[0, 1])
        lcx_th = thresholds['LCX']
        lcx_pred = bool(lcx_prob >= lcx_th)
        
        # 4. RCA Stenosis
        rca_cols = feature_subsets['RCA']
        rca_prob = float(rca_model.predict_proba(X_input[rca_cols])[0, 1])
        rca_th = thresholds['RCA']
        rca_pred = bool(rca_prob >= rca_th)
        
        # 5. SHAP Interpretability for CAD
        shap_values = shap_explainer.shap_values(cad_X)
        if isinstance(shap_values, list):
            # Binary classification list [class 0, class 1]
            sv = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            sv = shap_values[0, :, 1]
        else:
            sv = shap_values[0]
            
        shap_contributions = []
        for col, val, imp in zip(cad_cols, cad_X.iloc[0].values, sv):
            shap_contributions.append({
                "feature": col,
                "value": round(float(val), 2),
                "importance": round(float(imp), 4),
                "impact": "increases_risk" if imp > 0 else "decreases_risk"
            })
            
        # Sort by absolute importance
        shap_contributions.sort(key=lambda x: abs(x['importance']), reverse=True)
        top_shap = shap_contributions[:10]
        
        response = {
            "overall_cad": {
                "probability": round(cad_prob * 100, 1),
                "is_cad": cad_pred,
                "risk_category": "High Risk" if cad_prob > 0.65 else ("Moderate Risk" if cad_prob > 0.35 else "Low Risk"),
                "threshold": round(cad_th * 100, 1)
            },
            "vessels": {
                "LAD": {
                    "name": "Left Anterior Descending",
                    "territory": "Anterior Wall & Septum",
                    "probability": round(lad_prob * 100, 1),
                    "stenosis": lad_pred,
                    "risk_color": "#ef4444" if lad_prob >= 0.6 else ("#f59e0b" if lad_prob >= 0.35 else "#10b981")
                },
                "LCX": {
                    "name": "Left Circumflex",
                    "territory": "Lateral Wall & Posterior",
                    "probability": round(lcx_prob * 100, 1),
                    "stenosis": lcx_pred,
                    "risk_color": "#ef4444" if lcx_prob >= 0.6 else ("#f59e0b" if lcx_prob >= 0.35 else "#10b981")
                },
                "RCA": {
                    "name": "Right Coronary Artery",
                    "territory": "Right Ventricle & Inferior Wall",
                    "probability": round(rca_prob * 100, 1),
                    "stenosis": rca_pred,
                    "risk_color": "#ef4444" if rca_prob >= 0.6 else ("#f59e0b" if rca_prob >= 0.35 else "#10b981")
                }
            },
            "shap_explanation": top_shap,
            "calculated_biomarkers": {
                "TG_HDL_ratio": X_input['TG_HDL_ratio'].iloc[0],
                "LDL_HDL_ratio": X_input['LDL_HDL_ratio'].iloc[0],
                "ECG_burden": X_input['ECG_burden'].iloc[0],
                "Echo_score": X_input['Echo_score'].iloc[0]
            }
        }
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Server starting on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
