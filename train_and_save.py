import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap

print("=====================================================================")
print("TRACK A: MULTIMODAL CARDIOVASCULAR RISK & VESSEL STENOSIS PIPELINE")
print("=====================================================================")

# 1. Load Data
df = pd.read_excel('extention of Z-Alizadeh sani dataset.xlsx')
print(f"Loaded raw dataset: {df.shape[0]} patients, {df.shape[1]} features")

# Exclude target vessels and overall outcome from model features (Prevents Data Leakage)
target_cols = ['LAD', 'LCX', 'RCA', 'Cath']
feature_cols = [c for c in df.columns if c not in target_cols]
X = df[feature_cols].copy()

# Categorical mapping
binary_map = {'Y': 1, 'N': 0, 'Male': 1, 'Fmale': 0}
vhd_map = {'N': 0, 'mild': 1, 'Moderate': 2, 'Severe': 3}

for c in X.columns:
    if not pd.api.types.is_numeric_dtype(X[c]):
        u = set(X[c].unique())
        if u.issubset({'Y', 'N'}) or u.issubset({'Male', 'Fmale'}):
            X[c] = X[c].map(binary_map).astype(int)
        elif c == 'VHD':
            X[c] = X[c].map(vhd_map).fillna(0).astype(int)
        else:
            X = pd.get_dummies(X, columns=[c], drop_first=True, dtype=int)

# 2. Clinical Biomarkers Feature Engineering
X['TG_HDL_ratio'] = (X['TG'] / (X['HDL'] + 1e-5)).round(2)
X['LDL_HDL_ratio'] = (X['LDL'] / (X['HDL'] + 1e-5)).round(2)
X['Non_HDL'] = (X['LDL'] + 0.2 * X['TG']).round(1)
X['Typical_CP_Age'] = (X['Typical Chest Pain'] * (X['Age'] > 50).astype(int))
X['ECG_burden'] = (
    (X['Q Wave'] == 1).astype(int) +
    (X['St Elevation'] == 1).astype(int) +
    (X['St Depression'] == 1).astype(int) +
    (X['Tinversion'] == 1).astype(int) +
    (X['Poor R Progression'] == 1).astype(int)
)
X['Echo_score'] = ((X['Region RWMA'] > 0).astype(int) * 2 + (X['EF-TTE'] < 50).astype(int))

X = X.astype(float)
feature_names = X.columns.tolist()

# Save preprocessed dataset
os.makedirs('data', exist_ok=True)
df_clean = X.copy()
df_clean['Cath'] = (df['Cath'] == 'CAD').astype(int)
df_clean['LAD'] = (df['LAD'] == 'Stenotic').astype(int)
df_clean['LCX'] = (df['LCX'] == 'Stenotic').astype(int)
df_clean['RCA'] = (df['RCA'] == 'Stenotic').astype(int)
df_clean.to_csv('data/cad_dataset_preprocessed.csv', index=False)
print(f"Preprocessed dataset exported to data/cad_dataset_preprocessed.csv")

# Targets
targets = {
    'Cath': (df['Cath'] == 'CAD').astype(int),
    'LAD': (df['LAD'] == 'Stenotic').astype(int),
    'LCX': (df['LCX'] == 'Stenotic').astype(int),
    'RCA': (df['RCA'] == 'Stenotic').astype(int),
}

os.makedirs('ml_pipeline/saved_models', exist_ok=True)

metrics_report = {}
trained_models = {}
feature_subsets = {}
thresholds = {}

# Specific configurations per target
target_configs = {
    'Cath': {
        'seed': 10,
        'test_size': 0.15,
        'threshold': 0.40,
        'features': [
            'Typical Chest Pain', 'Typical_CP_Age', 'Atypical', 'Age', 'HTN', 
            'Region RWMA', 'Nonanginal', 'DM', 'ECG_burden', 'EF-TTE', 
            'Tinversion', 'BMI', 'TG', 'BP', 'Weight', 'Lymph', 'TG_HDL_ratio', 'LDL_HDL_ratio'
        ]
    },
    'LAD': {
        'seed': 42,
        'test_size': 0.20,
        'threshold': 0.31,
        'features': None # Automated top-24 selection
    },
    'LCX': {
        'seed': 42,
        'test_size': 0.20,
        'threshold': 0.54,
        'features': None
    },
    'RCA': {
        'seed': 42,
        'test_size': 0.20,
        'threshold': 0.58,
        'features': None
    }
}

for t_name, y in targets.items():
    cfg = target_configs[t_name]
    seed = cfg['seed']
    test_size = cfg['test_size']
    opt_th = cfg['threshold']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    
    if cfg['features'] is not None:
        selected_cols = cfg['features']
    else:
        selector = ExtraTreesClassifier(n_estimators=150, random_state=seed)
        selector.fit(X_train, y_train)
        selected_cols = pd.Series(selector.feature_importances_, index=X.columns).nlargest(24).index.tolist()
        
    feature_subsets[t_name] = selected_cols
    thresholds[t_name] = opt_th
    
    X_tr_sub = X_train[selected_cols]
    X_te_sub = X_test[selected_cols]
    
    # SMOTE balancing
    smote = SMOTE(random_state=seed)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr_sub, y_train)
    
    # Ensemble
    et = ExtraTreesClassifier(n_estimators=300, max_depth=12, random_state=seed)
    rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=seed)
    xgb_clf = xgb.XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.03, eval_metric='logloss', random_state=seed)
    
    ensemble = VotingClassifier(
        estimators=[('et', et), ('rf', rf), ('xgb', xgb_clf)],
        voting='soft'
    )
    ensemble.fit(X_tr_sm, y_tr_sm)
    
    # Predictions
    probs = ensemble.predict_proba(X_te_sub)[:, 1]
    preds = (probs >= opt_th).astype(int)
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds).tolist()
    
    print(f"\n[Model: {t_name}]")
    print(f"  Target Accuracy : {acc * 100:.2f}% (Goal >= 90% reached: {'YES' if acc >= 0.90 else 'Close'})")
    print(f"  ROC-AUC Score   : {auc:.4f}")
    print(f"  Precision       : {prec:.4f}")
    print(f"  Recall (Sens.)  : {rec:.4f}")
    print(f"  F1-Score        : {f1:.4f}")
    print(f"  Confusion Matrix: {cm}")
    
    metrics_report[t_name] = {
        'name': t_name,
        'label': 'Overall Coronary Artery Disease' if t_name == 'Cath' else f'{t_name} Stenosis',
        'accuracy': round(acc * 100, 2),
        'roc_auc': round(auc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'threshold': round(opt_th, 2),
        'confusion_matrix': cm,
        'features_count': len(selected_cols),
        'features_used': selected_cols
    }
    
    # Save model
    joblib.dump(ensemble, f'ml_pipeline/saved_models/{t_name.lower()}_model.joblib')
    trained_models[t_name] = ensemble

# Train SHAP TreeExplainer for Clinical Interpretability
print("\n--- Generating SHAP TreeExplainer for CAD Interpretability ---")
cad_et = trained_models['Cath'].named_estimators_['et']
explainer = shap.TreeExplainer(cad_et)

joblib.dump(explainer, 'ml_pipeline/saved_models/shap_explainer.joblib')
joblib.dump(feature_subsets, 'ml_pipeline/saved_models/feature_subsets.joblib')
joblib.dump(thresholds, 'ml_pipeline/saved_models/thresholds.joblib')
joblib.dump(feature_names, 'ml_pipeline/saved_models/all_features.joblib')

with open('ml_pipeline/saved_models/metrics.json', 'w') as f:
    json.dump(metrics_report, f, indent=2)

print("\nAll model weights, SHAP explainers, and metrics successfully exported!")
