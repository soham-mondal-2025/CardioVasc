# CardioVas — Track A Technical Report (maximum 6 pages)

## 1. Problem and contribution
Describe the challenge: overall CAD classification, LAD/LCX/RCA stenosis classification, and risk visualization on an interactive 3D heart. State the contribution honestly: integrated UI, input workflow, vessel overlays, interpretable model outputs, and evaluation pipeline.

## 2. Dataset and preprocessing
- Dataset source/version and license:
- Number of rows / columns:
- Target definitions and label mapping:
- Missing data and category handling:
- Leakage prevention: LAD, LCX, RCA, and Cath excluded from predictors:
- Train/test split and random seed:

## 3. Modeling and evaluation
Insert actual results from `ml/artifacts/metrics.json`. Include accuracy, precision, recall, F1 and ROC-AUC per target. Describe class balance and limitations. Do not fill this section with demo endpoint values.

## 4. 3D mapping and interface
Explain interaction (rotate, zoom, click a vessel), how model outputs map to vessel colors, and how clinical feature inputs update the dashboard. Clarify whether the mesh is a licensed anatomical asset or a stylized placeholder.

## 5. Interpretability and responsible use
Explain whether displayed values are true SHAP/LIME values or demo heuristic signals. State that physiological features are not anatomical coordinates and RWMA does not define a pixel-level lesion map. Include clinical safety disclaimer.

## 6. Limitations and next steps
Small sample size, external validation, calibration, uncertainty, dataset bias, and clinical review. Explain model-serving integration and reproducibility steps.
