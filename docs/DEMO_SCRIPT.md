# 5-minute demonstration script

1. **Opening (30 sec):** “CardioVas turns clinical feature inputs into a transparent educational view of overall CAD and three coronary vessel targets.”
2. **Safety / mode (20 sec):** Point out the DEMO MODE badge. Explain that this starter uses a demo heuristic and is not a clinical prediction.
3. **3D interaction (45 sec):** Rotate and zoom the heart. Click LAD, LCX, or RCA overlays to inspect the selected vessel. Explain that the current mesh is stylized and not patient-specific.
4. **Input workflow (60 sec):** Change age, systolic BP, cholesterol, pulse, and ECG/echo toggles. Run prediction and show vessel bars changing.
5. **Interpretability (45 sec):** Show input signal chart. Explain it is a heuristic signal, not SHAP. In final version, replace it with model-based SHAP on held-out examples.
6. **ML evidence (60 sec):** Show training code, leakage exclusions, and real metrics generated from the authorized dataset. Do not claim metrics until actually run.
7. **Closing (20 sec):** Emphasize integrated workflow, explicit safety, reproducibility, and next steps: licensed anatomical mesh, trained models, calibration, external validation.
