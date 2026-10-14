# ML pipeline (dataset required)

The Track A brief describes 303 records and overall CAD plus LAD/LCX/RCA targets. This folder intentionally contains no fabricated patient dataset or fake metrics.

## Configure
- Place the dataset you are authorized to use at `data/patients.csv`.
- Inspect the real headers and edit `config.json`.
- Target labels must represent the competition's actual definitions. Confirm whether labels are binary, textual, or coded.
- The input matrix excludes `LAD`, `LCX`, `RCA`, and `Cath` (leakage controls required by the brief). Targets are trained separately.
- The script uses stratified train/test splits where possible, a preprocessing pipeline, logistic regression, and reports accuracy, precision, recall, F1, ROC-AUC when computable.
- Small datasets make estimates unstable. Report the split, random seed, class balance, and metrics honestly. Consider repeated stratified cross-validation for final evaluation.

Run:
```bash
pip install -r requirements.txt
python train.py
```
Outputs go to `artifacts/metrics.json` and `artifacts/model_*.joblib`.

The Node API in this starter is not connected to these trained artifacts. Do not imply that the demo endpoint uses them.
