# *Lacticaseibacillus rhamnosus* research framework

This folder contains three independent Spyder-compatible scripts for study-level evidence generation, pooled predictive modeling, and researcher-oriented multi-objective optimization. The source workbook currently expected by the scripts is `rhamnosus.xlsx`, with `Dataset` and `Data Dictionary` worksheets.

The scripts generate evidence and candidate settings for researcher review. They do not establish causal effects, replace experimental validation, or make the researcher's final predictor or experimental decisions. The outputs folder includes the run you executed on November 26, 2026. Those results were inspected read-only; this agent did not rerun the analysis scripts. The updated scripts print progress through each stage and create a visual HTML report for each phase when run.

## Installation

Use a Python 3.10 or later environment available to Spyder. Install the listed packages into that same environment:

```powershell
python -m pip install -r requirements.txt
```

The dependencies are pandas, NumPy, SciPy, Matplotlib, openpyxl, scikit-learn, joblib, and pymoo. Phase 3 requires pymoo for NSGA-II. The Python command in the development shell was unavailable; install through the interpreter configured for Spyder.

## Files and execution order

Keep the scripts, `requirements.txt`, and `rhamnosus.xlsx` in this folder. The manuscript and `Articles` folder are not read by the scripts. Run the scripts from Spyder with this folder as the project location.

1. **`01_study_level_analysis.py`** reads the source workbook, creates study-level and integrated descriptive statistics, association evidence, data-quality summaries, figures, a browser-ready `phase1_visual_report.html`, and an editable decision workbook. Its console prints dataset counts, analysis stages, figure progress, and saved paths. Phase 1 preserves a matching workbook on reruns and creates a new version when the source or script version changes.
2. **`02_pooled_predictive_modelling.py`** reads the same source workbook and matching Phase 1 decision workbook. It prints the filtering counts, feature policies, every modeling scenario, and the held-out study for each validation fold. It saves comparisons, per-study performance, figures, and `phase2_visual_report.html` under a unique `outputs/phase2_runs/` directory.
3. **`03_researcher_oriented_optimization.py`** reads the matching source and decision workbook plus the matching Phase 2 run. It prints the selected models, support thresholds, search starts and completions, and artifact paths. It saves the workbook, CSV alternatives, a log-scale Pareto figure, candidate-setting and model-reliability figures, and `phase3_visual_report.html` under a unique `outputs/phase3_runs/` directory.

Each run directory includes metadata linking its outputs to the exact source and decision workbook. Output directories are created on execution; existing run folders are not reused.

## Scientific and data-handling choices

- Study identity, strain, experimental design, response basis, and response source status are retained as context. Study identifiers and run identifiers are not supplied as numeric predictors.
- Phase 1 reports Pearson and Spearman associations with sample counts and Benjamini–Hochberg adjusted p-values for pooled predictor pairs, response-basis strata, within-study, and within-study/design scopes. Pooled associations are labeled descriptive and are not interpreted as independent biological effects.
- Missing values stay missing in evidence outputs. No row is removed as an outlier in Phase 1, and no missing value is converted to zero. IQR fences are review flags only.
- Phase 2 defaults to `log10 CFU/mL` responses and excludes rows marked plot-derived or provisional. These settings are explicit near the top of `02_pooled_predictive_modelling.py`; changing them changes the modeled population and must be recorded in the run metadata.
- Phase 2 compares all available predictors with researcher-retained predictors and, where numeric limits were entered, a retained-plus-conditional policy. Conditional predictors without machine-readable numeric bounds are not added to that policy. Explicit numeric conditions filter rows to the stated range.
- Only the numeric condition bounds are applied automatically. Free-text conditions are preserved for review and are not parsed; do not rely on free text alone as an optimization constraint.
- Feature imputation, scaling, feature selection, and hyperparameter tuning occur inside study-grouped validation folds. The outer Leave-One-Study-Out loop estimates performance on studies not used for fitting; inner grouped folds tune settings. Scores are compared with a training-mean baseline.
- Phase 2 marks a model eligible for Phase 3 only when it has predictions for at least three held-out studies and beats the training-mean baseline on macro-average study LOSO RMSE. Pooled row-level metrics are also reported. This is a screening rule, not a general validation standard or guarantee of experimental reliability.
- Choosing the best feature policy or model from several LOSO scores can make the best observed score optimistic. Seek external-study or prospective validation before treating an optimization model as reliable.
- Phase 3 only considers eligible models built under researcher-retained or bounded-conditional feature policies. It maximizes predicted response while minimizing distance to the nearest observed predictor profile. The conservative scenario also constrains that distance to the 95th percentile of observed leave-one-profile-out distances; exploratory alternatives are flagged when they exceed that threshold.
- Predictor bounds are restricted to the observed modeling range. Predictors with few observed levels are snapped to observed levels. A point inside every single-variable range can still be a poorly supported combination; use the distance flag and the Phase 2 LOSO evidence when reviewing alternatives.

## Main outputs

- `outputs/research_visual_report.html`: consolidated visual review of the saved November 26, 2026 Phase 1–3 run, with embedded plots and links to the underlying files.
- `outputs/decision_workbooks/researcher_decision_workbook.xlsx` for the first source version, then fingerprinted workbook names for later source versions: editable Phase 1 evidence and researcher decisions.
- `outputs/phase1_runs/<fingerprint>_<timestamp>/`: CSV evidence, figures, study-aware data export, `phase1_visual_report.html`, and metadata.
- `outputs/phase2_runs/<fingerprint>_<timestamp>/`: comparison workbook and CSVs, per-study performance, predictions, figures, `phase2_visual_report.html`, model bundle, and manifest.
- `outputs/phase3_runs/<fingerprint>_<timestamp>/`: optimization workbook and CSVs, log-scale Pareto and candidate-setting figures, `phase3_visual_report.html`, and manifest.

The model bundle is a local Python joblib file and should only be loaded from a run created from this project and its trusted source data. Phase 3 verifies the source and decision workbook fingerprints before loading it.

## Known limits

The framework uses the cleaned values already present in `rhamnosus.xlsx`; it does not independently verify publications or reconstruct source measurements. Different ingredients, strains, designs, and measurement bases may remain scientifically non-comparable even when they share a displayed unit. Pooled predictions can be unstable with few studies, and LOSO performance based on a small number of studies has wide practical uncertainty. Multi-objective outputs are model-dependent candidates and require scientific review and experimental confirmation.
