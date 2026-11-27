# *Lactobacillus rhamnosus* culture analysis

This repository contains a three-stage Python workflow for analyzing literature-derived *L. rhamnosus* culture data, comparing predictive models across studies, and generating candidate culture conditions for researcher review. The source workbook, reference articles, manuscript draft, analysis scripts, and generated reports are kept together in this project.

## Project files

| Path | Purpose |
| --- | --- |
| `rhamnosus.xlsx` | Source data workbook. The scripts read the `Dataset` and `Data Dictionary` sheets. |
| `Articles/` | Reference papers used in the research dataset. |
| `01_study_level_analysis.py` | Phase 1: descriptive statistics, study-level summaries, association evidence, data-quality checks, and a researcher decision workbook. |
| `02_pooled_predictive_modelling.py` | Phase 2: compares feature policies and predictive models using leave-one-study-out evaluation. |
| `03_researcher_oriented_optimization.py` | Phase 3: searches for candidate conditions using eligible models and researcher-approved predictors. |
| `requirements.txt` | Python package requirements. |
| `outputs/` | Generated workbooks, tables, figures, reports, and run metadata. |
| `rhamnosus manuscript - Probiotics and Antimicrobial Proteins.docx` | Manuscript draft associated with this project. |

## Set up

Use Python with `pip`. From the project directory, create and activate a virtual environment, then install the requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you prefer not to activate the environment, install packages through its Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

In the run commands below, replace `python` with `.\.venv\Scripts\python.exe` when using the environment without activating it.

## Run the analysis

Run the scripts in order from the project directory:

```powershell
python 01_study_level_analysis.py
```

Review the generated researcher decision workbook in `outputs/decision_workbooks/`. In its `Predictor_Decisions` sheet, review the predictor decisions and any applicable bounds before modeling. Phase 1 preserves existing decision workbooks when rerun.

Then run model comparison and optimization:

```powershell
python 02_pooled_predictive_modelling.py
python 03_researcher_oriented_optimization.py
```

Each script prints progress and the path to its run directory and HTML report. The scripts use paths relative to their own location and do not require command-line arguments. Optional explicit paths and analysis settings are configured near the top of each script.

Phase 2 and Phase 3 match their inputs using SHA-256 fingerprints. If the source workbook or decision workbook changes, rerun the earlier phase(s) so later stages use matching artifacts. Phase 3 also requires a matching Phase 2 run and at least one researcher-approved model that passes the configured screening rule.

## Workflow and outputs

1. **Study-level evidence:** Phase 1 summarizes the workbook overall and by study, calculates association evidence, records data-quality review flags, and creates a decision workbook. It also writes CSV tables, figures, run metadata, and `phase1_visual_report.html` under `outputs/phase1_runs/`.
2. **Predictive model comparison:** Phase 2 applies the configured response filters, compares feature strategies and models with leave-one-study-out folds, and saves metrics, predictions, fitted models, figures, a workbook, and `phase2_visual_report.html` under `outputs/phase2_runs/`.
3. **Candidate condition search:** Phase 3 uses eligible Phase 2 models to explore candidate predictor settings. It writes an optimization workbook, CSV files, figures, run metadata, and `phase3_visual_report.html` under `outputs/phase3_runs/`.

Run directories are timestamped and include a source-data fingerprint. Decision workbooks are stored separately in `outputs/decision_workbooks/` so researcher review can carry forward between runs.

## Interpretation

- Phase 1 pooled associations are descriptive evidence; they do not establish causal effects or account for all differences between studies.
- Phase 2 uses leave-one-study-out evaluation so each evaluation fold holds out a study. Its results still depend on the number and comparability of studies in the workbook.
- By default, Phase 2 keeps responses recorded on the `log10 CFU/mL` basis and excludes provisional or plot-derived response rows when source-status information is present. Review the run manifest and report for the filters actually applied.
- Phase 3 outputs are model-based candidate conditions for researcher assessment and experimental testing. They are not experimentally confirmed optima.

Check the run-specific workbook, report, and manifest before interpreting results or selecting experiments.
