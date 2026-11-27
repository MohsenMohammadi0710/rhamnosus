"""Phase 2: compare feature strategies with nested study-held-out validation.

Spyder workflow: verify provenance -> apply response filters and researcher
decisions -> evaluate each policy/model with Leave-One-Study-Out folds -> write
metrics, predictions, figures, and a browser-ready report. Console output shows
every scenario and held-out fold, including row counts and baseline comparison.
Each run gets a new folder beneath outputs/phase2_runs.
"""

# %% Configuration — set response filters, minimum study count, and any explicit workbook path
from pathlib import Path
from datetime import datetime, timezone
import base64
from html import escape
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.feature_selection import VarianceThreshold
import joblib
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

SCRIPT_VERSION = "1.1.0"
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "rhamnosus.xlsx"
DECISION_WORKBOOK_PATH = None  # Set to a workbook path to bypass unique-match discovery.
OUTPUT_DIR = BASE_DIR / "outputs"
DECISION_DIR = OUTPUT_DIR / "decision_workbooks"
RUN_DIR = OUTPUT_DIR / "phase2_runs"
DATA_SHEET = "Dataset"
DATA_DICTIONARY_SHEET = "Data Dictionary"
DECISION_SHEET = "Predictor_Decisions"
METADATA_SHEET = "Workflow_Metadata"
STUDY_COLUMN = "Study ID"
RESPONSE_COLUMN = "Cell Density Log CFU/mL"
RESPONSE_BASIS_COLUMN = "Response Basis"
SOURCE_STATUS_COLUMN = "Response Source Status"
ALLOWED_RESPONSE_BASES = {"log10 CFU/mL"}
INCLUDE_PROVISIONAL_RESPONSES = False
MINIMUM_STUDIES_FOR_LOSO = 3
TOP_K_FEATURES = 5
N_JOBS = 1  # Stable in Spyder on Windows; increase after confirming local resources.


# %% Input and output helpers — match the decision workbook to the source fingerprint
def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_run_directory(root, fingerprint):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = root / f"{fingerprint[:10]}_{stamp}"
    candidate, suffix = base, 1
    while candidate.exists():
        candidate = root / f"{base.name}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def find_decision_workbook(fingerprint):
    if DECISION_WORKBOOK_PATH is not None:
        path = Path(DECISION_WORKBOOK_PATH)
        if not path.is_absolute():
            path = BASE_DIR / path
        candidates = [path]
    else:
        candidates = sorted(DECISION_DIR.glob("researcher_decision_workbook*.xlsx"))
    matches = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            metadata = pd.read_excel(path, sheet_name=METADATA_SHEET)
            if not metadata.empty and str(metadata.loc[0, "Source dataset SHA256"]) == fingerprint:
                analysis_date = str(metadata.loc[0, "Analysis date UTC"])
                matches.append((analysis_date, path))
        except Exception:
            continue
    if not matches:
        raise FileNotFoundError(
            "No readable Phase 1 decision workbook matches the current source dataset. "
            "Run 01_study_level_analysis.py and complete its Predictor_Decisions sheet."
        )
    matches.sort(key=lambda item: item[0], reverse=True)
    if len(matches) > 1 and matches[0][0] == matches[1][0]:
        raise ValueError(
            "Multiple matching decision workbooks have the same analysis date. Set "
            "DECISION_WORKBOOK_PATH explicitly so the reviewed workbook is unambiguous."
        )
    if len(matches) > 1:
        print(f"Multiple matching workbooks exist; using the newest evidence workbook: {matches[0][1]}")
    return matches[0][1]


def read_decisions(path, predictor_names):
    decisions = pd.read_excel(path, sheet_name=DECISION_SHEET)
    required = {"Predictor name", "Researcher decision", "Conditions or restrictions",
                "Condition Minimum", "Condition Maximum"}
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise ValueError(f"Decision workbook is missing required fields: {missing}")
    decisions = decisions[decisions["Predictor name"].astype(str).isin(predictor_names)].copy()
    missing_predictors = sorted(set(predictor_names).difference(decisions["Predictor name"].astype(str)))
    if missing_predictors:
        raise ValueError(f"Decision workbook has no row for predictors: {missing_predictors}")
    decisions["Researcher decision"] = decisions["Researcher decision"].fillna("").astype(str).str.strip()
    decisions["Decision normalized"] = decisions["Researcher decision"].str.casefold()
    invalid = decisions.loc[
        decisions["Decision normalized"].ne("")
        & ~decisions["Decision normalized"].isin({"retain", "exclude", "conditional"}),
        "Predictor name",
    ].tolist()
    if invalid:
        raise ValueError(f"Unrecognized researcher decision for predictors: {invalid}")
    unassigned = decisions.loc[decisions["Decision normalized"].eq(""), "Predictor name"].tolist()
    if unassigned:
        raise ValueError(f"Enter a decision for every predictor before modeling: {unassigned}")
    if decisions["Predictor name"].duplicated().any():
        raise ValueError("Decision workbook has duplicate rows for one or more predictors.")
    conditional_rows = decisions[decisions["Decision normalized"].eq("conditional")]
    undocumented = []
    for _, row in conditional_rows.iterrows():
        condition_value = row.get("Conditions or restrictions", "")
        condition_text = str(condition_value).strip() if pd.notna(condition_value) else ""
        lower = numeric_bound(row.get("Condition Minimum"))
        upper = numeric_bound(row.get("Condition Maximum"))
        if not condition_text and lower is None and upper is None:
            undocumented.append(row["Predictor name"])
    if undocumented:
        raise ValueError(f"Document a condition or numeric bound for Conditional predictors: {undocumented}")
    return decisions


def numeric_bound(value):
    return float(value) if pd.notna(value) and str(value).strip() != "" else None


def condition_rules(decisions, features):
    rules = {}
    for _, row in decisions[decisions["Predictor name"].isin(features)].iterrows():
        lower = numeric_bound(row.get("Condition Minimum"))
        upper = numeric_bound(row.get("Condition Maximum"))
        if lower is not None or upper is not None:
            if lower is not None and upper is not None and lower > upper:
                raise ValueError(f"Condition Minimum exceeds Condition Maximum for {row['Predictor name']}.")
            rules[row["Predictor name"]] = (lower, upper)
    return rules


def apply_numeric_conditions(frame, rules):
    filtered = frame.copy()
    descriptions = []
    for feature, (lower, upper) in rules.items():
        values = pd.to_numeric(filtered[feature], errors="coerce")
        keep = values.notna()
        if lower is not None:
            keep &= values.ge(lower)
        if upper is not None:
            keep &= values.le(upper)
        filtered = filtered.loc[keep].copy()
        descriptions.append(f"{feature}: {lower if lower is not None else '-inf'} to {upper if upper is not None else '+inf'}")
    return filtered, "; ".join(descriptions)


# %% Model and feature strategy definitions — keep preprocessing and selection inside each training fold
def model_definition(name):
    if name == "Ridge":
        estimator = Ridge()
        grid = {"model__alpha": [0.1, 1.0, 10.0, 100.0]}
        needs_scaling = True
    elif name == "RandomForest":
        estimator = RandomForestRegressor(n_estimators=300, random_state=41, n_jobs=1)
        grid = {"model__max_features": [0.7, 1.0], "model__min_samples_leaf": [1, 3]}
        needs_scaling = False
    elif name == "ExtraTrees":
        estimator = ExtraTreesRegressor(n_estimators=300, random_state=41, n_jobs=1)
        grid = {"model__max_features": [0.7, 1.0], "model__min_samples_leaf": [1, 3]}
        needs_scaling = False
    elif name == "SVR":
        estimator = SVR(kernel="rbf")
        grid = {"model__C": [0.1, 1.0, 10.0], "model__epsilon": [0.1, 0.5]}
        needs_scaling = True
    else:
        raise ValueError(f"Unknown model: {name}")
    return estimator, grid, needs_scaling


def build_pipeline(selector_name, model_name, input_feature_count):
    model, grid, needs_scaling = model_definition(model_name)
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if selector_name == "low_variance":
        steps.append(("selector", VarianceThreshold(threshold=0.0)))
    elif selector_name == "top_k":
        k = max(1, min(TOP_K_FEATURES, input_feature_count))
        steps.append(("selector", SelectKBest(score_func=f_regression, k=k)))
    elif selector_name == "pca_95_percent":
        steps.extend([("scale", StandardScaler()), ("selector", PCA(n_components=0.95, svd_solver="full"))])
    elif selector_name != "all_features":
        raise ValueError(f"Unknown selector strategy: {selector_name}")
    if needs_scaling and selector_name != "pca_95_percent":
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps), grid


def selected_features_from_estimator(estimator, selector_name, input_features):
    if selector_name == "all_features":
        return input_features
    selector = estimator.named_steps.get("selector")
    if selector_name in {"low_variance", "top_k"} and selector is not None:
        support = selector.get_support()
        return [feature for feature, keep in zip(input_features, support) if keep]
    return input_features  # PCA uses all listed inputs; component loadings are reported separately.


def model_contribution_summary(estimator, selector_name, input_features):
    model = estimator.named_steps["model"]
    raw = getattr(model, "feature_importances_", None)
    method = "tree impurity importance; predictive association only"
    if raw is None and hasattr(model, "coef_"):
        raw = np.abs(np.asarray(model.coef_).reshape(-1))
        method = "absolute standardized linear coefficient; predictive association only"
    if raw is None:
        return ""
    scores = np.asarray(raw, dtype=float).reshape(-1)
    if selector_name == "pca_95_percent":
        pca = estimator.named_steps.get("selector")
        if pca is not None and len(scores) == pca.components_.shape[0]:
            scores = np.abs(pca.components_).T @ scores
            method = "absolute PCA-loading-weighted model importance proxy; predictive association only"
        names = input_features
    elif selector_name in {"low_variance", "top_k"}:
        names = selected_features_from_estimator(estimator, selector_name, input_features)
    else:
        names = input_features
    if len(names) != len(scores):
        return ""
    return json.dumps({
        "method": method,
        "scores": [{"predictor": name, "score": float(score)}
                   for name, score in sorted(zip(names, scores), key=lambda item: item[1], reverse=True)],
    }, ensure_ascii=False)


def metric_summary(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "N predictions": len(actual),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2 pooled OOF": float(r2_score(actual, predicted)) if len(actual) >= 2 else np.nan,
    }


def write_summary_workbook(path, metrics, study_metrics, selections, policy_status,
                           failures, registry, phase1_stats, phase1_response_associations, metadata):
    instructions = pd.DataFrame([
        ["Purpose", "Compare predictor policies, feature-management methods, and regression models using study-held-out predictions."],
        ["Primary validation", "Outer Leave-One-Study-Out (LOSO); each test fold contains one study not used during fitting."],
        ["Tuning", "Hyperparameters and feature selection are tuned within inner study-grouped folds of the outer training studies."],
        ["Baseline", "Compare every scenario with an outer-fold training-mean predictor on the same feature-policy rows."],
        ["Interpretation", "Pooled out-of-fold scores summarize this dataset. Review per-study errors and sample counts; a small number of studies limits certainty."],
        ["Eligibility", "A model is screened for Phase 3 only if it covers at least the configured minimum number of held-out studies and beats the training-mean baseline on macro-average study RMSE. This is not experimental validation."],
        ["Model comparison limitation", "Choosing a feature policy or model by its LOSO score can make the best observed score optimistic. Treat comparisons as internal evidence and seek external-study or prospective validation."],
        ["Researcher decisions", "The researcher-retained policy uses only Retain decisions. Bounded Conditional predictors are evaluated separately after filtering rows to their stated numeric ranges."],
        ["Restrictions", "Only Condition Minimum and Condition Maximum are applied as computational row filters. Free-text conditions are preserved as context and are not interpreted by the scripts."],
        ["Association and causality", "Feature selection and prediction do not establish causal effects or biological importance."],
    ], columns=["Topic", "Guidance"])
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        instructions.to_excel(writer, sheet_name="Instructions", index=False)
        metrics.to_excel(writer, sheet_name="Model_Comparison_LOSO", index=False)
        study_metrics.to_excel(writer, sheet_name="Performance_By_Study", index=False)
        selections.to_excel(writer, sheet_name="Feature_Selection_By_Fold", index=False)
        policy_status.to_excel(writer, sheet_name="Feature_Policy_Status", index=False)
        failures.to_excel(writer, sheet_name="Model_Failures", index=False)
        registry.to_excel(writer, sheet_name="Model_Registry", index=False)
        phase1_stats.to_excel(writer, sheet_name="Phase1_Predictor_Evidence", index=False)
        phase1_response_associations.to_excel(writer, sheet_name="Phase1_Response_Associations", index=False)
        pd.DataFrame([metadata]).to_excel(writer, sheet_name="Workflow_Metadata", index=False)
    workbook = load_workbook(path)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2" if sheet.max_row > 1 else "A1"
        if sheet.max_row > 1:
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.fill = header_fill
                cell.font = Font(color="FFFFFF", bold=True)
                cell.alignment = Alignment(vertical="top", wrap_text=True)
        for cells in sheet.columns:
            letter = get_column_letter(cells[0].column)
            length = max((len(str(cell.value)) for cell in cells if cell.value is not None), default=10)
            sheet.column_dimensions[letter].width = min(max(12, length + 2), 48)
    workbook.save(path)


# %% Visual outputs
def save_phase2_figures(metrics, predictions, study_metrics, run_dir):
    """Summarize cross-study error and inspect out-of-fold predictions visually."""
    complete = metrics.loc[metrics["Status"].eq("complete")].copy()
    if complete.empty:
        print("  No completed scenarios to plot.")
        return {"comparison": None, "predictions": None, "study_errors": None, "best_scenario": ""}

    complete = complete.sort_values("LOSO RMSE macro by study", ascending=True)
    top = complete.head(12).copy()
    labels = top.apply(lambda row: f"{row['Feature policy']} | {row['Selector']} | {row['Model']}", axis=1)
    colors = ["#25855A" if bool(value) else "#9AA9B8"
              for value in top["Optimization eligible by screening rule"]]
    positions = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(12, max(6, len(top) * 0.48)))
    ax.barh(positions, top["LOSO RMSE macro by study"], color=colors, alpha=0.9)
    ax.scatter(top["Baseline mean RMSE macro by study"], positions, marker="D",
               color="#D1495B", edgecolor="white", zorder=3, label="Training-mean baseline")
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Macro-average RMSE across held-out studies (lower is better)")
    ax.set_title("Best 12 completed scenarios: study-level LOSO error")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    comparison_path = run_dir / "loso_model_comparison.png"
    fig.savefig(comparison_path, dpi=180)
    plt.close(fig)

    best_key = str(complete.iloc[0]["Scenario"])
    best_predictions = predictions.loc[predictions["Scenario"].eq(best_key)].copy()
    predictions_path = None
    if not best_predictions.empty:
        fig, ax = plt.subplots(figsize=(8, 7))
        studies = sorted(best_predictions["Study ID"].astype(str).unique())
        palette = plt.cm.tab10(np.linspace(0, 1, max(1, len(studies))))
        for color, study in zip(palette, studies):
            subset = best_predictions.loc[best_predictions["Study ID"].astype(str).eq(study)]
            ax.scatter(subset["Observed"], subset["Predicted"], color=color, alpha=0.75,
                       s=38, label=study)
        low = min(best_predictions["Observed"].min(), best_predictions["Predicted"].min())
        high = max(best_predictions["Observed"].max(), best_predictions["Predicted"].max())
        ax.plot([low, high], [low, high], color="#36454F", linestyle="--", linewidth=1)
        ax.set_xlabel("Observed held-out response (log CFU/mL)")
        ax.set_ylabel("Predicted held-out response (log CFU/mL)")
        ax.set_title("Best internal LOSO scenario, colored by held-out study")
        ax.legend(title="Held-out study", fontsize=7, frameon=False, loc="best")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        predictions_path = run_dir / "best_scenario_observed_vs_predicted.png"
        fig.savefig(predictions_path, dpi=180)
        plt.close(fig)

    study_path = None
    scenario_studies = study_metrics.loc[study_metrics["Scenario"].eq(best_key)].copy()
    if not scenario_studies.empty:
        scenario_studies = scenario_studies.sort_values("RMSE", ascending=False)
        fig, ax = plt.subplots(figsize=(10, max(4, len(scenario_studies) * 0.7)))
        ax.barh(scenario_studies["Study ID"].astype(str), scenario_studies["RMSE"], color="#277DA1")
        ax.invert_yaxis()
        ax.set_xlabel("RMSE for the held-out study")
        ax.set_title("Best scenario: error by held-out study")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        study_path = run_dir / "best_scenario_error_by_study.png"
        fig.savefig(study_path, dpi=180)
        plt.close(fig)
    return {"comparison": comparison_path, "predictions": predictions_path,
            "study_errors": study_path, "best_scenario": best_key}


def write_phase2_visual_report(run_dir, metrics, metadata, figure_paths, rows_after_filter):
    """Write a browser-ready report and embed PNGs so it can move as one file."""
    complete = metrics.loc[metrics["Status"].eq("complete")].copy()
    complete = complete.sort_values("LOSO RMSE macro by study", ascending=True)
    if complete.empty:
        best = None
        top = complete
        eligible_count = 0
    else:
        best = complete.iloc[0]
        top = complete.head(12)
        eligible_count = int(complete["Optimization eligible by screening rule"].fillna(False).astype(bool).sum())

    display_columns = [column for column in ["Feature policy", "Selector", "Model", "LOSO studies",
                       "LOSO RMSE macro by study", "Baseline mean RMSE macro by study",
                       "R2 pooled OOF", "Optimization eligible by screening rule"] if column in top.columns]
    display = top[display_columns].rename(columns={
        "LOSO studies": "Held-out studies", "LOSO RMSE macro by study": "LOSO RMSE",
        "Baseline mean RMSE macro by study": "Baseline RMSE", "R2 pooled OOF": "Pooled OOF R2",
        "Optimization eligible by screening rule": "Passed Phase 3 screen",
    })
    table_html = display.to_html(index=False, escape=True, border=0, classes="data-table",
                                 formatters={"LOSO RMSE": lambda value: f"{value:.3f}",
                                             "Baseline RMSE": lambda value: f"{value:.3f}",
                                             "Pooled OOF R2": lambda value: f"{value:.3f}"})

    def image_tag(key, alt, caption):
        image_path = figure_paths.get(key)
        if image_path is None or not Path(image_path).exists():
            return ""
        image_data = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        return (f'<figure><img src="data:image/png;base64,{image_data}" alt="{escape(alt)}">'
                f'<figcaption>{escape(caption)}</figcaption></figure>')

    if best is None:
        best_label = "No completed model scenarios"
        best_rmse = "n/a"
        baseline_rmse = "n/a"
    else:
        best_label = escape(str(best["Scenario"]))
        best_rmse = f"{best['LOSO RMSE macro by study']:.3f}"
        baseline_rmse = f"{best['Baseline mean RMSE macro by study']:.3f}"
    style = """body{margin:0;background:#f3f6f8;color:#18324a;font:16px/1.55 Segoe UI,Arial,sans-serif}
main{max-width:1120px;margin:auto;padding:32px 22px 56px}h1{margin:0}h2{margin-top:32px;border-bottom:1px solid #d9e2ea;padding-bottom:8px}
.sub,figcaption{color:#587083}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:20px 0}
.card,.panel,figure{background:white;border:1px solid #d9e2ea;border-radius:12px;padding:15px}.card b{display:block;font-size:1.55rem;color:#277da1}
.card span{color:#587083}.note{background:#fff3d8;border-left:4px solid #d89b25;padding:14px 16px;border-radius:5px}
figure{margin:20px 0 28px}figure img{display:block;max-width:100%;height:auto;margin:auto}figcaption{font-size:.92rem;margin-top:10px}
.data-table{border-collapse:collapse;width:100%;background:white;font-size:.9rem}.data-table th,.data-table td{border-bottom:1px solid #d9e2ea;padding:9px;text-align:left;vertical-align:top}.data-table th{background:#e7eff4}
a{color:#126782}li{margin:7px 0}"""
    report = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phase 2 visual report</title><style>{style}</style></head><body><main>
<h1>Study-held-out predictive modeling</h1><p class="sub">Phase 2 | Generated {escape(metadata['Analysis date UTC'])} | Basis: {escape(metadata['Response basis filter'])}</p>
<div class="cards"><div class="card"><b>{rows_after_filter}</b><span>rows after filters</span></div>
<div class="card"><b>{metadata['Studies after filters']}</b><span>held-out studies</span></div>
<div class="card"><b>{len(complete)}</b><span>completed scenarios</span></div>
<div class="card"><b>{eligible_count}</b><span>passed Phase 3 screen</span></div>
<div class="card"><b>{best_rmse}</b><span>best macro study RMSE</span></div></div>
<div class="note"><b>Interpretation:</b> best scenario: {best_label}; macro study RMSE {best_rmse}, compared with training-mean baseline {baseline_rmse}. These are internal study-held-out results. The small number of studies and selection of the best among many scenarios limit certainty; external or prospective validation is still needed.</div>
<h2>Model comparison</h2>{image_tag('comparison', 'Top model scenarios and training mean baseline', 'Bars show macro-average RMSE across held-out studies; red diamonds show each scenario’s training-mean baseline. Lower RMSE is better.')}
<h2>Observed and predicted</h2>{image_tag('predictions', 'Observed versus predicted values for the best scenario', 'Each point is a held-out prediction; colors identify the held-out study. The dashed line indicates perfect agreement.')}
<h2>Error across studies</h2>{image_tag('study_errors', 'Best scenario error for each held-out study', 'This plot shows whether the overall score hides study-specific differences.')}
<h2>Top completed scenarios</h2><div class="panel">{table_html}</div>
<h2>Files</h2><ul><li><a href="phase2_model_comparison.xlsx">Comparison workbook</a></li><li><a href="model_comparison_loso.csv">All scenario metrics</a></li><li><a href="loso_predictions.csv">Out-of-fold predictions</a></li><li><a href="performance_by_held_out_study.csv">Per-study performance</a></li></ul>
</main></body></html>"""
    report_path = run_dir / "phase2_visual_report.html"
    report_path.write_text(report, encoding="utf-8")
    return report_path


# %% Nested leave-one-study-out evaluation — hold out a complete study for every outer test fold
def evaluate_scenario(frame, features, selector_name, model_name, scenario_key,
                      feature_policy, condition_description):
    """Evaluate one model/selector with the study held out at the outer fold."""
    x = frame[features].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(frame[RESPONSE_COLUMN], errors="coerce")
    groups = frame[STUDY_COLUMN].astype(str)
    outer = LeaveOneGroupOut()
    predictions = []
    selections = []
    failures = []
    baseline_predictions = []
    for fold_number, (train_index, test_index) in enumerate(outer.split(x, y, groups), start=1):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        groups_train = groups.iloc[train_index]
        held_out_study = str(groups.iloc[test_index].iloc[0])
        print(f"      Fold {fold_number}/{groups.nunique()}: holding out study {held_out_study}; tuning on other studies...")
        pipeline, grid = build_pipeline(selector_name, model_name, len(features))
        try:
            n_inner_groups = int(groups_train.nunique())
            if n_inner_groups >= 2:
                n_splits = min(3, n_inner_groups)
                inner = GroupKFold(n_splits=n_splits)
                search = GridSearchCV(
                    pipeline, grid, cv=inner, scoring="neg_root_mean_squared_error",
                    n_jobs=N_JOBS, refit=True, error_score="raise",
                )
                search.fit(x_train, y_train, groups=groups_train)
                fitted = search.best_estimator_
                best_params = search.best_params_
            else:
                fitted = pipeline.fit(x_train, y_train)
                best_params = {"note": "Inner group tuning unavailable: fewer than two training studies."}
            test_pred = fitted.predict(x_test)
            selected = selected_features_from_estimator(fitted, selector_name, features)
            component_loadings = ""
            if selector_name == "pca_95_percent":
                pca = fitted.named_steps.get("selector")
                if pca is not None:
                    component_loadings = json.dumps([
                        {feature: float(loading) for feature, loading in zip(features, component)}
                        for component in pca.components_
                    ], ensure_ascii=False)
            contribution_summary = model_contribution_summary(fitted, selector_name, features)
            for index, actual, predicted in zip(test_index, y_test, test_pred):
                predictions.append({
                    "Scenario": scenario_key, "Feature policy": feature_policy,
                    "Selector": selector_name, "Model": model_name,
                    "Study ID": str(groups.iloc[index]), "Observed": float(actual),
                    "Predicted": float(predicted), "Fold": fold_number,
                })
            print(f"      Fold {fold_number}/{groups.nunique()} complete: {len(test_index)} held-out predictions.")
            selections.append({
                "Scenario": scenario_key, "Feature policy": feature_policy,
                "Selector": selector_name, "Model": model_name,
                "Held-out study": held_out_study,
                "Input features": "; ".join(features),
                "Selected original features": "; ".join(selected),
                "PCA component loadings JSON": component_loadings,
                "Model contribution summary JSON": contribution_summary,
                "Best parameters": json.dumps(best_params, ensure_ascii=False),
            })
            baseline_value = float(y_train.mean())
            for index, actual in zip(test_index, y_test):
                baseline_predictions.append({
                    "Feature policy": feature_policy, "Study ID": str(groups.iloc[index]),
                    "Observed": float(actual), "Predicted": baseline_value,
                })
        except Exception as error:
            failures.append(f"Fold {fold_number} ({held_out_study}): {type(error).__name__}: {error}")
            break
    if failures:
        return {"status": "failed", "error": " | ".join(failures),
                "predictions": predictions, "selections": selections,
                "baseline": baseline_predictions, "final_model": None,
                "metrics": {}}
    if not predictions:
        return {"status": "failed", "error": "No outer LOSO predictions were produced.",
                "predictions": [], "selections": [], "baseline": [],
                "final_model": None, "metrics": {}}
    pred_frame = pd.DataFrame(predictions)
    baseline_frame = pd.DataFrame(baseline_predictions)
    metrics = metric_summary(pred_frame["Observed"], pred_frame["Predicted"])
    baseline_metrics = metric_summary(baseline_frame["Observed"], baseline_frame["Predicted"])
    n_studies = int(pred_frame["Study ID"].nunique())
    study_metrics = []
    for study, subset in pred_frame.groupby("Study ID", sort=True):
        local = metric_summary(subset["Observed"], subset["Predicted"])
        study_metrics.append({"Scenario": scenario_key, "Study ID": study, **local})
    baseline_study_metrics = [
        metric_summary(subset["Observed"], subset["Predicted"])
        for _, subset in baseline_frame.groupby("Study ID", sort=True)
    ]
    macro_rmse = float(np.mean([item["RMSE"] for item in study_metrics]))
    macro_mae = float(np.mean([item["MAE"] for item in study_metrics]))
    baseline_macro_rmse = float(np.mean([item["RMSE"] for item in baseline_study_metrics]))
    baseline_macro_mae = float(np.mean([item["MAE"] for item in baseline_study_metrics]))
    metrics.update({
        "LOSO studies": n_studies,
        "LOSO RMSE macro by study": macro_rmse,
        "LOSO MAE macro by study": macro_mae,
        "Baseline mean RMSE": baseline_metrics["RMSE"],
        "Baseline mean MAE": baseline_metrics["MAE"],
        "Baseline mean RMSE macro by study": baseline_macro_rmse,
        "Baseline mean MAE macro by study": baseline_macro_mae,
        "Beats training-mean baseline": macro_rmse < baseline_macro_rmse,
        "Optimization eligible by screening rule": (
            n_studies >= MINIMUM_STUDIES_FOR_LOSO and macro_rmse < baseline_macro_rmse
        ),
    })

    # Refit with study-grouped tuning on all available development observations for Phase 3.
    print("      Outer folds complete; refitting the Phase 3 model with grouped tuning on all eligible rows...")
    try:
        final_pipeline, final_grid = build_pipeline(selector_name, model_name, len(features))
        if groups.nunique() >= 2:
            final_cv = GroupKFold(n_splits=min(3, int(groups.nunique())))
            final_search = GridSearchCV(
                final_pipeline, final_grid, cv=final_cv,
                scoring="neg_root_mean_squared_error", n_jobs=N_JOBS,
                refit=True, error_score="raise",
            )
            final_search.fit(x, y, groups=groups)
            final_model = final_search.best_estimator_
            final_params = final_search.best_params_
        else:
            final_model = final_pipeline.fit(x, y)
            final_params = {"note": "Inner group tuning unavailable."}
        print("      Final fit complete; fitted artifact is ready for the registry.")
    except Exception as error:
        final_model = None
        final_params = {"fit error": f"{type(error).__name__}: {error}"}
        metrics["Optimization eligible by screening rule"] = False
        print(f"      Final fit failed; this scenario will not be eligible for Phase 3: {final_params['fit error']}")
    return {
        "status": "complete", "error": "", "predictions": predictions,
        "selections": selections, "baseline": baseline_predictions,
        "study_metrics": study_metrics, "metrics": metrics,
        "final_model": final_model, "final_parameters": final_params,
        "features": features, "condition_description": condition_description,
        "feature_policy": feature_policy, "selector": selector_name,
        "model_name": model_name,
    }


# %% Run Phase 2 — print filters, policies, scenarios, fold progress, and output paths
if __name__ == "__main__":
    print("PHASE 2 — nested study-held-out model comparison")
    print(f"[1/6] Reading source dataset and decision workbook from: {BASE_DIR}")
    fingerprint = sha256_file(DATASET_PATH)
    dataset = pd.read_excel(DATASET_PATH, sheet_name=DATA_SHEET)
    dictionary = pd.read_excel(DATASET_PATH, sheet_name=DATA_DICTIONARY_SHEET)
    predictor_names = dictionary.loc[
        dictionary["Variable Category"].astype(str).str.strip().str.casefold().eq("predictor"),
        "Variable Name",
    ].dropna().astype(str).tolist()
    predictor_names = [feature for feature in predictor_names if feature in dataset.columns]
    decision_path = find_decision_workbook(fingerprint)
    print(f"      Source rows: {len(dataset)}; dictionary predictors: {len(predictor_names)}.")
    print(f"      Matched decision workbook: {decision_path}")
    decisions = read_decisions(decision_path, predictor_names)
    print("      Researcher decisions: " + ", ".join(
        f"{label}={count}" for label, count in decisions["Decision normalized"].value_counts().sort_index().items()
    ))
    if not decisions["Decision normalized"].eq("retain").any():
        raise ValueError("At least one predictor must be marked Retain before modeling.")
    decision_fingerprint = sha256_file(decision_path)
    phase1_stats = pd.read_excel(decision_path, sheet_name="Integrated_Predictor_Statistics")
    phase1_associations = pd.read_excel(decision_path, sheet_name="Association_Evidence")
    phase1_response_associations = phase1_associations[
        phase1_associations["Variable 2"].eq(RESPONSE_COLUMN)
    ].copy()

    print("[2/6] Applying configured response-basis and source-status filters...")
    usable = dataset.copy()
    for feature in predictor_names + [RESPONSE_COLUMN]:
        usable[feature] = pd.to_numeric(usable[feature], errors="coerce")
    basis_mask = usable[RESPONSE_BASIS_COLUMN].astype(str).isin(ALLOWED_RESPONSE_BASES)
    excluded_basis_rows = int((~basis_mask).sum())
    usable = usable.loc[basis_mask].copy()
    if not INCLUDE_PROVISIONAL_RESPONSES and SOURCE_STATUS_COLUMN in usable.columns:
        status = usable[SOURCE_STATUS_COLUMN].astype(str).str.casefold()
        provisional_mask = status.str.contains("plot-derived|provisional", regex=True, na=False)
        excluded_provisional_rows = int(provisional_mask.sum())
        usable = usable.loc[~provisional_mask].copy()
    else:
        excluded_provisional_rows = 0
    print(f"      Excluded {excluded_basis_rows} rows with a different response basis and "
          f"{excluded_provisional_rows} provisional/plot-derived rows.")
    usable = usable.dropna(subset=[STUDY_COLUMN, RESPONSE_COLUMN]).copy()
    study_count = int(usable[STUDY_COLUMN].nunique())
    print(f"      Modeling set: {len(usable)} rows from {study_count} studies after removing rows without a response or study ID.")
    if study_count < MINIMUM_STUDIES_FOR_LOSO:
        raise ValueError(
            f"Only {study_count} compatible, non-provisional studies remain. "
            "Study-held-out comparisons require at least "
            f"{MINIMUM_STUDIES_FOR_LOSO}; review the response-basis and provisional-source settings."
        )
    if not predictor_names:
        raise ValueError("No predictors from the Data Dictionary are present in the dataset.")

    print("[3/6] Building all-available and researcher-approved feature policies...")
    policies = []
    all_available = [feature for feature in predictor_names if usable[feature].notna().any()]
    policies.append({"name": "all_available", "features": all_available,
                     "frame": usable.copy(), "description": "All predictors with at least one compatible observed value.",
                     "condition_description": "No researcher numeric bounds applied."})

    retained = decisions.loc[decisions["Decision normalized"].eq("retain"), "Predictor name"].tolist()
    retained = [feature for feature in retained if feature in all_available]
    retain_rules = condition_rules(decisions, retained)
    retained_frame, retained_conditions = apply_numeric_conditions(usable, retain_rules)
    policies.append({"name": "researcher_retained", "features": retained,
                     "frame": retained_frame,
                     "description": "Predictors marked Retain; explicit numeric bounds are applied to the rows.",
                     "condition_description": retained_conditions or "No numeric bounds entered for retained predictors."})

    conditional_rows = decisions[decisions["Decision normalized"].eq("conditional")].copy()
    conditional_with_bounds = []
    conditional_unresolved = []
    for _, row in conditional_rows.iterrows():
        lower, upper = numeric_bound(row.get("Condition Minimum")), numeric_bound(row.get("Condition Maximum"))
        if (lower is not None or upper is not None) and row["Predictor name"] in all_available:
            conditional_with_bounds.append(row["Predictor name"])
        else:
            conditional_unresolved.append(row["Predictor name"])
    conditional_features = list(dict.fromkeys(retained + conditional_with_bounds))
    conditional_rules = condition_rules(decisions, conditional_features)
    conditional_frame, conditional_conditions = apply_numeric_conditions(usable, conditional_rules)
    if conditional_with_bounds:
        policies.append({
            "name": "researcher_retained_plus_bounded_conditionals",
            "features": conditional_features, "frame": conditional_frame,
            "description": "Retain predictors plus Conditional predictors with numeric limits; rows outside stated limits are omitted.",
            "condition_description": conditional_conditions or "No numeric bounds applied.",
        })

    run_dir = make_run_directory(RUN_DIR, fingerprint)
    dataset_dir = run_dir / "feature_strategy_datasets"
    dataset_dir.mkdir(parents=True, exist_ok=False)
    policy_status = []
    for policy in policies:
        policy_frame = policy["frame"]
        # A policy-level row filter can make a formerly available variable entirely missing.
        features = [feature for feature in policy["features"] if policy_frame[feature].notna().any()]
        policy["features"] = features
        unique_studies = int(policy_frame[STUDY_COLUMN].nunique())
        status = "ready"
        reason = ""
        if not features:
            status, reason = "skipped", "No predictor features are available under this policy."
        elif unique_studies < MINIMUM_STUDIES_FOR_LOSO:
            status, reason = "skipped", f"Only {unique_studies} studies remain after the policy restrictions."
        policy_status.append({
            "Feature policy": policy["name"], "Status": status, "Description": policy["description"],
            "Rows": len(policy_frame), "Studies": unique_studies,
            "Features": "; ".join(features), "Applied numeric conditions": policy.get("condition_description", ""),
            "Reason": reason,
        })
        if status == "ready":
            keep_columns = [STUDY_COLUMN, RESPONSE_BASIS_COLUMN, RESPONSE_COLUMN] + features
            policy_frame[keep_columns].to_csv(dataset_dir / f"{policy['name']}.csv", index=False)
        print(f"      Policy {policy['name']}: {status}; {len(policy_frame)} rows, "
              f"{unique_studies} studies, {len(features)} usable predictors. {reason}")

    metric_rows, prediction_rows, study_metric_rows = [], [], []
    selection_rows, failure_rows = [], []
    model_bundles = {}
    selector_names = ["all_features", "low_variance", "top_k", "pca_95_percent"]
    model_names = ["Ridge", "RandomForest", "ExtraTrees", "SVR"]
    ready_policies = [policy for policy in policies
                      if policy["features"] and policy["frame"][STUDY_COLUMN].nunique() >= MINIMUM_STUDIES_FOR_LOSO]
    planned_scenarios = len(ready_policies) * len(selector_names) * len(model_names)
    scenario_number = 0
    print(f"[4/6] Evaluating {planned_scenarios} policy/selector/model scenarios; "
          "each scenario runs one outer fold per study.")
    for policy in policies:
        if len(policy["features"]) == 0 or policy["frame"][STUDY_COLUMN].nunique() < MINIMUM_STUDIES_FOR_LOSO:
            continue
        print(f"Evaluating feature policy: {policy['name']} ({len(policy['frame'])} rows, "
              f"{policy['frame'][STUDY_COLUMN].nunique()} studies).")
        for selector_name in selector_names:
            for model_name in model_names:
                scenario_number += 1
                scenario_key = f"{policy['name']}::{selector_name}::{model_name}"
                print(f"  Scenario {scenario_number}/{planned_scenarios}: {scenario_key}")
                result = evaluate_scenario(
                    policy["frame"], policy["features"], selector_name, model_name,
                    scenario_key, policy["name"], policy.get("condition_description", "")
                )
                metric_rows.append({
                    "Scenario": scenario_key, "Feature policy": policy["name"],
                    "Selector": selector_name, "Model": model_name,
                    "Status": result["status"], "Error": result["error"],
                    "Rows": len(policy["frame"]),
                    "Studies available": int(policy["frame"][STUDY_COLUMN].nunique()),
                    **result.get("metrics", {}),
                })
                prediction_rows.extend(result["predictions"])
                selection_rows.extend(result.get("selections", []))
                study_metric_rows.extend(result.get("study_metrics", []))
                if result["status"] == "failed":
                    failure_rows.append({"Scenario": scenario_key, "Error": result["error"]})
                    print(f"    FAILED: {result['error']}")
                elif result.get("metrics"):
                    print(f"    Complete: macro study RMSE={result['metrics']['LOSO RMSE macro by study']:.3f}; "
                          f"training-mean baseline={result['metrics']['Baseline mean RMSE macro by study']:.3f}; "
                          f"Phase 3 screen={result['metrics']['Optimization eligible by screening rule']}.")
                if result["status"] != "failed" and result["final_model"] is not None:
                    model_bundles[scenario_key] = {
                        "estimator": result["final_model"],
                        "input_features": result["features"],
                        "feature_policy": result["feature_policy"],
                        "selector": result["selector"], "model": result["model_name"],
                        "source_sha256": fingerprint,
                        "decision_workbook_sha256": decision_fingerprint,
                        "condition_description": result["condition_description"],
                        "final_parameters": result["final_parameters"],
                    }

    metrics = pd.DataFrame(metric_rows)
    if not metrics.empty:
        eligible_keys = set(metrics.loc[
            metrics.get("Optimization eligible by screening rule", pd.Series(False, index=metrics.index)).fillna(False)
            & metrics["Status"].eq("complete"), "Scenario"
        ])
    else:
        eligible_keys = set()
    registry_rows = []
    for scenario_key, bundle in model_bundles.items():
        row = metrics.loc[metrics["Scenario"].eq(scenario_key)].iloc[0].to_dict()
        row.update({
            "Input features JSON": json.dumps(bundle["input_features"], ensure_ascii=False),
            "Condition description": bundle["condition_description"],
            "Final fit parameters": json.dumps(bundle["final_parameters"], ensure_ascii=False),
            "Model artifact key": scenario_key,
            "Optimization eligible": scenario_key in eligible_keys,
        })
        registry_rows.append(row)
    registry = pd.DataFrame(registry_rows)
    predictions_frame = pd.DataFrame(prediction_rows)
    study_metrics_frame = pd.DataFrame(study_metric_rows)
    selections_frame = pd.DataFrame(selection_rows)
    failures_frame = pd.DataFrame(failure_rows, columns=["Scenario", "Error"])
    print(f"[5/6] Writing tables, workbook, and fitted models for {len(metrics)} scenarios...")
    (run_dir / "feature_strategy_status.csv").write_text(
        pd.DataFrame(policy_status).to_csv(index=False), encoding="utf-8")
    metrics.to_csv(run_dir / "model_comparison_loso.csv", index=False)
    predictions_frame.to_csv(run_dir / "loso_predictions.csv", index=False)
    study_metrics_frame.to_csv(run_dir / "performance_by_held_out_study.csv", index=False)
    selections_frame.to_csv(run_dir / "feature_selection_by_outer_fold.csv", index=False)
    failures_frame.to_csv(run_dir / "model_failures.csv", index=False)
    registry.to_csv(run_dir / "model_registry.csv", index=False)
    joblib.dump(model_bundles, run_dir / "fitted_models.joblib")

    metadata = {
        "Source dataset SHA256": fingerprint,
        "Decision workbook": str(decision_path.relative_to(BASE_DIR)),
        "Decision workbook SHA256": decision_fingerprint,
        "Analysis date UTC": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "Script version": SCRIPT_VERSION,
        "Response basis filter": "; ".join(sorted(ALLOWED_RESPONSE_BASES)),
        "Included provisional responses": INCLUDE_PROVISIONAL_RESPONSES,
        "Excluded incompatible-basis rows": excluded_basis_rows,
        "Excluded provisional rows": excluded_provisional_rows,
        "Studies after filters": study_count,
        "Minimum studies for optimization screen": MINIMUM_STUDIES_FOR_LOSO,
        "Optimization eligibility screen": ">= minimum held-out studies and macro-average study LOSO RMSE below the training-mean baseline.",
        "Run directory": str(run_dir.relative_to(BASE_DIR)),
    }
    write_summary_workbook(
        run_dir / "phase2_model_comparison.xlsx", metrics,
        study_metrics_frame, selections_frame,
        pd.DataFrame(policy_status), failures_frame,
        registry, phase1_stats, phase1_response_associations, metadata,
    )
    print("[6/6] Creating visual report and browser-ready figures...")
    figure_paths = save_phase2_figures(metrics, predictions_frame, study_metrics_frame, run_dir)
    report_path = write_phase2_visual_report(run_dir, metrics, metadata, figure_paths, len(usable))
    run_manifest = {
        "script_version": SCRIPT_VERSION,
        "source_dataset_file": DATASET_PATH.name,
        "source_dataset_sha256": fingerprint,
        "decision_workbook": str(decision_path.relative_to(BASE_DIR)),
        "decision_workbook_sha256": decision_fingerprint,
        "run_directory": str(run_dir.relative_to(BASE_DIR)),
        "analysis_date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "response_basis_filter": sorted(ALLOWED_RESPONSE_BASES),
        "included_provisional_responses": INCLUDE_PROVISIONAL_RESPONSES,
        "excluded_incompatible_basis_rows": excluded_basis_rows,
        "excluded_provisional_rows": excluded_provisional_rows,
        "study_count_after_filters": study_count,
        "minimum_studies_for_optimization_screen": MINIMUM_STUDIES_FOR_LOSO,
        "conditional_predictors_without_machine_readable_bounds": conditional_unresolved,
        "optimization_eligibility_rule": (
            f"At least {MINIMUM_STUDIES_FOR_LOSO} held-out studies and lower macro-average study LOSO RMSE than "
            "the training-mean baseline; this is a screening rule, not experimental validation."
        ),
        "artifacts": {
            "summary_workbook": "phase2_model_comparison.xlsx",
            "metrics": "model_comparison_loso.csv",
            "predictions": "loso_predictions.csv",
            "registry": "model_registry.csv",
            "models": "fitted_models.joblib",
            "strategy_datasets": "feature_strategy_datasets/",
            "visual_report": "phase2_visual_report.html",
            "figures": [Path(value).name for value in figure_paths.values()
                        if isinstance(value, Path)],
        },
    }
    (run_dir / "phase2_run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Phase 2 outputs written to: {run_dir}")
    print(f"Visual report (open this HTML file in a browser): {report_path}")
    print(f"Compatible studies retained: {study_count}; incompatible-basis rows excluded: {excluded_basis_rows}; "
          f"provisional rows excluded: {excluded_provisional_rows}.")
    print("Optimization eligibility uses macro-average study LOSO RMSE versus a training-mean baseline; inspect the full LOSO tables before Phase 3.")
