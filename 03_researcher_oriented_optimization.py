"""Phase 3: researcher-oriented multi-objective culture optimization.

Spyder workflow: verify the source/workbook/Phase 2 fingerprints -> select only
researcher-approved models that passed the configured LOSO screen -> constrain
search to observed predictor ranges -> compare predicted response with distance
to observed profiles -> write a workbook, figures, and a browser-ready report.
Console messages identify each selected model and search scenario as it runs.
Candidates remain model estimates for researcher review and experimental testing.
"""

# %% Configuration — set explicit workbook/run paths, eligibility limit, and search settings
from pathlib import Path
from datetime import datetime, timezone
import base64
from html import escape
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import joblib
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

SCRIPT_VERSION = "1.1.0"
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "rhamnosus.xlsx"
DECISION_WORKBOOK_PATH = None  # Optional explicit path to the reviewed Phase 1 workbook.
PHASE2_RUN_DIR = None          # Optional explicit path; otherwise newest fingerprint match is used.
OUTPUT_DIR = BASE_DIR / "outputs"
DECISION_DIR = OUTPUT_DIR / "decision_workbooks"
PHASE2_DIR = OUTPUT_DIR / "phase2_runs"
RUN_DIR = OUTPUT_DIR / "phase3_runs"
ALLOWED_RESPONSE_BASES = {"log10 CFU/mL"}
INCLUDE_PROVISIONAL_RESPONSES = False
MAX_MODELS_TO_OPTIMIZE = 3
POPULATION_SIZE = 80
N_GENERATIONS = 150
RANDOM_SEED = 2026
DISCRETE_LEVEL_LIMIT = 8


# %% Provenance and input discovery — refuse mismatched source or decision fingerprints
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


def find_phase2_run(source_hash, decision_hash):
    if PHASE2_RUN_DIR is not None:
        path = Path(PHASE2_RUN_DIR)
        if not path.is_absolute():
            path = BASE_DIR / path
        candidates = [path]
    else:
        candidates = [path.parent for path in PHASE2_DIR.glob("*/phase2_run_manifest.json")]
    matches = []
    for path in candidates:
        manifest_path = path / "phase2_run_manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest.get("source_dataset_sha256") == source_hash
                    and manifest.get("decision_workbook_sha256") == decision_hash):
                matches.append((manifest_path.stat().st_mtime, path, manifest))
        except Exception:
            continue
    if not matches:
        raise FileNotFoundError(
            "No Phase 2 run matches the current source dataset and decision workbook. "
            "Run 02_pooled_predictive_modelling.py after completing Phase 1."
        )
    matches.sort(key=lambda item: item[0], reverse=True)
    if len(matches) > 1 and PHASE2_RUN_DIR is None:
        print("Several matching Phase 2 runs exist; using the newest run by file time. "
              "Set PHASE2_RUN_DIR to select a specific run.")
    _, path, manifest = matches[0]
    return path, manifest


def locate_decision_workbook(source_hash):
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
            metadata = pd.read_excel(path, sheet_name="Workflow_Metadata")
            if not metadata.empty and str(metadata.loc[0, "Source dataset SHA256"]) == source_hash:
                analysis_date = str(metadata.loc[0, "Analysis date UTC"])
                matches.append((analysis_date, path))
        except Exception:
            continue
    if not matches:
        raise ValueError(
            "A readable Phase 1 decision workbook matching the dataset is required."
        )
    matches.sort(key=lambda item: item[0], reverse=True)
    if len(matches) > 1 and matches[0][0] == matches[1][0]:
        raise ValueError(
            "Multiple matching decision workbooks have the same analysis date. "
            "Set DECISION_WORKBOOK_PATH explicitly."
        )
    if len(matches) > 1:
        print(f"Multiple matching workbooks exist; using the newest evidence workbook: {matches[0][1]}")
    return matches[0][1]


def boolean_value(value):
    return str(value).strip().casefold() in {"true", "1", "yes"}


# %% Optimization model and domain setup — bound each factor by observations and researcher conditions
def snap_discrete(values, features, levels_by_feature):
    snapped = np.asarray(values, dtype=float).copy()
    for column, feature in enumerate(features):
        levels = levels_by_feature.get(feature)
        if levels:
            level_values = np.asarray(levels, dtype=float)
            nearest = np.abs(snapped[:, column, None] - level_values[None, :]).argmin(axis=1)
            snapped[:, column] = level_values[nearest]
    return snapped


class CultureOptimizationProblem(Problem):
    def __init__(self, model_bundle, features, variable_features, fixed_values,
                 lower_bounds, upper_bounds, levels_by_feature, training_scaled,
                 scale_values, max_supported_distance, conservative):
        self.model_bundle = model_bundle
        self.features = features
        self.variable_features = variable_features
        self.fixed_values = fixed_values
        self.lower_bounds = np.asarray(lower_bounds, dtype=float)
        self.upper_bounds = np.asarray(upper_bounds, dtype=float)
        self.levels_by_feature = levels_by_feature
        self.training_scaled = training_scaled
        self.scale_values = scale_values
        self.max_supported_distance = max_supported_distance
        self.conservative = conservative
        super().__init__(
            n_var=len(variable_features), n_obj=2,
            n_ieq_constr=1 if conservative else 0,
            xl=np.asarray(lower_bounds, dtype=float), xu=np.asarray(upper_bounds, dtype=float),
        )

    def evaluate_candidates(self, decision_matrix):
        decision_matrix = np.atleast_2d(np.asarray(decision_matrix, dtype=float))
        snapped_variables = snap_discrete(decision_matrix, self.variable_features,
                                          self.levels_by_feature)
        full = np.tile(np.asarray([self.fixed_values[name] for name in self.features], dtype=float),
                       (len(snapped_variables), 1))
        feature_index = {name: index for index, name in enumerate(self.features)}
        for variable_index, name in enumerate(self.variable_features):
            full[:, feature_index[name]] = snapped_variables[:, variable_index]
        predictions = self.model_bundle["estimator"].predict(pd.DataFrame(full, columns=self.features))
        distances = nearest_training_distances(full, self.training_scaled, self.scale_values)
        return full, np.asarray(predictions, dtype=float), distances

    def _evaluate(self, x, out, *args, **kwargs):
        _, predictions, distances = self.evaluate_candidates(x)
        out["F"] = np.column_stack([-predictions, distances])
        if self.conservative:
            out["G"] = (distances - self.max_supported_distance).reshape(-1, 1)


def nearest_training_distances(candidates, training_scaled, scales):
    candidate_scaled = np.asarray(candidates, dtype=float) / scales
    if training_scaled.shape[1] != candidate_scaled.shape[1]:
        raise ValueError("Training and optimization feature dimensions do not match.")
    distances = cdist(candidate_scaled, training_scaled, metric="euclidean")
    return np.min(distances / max(1, training_scaled.shape[1]) ** 0.5, axis=1)


def nearest_neighbor_support_threshold(training_matrix, scales):
    scaled = np.asarray(training_matrix, dtype=float) / scales
    if len(scaled) < 3:
        return np.nan, np.array([], dtype=float)
    distances = cdist(scaled, scaled, metric="euclidean") / max(1, scaled.shape[1]) ** 0.5
    np.fill_diagonal(distances, np.inf)
    leave_one_out = distances.min(axis=1)
    return float(np.quantile(leave_one_out, 0.95)), leave_one_out


def build_domain(raw_training, features, decision_rows):
    decision_index = decision_rows.set_index("Predictor name")
    lower_by_feature, upper_by_feature, levels_by_feature = {}, {}, {}
    fixed_values = {}
    variable_features = []
    constraint_rows = []
    for feature in features:
        observed = pd.to_numeric(raw_training[feature], errors="coerce").dropna()
        if observed.empty:
            raise ValueError(f"No observed training range is available for model feature: {feature}")
        data_low, data_high = float(observed.min()), float(observed.max())
        low, high = data_low, data_high
        if feature in decision_index.index:
            decision = decision_index.loc[feature]
            user_low = float(decision["Condition Minimum"]) if pd.notna(decision["Condition Minimum"]) else None
            user_high = float(decision["Condition Maximum"]) if pd.notna(decision["Condition Maximum"]) else None
            if user_low is not None:
                low = max(low, user_low)
            if user_high is not None:
                high = min(high, user_high)
            status = str(decision.get("Researcher decision", ""))
            condition_value = decision.get("Conditions or restrictions", "")
            condition_text = str(condition_value) if pd.notna(condition_value) else ""
        else:
            user_low = user_high = None
            status = "Not found"
            condition_text = ""
        if low > high:
            raise ValueError(f"Researcher bounds do not overlap the observed range for {feature}.")
        allowed_observed = np.sort(observed[(observed >= low) & (observed <= high)].unique())
        if len(allowed_observed) == 0:
            raise ValueError(f"No observed levels remain inside the allowed range for {feature}.")
        if len(allowed_observed) == 1:
            fixed_values[feature] = float(allowed_observed[0])
            low = high = float(allowed_observed[0])
        else:
            variable_features.append(feature)
            if len(allowed_observed) <= DISCRETE_LEVEL_LIMIT:
                levels_by_feature[feature] = allowed_observed.tolist()
        lower_by_feature[feature] = low
        upper_by_feature[feature] = high
        constraint_rows.append({
            "Predictor": feature, "Researcher decision": status,
            "Observed minimum in modeling data": data_low,
            "Observed maximum in modeling data": data_high,
            "Applied minimum": low, "Applied maximum": high,
            "Observed levels within bounds": "; ".join(map(str, allowed_observed.tolist())),
            "Optimization representation": (
                "Fixed at sole observed level" if feature not in variable_features
                else ("Snapped to observed levels" if feature in levels_by_feature else "Continuous within observed range")
            ),
            "Researcher conditions": condition_text,
            "Extrapolation control": "Bounds intersect observed modeling range",
        })
    # If every feature is fixed, Phase 3 evaluates that single supported profile.
    return variable_features, fixed_values, lower_by_feature, upper_by_feature, levels_by_feature, constraint_rows


# %% Workbook styling and figures — show distance on a log scale when values cluster near zero
def style_workbook(path):
    workbook = load_workbook(path)
    fill = PatternFill("solid", fgColor="1F4E78")
    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2" if sheet.max_row > 1 else "A1"
        if sheet.max_row > 1:
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.fill = fill
                cell.font = Font(color="FFFFFF", bold=True)
                cell.alignment = Alignment(vertical="top", wrap_text=True)
        for cells in sheet.columns:
            letter = get_column_letter(cells[0].column)
            length = max((len(str(cell.value)) for cell in cells if cell.value is not None), default=10)
            sheet.column_dimensions[letter].width = min(max(12, length + 2), 45)
    workbook.save(path)


def save_pareto_plot(alternatives, path):
    if alternatives.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    for scenario, frame in alternatives.groupby("Optimization scenario", sort=True):
        # Distances can be orders of magnitude smaller than the support threshold.
        # A logarithmic axis keeps close alternatives visible without hiding that gap.
        distances = pd.to_numeric(frame["Nearest observed-profile distance"], errors="coerce").clip(lower=1e-8)
        ax.scatter(distances, frame["Predicted cell density"],
                   label=scenario, alpha=0.8, s=55, edgecolor="white", linewidth=0.4)
    ax.axvline(alternatives["Supported distance threshold"].dropna().min()
               if alternatives["Supported distance threshold"].notna().any() else np.nan,
               color="gray", linestyle="--", linewidth=1,
               label="95th percentile observed leave-one-out distance")
    ax.set_xscale("log")
    ax.set_xlabel("Distance to nearest observed profile (log scale; lower is closer)")
    ax.set_ylabel("Predicted cell density (reported log CFU/mL scale)")
    ax.set_title("Pareto alternatives: predicted density and similarity to evidence")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_candidate_profile_plot(alternatives, path):
    """Show how the numeric settings differ among the displayed candidates."""
    metadata_columns = {
        "Pareto alternative", "Predicted cell density", "Nearest observed-profile distance",
        "Supported distance threshold", "LOSO RMSE pooled", "LOSO RMSE macro by study", "LOSO studies",
    }
    feature_columns = []
    for column in alternatives.columns:
        if column in metadata_columns:
            continue
        values = pd.to_numeric(alternatives[column], errors="coerce")
        if values.notna().any() and values.nunique(dropna=True) > 1:
            feature_columns.append(column)
    if not feature_columns:
        return None, []
    values = alternatives[feature_columns].apply(pd.to_numeric, errors="coerce")
    minimum = values.min(axis=0)
    maximum = values.max(axis=0)
    spread = (maximum - minimum).replace(0, np.nan)
    scaled = ((values - minimum) / spread).fillna(0.5)
    row_labels = [f"{row['Model']} | {row['Optimization scenario']} | {row['Pareto alternative']}"
                  for _, row in alternatives.iterrows()]
    fig, ax = plt.subplots(figsize=(max(9, len(feature_columns) * 1.1), max(4.5, len(alternatives) * 0.55)))
    image = ax.imshow(scaled.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(feature_columns)), feature_columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    for row in range(len(alternatives)):
        for column, feature in enumerate(feature_columns):
            value = values.iloc[row, column]
            if pd.notna(value):
                ax.text(column, row, f"{value:.3g}", ha="center", va="center", fontsize=8,
                        color="white" if scaled.iloc[row, column] > 0.65 else "#18324a")
    ax.set_title("Candidate settings (cell color is relative within this candidate set)")
    fig.colorbar(image, ax=ax, label="Relative setting value, 0 to 1")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path, feature_columns


def save_model_reliability_plot(reliability, path):
    """Compare selected model errors with their training-mean baselines."""
    if reliability.empty:
        return None
    frame = reliability.sort_values("LOSO RMSE macro by study", ascending=True).copy()
    labels = frame.apply(lambda row: f"{row['Feature policy']} | {row['Selector']} | {row['Model']}", axis=1)
    positions = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(11, max(4, len(frame) * 0.65)))
    ax.barh(positions, frame["LOSO RMSE macro by study"], color="#277DA1", alpha=0.9)
    ax.scatter(frame["Training-mean baseline RMSE macro by study"], positions,
               marker="D", color="#D1495B", edgecolor="white", zorder=3,
               label="Training-mean baseline")
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Macro-average RMSE across held-out studies")
    ax.set_title("Predictive error for models used in optimization")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_phase3_visual_report(run_dir, alternatives, reliability, metadata,
                               profile_path, reliability_path, feature_columns):
    """Create a self-contained report for the proposed settings and their limits."""
    def image_tag(path, alt, caption):
        if path is None or not Path(path).exists():
            return ""
        image_data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        return (f'<figure><img src="data:image/png;base64,{image_data}" alt="{escape(alt)}">'
                f'<figcaption>{escape(caption)}</figcaption></figure>')

    metadata_columns = {
        "Pareto alternative", "Predicted cell density", "Nearest observed-profile distance",
        "Supported distance threshold", "LOSO RMSE pooled", "LOSO RMSE macro by study", "LOSO studies",
    }
    report_features = [column for column in alternatives.columns
                       if column not in metadata_columns
                       and pd.to_numeric(alternatives[column], errors="coerce").notna().any()
                       and column not in {"Optimization scenario", "Model artifact key", "Feature policy",
                                          "Selector", "Model"}]
    display_columns = ["Optimization scenario", "Model", "Pareto alternative",
                       "Predicted cell density", "Nearest observed-profile distance",
                       "Supported distance threshold", "Support status", "LOSO RMSE macro by study"]
    display_columns.extend(column for column in report_features if column not in display_columns)
    table = alternatives[display_columns].copy()
    table = table.rename(columns={"Predicted cell density": "Predicted response",
                                  "Nearest observed-profile distance": "Distance to data",
                                  "Supported distance threshold": "Support threshold",
                                  "LOSO RMSE macro by study": "LOSO RMSE (macro study)"})
    table_html = table.to_html(index=False, escape=True, border=0, classes="data-table",
                               float_format=lambda value: f"{value:.4g}")
    within_threshold = int(alternatives["Support status"].eq("Within observed neighborhood threshold").sum())
    predicted_low = float(alternatives["Predicted cell density"].min())
    predicted_high = float(alternatives["Predicted cell density"].max())
    style = """body{margin:0;background:#f3f6f8;color:#18324a;font:16px/1.55 Segoe UI,Arial,sans-serif}
main{max-width:1120px;margin:auto;padding:32px 22px 56px}h1{margin:0}h2{margin-top:32px;border-bottom:1px solid #d9e2ea;padding-bottom:8px}
.sub,figcaption{color:#587083}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:20px 0}
.card,.panel,figure{background:white;border:1px solid #d9e2ea;border-radius:12px;padding:15px}.card b{display:block;font-size:1.55rem;color:#277da1}.card span{color:#587083}
.note{background:#fff3d8;border-left:4px solid #d89b25;padding:14px 16px;border-radius:5px}figure{margin:20px 0 28px}figure img{display:block;max-width:100%;height:auto;margin:auto}figcaption{font-size:.92rem;margin-top:10px}
.table-wrap{overflow-x:auto}.data-table{border-collapse:collapse;width:100%;background:white;font-size:.9rem}.data-table th,.data-table td{border-bottom:1px solid #d9e2ea;padding:9px;text-align:left;vertical-align:top;white-space:nowrap}.data-table th{background:#e7eff4}a{color:#126782}li{margin:7px 0}"""
    report = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phase 3 visual report</title><style>{style}</style></head><body><main>
<h1>Researcher-reviewed optimization candidates</h1><p class="sub">Phase 3 | Generated {escape(metadata['Analysis date UTC'])}</p>
<div class="cards"><div class="card"><b>{metadata['Models optimized']}</b><span>screened models used</span></div>
<div class="card"><b>{len(alternatives)}</b><span>candidate rows</span></div>
<div class="card"><b>{within_threshold}/{len(alternatives)}</b><span>within the support threshold</span></div>
<div class="card"><b>{predicted_low:.3f}–{predicted_high:.3f}</b><span>predicted response range</span></div></div>
<div class="note"><b>These are model-generated candidates.</b> The predicted response is not an experimental result or causal effect. LOSO RMSE is a dataset-level error summary, not an individual prediction interval. Inspect the support distance and study-held-out evidence before deciding whether any setting is worth testing.</div>
<h2>Predicted response and distance to observed profiles</h2>{image_tag(run_dir / 'pareto_tradeoffs.png', 'Predicted response and distance on a logarithmic horizontal axis', 'The horizontal axis is logarithmic so candidates close to observed profiles remain visible. The dashed line marks the 95th-percentile support threshold.')}
<h2>Candidate settings</h2>{image_tag(profile_path, 'Heatmap of settings across optimization candidates', 'Cell labels show the actual setting. Color indicates relative position within the displayed candidate set, not a biological desirability score.')}
<h2>Reliability of models used</h2>{image_tag(reliability_path, 'Macro-average LOSO RMSE compared with training-mean baselines', 'Bars show selected model error; red diamonds show training-mean baseline error. A lower value is better.')}
<h2>Candidate table</h2><div class="panel table-wrap">{table_html}</div>
<h2>Files</h2><ul><li><a href="researcher_optimization_results.xlsx">Optimization workbook</a></li><li><a href="pareto_alternatives.csv">Candidate settings (CSV)</a></li><li><a href="predictor_constraints_and_ranges.csv">Predictor ranges and constraints</a></li><li><a href="model_reliability_used.csv">Model reliability evidence</a></li></ul>
</main></body></html>"""
    report_path = run_dir / "phase3_visual_report.html"
    report_path.write_text(report, encoding="utf-8")
    return report_path


# %% Run Phase 3 — print selected models, support thresholds, search starts, and saved artifacts
if __name__ == "__main__":
    print("PHASE 3 — supported multi-objective candidate search")
    print("[1/6] Verifying source data, decision workbook, and Phase 2 run fingerprints...")
    source_hash = sha256_file(DATASET_PATH)
    decision_path = locate_decision_workbook(source_hash)
    decision_hash = sha256_file(decision_path)
    phase2_path, phase2_manifest = find_phase2_run(source_hash, decision_hash)
    print(f"      Decision workbook: {decision_path}")
    print(f"      Matching Phase 2 run: {phase2_path}")
    if phase2_manifest.get("response_basis_filter") != sorted(ALLOWED_RESPONSE_BASES):
        raise ValueError("Phase 2 response-basis filter differs from Phase 3 configuration.")
    if bool(phase2_manifest.get("included_provisional_responses")) != INCLUDE_PROVISIONAL_RESPONSES:
        raise ValueError("Phase 2 provisional-response setting differs from Phase 3 configuration.")

    decisions = pd.read_excel(decision_path, sheet_name="Predictor_Decisions")
    phase1_stats = pd.read_excel(decision_path, sheet_name="Integrated_Predictor_Statistics")
    model_registry = pd.read_csv(phase2_path / "model_registry.csv")
    if model_registry.empty:
        raise ValueError("Phase 2 produced no fitted models to consider.")
    model_registry["Optimization eligible parsed"] = model_registry["Optimization eligible"].map(boolean_value)
    model_registry = model_registry[model_registry["Optimization eligible parsed"]].copy()
    allowed_policies = {"researcher_retained", "researcher_retained_plus_bounded_conditionals"}
    model_registry = model_registry[model_registry["Feature policy"].isin(allowed_policies)]
    if model_registry.empty:
        raise ValueError(
            "No researcher-decision model passed Phase 2 screening. Inspect LOSO results and baseline comparisons; "
            "Phase 3 will not treat an unqualified model as an optimization engine."
        )
    model_registry = model_registry.sort_values("LOSO RMSE macro by study", ascending=True).head(MAX_MODELS_TO_OPTIMIZE)
    print(f"      {len(model_registry)} approved model(s) passed screening and will be optimized (limit {MAX_MODELS_TO_OPTIMIZE}).")
    for _, selected_row in model_registry.iterrows():
        print(f"        {selected_row['Model artifact key']}: macro LOSO RMSE "
              f"{selected_row['LOSO RMSE macro by study']:.3f} across {int(selected_row['LOSO studies'])} studies.")
    model_bundles = joblib.load(phase2_path / "fitted_models.joblib")
    run_dir = make_run_directory(RUN_DIR, source_hash)
    all_alternatives = []
    constraint_rows = []
    model_reliability_rows = []
    print("[2/6] Loading selected models, setting predictor bounds, and estimating support thresholds...")
    for model_number, (_, registry_row) in enumerate(model_registry.iterrows(), start=1):
        key = str(registry_row["Model artifact key"])
        print(f"  Model {model_number}/{len(model_registry)}: {key}")
        if key not in model_bundles:
            print(f"Skipping missing fitted-model artifact: {key}")
            continue
        bundle = model_bundles[key]
        features = list(bundle["input_features"])
        policy_name = str(bundle["feature_policy"])
        strategy_data_path = phase2_path / "feature_strategy_datasets" / f"{policy_name}.csv"
        if not strategy_data_path.exists():
            print(f"Skipping missing feature-strategy dataset: {strategy_data_path}")
            continue
        training = pd.read_csv(strategy_data_path)
        raw_training = training[features].apply(pd.to_numeric, errors="coerce")
        imputer = bundle["estimator"].named_steps.get("imputer")
        if imputer is not None:
            training_imputed = imputer.transform(raw_training)
        else:
            training_imputed = raw_training.fillna(raw_training.median()).to_numpy()
        training_imputed = np.asarray(training_imputed, dtype=float)
        training_imputed = np.where(np.isfinite(training_imputed), training_imputed, 0.0)
        scales = np.nanmax(training_imputed, axis=0) - np.nanmin(training_imputed, axis=0)
        scales = np.where(np.isfinite(scales) & (scales > 0), scales, 1.0)
        threshold, loo_distances = nearest_neighbor_support_threshold(training_imputed, scales)
        domain = build_domain(raw_training, features, decisions)
        (variable_features, fixed_values, lower_by_feature, upper_by_feature,
         levels_by_feature, these_constraints) = domain
        for row in these_constraints:
            row.update({"Model artifact key": key, "Feature policy": policy_name})
            stat_rows = phase1_stats[phase1_stats["Variable"].eq(row["Predictor"])]
            if not stat_rows.empty:
                row["Phase 1 overall missing percent"] = stat_rows.iloc[0].get("Missing percent", np.nan)
                row["Phase 1 overall unique values"] = stat_rows.iloc[0].get("N unique", np.nan)
        constraint_rows.extend(these_constraints)
        print(f"    Using {len(features)} predictors across {len(training)} rows; "
              f"observed-neighborhood threshold={threshold:.6g} (scaled distance).")
        model_reliability_rows.append({
            "Model artifact key": key, "Feature policy": policy_name,
            "Selector": bundle["selector"], "Model": bundle["model"],
            "LOSO RMSE pooled": registry_row["RMSE"],
            "LOSO RMSE macro by study": registry_row["LOSO RMSE macro by study"],
            "LOSO MAE": registry_row.get("MAE", np.nan),
            "LOSO pooled R2": registry_row.get("R2 pooled OOF", np.nan),
            "Training-mean baseline RMSE": registry_row.get("Baseline mean RMSE", np.nan),
            "Training-mean baseline RMSE macro by study": registry_row.get("Baseline mean RMSE macro by study", np.nan),
            "LOSO study count": registry_row.get("LOSO studies", np.nan),
            "Screening status": "Passed Phase 2 macro-average-study baseline and study-count screen",
            "Decision workbook conditions": bundle.get("condition_description", ""),
        })
        for index, feature in enumerate(features):
            if feature in fixed_values:
                continue
            else:
                median = float(np.nanmedian(training_imputed[:, index]))
                fixed_values[feature] = float(np.clip(median, lower_by_feature[feature], upper_by_feature[feature]))
        scaled_training = training_imputed / scales

        for mode, conservative in [("Conservative supported", True), ("Exploratory", False)]:
            print(f"    Searching {mode.lower()} alternatives: population={POPULATION_SIZE}, "
                  f"generations={N_GENERATIONS}, variable predictors={len(variable_features)}...")
            if variable_features:
                if conservative and not np.isfinite(threshold):
                    print(f"Skipping conservative scenario for {key}: no empirical support threshold can be estimated.")
                    continue
                lower = [lower_by_feature[name] for name in variable_features]
                upper = [upper_by_feature[name] for name in variable_features]
                problem = CultureOptimizationProblem(
                    bundle, features, variable_features, fixed_values,
                    lower, upper, levels_by_feature, scaled_training,
                    scales, threshold, conservative,
                )
                algorithm = NSGA2(
                    pop_size=POPULATION_SIZE,
                    sampling=FloatRandomSampling(),
                    crossover=SBX(prob=0.9, eta=15),
                    mutation=PM(eta=20),
                    eliminate_duplicates=True,
                )
                # pymoo prints generation-level progress so Spyder does not appear idle
                # during the population search.
                result = minimize(problem, algorithm, ("n_gen", N_GENERATIONS),
                                  seed=RANDOM_SEED, verbose=True)
                if result.X is None:
                    print(f"      No feasible alternatives were found for {key} / {mode}.")
                    continue
                decision_vectors = np.atleast_2d(result.X)
                full_candidates, predictions, distances = problem.evaluate_candidates(decision_vectors)
                if conservative:
                    supported = distances <= threshold + 1e-12
                    full_candidates = full_candidates[supported]
                    predictions = predictions[supported]
                    distances = distances[supported]
                    if len(full_candidates) == 0:
                        print(f"      No empirically supported alternatives were found for {key}.")
                        continue
            else:
                if conservative and not np.isfinite(threshold):
                    print(f"Skipping conservative scenario for {key}: no empirical support threshold can be estimated.")
                    continue
                full_candidates = np.asarray([[fixed_values[name] for name in features]], dtype=float)
                predictions = bundle["estimator"].predict(pd.DataFrame(full_candidates, columns=features))
                distances = nearest_training_distances(full_candidates, scaled_training, scales)
                if conservative and distances[0] > threshold + 1e-12:
                    print(f"      Skipping conservative scenario: the only profile is outside the support threshold.")
                    continue

            if len(full_candidates):
                print(f"      Search complete: {len(full_candidates)} candidate(s); "
                      f"distance range {float(np.min(distances)):.6g} to {float(np.max(distances)):.6g}.")

            for candidate_number, (candidate, prediction, distance) in enumerate(
                    zip(full_candidates, predictions, distances), start=1):
                row = {
                    "Optimization scenario": mode,
                    "Model artifact key": key,
                    "Feature policy": policy_name,
                    "Selector": bundle["selector"], "Model": bundle["model"],
                    "Pareto alternative": candidate_number,
                    "Predicted cell density": float(prediction),
                    "Nearest observed-profile distance": float(distance),
                    "Supported distance threshold": threshold,
                    "Support status": (
                        "Within observed neighborhood threshold" if pd.notna(threshold) and distance <= threshold
                        else "Potentially unsupported combination; review before use"
                    ),
                    "LOSO RMSE pooled": registry_row["RMSE"],
                    "LOSO RMSE macro by study": registry_row["LOSO RMSE macro by study"],
                    "LOSO studies": registry_row.get("LOSO studies", np.nan),
                    "Objective interpretation": "Maximize predicted response while minimizing distance to observed predictor profiles.",
                    "Biological interpretation": "No causal or biological effect is assigned automatically; compare settings with the Phase 1 study-level evidence.",
                    "Practical feasibility": "Ingredient availability, cost, and laboratory feasibility require researcher assessment.",
                    "Prediction uncertainty limitation": "No individual prediction interval is computed; use LOSO RMSE as a dataset-level error summary.",
                    "Important limitation": "Prediction is a model estimate; no experimental response or causal effect is established. LOSO model selection has no independent external validation, and nearest-profile distance uses median-imputed predictors as a support proxy.",
                }
                row.update({feature: float(value) for feature, value in zip(features, candidate)})
                all_alternatives.append(row)

    alternatives = pd.DataFrame(all_alternatives)
    if alternatives.empty:
        raise RuntimeError("No optimization alternatives were generated.")
    # Exact duplicate profiles can arise after snapping discrete factors; keep one copy per model and scenario.
    feature_columns = sorted({feature for bundle in model_bundles.values()
                              for feature in bundle.get("input_features", [])})
    present_features = [feature for feature in feature_columns if feature in alternatives.columns]
    alternatives = alternatives.drop_duplicates(
        subset=["Optimization scenario", "Model artifact key"] + present_features,
        keep="first",
    ).reset_index(drop=True)
    constraints = pd.DataFrame(constraint_rows)
    reliability = pd.DataFrame(model_reliability_rows)
    print(f"[3/6] Saving {len(alternatives)} candidate rows and their predictor constraints...")
    alternatives.to_csv(run_dir / "pareto_alternatives.csv", index=False)
    constraints.to_csv(run_dir / "predictor_constraints_and_ranges.csv", index=False)
    reliability.to_csv(run_dir / "model_reliability_used.csv", index=False)
    print("[4/6] Creating Pareto, candidate-setting, and model-reliability figures...")
    save_pareto_plot(alternatives, run_dir / "pareto_tradeoffs.png")
    profile_plot, profile_features = save_candidate_profile_plot(
        alternatives, run_dir / "candidate_settings_heatmap.png"
    )
    reliability_plot = save_model_reliability_plot(
        reliability, run_dir / "optimization_model_reliability.png"
    )

    instructions = pd.DataFrame([
        ["Purpose", "Compare Pareto alternatives for predicted cell density and similarity to observed experimental profiles."],
        ["Objectives", "Maximize the modeled response and minimize nearest observed-profile distance. Distance is a support diagnostic, not a biological response."],
        ["Conservative supported", "Adds a distance constraint based on the 95th percentile of observed leave-one-profile-out distances."],
        ["Exploratory", "Uses the same univariate observed ranges but allows combinations beyond the empirical neighborhood; review support flags carefully."],
        ["Discrete factors", "Predictors with few observed values are snapped to those observed levels."],
        ["Model screening", "Only researcher-decision feature policies that passed the Phase 2 LOSO study-count and macro-average study RMSE baseline screen are used."],
        ["Interpretation", "Pareto optimality is a mathematical trade-off, not experimental optimality. The researcher chooses alternatives and validates them experimentally."],
        ["Response basis", "This run uses the configured CFU/mL response basis. CFU/g observations are not pooled with CFU/mL."],
    ], columns=["Topic", "Guidance"])
    metadata = {
        "Source dataset SHA256": source_hash,
        "Decision workbook": str(decision_path.relative_to(BASE_DIR)),
        "Decision workbook SHA256": decision_hash,
        "Phase 2 run directory": str(phase2_path.relative_to(BASE_DIR)),
        "Phase 2 manifest SHA256": sha256_file(phase2_path / "phase2_run_manifest.json"),
        "Phase 3 script version": SCRIPT_VERSION,
        "Analysis date UTC": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "Models optimized": len(reliability),
        "Alternatives": len(alternatives),
        "Population size": POPULATION_SIZE,
        "Generations": N_GENERATIONS,
        "Random seed": RANDOM_SEED,
        "Objective 1": "Maximize predicted Cell Density Log CFU/mL.",
        "Objective 2": "Minimize distance to nearest observed predictor profile after range scaling.",
        "Support threshold": "95th percentile of observed leave-one-profile-out nearest-neighbor distances.",
    }
    workbook_path = run_dir / "researcher_optimization_results.xlsx"
    print("[5/6] Writing the optimization workbook...")
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        instructions.to_excel(writer, sheet_name="Instructions", index=False)
        alternatives.to_excel(writer, sheet_name="Optimization_Alternatives", index=False)
        reliability.to_excel(writer, sheet_name="Model_Reliability", index=False)
        constraints.to_excel(writer, sheet_name="Predictor_Constraints", index=False)
        pd.DataFrame([metadata]).to_excel(writer, sheet_name="Workflow_Metadata", index=False)
    style_workbook(workbook_path)
    print("[6/6] Building the browser-ready visual report...")
    report_path = write_phase3_visual_report(
        run_dir, alternatives, reliability, metadata, profile_plot, reliability_plot, profile_features
    )
    run_manifest = {
        **metadata,
        "run_directory": str(run_dir.relative_to(BASE_DIR)),
        "artifacts": {
            "workbook": workbook_path.name,
            "alternatives": "pareto_alternatives.csv",
            "constraints": "predictor_constraints_and_ranges.csv",
            "reliability": "model_reliability_used.csv",
            "figure": "pareto_tradeoffs.png",
            "candidate_settings_figure": "candidate_settings_heatmap.png" if profile_plot else "",
            "model_reliability_figure": "optimization_model_reliability.png" if reliability_plot else "",
            "visual_report": "phase3_visual_report.html",
        },
    }
    (run_dir / "phase3_run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Phase 3 outputs written to: {run_dir}")
    print(f"Visual report (open this HTML file in a browser): {report_path}")
    print("Review the workbook's model reliability, support flags, bounds, and alternatives before selecting experiments.")
