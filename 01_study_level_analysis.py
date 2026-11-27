"""Phase 1: study-aware evidence generation for the rhamnosus dataset.

Spyder workflow: read and validate the workbook -> calculate descriptive and
association evidence -> create a researcher decision workbook -> export tables,
figures, and a browser-ready HTML report. Console messages mark each stage and
report counts and output paths so a long run is visible while it is working.

Existing decision workbooks are preserved. New run artifacts are written below
outputs/phase1_runs using a unique run folder.
"""

# %% Configuration — edit file paths or column names here if the workbook changes
from pathlib import Path
from datetime import datetime, timezone
import base64
from html import escape
import hashlib
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.utils import get_column_letter

SCRIPT_VERSION = "1.1.0"
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "rhamnosus.xlsx"
OUTPUT_DIR = BASE_DIR / "outputs"
DECISION_DIR = OUTPUT_DIR / "decision_workbooks"
RUN_DIR = OUTPUT_DIR / "phase1_runs"
DATA_SHEET = "Dataset"
DICTIONARY_SHEET = "Data Dictionary"
STUDY_COLUMN = "Study ID"
RESPONSE_COLUMN = "Cell Density Log CFU/mL"
RESPONSE_BASIS_COLUMN = "Response Basis"
DESIGN_COLUMN = "Experimental Design"


# %% File and dataset helpers — check the workbook fingerprint and required columns
def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_run_directory(root, fingerprint):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = root / f"{fingerprint[:10]}_{stamp}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = root / f"{base.name}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def read_inputs(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    dataset = pd.read_excel(path, sheet_name=DATA_SHEET)
    dictionary = pd.read_excel(path, sheet_name=DICTIONARY_SHEET)
    required = {STUDY_COLUMN, RESPONSE_COLUMN, RESPONSE_BASIS_COLUMN}
    missing = sorted(required.difference(dataset.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    predictor_rows = dictionary.loc[
        dictionary["Variable Category"].astype(str).str.strip().str.casefold().eq("predictor")
    ]
    predictors = [name for name in predictor_rows["Variable Name"].dropna().astype(str)
                  if name in dataset.columns]
    if not predictors:
        raise ValueError("No numeric predictors listed in the Data Dictionary were found.")
    numeric_columns = predictors + [RESPONSE_COLUMN]
    for column in numeric_columns:
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce")
    return dataset, dictionary, predictors, numeric_columns


# %% Descriptive statistics — summarize each variable overall and within each study
def describe_series(series):
    values = pd.to_numeric(series, errors="coerce")
    observed = values.dropna()
    n_total = int(len(values))
    n = int(observed.size)
    unique = int(observed.nunique())
    minimum = observed.min() if n else np.nan
    maximum = observed.max() if n else np.nan
    q1 = observed.quantile(0.25) if n else np.nan
    q3 = observed.quantile(0.75) if n else np.nan
    sd = observed.std(ddof=1) if n > 1 else np.nan
    mean = observed.mean() if n else np.nan
    skewness = observed.skew() if n >= 3 and unique > 2 else np.nan
    excess_kurtosis = observed.kurt() if n >= 4 and unique > 2 else np.nan
    if n == 0:
        distribution = "No observed values"
    elif unique == 1:
        distribution = "Constant among observed values"
    elif unique <= 5:
        distribution = "Few discrete levels"
    elif pd.notna(skewness) and abs(skewness) >= 1:
        distribution = "Potentially skewed (|skew| >= 1)"
    else:
        distribution = "Continuous or multi-level; inspect histogram"
    return {
        "N rows": n_total,
        "N observed": n,
        "N missing": n_total - n,
        "Missing percent": (100 * (n_total - n) / n_total) if n_total else np.nan,
        "N unique": unique,
        "N zero": int(observed.eq(0).sum()),
        "Zero percent of observed": (100 * observed.eq(0).mean()) if n else np.nan,
        "Mean": mean,
        "Median": observed.median() if n else np.nan,
        "SD": sd,
        "Variance": observed.var(ddof=1) if n > 1 else np.nan,
        "Minimum": minimum,
        "Maximum": maximum,
        "Range": (maximum - minimum) if n else np.nan,
        "Q1": q1,
        "Q3": q3,
        "IQR": (q3 - q1) if n else np.nan,
        "Coefficient of variation percent": (100 * sd / abs(mean))
            if n > 1 and pd.notna(sd) and pd.notna(mean) and mean != 0 else np.nan,
        "Skewness": skewness,
        "Excess kurtosis": excess_kurtosis,
        "Distribution note": distribution,
    }


def build_statistics(dataset, columns):
    overall_rows = []
    study_rows = []
    for column in columns:
        if column == RESPONSE_COLUMN:
            # Cell-density values on CFU/mL and CFU/g bases are never summarized together.
            for basis, subset in dataset.groupby(RESPONSE_BASIS_COLUMN, dropna=False, sort=True):
                basis_label = str(basis) if pd.notna(basis) else "Unspecified response basis"
                overall_rows.append({"Variable": column, "Response basis": basis_label,
                                     **describe_series(subset[column])})
                for key, study_subset in subset.groupby(STUDY_COLUMN, dropna=False, sort=True):
                    study_rows.append({"Study ID": key, "Response basis": basis_label,
                                       "Variable": column, **describe_series(study_subset[column])})
        else:
            overall_rows.append({"Variable": column, **describe_series(dataset[column])})
            for study, subset in dataset.groupby(STUDY_COLUMN, dropna=False, sort=True):
                study_rows.append({"Study ID": study, "Variable": column,
                                   **describe_series(subset[column])})
    return pd.DataFrame(overall_rows), pd.DataFrame(study_rows)


# %% Association evidence — keep pooled, basis-specific, within-study, and design-block scopes separate
def benjamini_hochberg(p_values):
    p = np.asarray(p_values, dtype=float)
    result = np.full(p.shape, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return result
    selected = p[valid]
    order = np.argsort(selected)
    ranked = selected[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    restored = np.empty(m)
    restored[order] = adjusted
    result[valid] = restored
    return result


def build_associations(dataset, columns):
    records = []
    scopes = [("Pooled descriptive", "All studies", dataset)]
    scopes.extend(("Pooled response-basis stratum", str(basis), subset)
                  for basis, subset in dataset.groupby(RESPONSE_BASIS_COLUMN, dropna=False, sort=True))
    group_columns = [STUDY_COLUMN]
    if RESPONSE_BASIS_COLUMN in dataset.columns:
        group_columns.append(RESPONSE_BASIS_COLUMN)
    for key, subset in dataset.groupby(group_columns, dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        scope_name = " | ".join(str(value) for value in key_tuple)
        scopes.append(("Within study", scope_name, subset))
    if DESIGN_COLUMN in dataset.columns:
        design_group_columns = [STUDY_COLUMN, DESIGN_COLUMN]
        if RESPONSE_BASIS_COLUMN in dataset.columns:
            design_group_columns.append(RESPONSE_BASIS_COLUMN)
        for key, subset in dataset.groupby(design_group_columns, dropna=False, sort=True):
            key_tuple = key if isinstance(key, tuple) else (key,)
            scope_name = " | ".join(str(value) for value in key_tuple)
            scopes.append(("Within study/design block", scope_name, subset))
    for scope_type, scope_name, frame in scopes:
        for left_index, left in enumerate(columns):
            for right in columns[left_index + 1:]:
                # Never pool response values across CFU/mL and CFU/g bases.
                if scope_type == "Pooled descriptive" and RESPONSE_COLUMN in {left, right}:
                    continue
                pair = frame[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
                n = len(pair)
                pearson_r = pearson_p = spearman_rho = spearman_p = np.nan
                if n >= 3 and pair[left].nunique() > 1 and pair[right].nunique() > 1:
                    pearson = stats.pearsonr(pair[left], pair[right])
                    spearman = stats.spearmanr(pair[left], pair[right])
                    pearson_r, pearson_p = float(pearson.statistic), float(pearson.pvalue)
                    spearman_rho, spearman_p = float(spearman.statistic), float(spearman.pvalue)
                records.append({
                    "Scope type": scope_type, "Scope": scope_name,
                    "Variable 1": left, "Variable 2": right, "N pairs": n,
                    "Pearson r": pearson_r, "Pearson p": pearson_p,
                    "Spearman rho": spearman_rho, "Spearman p": spearman_p,
                })
    result = pd.DataFrame(records)
    result["Pearson q (BH within scope)"] = np.nan
    result["Spearman q (BH within scope)"] = np.nan
    for _, indices in result.groupby(["Scope type", "Scope"]).groups.items():
        result.loc[indices, "Pearson q (BH within scope)"] = benjamini_hochberg(
            result.loc[indices, "Pearson p"])
        result.loc[indices, "Spearman q (BH within scope)"] = benjamini_hochberg(
            result.loc[indices, "Spearman p"])
    result["Pooled high-rank-correlation flag"] = False
    pooled = result["Scope type"].eq("Pooled descriptive")
    result.loc[pooled, "Pooled high-rank-correlation flag"] = (
        result.loc[pooled, "Spearman rho"].abs().ge(0.85)
    )
    return result


# %% Data quality and study context — create review flags without deleting or altering rows
def build_quality_table(dataset, dictionary, predictors):
    notes = {str(row["Variable Name"]): str(row.get("Notes", ""))
             for _, row in dictionary.iterrows() if pd.notna(row.get("Variable Name"))}
    unit = {str(row["Variable Name"]): row.get("Unit", "")
            for _, row in dictionary.iterrows() if pd.notna(row.get("Variable Name"))}
    records = []
    numeric = predictors + [RESPONSE_COLUMN]
    for variable in numeric:
        if variable == RESPONSE_COLUMN:
            scoped_frames = [
                (str(basis) if pd.notna(basis) else "Unspecified response basis", subset)
                for basis, subset in dataset.groupby(RESPONSE_BASIS_COLUMN, dropna=False, sort=True)
            ]
        else:
            scoped_frames = [("", dataset)]
        for basis_label, scoped_frame in scoped_frames:
            s = pd.to_numeric(scoped_frame[variable], errors="coerce")
            observed = s.dropna()
            if len(observed) >= 4:
                q1, q3 = observed.quantile([0.25, 0.75])
                spread = q3 - q1
                low, high = q1 - 1.5 * spread, q3 + 1.5 * spread
                outlier_count = int(((observed < low) | (observed > high)).sum())
            else:
                low = high = np.nan
                outlier_count = 0
            missing_pct = 100 * s.isna().mean() if len(s) else np.nan
            unique = int(observed.nunique())
            concerns = []
            if len(observed) < 3:
                concerns.append("Insufficient observations for stable distribution or association summaries")
            if unique <= 1 and len(observed):
                concerns.append("No observed variation")
            elif unique <= 3 and len(observed):
                concerns.append("Limited number of observed levels")
            if missing_pct >= 50:
                concerns.append("At least 50% missing")
            if outlier_count:
                concerns.append(f"{outlier_count} value(s) beyond the 1.5×IQR screening fences; review, do not auto-remove")
            if variable == RESPONSE_COLUMN and dataset[RESPONSE_BASIS_COLUMN].nunique(dropna=True) > 1:
                concerns.append("Statistics are stratified by response basis; bases are not combined")
            if not concerns:
                concerns.append("No automatic screening flag; inspect source context and distributions")
            records.append({
                "Variable": variable, "Response basis": basis_label,
                "Unit": unit.get(variable, "study-specific response basis"),
                "Missing count": int(s.isna().sum()), "Missing percent": missing_pct,
                "Observed unique values": unique, "IQR lower fence": low, "IQR upper fence": high,
                "Values beyond IQR fences": outlier_count,
                "Data dictionary notes": notes.get(variable, ""),
                "Quality review note": "; ".join(concerns),
            })
    quality = pd.DataFrame(records)
    constant_records = []
    design_keys = [STUDY_COLUMN]
    if DESIGN_COLUMN in dataset.columns:
        design_keys.append(DESIGN_COLUMN)
    if RESPONSE_BASIS_COLUMN in dataset.columns:
        design_keys.append(RESPONSE_BASIS_COLUMN)
    for key, frame in dataset.groupby(design_keys, dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        context = " | ".join(str(value) for value in key_tuple)
        for variable in numeric:
            observed = pd.to_numeric(frame[variable], errors="coerce").dropna()
            if len(observed) >= 2 and observed.nunique() == 1:
                constant_records.append({
                    "Variable": variable,
                    "Response basis": str(key_tuple[-1]) if variable == RESPONSE_COLUMN else "",
                    "Unit": unit.get(variable, "study-specific response basis"),
                    "Missing count": int(frame[variable].isna().sum()),
                    "Missing percent": 100 * frame[variable].isna().mean(),
                    "Observed unique values": 1, "IQR lower fence": np.nan,
                    "IQR upper fence": np.nan, "Values beyond IQR fences": 0,
                    "Data dictionary notes": f"Study/design/basis block: {context}",
                    "Quality review note": f"Constant among {len(observed)} observed value(s) within this study/design block; independent effects may not be separable within the block.",
                })
    if constant_records:
        quality = pd.concat([quality, pd.DataFrame(constant_records)], ignore_index=True)
    duplicate_records = []
    profile_columns = predictors
    if profile_columns:
        for (study, design), frame in dataset.groupby([STUDY_COLUMN, DESIGN_COLUMN], dropna=False):
            if frame.empty:
                continue
            complete_profiles = frame[profile_columns].notna().all(axis=1)
            duplicate_mask = pd.Series(False, index=frame.index)
            duplicate_mask.loc[complete_profiles] = frame.loc[complete_profiles].duplicated(
                subset=profile_columns, keep=False
            )
            n_rows = int(duplicate_mask.sum())
            if n_rows:
                duplicate_records.append({
                    "Variable": "; ".join(profile_columns), "Response basis": "",
                    "Unit": "mixed predictor units",
                    "Missing count": np.nan, "Missing percent": np.nan,
                    "Observed unique values": np.nan, "IQR lower fence": np.nan,
                    "IQR upper fence": np.nan, "Values beyond IQR fences": np.nan,
                    "Data dictionary notes": f"Experimental profiles within {study} / {design}",
                    "Quality review note": f"{n_rows} row(s) share a complete predictor profile; could represent design repeats. Review with run IDs and response values.",
                })
    if duplicate_records:
        quality = pd.concat([quality, pd.DataFrame(duplicate_records)], ignore_index=True)
    return quality


def build_study_characteristics(dataset):
    rows = []
    group_columns = [STUDY_COLUMN, RESPONSE_BASIS_COLUMN]
    for key, frame in dataset.groupby(group_columns, dropna=False, sort=True):
        study, basis = key
        y = pd.to_numeric(frame[RESPONSE_COLUMN], errors="coerce")
        rows.append({
            "Study ID": study, "N observations": len(frame),
            "Response basis": str(basis) if pd.notna(basis) else "Unspecified response basis",
            "N response observed": int(y.notna().sum()),
            "Study year(s)": "; ".join(sorted(frame["Study Year"].dropna().astype(str).unique()))
                if "Study Year" in frame else "",
            "Strains": "; ".join(sorted(frame["Strain Name"].dropna().astype(str).unique()))
                if "Strain Name" in frame else "",
            "Experimental designs": "; ".join(sorted(frame[DESIGN_COLUMN].dropna().astype(str).unique()))
                if DESIGN_COLUMN in frame else "",
            "Response source statuses": "; ".join(sorted(frame["Response Source Status"].dropna().astype(str).unique()))
                if "Response Source Status" in frame else "",
            **{f"Response {key}": value for key, value in describe_series(y).items()},
        })
    return pd.DataFrame(rows)


# %% Researcher decision workbook — format evidence and leave final decisions to the researcher
def make_decision_rows(dataset, dictionary, predictors, overall_stats, study_stats, associations):
    dict_rows = dictionary.set_index("Variable Name", drop=False)
    overall_lookup = overall_stats.set_index("Variable")
    stats_lookup = study_stats[study_stats["Variable"].isin(predictors)]
    pooled_assoc = associations[
        associations["Scope type"].eq("Pooled response-basis stratum")
        & associations["Variable 2"].eq(RESPONSE_COLUMN)
    ]
    response_assoc = associations[
        associations["Scope type"].isin({"Within study", "Within study/design block"})
        & associations["Variable 2"].eq(RESPONSE_COLUMN)
    ]
    high_pairs = associations[
        associations["Scope type"].eq("Pooled descriptive")
        & associations["Pooled high-rank-correlation flag"]
    ]
    redundancy = {name: [] for name in predictors}
    for _, row in high_pairs.iterrows():
        left, right = row["Variable 1"], row["Variable 2"]
        if left in redundancy:
            redundancy[left].append(f"{right} (pooled Spearman rho={row['Spearman rho']:.3f})")
        if right in redundancy:
            redundancy[right].append(f"{left} (pooled Spearman rho={row['Spearman rho']:.3f})")
    rows = []
    for index, predictor in enumerate(predictors, start=1):
        metadata = dict_rows.loc[predictor]
        overall = overall_lookup.loc[predictor]
        by_study = stats_lookup[stats_lookup["Variable"].eq(predictor)]
        coverage = int(by_study.loc[by_study["N observed"].gt(0), "Study ID"].nunique())
        missing_by_study = "; ".join(
            f"{r['Study ID']}: {r['Missing percent']:.1f}% missing"
            for _, r in by_study.iterrows()
        )
        within = response_assoc[response_assoc["Variable 1"].eq(predictor)]
        within_text = "; ".join(
            f"{r['Scope']}: Spearman rho={r['Spearman rho']:.3f} (n={int(r['N pairs'])})"
            for _, r in within.iterrows() if pd.notna(r["Spearman rho"])
        ) or "Insufficient or no within-study correlation estimates"
        pooled_rows = pooled_assoc[pooled_assoc["Variable 1"].eq(predictor)]
        pooled_text = "; ".join(
            f"Basis {row['Scope']}: Pearson r={row['Pearson r']:.3f}, p={row['Pearson p']:.3g}; "
            f"Spearman rho={row['Spearman rho']:.3f}, q(BH)={row['Spearman q (BH within scope)']:.3g}; n={int(row['N pairs'])}"
            for _, row in pooled_rows.iterrows() if pd.notna(row["Spearman rho"])
        ) or "Insufficient or no response-basis-stratified correlation estimate"
        flags = []
        if overall["N observed"] < 3:
            flags.append("insufficient overall observations")
        if overall["N unique"] <= 1:
            flags.append("no observed variation")
        if overall["Missing percent"] >= 50:
            flags.append("substantial missingness (>=50%)")
        if not flags:
            flags.append("review study coverage, units, design, and potential confounding")
        rows.append({
            "Predictor ID": f"P{index:02d}", "Predictor name": predictor,
            "Biological or technological role": metadata.get("Scientific Description", ""),
            "Measurement unit": metadata.get("Unit", ""),
            "Overall N observed": overall["N observed"], "Overall missing percent": overall["Missing percent"],
            "Overall unique values": overall["N unique"], "Overall mean": overall["Mean"],
            "Overall median": overall["Median"], "Overall SD": overall["SD"],
            "Overall minimum": overall["Minimum"], "Overall maximum": overall["Maximum"],
            "Study coverage (of all studies)": f"{coverage} of {dataset[STUDY_COLUMN].nunique()}",
            "Study-level missingness": missing_by_study,
            "Variation and distribution": overall["Distribution note"],
            "Within-study response associations": within_text,
            "Pooled response association (descriptive only)": pooled_text,
            "Between-study characteristics": "See Predictor_Statistics_By_Study; study identity and design may confound pooled patterns.",
            "Redundancy information": "; ".join(redundancy[predictor]) or "No pooled |Spearman rho| >= 0.85 flag",
            "Biological relevance evidence": metadata.get("Notes", ""),
            "Potential predictive usefulness": "Association is descriptive evidence only; assess study-held-out predictive performance in Phase 2.",
            "Potential limitations": "; ".join(flags) + "; units and experimental contexts must be checked before pooling.",
            "Automated preliminary recommendation": "Review evidence; no automatic retain/exclude decision is made.",
            "Researcher decision": "", "Researcher rationale": "",
            "Conditions or restrictions": "", "Condition Minimum": "", "Condition Maximum": "",
            "Final researcher status": "Pending researcher review",
        })
    return pd.DataFrame(rows)


def style_workbook(path):
    workbook = load_workbook(path)
    header_fill = PatternFill("solid", fgColor="17365D")
    edit_fill = PatternFill("solid", fgColor="FFF2CC")
    for sheet in workbook.worksheets:
        # Freeze the identifying fields so they stay visible while reviewing the wide evidence sheet.
        sheet.sheet_view.showGridLines = False
        sheet.freeze_panes = "C2" if sheet.title == "Predictor_Decisions" else ("A2" if sheet.max_row > 1 else "A1")
        if sheet.max_row > 1:
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.fill = header_fill
                cell.font = Font(color="FFFFFF", bold=True)
                cell.alignment = Alignment(vertical="top", wrap_text=True)
        if sheet.title == "Predictor_Decisions":
            widths = {
                "Predictor ID": 10, "Predictor name": 32,
                "Biological or technological role": 38, "Measurement unit": 16,
                "Overall N observed": 13, "Overall missing percent": 16,
                "Overall unique values": 15, "Overall mean": 13,
                "Overall median": 13, "Overall SD": 13,
                "Overall minimum": 13, "Overall maximum": 13,
                "Study coverage (of all studies)": 20, "Study-level missingness": 34,
                "Variation and distribution": 30, "Within-study response associations": 38,
                "Pooled response association (descriptive only)": 42,
                "Between-study characteristics": 35, "Redundancy information": 35,
                "Biological relevance evidence": 35, "Potential predictive usefulness": 34,
                "Potential limitations": 38, "Automated preliminary recommendation": 30,
                "Researcher decision": 18, "Researcher rationale": 30,
                "Conditions or restrictions": 30, "Condition Minimum": 14,
                "Condition Maximum": 14, "Final researcher status": 22,
            }
            header_to_column = {cell.value: cell.column for cell in sheet[1]}
            for title, width in widths.items():
                if title in header_to_column:
                    sheet.column_dimensions[get_column_letter(header_to_column[title])].width = width
            input_titles = {"Researcher decision", "Researcher rationale", "Conditions or restrictions",
                            "Condition Minimum", "Condition Maximum", "Final researcher status"}
            for title in input_titles:
                column = header_to_column.get(title)
                if column:
                    for row in range(2, sheet.max_row + 1):
                        sheet.cell(row, column).fill = edit_fill
            sheet.sheet_view.zoomScale = 70
            sheet.sheet_properties.pageSetUpPr.fitToPage = True
            sheet.page_setup.orientation = "landscape"
            sheet.page_setup.fitToWidth = 1
            sheet.page_setup.fitToHeight = 0
            for row in sheet.iter_rows(min_row=2):
                for cell in row:
                    cell.alignment = Alignment(vertical="top", wrap_text=True)
            continue
        if sheet.title == "Instructions":
            sheet.column_dimensions["A"].width = 30
            sheet.column_dimensions["B"].width = 105
            sheet.sheet_view.zoomScale = 90
            for row in sheet.iter_rows(min_row=2):
                for cell in row:
                    cell.alignment = Alignment(vertical="top", wrap_text=True)
            continue
        for column_cells in sheet.columns:
            letter = get_column_letter(column_cells[0].column)
            max_length = max((len(str(cell.value)) for cell in column_cells if cell.value is not None), default=10)
            sheet.column_dimensions[letter].width = min(max(max_length + 2, 12), 48)
        sheet.sheet_view.zoomScale = 85
    decisions = workbook["Predictor_Decisions"]
    decision_column = next(i for i, cell in enumerate(decisions[1], start=1)
                           if cell.value == "Researcher decision")
    validation = DataValidation(type="list", formula1='"Retain,Exclude,Conditional"', allow_blank=True)
    validation.error = "Choose Retain, Exclude, or Conditional."
    validation.errorTitle = "Select a listed decision"
    validation.prompt = "For Conditional, document conditions and numeric bounds where applicable."
    validation.promptTitle = "Researcher decision"
    decisions.add_data_validation(validation)
    validation.add(f"{get_column_letter(decision_column)}2:{get_column_letter(decision_column)}{decisions.max_row}")
    status_column = next(i for i, cell in enumerate(decisions[1], start=1)
                         if cell.value == "Final researcher status")
    status_validation = DataValidation(
        type="list", formula1='"Pending researcher review,Reviewed,Needs discussion"', allow_blank=True
    )
    status_validation.prompt = "Mark the review status after considering the evidence."
    status_validation.promptTitle = "Researcher review status"
    decisions.add_data_validation(status_validation)
    status_validation.add(f"{get_column_letter(status_column)}2:{get_column_letter(status_column)}{decisions.max_row}")
    workbook.save(path)


def write_decision_workbook(path, decision_rows, study_stats, overall_stats,
                            study_characteristics, associations, quality, metadata):
    instructions = pd.DataFrame([
        ["Purpose", "Choose which predictors should enter the modeling question. Evidence is provided for review; no decision is made automatically."],
        ["1. Review", "Use Predictor_Decisions as the main sheet. Filter or sort predictors, then consult Predictor_Statistics_By_Study, Integrated_Predictor_Statistics, Association_Evidence, Study_Characteristics, or Data_Quality as needed."],
        ["2. Decide", "Use the dropdown to choose Retain, Exclude, or Conditional. Yellow cells are for your decision, rationale, restrictions, and review status."],
        ["3. Conditional", "Explain the condition in Conditions or restrictions. Enter inclusive numeric bounds when the condition is a numeric range. Phase 2 applies numeric bounds only; it does not interpret free text."],
        ["4. Mark reviewed", "Set Final researcher status to Reviewed after checking the evidence. Use Needs discussion if you have not settled the decision."],
        ["Interpret associations", "Pooled correlations may reflect study, strain, or design differences. Review response basis, sample size, and within-study evidence. Correlation does not establish causation."],
        ["Missing and unusual values", "Missing values remain missing. IQR flags prompt a review; this workflow does not remove rows or change values to zero."],
        ["Phase 2 handoff", "Save your decisions. Phase 2 checks the source fingerprint and reads the Predictor_Decisions sheet."],
        ["Preservation", "Phase 1 keeps an existing decision workbook unchanged. A new source fingerprint or script version produces a separate workbook file."],
    ], columns=["Topic", "Instructions"])
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        instructions.to_excel(writer, sheet_name="Instructions", index=False)
        decision_rows.to_excel(writer, sheet_name="Predictor_Decisions", index=False)
        study_stats.to_excel(writer, sheet_name="Predictor_Statistics_By_Study", index=False)
        overall_stats[overall_stats["Variable"].isin(decision_rows["Predictor name"].tolist())].to_excel(
            writer, sheet_name="Integrated_Predictor_Statistics", index=False)
        overall_stats[overall_stats["Variable"].eq(RESPONSE_COLUMN)].to_excel(
            writer, sheet_name="Integrated_Response_By_Basis", index=False)
        study_characteristics.to_excel(writer, sheet_name="Study_Characteristics", index=False)
        associations.to_excel(writer, sheet_name="Association_Evidence", index=False)
        quality.to_excel(writer, sheet_name="Data_Quality", index=False)
        pd.DataFrame([metadata]).to_excel(writer, sheet_name="Workflow_Metadata", index=False)
    style_workbook(path)


def find_or_create_decision_workbook(fingerprint, decision_rows, study_stats,
                                     overall_stats, study_characteristics,
                                     associations, quality, metadata):
    DECISION_DIR.mkdir(parents=True, exist_ok=True)
    standard_path = DECISION_DIR / "researcher_decision_workbook.xlsx"
    existing_paths = [standard_path] if standard_path.exists() else []
    existing_paths += sorted(DECISION_DIR.glob(f"researcher_decision_workbook_{fingerprint[:10]}*.xlsx"))
    for path in existing_paths:
        try:
            meta = pd.read_excel(path, sheet_name="Workflow_Metadata")
            stored = str(meta.loc[0, "Source dataset SHA256"])
            stored_version = str(meta.loc[0, "Script version"])
        except Exception:
            stored = "unreadable"
            stored_version = "unreadable"
        if stored == fingerprint and stored_version == SCRIPT_VERSION:
            print(f"Preserving existing researcher workbook with matching source fingerprint: {path}")
            return path
        if stored == fingerprint:
            print(f"Preserving older evidence workbook from script version {stored_version}: {path}")
        if path == standard_path:
            print("Existing standard decision workbook is stale or unreadable; preserving it.")
    if not standard_path.exists():
        write_decision_workbook(standard_path, decision_rows, study_stats, overall_stats,
                                study_characteristics, associations, quality, metadata)
        print(f"Created fresh researcher decision workbook: {standard_path}")
        return standard_path
    version_tag = SCRIPT_VERSION.replace(".", "_")
    for prefix_length in (10, 16, 24, 32, 64):
        stem = f"researcher_decision_workbook_{fingerprint[:prefix_length]}_v{version_tag}"
        path = DECISION_DIR / f"{stem}.xlsx"
        suffix = 1
        while path.exists():
            try:
                existing_metadata = pd.read_excel(path, sheet_name="Workflow_Metadata")
                existing_hash = str(existing_metadata.loc[0, "Source dataset SHA256"])
                existing_version = str(existing_metadata.loc[0, "Script version"])
                if existing_hash == fingerprint and existing_version == SCRIPT_VERSION:
                    print(f"Preserving existing versioned workbook without modification: {path}")
                    return path
            except Exception:
                pass
            path = DECISION_DIR / f"{stem}_{suffix:02d}.xlsx"
            suffix += 1
        write_decision_workbook(path, decision_rows, study_stats, overall_stats,
                                study_characteristics, associations, quality, metadata)
        print(f"Created fresh researcher decision workbook: {path}")
        return path
    raise RuntimeError("Could not create a unique decision workbook without overwriting an existing file.")


# %% Outputs and figures — make publication-review plots and a self-contained HTML report
def save_figures(dataset, predictors, run_dir):
    """Save figures that help the researcher inspect study heterogeneity and coverage."""
    print("  [Figures 1/4] Comparing reported response distributions by study and basis...")
    bases = sorted(dataset[RESPONSE_BASIS_COLUMN].dropna().astype(str).unique())
    fig, axes = plt.subplots(len(bases), 1, figsize=(max(8, dataset[STUDY_COLUMN].nunique() * 1.2), 4.5 * len(bases)), squeeze=False)
    for ax, basis in zip(axes[:, 0], bases):
        subset = dataset.loc[dataset[RESPONSE_BASIS_COLUMN].astype(str).eq(basis)]
        studies = sorted(subset[STUDY_COLUMN].dropna().astype(str).unique())
        response_data = [pd.to_numeric(
            subset.loc[subset[STUDY_COLUMN].astype(str).eq(study), RESPONSE_COLUMN], errors="coerce"
        ).dropna().to_numpy() for study in studies]
        ax.boxplot(response_data, showmeans=True)
        ax.set_xticks(range(1, len(studies) + 1), studies)
        ax.set_title(f"Reported response by study; response basis: {basis}")
        ax.set_ylabel("Reported cell density (log units; basis shown in title)")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(run_dir / "response_by_study_and_basis.png", dpi=180)
    plt.close(fig)

    print("  [Figures 2/4] Mapping predictor missingness and study coverage...")
    study_coverage = dataset.groupby(STUDY_COLUMN)[predictors].count().gt(0).sum(axis=0)
    missing_percent = dataset[predictors].isna().mean().mul(100)
    order = missing_percent.sort_values(ascending=True).index
    values = missing_percent.loc[order]
    coverage_values = study_coverage.loc[order]
    colors = plt.cm.viridis(coverage_values.to_numpy() / max(1, dataset[STUDY_COLUMN].nunique()))
    fig, ax = plt.subplots(figsize=(10, max(6, len(predictors) * 0.32)))
    positions = np.arange(len(order))
    ax.barh(positions, values.to_numpy(), color=colors, alpha=0.9)
    ax.set_yticks(positions, order)
    ax.set_xlim(0, 125)
    ax.set_xlabel("Rows missing (%)")
    ax.set_title("Predictor completeness; label shows number of studies with observed values")
    for position, percent, coverage in zip(positions, values.to_numpy(), coverage_values.to_numpy()):
        ax.text(min(percent + 2, 100), position, f"{int(coverage)} studies", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(run_dir / "predictor_coverage_and_missingness.png", dpi=180)
    plt.close(fig)

    print("  [Figures 3/4] Counting observations by study and response basis...")
    study_counts = dataset.groupby([STUDY_COLUMN, RESPONSE_BASIS_COLUMN], dropna=False).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(max(9, len(study_counts) * 1.5), 5.5))
    study_counts.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
    ax.set_ylabel("Dataset rows")
    ax.set_xlabel("Study ID")
    ax.set_title("Rows available from each study, separated by reported response basis")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Response basis", frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(run_dir / "study_rows_by_response_basis.png", dpi=180)
    plt.close(fig)

    print("  [Figures 4/4] Calculating pooled predictor correlations (descriptive only)...")
    matrix = dataset[predictors].corr(method="spearman", min_periods=3)
    fig, ax = plt.subplots(figsize=(max(9, len(matrix.columns) * 0.55), max(7, len(matrix.columns) * 0.5)))
    image = ax.imshow(matrix.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(matrix.columns)), matrix.columns, fontsize=7)
    ax.set_title("Pooled predictor-only Spearman correlations; study-confounded descriptive evidence")
    fig.colorbar(image, ax=ax, label="Spearman rho")
    fig.tight_layout()
    fig.savefig(run_dir / "pooled_spearman_correlations.png", dpi=180)
    plt.close(fig)

    print("  Writing predictor distribution appendix (PDF)...")
    scopes = [("All studies", dataset)]
    scopes.extend((f"Study {study}", frame) for study, frame in
                  dataset.groupby(STUDY_COLUMN, dropna=False, sort=True))
    with PdfPages(run_dir / "predictor_distributions_by_scope.pdf") as pdf:
        for scope_number, (scope_name, frame) in enumerate(scopes, start=1):
            print(f"    Distribution page {scope_number}/{len(scopes)}: {scope_name}")
            nrows = max(1, (len(predictors) + 3) // 4)
            fig, axes = plt.subplots(nrows, 4, figsize=(11, 2.6 * nrows), squeeze=False)
            for ax, predictor in zip(axes.flat, predictors):
                values = pd.to_numeric(frame[predictor], errors="coerce").dropna()
                if len(values):
                    ax.hist(values, bins="auto", color="#4C78A8", alpha=0.85)
                ax.set_title(predictor, fontsize=8)
                ax.tick_params(labelsize=7)
            for ax in list(axes.flat)[len(predictors):]:
                ax.set_visible(False)
            fig.suptitle(f"Predictor distributions: {scope_name}", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.98))
            pdf.savefig(fig)
            plt.close(fig)


def write_visual_report(run_dir, dataset, predictors, decision_rows, metadata, workbook_path):
    """Create a self-contained HTML report that opens directly in a browser."""
    def image_tag(filename, alt, caption):
        image_path = run_dir / filename
        image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return (
            f'<figure><img src="data:image/png;base64,{image_data}" alt="{escape(alt)}">'
            f'<figcaption>{escape(caption)}</figcaption></figure>'
        )

    basis_counts = dataset[RESPONSE_BASIS_COLUMN].value_counts(dropna=False)
    basis_text = " | ".join(f"{escape(str(basis))}: {int(count)} rows" for basis, count in basis_counts.items())
    summary_rows = decision_rows[["Predictor ID", "Predictor name", "Measurement unit",
                                  "Study coverage (of all studies)", "Overall N observed",
                                  "Overall missing percent", "Potential limitations"]].copy()
    summary_rows = summary_rows.sort_values("Overall missing percent", ascending=False)
    summary_table = summary_rows.to_html(index=False, escape=True, border=0, classes="data-table",
                                         formatters={"Overall missing percent": lambda value: f"{value:.1f}%"})
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Phase 1 visual report</title><style>
:root{{color-scheme:light;--ink:#18324a;--muted:#587083;--line:#d9e2ea;--accent:#277da1;--bg:#f3f6f8}}
body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 Segoe UI,Arial,sans-serif}}
main{{max-width:1120px;margin:0 auto;padding:34px 22px 60px}}h1{{font-size:2rem;margin:0 0 4px}}h2{{margin-top:34px;border-bottom:1px solid var(--line);padding-bottom:8px}}
.sub{{color:var(--muted);margin:0 0 22px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:20px 0}}
.card,.panel{{background:white;border:1px solid var(--line);border-radius:12px;padding:16px;box-shadow:0 3px 12px #19324a0a}}
.card b{{display:block;font-size:1.65rem;color:var(--accent)}}.card span{{color:var(--muted)}}.note{{background:#e8f3f7;border-left:4px solid var(--accent);padding:13px 16px;border-radius:5px}}
figure{{margin:20px 0 28px;background:white;border:1px solid var(--line);border-radius:12px;padding:14px}}figure img{{display:block;max-width:100%;height:auto;margin:auto}}figcaption{{color:var(--muted);font-size:.92rem;margin-top:10px}}
.data-table{{border-collapse:collapse;width:100%;background:white;font-size:.9rem}}.data-table th,.data-table td{{border-bottom:1px solid var(--line);padding:9px;text-align:left;vertical-align:top}}.data-table th{{background:#e7eff4;position:sticky;top:0}}
a{{color:#126782}}.links li{{margin:7px 0}}small{{color:var(--muted)}}
</style></head><body><main>
<h1>Study-level evidence report</h1>
<p class="sub">Phase 1 | Source: {escape(metadata['Source dataset file'])} | Generated {escape(metadata['Analysis date UTC'])}</p>
<div class="cards"><div class="card"><b>{int(len(dataset))}</b><span>dataset rows</span></div>
<div class="card"><b>{int(dataset[STUDY_COLUMN].nunique())}</b><span>studies</span></div>
<div class="card"><b>{len(predictors)}</b><span>predictors</span></div>
<div class="card"><b>{int(dataset[RESPONSE_COLUMN].notna().sum())}</b><span>observed response values</span></div></div>
<div class="note"><b>Response basis counts:</b> {basis_text}. These reported bases remain separate in the plots. Associations are descriptive and may reflect differences between studies, strains, or designs.</div>
<h2>Study coverage</h2>
{image_tag('study_rows_by_response_basis.png', 'Stacked bar chart of dataset rows by study and response basis', 'The bar heights show how much data each study contributes and which response basis each row reports.')}
<h2>Response distributions</h2>
{image_tag('response_by_study_and_basis.png', 'Response boxplots by study and response basis', 'Compare distributions within each response basis. A separation between studies is not by itself evidence of a predictor effect.')}
<h2>Predictor completeness</h2>
{image_tag('predictor_coverage_and_missingness.png', 'Predictor missingness and study coverage', 'Longer bars indicate more missing rows. Labels report the number of studies with at least one observed value.')}
<h2>Predictor relationships</h2>
{image_tag('pooled_spearman_correlations.png', 'Pooled Spearman correlation heatmap for predictors', 'Pooled correlations are a screening view; study, design, and strain can confound pooled patterns.')}
<h2>Predictor review table</h2>
<div class="panel"><p>Use the workbook dropdown and rationale fields to record decisions. Sort this table by missingness to prioritize review, then check the study-specific statistics and response association evidence.</p>{summary_table}</div>
<h2>Files from this run</h2><ul class="links"><li><a href="{escape(os.path.relpath(workbook_path, run_dir).replace(os.sep, '/'))}">Open the researcher decision workbook</a></li>
<li><a href="predictor_distributions_by_scope.pdf">Open the predictor distribution appendix (PDF)</a></li>
<li><a href="association_evidence.csv">Download complete association evidence (CSV)</a></li>
<li><a href="data_quality_review.csv">Download data-quality review (CSV)</a></li></ul>
<small>Missing values are preserved. This report does not select predictors or establish causal effects.</small>
</main></body></html>"""
    report_path = run_dir / "phase1_visual_report.html"
    report_path.write_text(report, encoding="utf-8")
    return report_path


# %% Run Phase 1 — visible progress messages mark input, analysis, export, and reporting stages
if __name__ == "__main__":
    print("PHASE 1 / 7 — source data and study-level evidence")
    print(f"[1/7] Reading source workbook: {DATASET_PATH}")
    fingerprint = sha256_file(DATASET_PATH)
    dataset, dictionary, predictors, numeric_columns = read_inputs(DATASET_PATH)
    print(f"      Loaded {len(dataset)} rows, {dataset[STUDY_COLUMN].nunique()} studies, and {len(predictors)} predictors.")
    print(f"      Observed responses: {dataset[RESPONSE_COLUMN].notna().sum()} / {len(dataset)}; response bases: "
          f"{', '.join(sorted(dataset[RESPONSE_BASIS_COLUMN].dropna().astype(str).unique()))}.")

    print("[2/7] Calculating integrated and study-specific descriptive statistics...")
    overall_stats, study_stats = build_statistics(dataset, numeric_columns)
    print(f"      Statistics complete: {len(overall_stats)} variables overall; {len(study_stats)} study-variable rows.")

    print("[3/7] Calculating association evidence by pooled basis, study, and design block...")
    association_columns = predictors + [RESPONSE_COLUMN]
    associations = build_associations(dataset, association_columns)
    print(f"      Association table contains {len(associations):,} rows.")

    print("[4/7] Summarizing study characteristics and data-quality review flags...")
    study_characteristics = build_study_characteristics(dataset)
    quality = build_quality_table(dataset, dictionary, predictors)
    print(f"      Review tables ready: {len(study_characteristics)} study groups; {len(quality)} quality checks.")

    created_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metadata = {
        "Source dataset file": DATASET_PATH.name,
        "Source dataset SHA256": fingerprint,
        "Analysis date UTC": created_utc,
        "Script version": SCRIPT_VERSION,
        "Dataset rows": len(dataset),
        "Study count": int(dataset[STUDY_COLUMN].nunique(dropna=True)),
        "Predictor count": len(predictors),
        "Response column": RESPONSE_COLUMN,
        "Response bases present": "; ".join(sorted(dataset[RESPONSE_BASIS_COLUMN].dropna().astype(str).unique())),
        "Notes": "Missing values preserved; no imputation or outlier removal; pooled associations are descriptive.",
    }
    decision_rows = make_decision_rows(dataset, dictionary, predictors, overall_stats,
                                       study_stats, associations)
    print("[5/7] Creating or locating the versioned decision workbook (existing decisions are preserved)...")
    workbook_path = find_or_create_decision_workbook(
        fingerprint, decision_rows, study_stats, overall_stats,
        study_characteristics, associations, quality, metadata
    )
    run_dir = make_run_directory(RUN_DIR, fingerprint)
    print(f"[6/7] Writing Phase 1 tables to: {run_dir}")
    overall_stats.to_csv(run_dir / "integrated_numeric_statistics.csv", index=False)
    study_stats.to_csv(run_dir / "numeric_statistics_by_study.csv", index=False)
    associations.to_csv(run_dir / "association_evidence.csv", index=False)
    quality.to_csv(run_dir / "data_quality_review.csv", index=False)
    study_characteristics.to_csv(run_dir / "study_characteristics.csv", index=False)
    dataset[[STUDY_COLUMN, RESPONSE_BASIS_COLUMN] + predictors + [RESPONSE_COLUMN]].to_csv(
        run_dir / "study_aware_analysis_data.csv", index=False)
    save_figures(dataset, predictors, run_dir)
    print("  Creating the browser-ready visual report with embedded figures...")
    report_path = write_visual_report(run_dir, dataset, predictors, decision_rows, metadata, workbook_path)
    run_manifest = {
        **metadata,
        "Decision workbook": str(workbook_path.relative_to(BASE_DIR)),
        "Decision workbook SHA256": sha256_file(workbook_path),
        "Phase 1 output directory": str(run_dir.relative_to(BASE_DIR)),
        "Visual report": str(report_path.relative_to(BASE_DIR)),
    }
    (run_dir / "phase1_run_metadata.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Generated Phase 1 evidence in: {run_dir}")
    print(f"Decision workbook (preserved across reruns): {workbook_path}")
    print(f"Visual report (open this HTML file in a browser): {report_path}")
    print("No researcher decisions or numerical conclusions were made by this script.")
