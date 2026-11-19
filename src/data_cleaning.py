"""
data_cleaning.py
Two jobs:
  1. detect_issues()  - look at a dataset and report problems (changes nothing).
  2. apply_cleaning() - make a CLEANED COPY of the dataset using options the user chose.

The original (raw) DataFrame is never modified.
"""

import warnings

import numpy as np
import pandas as pd

from src.config import (
    CONVERSION_THRESHOLD,
    MAX_UNIQUE_FOR_TEXT_CHECK,
    MIN_ROWS_FOR_OUTLIERS,
    NON_NEGATIVE_KEYWORDS,
    OUTLIER_IQR_MULTIPLIER,
    SAMPLE_SIZE,
)

SEVERITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}


# ---------------------------------------------------------------------------
# Small helper functions (names start with _ because they are used only here)
# ---------------------------------------------------------------------------
def _is_text_column(series: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def _text_to_numeric(series: pd.Series) -> pd.Series:
    """Turn text like '₹1,200' into the number 1200. Unreadable values become NaN."""
    was_missing = series.isna()
    text = series.astype(str).str.replace(r"[,\s₹$€£]", "", regex=True)
    numbers = pd.to_numeric(text, errors="coerce").astype("float64")
    return numbers.where(~was_missing)


def _text_to_dates(series: pd.Series) -> pd.Series:
    """Turn text into dates. Unreadable values become NaT (missing date)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # hide Pandas' "could not infer format" warning
        return pd.to_datetime(series, errors="coerce")


def _sample(series: pd.Series) -> pd.Series:
    """A small random sample of non-missing values, so tests on big files stay fast."""
    values = series.dropna()
    if len(values) > SAMPLE_SIZE:
        values = values.sample(SAMPLE_SIZE, random_state=0)
    return values


def _issue(category, column, count, severity, explanation, recommendation) -> dict:
    return {
        "category": category,
        "column": column,
        "count": int(count),
        "severity": severity,
        "explanation": explanation,
        "recommendation": recommendation,
    }


def _tidy_text(value):
    """Remove spaces around text. Blank text becomes a missing value."""
    if isinstance(value, str):
        value = value.strip()
        return np.nan if value == "" else value
    return value


# ---------------------------------------------------------------------------
# Detection: each _check_ function looks for one kind of problem
# ---------------------------------------------------------------------------
def _check_missing(df):
    issues = []
    for column in df.columns:
        missing = int(df[column].isna().sum())
        if missing == 0:
            continue
        percent = missing / len(df) * 100
        severity = "High" if percent >= 20 else "Medium" if percent >= 5 else "Low"
        issues.append(
            _issue(
                "Missing values", column, missing, severity,
                f"{missing:,} of {len(df):,} values ({percent:.1f}%) are empty.",
                "Leave them empty, fill them (median/mean for numbers, 'Unknown' for text), "
                "or drop those rows. Filling numbers creates estimates, so use it carefully.",
            )
        )
    return issues


def _check_duplicates(df):
    duplicates = int(df.duplicated().sum())
    if duplicates == 0:
        return []
    return [
        _issue(
            "Duplicate rows", "(whole row)", duplicates, "High",
            f"{duplicates:,} rows are exact copies of another row. "
            "Duplicates can inflate totals such as revenue.",
            "Remove duplicate rows (the first copy is kept).",
        )
    ]


def _check_numbers_as_text(df):
    issues = []
    for column in df.columns:
        if not _is_text_column(df[column]):
            continue
        sample = _sample(df[column])
        if sample.empty:
            continue
        if _text_to_numeric(sample).notna().mean() < CONVERSION_THRESHOLD:
            continue
        values = df[column].dropna()
        unreadable = int(_text_to_numeric(values).isna().sum())
        extra = f" {unreadable:,} values cannot be read as numbers and would become missing." if unreadable else ""
        issues.append(
            _issue(
                "Numbers stored as text", column, len(values), "High",
                "This column is stored as text but looks like numbers, so it cannot be "
                f"summed or averaged correctly.{extra}",
                "Convert the column to numbers. Check that it is not an ID or phone number first.",
            )
        )
    return issues


def _check_dates_as_text(df):
    issues = []
    for column in df.columns:
        if not _is_text_column(df[column]):
            continue
        sample = _sample(df[column])
        if sample.empty:
            continue
        if _text_to_numeric(sample).notna().mean() >= CONVERSION_THRESHOLD:
            continue  # it is numbers, not dates
        if _text_to_dates(sample).notna().mean() < CONVERSION_THRESHOLD:
            continue
        values = df[column].dropna()
        invalid = int(_text_to_dates(values).isna().sum())
        extra = f" {invalid:,} values are not valid dates and would become missing." if invalid else ""
        issues.append(
            _issue(
                "Dates stored as text", column, len(values), "Medium" if invalid == 0 else "High",
                f"This column holds dates as text, so monthly trends cannot be calculated.{extra}",
                "Convert the column to a date type. Invalid dates become missing so you can review them.",
            )
        )
    return issues


def _check_whitespace(df):
    issues = []
    for column in df.columns:
        if not _is_text_column(df[column]):
            continue
        values = df[column].dropna()
        values = values[values.map(lambda v: isinstance(v, str))]
        if values.empty:
            continue
        count = int((values != values.str.strip()).sum())
        if count:
            issues.append(
                _issue(
                    "Extra spaces / blank text", column, count, "Low",
                    f"{count:,} values have spaces at the start or end (or are blank). "
                    "'Mysuru' and 'Mysuru ' would be counted as different regions.",
                    "Trim the spaces and treat blank text as missing.",
                )
            )
    return issues


def _check_inconsistent_text(df):
    issues = []
    for column in df.columns:
        if not _is_text_column(df[column]):
            continue
        values = df[column].dropna().astype(str).str.strip()
        if values.empty or values.nunique() > MAX_UNIQUE_FOR_TEXT_CHECK:
            continue
        keys = values.str.lower()
        variants = values.groupby(keys).nunique()
        bad_keys = variants[variants > 1].index
        if len(bad_keys) == 0:
            continue
        examples = [" / ".join(sorted(values[keys == key].unique())) for key in bad_keys[:3]]
        affected = int(keys.isin(bad_keys).sum())
        issues.append(
            _issue(
                "Inconsistent spelling", column, affected, "Medium",
                f"The same value is written in different ways, for example: {'; '.join(examples)}. "
                "Grouping would split them into separate categories.",
                "Unify each group to its most common spelling.",
            )
        )
    return issues


def _check_negative_values(df):
    issues = []
    for column in df.select_dtypes(include="number").columns:
        if not any(word in column.lower() for word in NON_NEGATIVE_KEYWORDS):
            continue
        negatives = int((df[column] < 0).sum())
        if negatives:
            issues.append(
                _issue(
                    "Potentially invalid values", column, negatives, "High",
                    f"{negatives:,} values are negative, but a column named '{column}' "
                    "normally should not be. They may be returns, refunds or data-entry errors.",
                    "Check these rows manually. They are NOT changed automatically.",
                )
            )
    return issues


def _check_outliers(df):
    issues = []
    for column in df.select_dtypes(include="number").columns:
        values = df[column].dropna()
        if len(values) < MIN_ROWS_FOR_OUTLIERS:
            continue
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        low = q1 - OUTLIER_IQR_MULTIPLIER * iqr
        high = q3 + OUTLIER_IQR_MULTIPLIER * iqr
        count = int(((values < low) | (values > high)).sum())
        if count:
            issues.append(
                _issue(
                    "Outliers", column, count, "Low",
                    f"{count:,} values fall outside the usual range ({low:,.2f} to {high:,.2f}). "
                    "Outliers may be errors, or they may be genuine big orders.",
                    "Review them before deciding. They are NOT removed automatically.",
                )
            )
    return issues


def detect_issues(df: pd.DataFrame) -> list:
    """Run every check and return a list of issues, most serious first."""
    if df.empty:
        return []
    issues = []
    for check in (
        _check_missing,
        _check_duplicates,
        _check_numbers_as_text,
        _check_dates_as_text,
        _check_whitespace,
        _check_inconsistent_text,
        _check_negative_values,
        _check_outliers,
    ):
        issues.extend(check(df))
    issues.sort(key=lambda issue: SEVERITY_ORDER[issue["severity"]])
    return issues


# ---------------------------------------------------------------------------
# Cleaning: works on a COPY of the data
# ---------------------------------------------------------------------------
def apply_cleaning(df: pd.DataFrame, options: dict):
    """
    Create a cleaned copy of df.

    options may contain:
      trim_whitespace (bool), unify_text (bool),
      numeric_columns (list), date_columns (list),
      remove_duplicates (bool),
      missing_numeric ("leave" | "median" | "mean" | "drop"),
      missing_text    ("leave" | "unknown" | "drop")

    Returns (cleaned_df, log) where log is a list of sentences describing what was done.
    """
    cleaned = df.copy()  # the original stays untouched
    log = []

    # 1. Trim spaces, blank text -> missing
    if options.get("trim_whitespace"):
        for column in [c for c in cleaned.columns if _is_text_column(cleaned[c])]:
            before = cleaned[column]
            after = before.map(_tidy_text)
            changed = int((~((before == after) | (before.isna() & after.isna()))).sum())
            if changed:
                cleaned[column] = after
                log.append(f"Trimmed spaces / blanks in '{column}' ({changed:,} values changed).")

    # 2. Unify spelling to the most common version
    if options.get("unify_text"):
        for column in [c for c in cleaned.columns if _is_text_column(cleaned[c])]:
            values = cleaned[column].dropna().astype(str)
            if values.empty or values.nunique() > MAX_UNIQUE_FOR_TEXT_CHECK:
                continue
            best_spelling, mapping = {}, {}
            for value in values.value_counts().index:  # most common first
                key = value.strip().lower()
                best_spelling.setdefault(key, value)
                mapping[value] = best_spelling[key]
            changed = sum(1 for v in values if mapping[v] != v)
            if changed:
                cleaned[column] = cleaned[column].map(
                    lambda v: mapping.get(v, v) if isinstance(v, str) else v
                )
                log.append(f"Unified spelling in '{column}' ({changed:,} values changed).")

    # 3. Text -> numbers
    for column in options.get("numeric_columns", []):
        if column in cleaned.columns and _is_text_column(cleaned[column]):
            missing_before = int(cleaned[column].isna().sum())
            cleaned[column] = _text_to_numeric(cleaned[column])
            new_missing = int(cleaned[column].isna().sum()) - missing_before
            note = f" {new_missing:,} unreadable values became missing." if new_missing else ""
            log.append(f"Converted '{column}' to numbers.{note}")

    # 4. Text -> dates
    for column in options.get("date_columns", []):
        if column in cleaned.columns and _is_text_column(cleaned[column]):
            missing_before = int(cleaned[column].isna().sum())
            cleaned[column] = _text_to_dates(cleaned[column])
            new_missing = int(cleaned[column].isna().sum()) - missing_before
            note = f" {new_missing:,} invalid dates became missing." if new_missing else ""
            log.append(f"Converted '{column}' to dates.{note}")

    # 5. Duplicates (done after trimming/unifying so near-duplicates are caught too)
    if options.get("remove_duplicates"):
        rows_before = len(cleaned)
        cleaned = cleaned.drop_duplicates()
        removed = rows_before - len(cleaned)
        log.append(f"Removed {removed:,} duplicate rows.")

    # 6. Missing numeric values
    numeric_columns = cleaned.select_dtypes(include="number").columns.tolist()
    mode = options.get("missing_numeric", "leave")
    if mode in ("median", "mean"):
        for column in numeric_columns:
            missing = int(cleaned[column].isna().sum())
            value = cleaned[column].median() if mode == "median" else cleaned[column].mean()
            if missing and pd.notna(value):
                cleaned[column] = cleaned[column].fillna(value)
                log.append(f"Filled {missing:,} missing values in '{column}' with the {mode} ({value:,.2f}).")
    elif mode == "drop" and numeric_columns:
        rows_before = len(cleaned)
        cleaned = cleaned.dropna(subset=numeric_columns)
        log.append(f"Dropped {rows_before - len(cleaned):,} rows with missing numeric values.")

    # 7. Missing text values
    text_columns = [c for c in cleaned.columns if _is_text_column(cleaned[c])]
    mode = options.get("missing_text", "leave")
    if mode == "unknown":
        for column in text_columns:
            missing = int(cleaned[column].isna().sum())
            if missing:
                cleaned[column] = cleaned[column].fillna("Unknown")
                log.append(f"Filled {missing:,} missing values in '{column}' with 'Unknown'.")
    elif mode == "drop" and text_columns:
        rows_before = len(cleaned)
        cleaned = cleaned.dropna(subset=text_columns)
        log.append(f"Dropped {rows_before - len(cleaned):,} rows with missing text values.")

    if not log:
        log.append("No changes were made (nothing selected, or nothing needed fixing).")

    return cleaned.reset_index(drop=True), log