"""Tests for data-quality detection and the cleaning options."""

import pandas as pd
import pytest

from src.data_cleaning import apply_cleaning, detect_issues
from src.data_exploration import get_overview

DEFAULT_OPTIONS = {
    "remove_duplicates": True,
    "trim_whitespace": True,
    "unify_text": True,
    "numeric_columns": [],
    "date_columns": ["Order_Date"],
    "missing_numeric": "leave",
    "missing_text": "leave",
}


def issue_keys(df):
    return {(issue["category"], issue["column"]) for issue in detect_issues(df)}


# ---------- Detection ----------
def test_overview_counts(messy_df):
    overview = get_overview(messy_df)
    assert overview["rows"] == 12
    assert overview["columns"] == 10
    assert overview["missing_cells"] == 1
    assert overview["duplicate_rows"] == 1


def test_detects_missing_values(messy_df):
    assert ("Missing values", "Revenue") in issue_keys(messy_df)


def test_detects_duplicate_rows(messy_df):
    duplicates = [i for i in detect_issues(messy_df) if i["category"] == "Duplicate rows"]
    assert len(duplicates) == 1
    assert duplicates[0]["count"] == 1


def test_detects_dates_stored_as_text(messy_df):
    assert ("Dates stored as text", "Order_Date") in issue_keys(messy_df)


def test_detects_extra_spaces_and_inconsistent_spelling(messy_df):
    keys = issue_keys(messy_df)
    assert ("Extra spaces / blank text", "Region") in keys
    assert ("Inconsistent spelling", "Region") in keys


def test_detects_negative_values(messy_df):
    keys = issue_keys(messy_df)
    assert ("Potentially invalid values", "Quantity") in keys
    assert ("Potentially invalid values", "Revenue") in keys


def test_detects_outliers(messy_df):
    assert ("Outliers", "Quantity") in issue_keys(messy_df)


def test_detects_numbers_stored_as_text():
    df = pd.DataFrame({"Price": ["100", "200", "300", "400", "500", "600", "700", "800", "900", "1,000"]})
    assert ("Numbers stored as text", "Price") in issue_keys(df)


def test_outlier_detection_on_simple_numbers():
    df = pd.DataFrame({"Score": list(range(10, 30)) + [1000]})
    outliers = [i for i in detect_issues(df) if i["category"] == "Outliers"]
    assert len(outliers) == 1
    assert outliers[0]["count"] == 1


def test_clean_dataset_has_no_missing_or_duplicate_issues(kpi_df):
    categories = {issue["category"] for issue in detect_issues(kpi_df)}
    assert "Missing values" not in categories
    assert "Duplicate rows" not in categories


def test_empty_dataset_returns_no_issues():
    assert detect_issues(pd.DataFrame()) == []


# ---------- Cleaning ----------
def test_cleaning_never_changes_the_original(messy_df):
    before = messy_df.copy()
    apply_cleaning(messy_df, DEFAULT_OPTIONS)
    pd.testing.assert_frame_equal(messy_df, before)


def test_default_cleaning_result(messy_df):
    cleaned, log = apply_cleaning(messy_df, DEFAULT_OPTIONS)
    assert len(cleaned) == 11                                              # one duplicate removed
    assert set(cleaned["Region"].unique()) == {"Bengaluru", "Mysuru", "Hubballi"}
    assert pd.api.types.is_datetime64_any_dtype(cleaned["Order_Date"])
    assert cleaned["Order_Date"].isna().sum() == 1                         # "not a date" became missing
    assert cleaned["Revenue"].sum() == 660000                              # 672000 - 12000 (duplicate)
    assert cleaned["Quantity"].sum() == 1019                               # 1020 - 1 (duplicate)
    assert len(log) >= 3


def test_no_options_means_no_changes(messy_df):
    cleaned, log = apply_cleaning(messy_df, {})
    pd.testing.assert_frame_equal(cleaned, messy_df)
    assert "No changes" in log[0]


def test_fill_missing_numbers_with_median(messy_df):
    options = {"remove_duplicates": True, "missing_numeric": "median"}
    cleaned, _ = apply_cleaning(messy_df, options)
    assert cleaned["Revenue"].isna().sum() == 0
    filled = cleaned.loc[cleaned["Order_ID"] == "ORD009", "Revenue"].iloc[0]
    assert filled == 16000     # median of the 10 known revenues after removing the duplicate


def test_drop_rows_with_missing_numbers(messy_df):
    options = {"remove_duplicates": True, "missing_numeric": "drop"}
    cleaned, _ = apply_cleaning(messy_df, options)
    assert len(cleaned) == 10  # 12 - 1 duplicate - 1 row with missing Revenue


def test_text_to_number_conversion():
    df = pd.DataFrame({"Amount": ["1,200", "₹300", "abc", None]})
    cleaned, log = apply_cleaning(df, {"numeric_columns": ["Amount"]})
    assert cleaned["Amount"].sum() == 1500
    assert cleaned["Amount"].isna().sum() == 2   # "abc" and the original blank
    assert "unreadable" in log[0]