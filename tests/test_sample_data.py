"""End-to-end tests on the generated sample dataset (data/sample_sales.csv)."""

from pathlib import Path

import pandas as pd
import pytest

from src.analysis_engine import run_operation
from src.data_cleaning import apply_cleaning, detect_issues

SAMPLE_FILE = Path(__file__).resolve().parent.parent / "data" / "sample_sales.csv"

pytestmark = pytest.mark.skipif(
    not SAMPLE_FILE.exists(), reason="Run  python scripts\\generate_sample_data.py  first"
)


@pytest.fixture(scope="module")
def sample():
    return pd.read_csv(SAMPLE_FILE)


def test_sample_size_and_columns(sample):
    assert 5000 <= len(sample) <= 10000
    assert list(sample.columns) == [
        "Order_ID", "Order_Date", "Customer", "Product", "Category",
        "Quantity", "Unit_Price", "Revenue", "Region", "Payment_Method",
    ]


def test_sample_contains_planted_problems(sample):
    assert sample.duplicated().sum() == 40
    assert sample["Revenue"].isna().sum() >= 60
    assert (sample["Order_Date"] == "not available").sum() == 8
    assert (sample["Quantity"] < 0).sum() == 5


def test_revenue_equals_quantity_times_price_where_present(sample):
    known = sample.dropna(subset=["Revenue"])
    assert (known["Revenue"] == known["Quantity"] * known["Unit_Price"]).all()


def test_quality_checks_find_the_planted_problems(sample):
    found = {(i["category"], i["column"]) for i in detect_issues(sample)}
    assert ("Missing values", "Revenue") in found
    assert ("Duplicate rows", "(whole row)") in found
    assert ("Dates stored as text", "Order_Date") in found
    assert ("Inconsistent spelling", "Region") in found
    assert ("Extra spaces / blank text", "Region") in found
    assert ("Potentially invalid values", "Quantity") in found


def test_engine_total_matches_pandas(sample):
    result = run_operation(sample, "calculate_metric", {"agg": "sum", "column": "Revenue"})
    assert result["table"]["Value"].iloc[0] == pytest.approx(sample["Revenue"].sum())


def test_cleaning_removes_exactly_the_duplicates(sample):
    options = {"remove_duplicates": True, "trim_whitespace": True, "unify_text": True,
               "date_columns": ["Order_Date"]}
    cleaned, _ = apply_cleaning(sample, options)
    assert len(cleaned) == len(sample) - 40
    assert cleaned["Order_Date"].isna().sum() >= 1
    regions = set(cleaned["Region"].dropna().unique())
    assert regions == {"Bengaluru", "Mysuru", "Hubballi", "Belagavi", "Mangaluru", "Dharwad"}