"""Tests for the safe calculation engine: correct maths, filters, and rejection of bad requests."""

import pytest

from src.analysis_engine import describe_dataset, run_operation


def value_of(result):
    return result["table"]["Value"].iloc[0]


# ---------- Correct calculations ----------
def test_metric_sum_count_unique(kpi_df):
    assert value_of(run_operation(kpi_df, "calculate_metric", {"agg": "sum", "column": "Revenue"})) == 143000
    assert value_of(run_operation(kpi_df, "calculate_metric", {"agg": "count"})) == 6
    assert value_of(run_operation(kpi_df, "calculate_metric", {"agg": "nunique", "column": "Customer"})) == 6
    assert value_of(run_operation(kpi_df, "calculate_metric", {"agg": "mean", "column": "Revenue"})) == pytest.approx(143000 / 6)


def test_metric_date_range(kpi_df):
    assert value_of(run_operation(kpi_df, "calculate_metric", {"agg": "min", "column": "Order_Date"})) == "2026-01-05"
    assert value_of(run_operation(kpi_df, "calculate_metric", {"agg": "max", "column": "Order_Date"})) == "2026-03-21"


def test_group_with_share_of_total(kpi_df):
    result = run_operation(kpi_df, "group_and_aggregate",
                           {"group_column": "Region", "agg": "sum", "value_column": "Revenue", "top_n": 3})
    table = result["table"]
    assert table["Region"].tolist() == ["Bengaluru", "Mysuru", "Hubballi"]
    assert table["Sum of Revenue"].tolist() == [112000, 26500, 4500]
    assert table["Share of total (%)"].iloc[0] == pytest.approx(78.3)


def test_group_lowest_first(kpi_df):
    result = run_operation(kpi_df, "group_and_aggregate",
                           {"group_column": "Region", "agg": "sum", "value_column": "Revenue",
                            "top_n": 1, "order": "lowest"})
    assert result["table"]["Region"].iloc[0] == "Hubballi"
    assert result["table"]["Sum of Revenue"].iloc[0] == 4500


def test_monthly_trend_with_changes(kpi_df):
    result = run_operation(kpi_df, "monthly_trend",
                           {"date_column": "Order_Date", "agg": "sum", "value_column": "Revenue"})
    table = result["table"]
    assert table["Sum of Revenue"].tolist() == [92500, 24500, 26000]
    changes = table["Change vs previous month (%)"]
    assert changes.iloc[1] == pytest.approx(-73.5)
    assert changes.iloc[2] == pytest.approx(6.1)


# ---------- Filters ----------
def test_text_filter_ignores_case(kpi_df):
    args = {"agg": "sum", "column": "Revenue", "filters": [{"column": "Region", "op": "==", "value": "bengaluru"}]}
    assert value_of(run_operation(kpi_df, "calculate_metric", args)) == 112000


def test_in_filter(kpi_df):
    args = {"agg": "sum", "column": "Revenue",
            "filters": [{"column": "Region", "op": "in", "value": ["Mysuru", "Hubballi"]}]}
    assert value_of(run_operation(kpi_df, "calculate_metric", args)) == 31000


def test_number_filter(kpi_df):
    args = {"agg": "count", "filters": [{"column": "Revenue", "op": ">=", "value": 20000}]}
    assert value_of(run_operation(kpi_df, "calculate_metric", args)) == 3


def test_date_filter_for_one_month(kpi_df):
    args = {"agg": "sum", "column": "Revenue", "filters": [
        {"column": "Order_Date", "op": ">=", "value": "2026-02-01"},
        {"column": "Order_Date", "op": "<=", "value": "2026-02-28"},
    ]}
    assert value_of(run_operation(kpi_df, "calculate_metric", args)) == 24500


def test_filters_work_inside_group_and_trend(kpi_df):
    where_bengaluru = [{"column": "Region", "op": "==", "value": "Bengaluru"}]
    trend = run_operation(kpi_df, "monthly_trend", {
        "date_column": "Order_Date", "agg": "sum", "value_column": "Revenue", "filters": where_bengaluru})
    assert trend["table"]["Sum of Revenue"].tolist() == [90000, 20000, 2000]
    group = run_operation(kpi_df, "group_and_aggregate", {
        "group_column": "Product", "agg": "sum", "value_column": "Revenue", "filters": where_bengaluru})
    assert group["table"]["Product"].iloc[0] == "Laptop"


# ---------- Safety: bad or malicious requests are rejected ----------
BAD_REQUESTS = [
    ("exec", {"agg": "sum", "column": "Revenue"}),                                    # unknown operation
    ("__import__('os').system('dir')", {}),
    ("calculate_metric", {"agg": "sum", "column": "Nope"}),                           # unknown column
    ("calculate_metric", {"agg": "sum", "column": "__import__('os')"}),
    ("calculate_metric", {"agg": "eval", "column": "Revenue"}),                       # unknown calculation
    ("calculate_metric", {"agg": "sum", "column": "Customer"}),                       # text column
    ("calculate_metric", {"agg": "sum", "column": "Revenue",
                          "filters": [{"column": "Region", "op": "like", "value": "x"}]}),   # bad operator
    ("calculate_metric", {"agg": "sum", "column": "Revenue",
                          "filters": [{"column": "Nope", "op": "==", "value": "x"}]}),       # bad filter column
    ("calculate_metric", {"agg": "sum", "column": "Revenue",
                          "filters": [{"column": "Revenue", "op": "==", "value": "abc"}]}),  # text for a number
    ("calculate_metric", {"agg": "sum", "column": "Revenue",
                          "filters": [{"column": "Region", "op": "==", "value": "Atlantis"}]}),  # matches nothing
    ("calculate_metric", {"agg": "sum", "column": "Revenue",
                          "filters": [{"column": "Region", "op": "!=", "value": "x"}] * 6}),     # too many filters
    ("group_and_aggregate", {"group_column": "Region", "agg": "sum"}),                # value column missing
    ("group_and_aggregate", {"group_column": "Nope", "agg": "count"}),
    ("monthly_trend", {"date_column": "Region", "agg": "count"}),                     # not a date column
]


@pytest.mark.parametrize("name, args", BAD_REQUESTS)
def test_bad_requests_are_rejected(kpi_df, name, args):
    with pytest.raises(ValueError):
        run_operation(kpi_df, name, args)


def test_arguments_must_be_an_object(kpi_df):
    with pytest.raises(ValueError):
        run_operation(kpi_df, "calculate_metric", "sum of revenue")


def test_dataset_description_lists_columns(kpi_df):
    text = describe_dataset(kpi_df)
    assert "Rows: 6" in text
    assert "Revenue [number]" in text
    assert "Order_Date [date]" in text
    assert "Region [text]" in text