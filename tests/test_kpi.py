"""Tests for column detection, KPI cards, the safe group engine and monthly trends."""

import pandas as pd
import pytest

from src.kpi import aggregate_by_group, calculate_kpis, detect_roles, monthly_trend


def card(cards, label):
    """Find one KPI card by its label."""
    return next(c for c in cards if c["label"] == label)


def test_detect_roles(kpi_df):
    roles = detect_roles(kpi_df)
    assert roles["order_id"] == "Order_ID"
    assert roles["date"] == "Order_Date"
    assert roles["revenue"] == "Revenue"
    assert roles["quantity"] == "Quantity"
    assert roles["unit_price"] == "Unit_Price"
    assert roles["product"] == "Product"
    assert roles["category"] == "Category"
    assert roles["region"] == "Region"
    assert roles["customer"] == "Customer"


def test_kpi_cards_match_hand_calculation(kpi_df):
    cards, skipped = calculate_kpis(kpi_df)
    assert skipped == []
    assert card(cards, "Total records")["value"] == "6"
    assert card(cards, "Total revenue")["value"] == "143,000"
    assert card(cards, "Total orders")["value"] == "6"
    assert card(cards, "Average order value")["value"] == "23,833"   # 143000 / 6
    assert card(cards, "Total quantity sold")["value"] == "17"
    assert card(cards, "Unique customers")["value"] == "6"
    assert card(cards, "Date range")["value"] == "05 Jan 2026 – 21 Mar 2026"
    assert card(cards, "Top product")["value"] == "Laptop"
    assert card(cards, "Top category")["value"] == "Electronics"
    assert card(cards, "Best region")["value"] == "Bengaluru"


def test_missing_values_are_ignored_and_reported(messy_df):
    cards, _ = calculate_kpis(messy_df)
    assert card(cards, "Total revenue")["value"] == "672,000"
    assert "missing" in card(cards, "Total revenue")["detail"]
    assert card(cards, "Total orders")["value"] == "11"          # unique Order_ID values
    assert card(cards, "Average order value")["value"] == "61,091"  # 672000 / 11


def test_missing_revenue_column_is_explained(kpi_df):
    cards, skipped = calculate_kpis(kpi_df.drop(columns="Revenue"))
    labels = [c["label"] for c in cards]
    assert "Total revenue" not in labels
    assert any(message.startswith("Total revenue") for message in skipped)
    assert any(message.startswith("Top product") for message in skipped)


def test_non_sales_dataset_gets_generic_kpis(generic_df):
    cards, skipped = calculate_kpis(generic_df)
    assert card(cards, "Total Marks")["value"] == "295"
    assert card(cards, "Average Marks")["value"] == "73.75"
    assert card(cards, "Total Hours_Studied")["value"] == "32"
    assert card(cards, "Average Hours_Studied")["value"] == "8"
    assert any("revenue" in message.lower() for message in skipped)


def test_aggregate_by_region(kpi_df):
    result = aggregate_by_group(kpi_df, "Region", "Revenue", "sum", top_n=None)
    assert result["Region"].tolist() == ["Bengaluru", "Mysuru", "Hubballi"]
    assert result["Sum of Revenue"].tolist() == [112000, 26500, 4500]


def test_aggregate_top_n_and_order(kpi_df):
    top = aggregate_by_group(kpi_df, "Product", "Revenue", "sum", top_n=2)
    assert top["Product"].tolist() == ["Laptop", "Monitor"]
    lowest = aggregate_by_group(kpi_df, "Category", "Revenue", "sum", top_n=1, ascending=True)
    assert lowest.iloc[0, 0] == "Accessories"
    assert lowest.iloc[0, 1] == 9000


def test_aggregate_count_and_mean(kpi_df):
    counts = aggregate_by_group(kpi_df, "Region", agg="count", top_n=None)
    assert counts.set_index("Region")["Number of rows"].to_dict() == {"Bengaluru": 3, "Mysuru": 2, "Hubballi": 1}
    means = aggregate_by_group(kpi_df, "Category", "Revenue", "mean", top_n=None).set_index("Category")
    assert means.loc["Electronics"].iloc[0] == pytest.approx(134000 / 3)


@pytest.mark.parametrize(
    "group_col, value_col, agg",
    [
        ("Region", "Revenue", "eval"),            # calculation not on the allowed list
        ("Region", "Revenue", "__import__('os')"),
        ("Nope", "Revenue", "sum"),               # group column does not exist
        ("Region", "Nope", "sum"),                # value column does not exist
        ("Region", "Customer", "sum"),            # value column is text, not numeric
        ("Region", "Region", "sum"),              # same column twice
    ],
)
def test_aggregate_rejects_invalid_requests(kpi_df, group_col, value_col, agg):
    with pytest.raises(ValueError):
        aggregate_by_group(kpi_df, group_col, value_col, agg)


def test_monthly_trend(kpi_df):
    result, invalid = monthly_trend(kpi_df, "Order_Date", "Revenue", "sum")
    assert invalid == 0
    assert result["Month"].dt.strftime("%Y-%m").tolist() == ["2026-01", "2026-02", "2026-03"]
    assert result["Sum of Revenue"].tolist() == [92500, 24500, 26000]


def test_monthly_trend_leaves_out_invalid_dates(messy_df):
    result, invalid = monthly_trend(messy_df, "Order_Date", "Revenue", "sum")
    assert invalid == 1                                   # the "not a date" row
    assert len(result) == 1                               # everything else is January
    assert result["Sum of Revenue"].iloc[0] == 627000     # 672000 - 45000 (the invalid-date row)


def test_monthly_trend_rejects_bad_columns(kpi_df):
    with pytest.raises(ValueError):
        monthly_trend(kpi_df, "Nope", "Revenue", "sum")
    with pytest.raises(ValueError):
        monthly_trend(kpi_df, "Order_Date", "Customer", "sum")