"""Tests for the business insights page content and the export files."""

import io

import pandas as pd
import pytest
from openpyxl import load_workbook

from src.insights import build_insights
from src.report_export import (
    build_excel,
    build_text_report,
    df_to_csv_bytes,
    make_export_safe,
    safe_stem,
)
from src.visualizations import sample_for_scatter


# ---------- Insights ----------
def test_insight_facts_match_hand_calculation(kpi_df):
    facts = build_insights(kpi_df)["facts"]
    assert "Total Revenue: 143,000." in facts
    assert any("Laptop (90,000" in text for text in facts)
    assert any("Highest Revenue by Region: Bengaluru (112,000" in text for text in facts)
    assert any("Lowest Revenue by Region: Hubballi (4,500)" in text for text in facts)
    assert any("Highest month: Jan 2026 (92,500)" in text for text in facts)


def test_observations_flag_concentration(kpi_df):
    observations = build_insights(kpi_df)["observations"]
    assert any("Bengaluru accounts for 78.3%" in text for text in observations)
    assert any("Laptop accounts for 62.9%" in text for text in observations)


def test_clean_data_has_nothing_needing_attention(kpi_df):
    assert build_insights(kpi_df)["attention"] == ["The automatic checks found no major problems."]


def test_messy_data_lists_attention_items(messy_df):
    attention = " ".join(build_insights(messy_df)["attention"])
    assert "duplicate" in attention
    assert "Revenue" in attention
    assert "Quantity" in attention


# ---------- Exports ----------
def test_safe_stem():
    assert safe_stem("My Sales (2026).csv") == "My_Sales_2026"
    assert safe_stem("???.csv") == "dataset"


def test_formula_like_text_is_neutralised():
    df = pd.DataFrame({"Customer": ["=1+1", "+5", "@cmd", "Normal"], "Amount": [1, 2, 3, 4]})
    safe = make_export_safe(df)
    assert safe["Customer"].tolist() == ["'=1+1", "'+5", "'@cmd", "Normal"]
    assert df["Customer"].iloc[0] == "=1+1"          # the original is untouched
    assert "'=1+1" in df_to_csv_bytes(df).decode("utf-8-sig")


def test_csv_round_trip(kpi_df):
    data = df_to_csv_bytes(kpi_df)
    loaded = pd.read_csv(io.BytesIO(data))
    assert loaded.shape == kpi_df.shape
    assert loaded["Revenue"].sum() == 143000


def test_excel_has_bold_headers_and_clean_months(kpi_df):
    trend = pd.DataFrame({"Month": pd.to_datetime(["2026-01-01", "2026-02-01"]), "Sum of Revenue": [1, 2]})
    workbook = load_workbook(io.BytesIO(build_excel({"Data": kpi_df, "Trend": trend})))
    assert workbook.sheetnames == ["Data", "Trend"]
    assert workbook["Data"]["A1"].font.bold is True
    assert workbook["Trend"]["A2"].value == "2026-01"
    assert workbook["Data"].max_row == 7             # header + 6 rows


def test_excel_sheet_names_are_cleaned_and_unique(kpi_df):
    workbook = load_workbook(io.BytesIO(build_excel({"A/B:C": kpi_df, "Data": kpi_df, "data": kpi_df})))
    assert workbook.sheetnames == ["ABC", "Data", "data_2"]


def test_excel_export_needs_something_to_export():
    with pytest.raises(ValueError):
        build_excel({})


def test_text_report_keeps_facts_and_observations_apart(kpi_df):
    report = build_text_report("sales.csv", "original", build_insights(kpi_df)).decode("utf-8")
    for heading in ("## Key metrics", "## Calculated facts", "## Rule-based observations", "## Areas needing attention"):
        assert heading in report
    assert "143,000" in report


# ---------- Large dataset safeguard ----------
def test_scatter_plots_are_sampled_for_big_data():
    big = pd.DataFrame({"x": range(6000), "y": range(6000)})
    sampled, was_sampled = sample_for_scatter(big)
    assert was_sampled and len(sampled) == 5000
    small, was_sampled = sample_for_scatter(big.head(100))
    assert not was_sampled and len(small) == 100