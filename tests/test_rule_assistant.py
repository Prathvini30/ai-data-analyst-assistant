"""
Tests for the free rule-based assistant.
The key idea: every number in an answer must exist in a table that Pandas calculated.
"""

import math
import re

import pandas as pd
import pytest

from src.rule_assistant import AssistantError, ask_assistant

NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers_in(text):
    found = []
    for match in NUMBER.findall(text):
        try:
            found.append(float(match.replace(",", "")))
        except ValueError:
            continue
    return found


def calculated_numbers(calculations):
    """Every number in every calculated table (absolute values, so '-73.5' matches '73.5')."""
    values = []
    for calc in calculations:
        for column in calc["table"].columns:
            series = calc["table"][column]
            if pd.api.types.is_numeric_dtype(series):
                values += [abs(float(v)) for v in series.dropna()]
    return values


def ungrounded_numbers(result):
    """Numbers in the answer that appear in NO calculated table (small counts and years are ignored)."""
    known = calculated_numbers(result["calculations"])
    missing = []
    for number in numbers_in(result["answer"]):
        if number.is_integer() and (number < 100 or 1900 <= number <= 2100):
            continue
        if not any(math.isclose(number, k, rel_tol=0.005, abs_tol=0.051) for k in known):
            missing.append(number)
    return missing


# ---------- Correct answers ----------
def test_total_revenue(kpi_df):
    result = ask_assistant(kpi_df, "What is the total revenue?")
    assert "143,000" in result["answer"]
    assert result["calculations"][0]["table"]["Value"].iloc[0] == kpi_df["Revenue"].sum()


def test_top_product(kpi_df):
    result = ask_assistant(kpi_df, "Which product generated the highest sales?")
    assert "Laptop" in result["answer"] and "90,000" in result["answer"]


def test_best_region_matches_independent_pandas_calculation(kpi_df):
    result = ask_assistant(kpi_df, "Which region performed best?")
    expected = kpi_df.groupby("Region")["Revenue"].sum().sort_values(ascending=False)
    table = result["calculations"][0]["table"]
    assert table["Region"].tolist() == expected.index.tolist()
    assert table["Sum of Revenue"].tolist() == expected.tolist()
    assert "Bengaluru" in result["answer"] and "112,000" in result["answer"]


def test_lowest_category(kpi_df):
    result = ask_assistant(kpi_df, "Which category has the lowest sales?")
    assert "Accessories" in result["answer"] and "9,000" in result["answer"]


def test_monthly_trend(kpi_df):
    result = ask_assistant(kpi_df, "Show monthly revenue trend")
    for text in ("92,500", "24,500", "26,000", "-73.5%", "+6.1%"):
        assert text in result["answer"]


def test_average_order_value(kpi_df):
    result = ask_assistant(kpi_df, "What is the average revenue per order?")
    assert "23,833.33" in result["answer"]


def test_filter_by_value_and_by_month(kpi_df):
    by_region = ask_assistant(kpi_df, "Total revenue for Bengaluru")
    assert by_region["calculations"][0]["table"]["Value"].iloc[0] == 112000
    by_month = ask_assistant(kpi_df, "Total revenue in February")
    assert by_month["calculations"][0]["table"]["Value"].iloc[0] == 24500


def test_count_unique_customers(kpi_df):
    result = ask_assistant(kpi_df, "How many customers are there?")
    assert result["calculations"][0]["table"]["Value"].iloc[0] == 6


def test_follow_up_question_keeps_context(kpi_df):
    first = ask_assistant(kpi_df, "Which region performed best?")
    second = ask_assistant(kpi_df, "And which was the worst?", first["context"])
    assert "Hubballi" in second["answer"] and "4,500" in second["answer"]


def test_non_sales_dataset(generic_df):
    result = ask_assistant(generic_df, "Which subject has the highest average marks?")
    assert "Maths" in result["answer"] and "74.5" in result["answer"]


# ---------- Grounding: no invented numbers ----------
GROUNDING_QUESTIONS = [
    "What is the total revenue?",
    "Which product generated the highest sales?",
    "Top 3 products by revenue",
    "Which region performed best?",
    "Which category has the lowest sales?",
    "Show monthly revenue trend",
    "What is the average revenue per order?",
    "Total revenue for Bengaluru",
    "Total revenue in February",
    "How many customers are there?",
    "Give me a summary of this dataset",
]


@pytest.mark.parametrize("question", GROUNDING_QUESTIONS)
def test_every_number_in_the_answer_comes_from_a_calculation(kpi_df, question):
    result = ask_assistant(kpi_df, question)
    assert result["calculations"], "an answer with numbers must have calculations behind it"
    assert ungrounded_numbers(result) == []


# ---------- Honesty and error handling ----------
def test_refuses_profit_when_no_cost_column(kpi_df):
    result = ask_assistant(kpi_df, "What is the profit margin?")
    assert "profit" in result["answer"].lower()
    assert result["calculations"] == []


def test_refuses_predictions_and_causes(kpi_df):
    assert "predict" in ask_assistant(kpi_df, "What will sales be next month?")["answer"]
    assert "why" in ask_assistant(kpi_df, "Why did sales drop in February?")["answer"].lower()


def test_gibberish_gets_a_friendly_reply(kpi_df):
    result = ask_assistant(kpi_df, "asdf")
    assert "didn't understand" in result["answer"]
    assert result["calculations"] == []


def test_help_message(kpi_df):
    result = ask_assistant(kpi_df, "help")
    assert "safe calculations" in result["answer"]


def test_invalid_input_raises_friendly_errors(kpi_df):
    with pytest.raises(AssistantError):
        ask_assistant(kpi_df, "   ")
    with pytest.raises(AssistantError):
        ask_assistant(kpi_df, "a" * 1001)
    with pytest.raises(AssistantError):
        ask_assistant(pd.DataFrame(), "What is the total revenue?")