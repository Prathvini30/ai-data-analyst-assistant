"""
analysis_engine.py
The SAFE bridge between the assistant and Pandas.

The assistant never runs code. It only asks for one of three operations, with
arguments (column names, calculation type, filters). This file checks every
argument, then lets Pandas do the real calculation:

  calculate_metric     - one number (total, average, count ...)
  group_and_aggregate  - a number per category (top 5 products ...)
  monthly_trend        - a number per month

Anything not on the allowed lists raises a ValueError with a clear message.
"""

import operator
import warnings

import numpy as np
import pandas as pd

from src.config import (
    MAX_CATEGORY_VALUES_SHOWN,
    MAX_FILTERS,
    MAX_RESULT_ROWS,
    SAMPLE_VALUES_SHOWN,
)
from src.kpi import (
    ALLOWED_AGGREGATIONS,
    aggregate_by_group,
    find_date_columns,
    monthly_trend,
    to_datetime_safe,
)

SCALAR_AGGREGATIONS = ALLOWED_AGGREGATIONS + ("nunique",)
FILTER_OPERATORS = ("==", "!=", ">", ">=", "<", "<=", "in", "contains")

_COMPARE = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}
_AGG_WORDS = {"sum": "Total", "mean": "Average", "median": "Median", "min": "Minimum", "max": "Maximum"}


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------
def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def get_column_kinds(df: pd.DataFrame) -> dict:
    """Label every column as 'number', 'date' or 'text'."""
    date_columns = set(find_date_columns(df))
    kinds = {}
    for column in df.columns:
        if column in date_columns:
            kinds[column] = "date"
        elif _is_numeric(df[column]):
            kinds[column] = "number"
        else:
            kinds[column] = "text"
    return kinds


def _require_column(df: pd.DataFrame, column, what: str = "Column") -> None:
    if not isinstance(column, str) or column not in df.columns:
        raise ValueError(
            f"{what} '{column}' does not exist. Available columns: {', '.join(map(str, df.columns))}."
        )


def _require_numeric(df: pd.DataFrame, kinds: dict, column) -> None:
    _require_column(df, column, "Value column")
    if kinds[column] != "number":
        raise ValueError(
            f"Column '{column}' is {kinds[column]}, not numeric, so it cannot be summed or averaged. "
            "(If it holds numbers stored as text, convert it in Data Quality → Cleaning.)"
        )


def _where_text(descriptions: list) -> str:
    return f" (where {' AND '.join(descriptions)})" if descriptions else ""


def _plain(value: float) -> str:
    return f"{value:,.2f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------
# Dataset description (no raw rows, only a summary of the columns)
# ---------------------------------------------------------------------------
def describe_dataset(df: pd.DataFrame) -> str:
    """Describe the columns of a dataset in plain text."""
    kinds = get_column_kinds(df)
    lines = [f"Rows: {len(df):,}", f"Columns: {len(df.columns)}", "Column details:"]
    for column in df.columns:
        series = df[column]
        kind = kinds[column]
        missing = int(series.isna().sum())

        if kind == "number":
            detail = (
                f"min {_plain(series.min())}, max {_plain(series.max())}"
                if series.notna().any() else "no values"
            )
        elif kind == "date":
            dates = to_datetime_safe(series).dropna()
            detail = f"{dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}" if not dates.empty else "no valid dates"
        else:
            values = series.dropna().astype(str).str.strip()
            unique = values.unique()
            if len(unique) == 0:
                detail = "no values"
            elif len(unique) <= MAX_CATEGORY_VALUES_SHOWN:
                detail = "values: " + ", ".join(sorted(v[:40] for v in unique))
            else:
                examples = ", ".join(v[:40] for v in unique[:SAMPLE_VALUES_SHOWN])
                detail = f"{len(unique):,} distinct values, for example: {examples}"

        lines.append(f"- {column} [{kind}], {missing:,} missing: {detail}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
def _as_list(value) -> list:
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _as_number(value, column) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Filter value for numeric column '{column}' must be a number.")
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        raise ValueError(
            f"Filter value '{value}' is not a number, but column '{column}' is numeric."
        ) from None


def _as_date(value, column) -> pd.Timestamp:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(str(value), errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Filter value '{value}' is not a valid date for column '{column}'. Use YYYY-MM-DD.")
    return parsed.normalize()


def _filter_mask(series: pd.Series, kind: str, column: str, op: str, value) -> pd.Series:
    """Return True/False per row for ONE filter."""
    if kind == "number":
        if op == "contains":
            raise ValueError(f"'contains' only works on text columns, but '{column}' is numeric.")
        valid = series.notna()
        if op == "in":
            return valid & series.isin([_as_number(v, column) for v in _as_list(value)])
        return valid & _COMPARE[op](series, _as_number(value, column))

    if kind == "date":
        if op in ("in", "contains"):
            raise ValueError(f"'{op}' does not work on date column '{column}'. Use ==, !=, >, >=, < or <=.")
        dates = to_datetime_safe(series).dt.normalize()
        return dates.notna() & _COMPARE[op](dates, _as_date(value, column))

    # text column (comparison ignores upper/lower case and spaces at the ends)
    if op in (">", ">=", "<", "<="):
        raise ValueError(f"'{op}' only works on numbers and dates, but '{column}' is text.")
    text = series.astype("string").str.strip().str.lower()
    if op == "==":
        result = text == str(value).strip().lower()
    elif op == "!=":
        result = text != str(value).strip().lower()
    elif op == "in":
        result = text.isin([str(v).strip().lower() for v in _as_list(value)])
    else:  # contains
        result = text.str.contains(str(value).strip().lower(), regex=False)
    return result.fillna(False).astype(bool)


def apply_filters(df: pd.DataFrame, filters, kinds: dict):
    """Validate the filters, then return (filtered_rows, list_of_filter_descriptions)."""
    if filters is None:
        filters = []
    if not isinstance(filters, list):
        raise ValueError("'filters' must be a list.")
    if len(filters) > MAX_FILTERS:
        raise ValueError(f"At most {MAX_FILTERS} filters are allowed.")

    mask = pd.Series(True, index=df.index)
    descriptions = []
    for item in filters:
        if not isinstance(item, dict):
            raise ValueError("Each filter must be an object with 'column', 'op' and 'value'.")
        column, op, value = item.get("column"), item.get("op"), item.get("value")
        _require_column(df, column, "Filter column")
        if op not in FILTER_OPERATORS:
            raise ValueError(f"Unsupported filter operator '{op}'. Allowed: {', '.join(FILTER_OPERATORS)}.")
        if value is None:
            raise ValueError(f"The filter on '{column}' needs a value.")
        mask &= _filter_mask(df[column], kinds[column], column, op, value)
        descriptions.append(f"{column} {op} {value}")
    return df[mask], descriptions


def _filtered_rows(df: pd.DataFrame, args: dict, kinds: dict):
    rows, descriptions = apply_filters(df, args.get("filters"), kinds)
    notes = []
    if descriptions:
        notes.append(f"Filters: {' AND '.join(descriptions)}. {len(rows):,} of {len(df):,} rows matched.")
    if rows.empty:
        raise ValueError(
            "No rows match the filters" + (f" ({' AND '.join(descriptions)})" if descriptions else "")
            + ". Check the exact values in your dataset."
        )
    return rows, descriptions, notes


def _as_top_n(value) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError("top_n must be a whole number.") from None
    return max(1, min(number, MAX_RESULT_ROWS))


# ---------------------------------------------------------------------------
# The three operations
# ---------------------------------------------------------------------------
def _op_metric(df: pd.DataFrame, args: dict, kinds: dict) -> dict:
    agg, column = args.get("agg"), args.get("column")
    if agg not in SCALAR_AGGREGATIONS:
        raise ValueError(f"Unsupported calculation '{agg}'. Allowed: {', '.join(SCALAR_AGGREGATIONS)}.")

    rows, descriptions, notes = _filtered_rows(df, args, kinds)

    if agg == "count":
        value, label = len(rows), "Number of rows"
    elif agg == "nunique":
        _require_column(rows, column)
        value, label = int(rows[column].nunique()), f"Unique values in {column}"
    elif agg in ("min", "max") and column in df.columns and kinds[column] == "date":
        dates = to_datetime_safe(rows[column]).dropna()
        if dates.empty:
            raise ValueError(f"Column '{column}' has no valid dates.")
        value = (dates.min() if agg == "min" else dates.max()).strftime("%Y-%m-%d")
        label = f"{'Earliest' if agg == 'min' else 'Latest'} date in {column}"
    else:
        _require_numeric(rows, kinds, column)
        values = rows[column]
        if not values.notna().any():
            raise ValueError(f"Column '{column}' has no values to calculate on.")
        missing = int(values.isna().sum())
        if missing:
            notes.append(f"{missing:,} missing values in '{column}' were ignored.")
        value = values.agg(agg)
        label = f"{_AGG_WORDS[agg]} of {column}"

    if hasattr(value, "item"):
        value = value.item()
    title = f"{label}{_where_text(descriptions)}"
    table = pd.DataFrame({"Metric": [title], "Value": [value]})
    return {"kind": "scalar", "title": title, "table": table, "notes": notes}


def _op_group(df: pd.DataFrame, args: dict, kinds: dict) -> dict:
    group_col, agg, value_col = args.get("group_column"), args.get("agg"), args.get("value_column")
    if agg not in ALLOWED_AGGREGATIONS:
        raise ValueError(f"Unsupported calculation '{agg}'. Allowed: {', '.join(ALLOWED_AGGREGATIONS)}.")
    _require_column(df, group_col, "Group column")
    if agg != "count":
        if not value_col:
            raise ValueError("'value_column' is required unless agg is 'count'.")
        _require_numeric(df, kinds, value_col)
    top_n = _as_top_n(args.get("top_n", 10))
    order = args.get("order", "highest")
    if order not in ("highest", "lowest"):
        raise ValueError("'order' must be 'highest' or 'lowest'.")

    rows, descriptions, notes = _filtered_rows(df, args, kinds)

    everything = aggregate_by_group(rows, group_col, value_col, agg, top_n=None, ascending=(order == "lowest"))
    if everything.empty:
        raise ValueError(f"Column '{group_col}' has no usable values.")
    label = everything.columns[1]

    # Share of total is calculated here by Python
    if agg in ("sum", "count"):
        total = everything[label].sum()
        if total > 0 and (everything[label] >= 0).all():
            everything["Share of total (%)"] = (everything[label] / total * 100).round(1)

    table = everything.head(top_n).reset_index(drop=True)
    if len(table) < len(everything):
        notes.append(f"Showing {len(table)} of {len(everything):,} groups.")
    missing_group = int(rows[group_col].isna().sum())
    if missing_group:
        notes.append(f"{missing_group:,} rows have no value in '{group_col}' and were left out.")

    title = f"{label} by {group_col}{_where_text(descriptions)}"
    return {"kind": "group", "title": title, "table": table, "notes": notes}


def _op_trend(df: pd.DataFrame, args: dict, kinds: dict) -> dict:
    date_col, agg, value_col = args.get("date_column"), args.get("agg"), args.get("value_column")
    if agg not in ALLOWED_AGGREGATIONS:
        raise ValueError(f"Unsupported calculation '{agg}'. Allowed: {', '.join(ALLOWED_AGGREGATIONS)}.")
    _require_column(df, date_col, "Date column")
    if kinds[date_col] != "date":
        date_columns = [c for c, k in kinds.items() if k == "date"]
        raise ValueError(
            f"Column '{date_col}' is not a date column. Date columns: {', '.join(date_columns) or 'none'}."
        )
    if agg != "count":
        if not value_col:
            raise ValueError("'value_column' is required unless agg is 'count'.")
        _require_numeric(df, kinds, value_col)

    rows, descriptions, notes = _filtered_rows(df, args, kinds)
    table, invalid = monthly_trend(rows, date_col, value_col, agg)
    if table.empty:
        raise ValueError(f"Column '{date_col}' has no valid dates, so no monthly trend can be calculated.")
    if invalid:
        notes.append(f"{invalid:,} rows with invalid dates were left out.")

    first, last = table["Month"].iloc[0], table["Month"].iloc[-1]
    months_expected = (last.year - first.year) * 12 + last.month - first.month + 1
    if months_expected > len(table):
        notes.append("Some months have no data and are not listed; changes are versus the previous month listed.")

    label = table.columns[1]
    change = table[label].pct_change(fill_method=None) * 100
    table["Change vs previous month (%)"] = change.replace([np.inf, -np.inf], np.nan).round(1)

    title = f"{label} per month{_where_text(descriptions)}"
    return {"kind": "trend", "title": title, "table": table, "notes": notes}


OPERATIONS = {
    "calculate_metric": _op_metric,
    "group_and_aggregate": _op_group,
    "monthly_trend": _op_trend,
}


def run_operation(df: pd.DataFrame, name: str, args) -> dict:
    """
    The ONLY entry point the assistant can use.
    Returns {"kind", "title", "table", "notes", "tool", "inputs"} or raises ValueError.
    """
    if name not in OPERATIONS:
        raise ValueError(f"Unknown operation '{name}'. Allowed: {', '.join(OPERATIONS)}.")
    if not isinstance(args, dict):
        raise ValueError("The operation arguments must be an object.")
    result = OPERATIONS[name](df, args, get_column_kinds(df))
    result["tool"] = name
    result["inputs"] = args
    return result


def result_to_text(result: dict) -> str:
    """Turn a result into compact text."""
    table = result["table"].copy()
    for column in table.columns:
        if pd.api.types.is_datetime64_any_dtype(table[column]):
            table[column] = table[column].dt.strftime("%Y-%m")
    lines = [f"Result: {result['title']}"]
    lines += [f"Note: {note}" for note in result["notes"]]
    lines.append(table.round(2).to_csv(index=False).strip())
    return "\n".join(lines)