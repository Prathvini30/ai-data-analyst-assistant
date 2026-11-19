"""
insights.py
Builds the Business Insights page content.

Three kinds of statements, kept separate on purpose:
  facts        - numbers calculated by Pandas through the safe analysis engine
  observations - simple rule-based patterns (a share above a threshold, a big change).
                 They describe the numbers only: no causes, no predictions.
  attention    - data problems that could distort the numbers
"""

import pandas as pd

from src.analysis_engine import get_column_kinds, run_operation
from src.config import CHANGE_ALERT_PCT, CONCENTRATION_PCT, MISSING_ALERT_PCT
from src.data_cleaning import detect_issues
from src.kpi import calculate_kpis, detect_roles

SHARE_COL = "Share of total (%)"
CHANGE_COL = "Change vs previous month (%)"


def _fmt(value) -> str:
    """143000.0 -> '143,000'; 23833.333 -> '23,833.33'."""
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return "—"
    return f"{float(value):,.2f}".rstrip("0").rstrip(".")


def _scalar(result):
    return result["table"]["Value"].iloc[0]


def _try(df, name, args):
    """Run a safe operation; return None instead of failing."""
    try:
        return run_operation(df, name, args)
    except ValueError:
        return None


def _group_section(df, role, column, value_col, facts, observations, groups) -> None:
    """Highest and lowest group for one category column (product, category or region)."""
    top = _try(df, "group_and_aggregate",
               {"group_column": column, "agg": "sum", "value_column": value_col,
                "top_n": 50, "order": "highest"})
    if top is None:
        return
    table = top["table"]
    label = table.columns[1]
    groups.append({"role": role, "column": column, "table": table, "label": label})

    first = table.iloc[0]
    share = first[SHARE_COL] if SHARE_COL in table.columns else None
    text = f"Highest {value_col} by {column}: {first[column]} ({_fmt(first[label])}"
    if share is not None and pd.notna(share):
        text += f", {share:.1f}% of the total"
    facts.append(text + ").")

    low = _try(df, "group_and_aggregate",
               {"group_column": column, "agg": "sum", "value_column": value_col,
                "top_n": 1, "order": "lowest"})
    if low is not None:
        row = low["table"].iloc[0]
        if row[column] != first[column]:
            facts.append(f"Lowest {value_col} by {column}: {row[column]} ({_fmt(row.iloc[1])}).")

    group_count = int(df[column].nunique())
    if share is not None and pd.notna(share) and share >= CONCENTRATION_PCT and group_count >= 3:
        observations.append(
            f"{first[column]} accounts for {share:.1f}% of total {value_col} across {group_count} "
            f"{column} values, so the overall result depends heavily on it."
        )


def _trend_section(df, date_col, value_col, facts, observations, attention):
    """Monthly pattern. Returns the trend table info, or None."""
    result = _try(df, "monthly_trend",
                  {"date_column": date_col, "agg": "sum", "value_column": value_col})
    if result is None:
        return None
    table = result["table"]
    label = table.columns[1]
    for note in result["notes"]:
        attention.append(f"Monthly trend: {note}")

    first_month, last_month = table["Month"].iloc[0], table["Month"].iloc[-1]
    facts.append(f"Monthly {value_col} covers {len(table)} month(s), "
                 f"from {first_month:%b %Y} to {last_month:%b %Y}.")

    if len(table) > 1 and table[label].notna().any():
        high, low = table.loc[table[label].idxmax()], table.loc[table[label].idxmin()]
        facts.append(f"Highest month: {high['Month']:%b %Y} ({_fmt(high[label])}). "
                     f"Lowest month: {low['Month']:%b %Y} ({_fmt(low[label])}).")

    if len(table) >= 2:
        latest = table.iloc[-1]
        change = latest[CHANGE_COL]
        if pd.notna(change):
            facts.append(f"The latest month ({latest['Month']:%b %Y}) is {change:+.1f}% "
                         "versus the previous month listed.")
            if abs(change) >= CHANGE_ALERT_PCT:
                direction = "lower" if change < 0 else "higher"
                observations.append(
                    f"{value_col} in {latest['Month']:%b %Y} is {abs(change):.1f}% {direction} than in the "
                    "previous month. If the data ends part-way through a month, that month may be "
                    "incomplete, so check this before drawing conclusions."
                )
        changes = table[CHANGE_COL].dropna()
        if not changes.empty:
            worst, best = table.loc[changes.idxmin()], table.loc[changes.idxmax()]
            if worst[CHANGE_COL] <= -CHANGE_ALERT_PCT:
                facts.append(f"Largest month-over-month drop: {worst['Month']:%b %Y} "
                             f"({worst[CHANGE_COL]:+.1f}%).")
            if best[CHANGE_COL] >= CHANGE_ALERT_PCT:
                facts.append(f"Largest month-over-month rise: {best['Month']:%b %Y} "
                             f"({best[CHANGE_COL]:+.1f}%).")
    return {"column": value_col, "table": table, "label": label}


def build_insights(df: pd.DataFrame) -> dict:
    """
    Returns a dictionary with:
      cards, skipped      - KPI cards (from kpi.py)
      facts               - calculated statements
      observations        - rule-based patterns
      attention           - things that may distort the numbers
      issues              - full data-quality issue list
      groups, trend       - tables used for charts and exports
    """
    kinds, roles = get_column_kinds(df), detect_roles(df)
    cards, skipped = calculate_kpis(df, roles)
    facts, observations, attention = [], [], []
    groups, trend = [], None

    facts.append(f"The dataset has {len(df):,} rows and {len(df.columns)} columns.")

    revenue = roles["revenue"]
    if revenue and kinds.get(revenue) == "number":
        total = _try(df, "calculate_metric", {"agg": "sum", "column": revenue})
        if total is not None:
            facts.append(f"Total {revenue}: {_fmt(_scalar(total))}.")
        for role in ("product", "category", "region"):
            column = roles[role]
            if column and kinds.get(column) == "text":
                _group_section(df, role, column, revenue, facts, observations, groups)
        date_col = roles["date"]
        if date_col and kinds.get(date_col) == "date":
            trend = _trend_section(df, date_col, revenue, facts, observations, attention)
    else:
        numeric = [c for c, k in kinds.items() if k == "number"]
        for column in numeric[:3]:
            total = _try(df, "calculate_metric", {"agg": "sum", "column": column})
            mean = _try(df, "calculate_metric", {"agg": "mean", "column": column})
            if total is not None and mean is not None:
                facts.append(f"{column}: total {_fmt(_scalar(total))}, average {_fmt(_scalar(mean))}.")
        if not numeric:
            facts.append("This dataset has no numeric columns, so no totals can be calculated.")

    if not observations:
        observations.append(
            f"No pattern crossed the alert thresholds (a group above {CONCENTRATION_PCT}% of the total, "
            f"or a monthly change of {CHANGE_ALERT_PCT}% or more)."
        )

    # ---- Areas needing attention ----
    issues = detect_issues(df)
    duplicates = int(df.duplicated().sum())
    if duplicates:
        attention.append(f"{duplicates:,} rows are exact duplicates, so totals may be inflated. "
                         "Remove them in Data Quality → Cleaning.")
    for column in df.columns:
        missing = int(df[column].isna().sum())
        percent = missing / len(df) * 100 if len(df) else 0
        if percent >= MISSING_ALERT_PCT:
            attention.append(f"'{column}' has {missing:,} empty values ({percent:.1f}%). "
                             "Calculations on this column ignore those rows.")
    for issue in issues:
        if issue["severity"] == "High" and issue["category"] not in ("Duplicate rows", "Missing values"):
            attention.append(f"{issue['category']} in '{issue['column']}': {issue['explanation']}")
    if skipped:
        attention.append(f"{len(skipped)} KPI(s) could not be calculated because the needed columns "
                         "were not found (see the Dashboard page).")
    if not attention:
        attention.append("The automatic checks found no major problems.")

    return {
        "cards": cards, "skipped": skipped,
        "facts": facts, "observations": observations, "attention": attention,
        "issues": issues, "groups": groups, "trend": trend,
    }