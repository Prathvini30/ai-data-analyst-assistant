"""
kpi.py
Everything that CALCULATES numbers from the dataset:

  1. detect_roles()        - guess what each column means (revenue, product, region ...)
  2. calculate_kpis()      - build KPI cards, only for metrics that can be calculated reliably
  3. aggregate_by_group()  - a SAFE calculation engine (validated columns + allowed operations)
  4. monthly_trend()       - totals per month

Stage 5 will let Claude request operations from this engine, so Python always
does the math and the AI never invents numbers.
"""

import re
import warnings

import pandas as pd

from src.config import DATE_PARSE_MIN_RATIO, DATE_SAMPLE_SIZE

# The ONLY calculations the engine will perform
ALLOWED_AGGREGATIONS = ("sum", "mean", "median", "min", "max", "count")

# Each rule: (role, name keywords in priority order, required column type)
ROLE_RULES = [
    ("order_id", ["order_id", "orderid", "order_no", "order_number", "invoice", "transaction_id"], "any"),
    ("date", ["order_date", "invoice_date", "date", "timestamp"], "date"),
    ("revenue", ["revenue", "sales", "total_amount", "amount", "total"], "number"),
    ("quantity", ["quantity", "qty", "units"], "number"),
    ("unit_price", ["unit_price", "price"], "number"),
    ("product", ["product", "item"], "text"),
    ("category", ["category"], "text"),
    ("region", ["region", "state", "city", "zone", "location", "area"], "text"),
    ("customer", ["customer", "client"], "any"),
    ("payment", ["payment"], "text"),
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _normalize(name) -> str:
    """'Unit Price' and 'Unit_Price' both become 'unit_price'."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _is_number(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _is_text(series: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def to_datetime_safe(series: pd.Series) -> pd.Series:
    """Convert to dates. Values that are not valid dates become NaT (missing)."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(series, errors="coerce")


def _looks_like_dates(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not _is_text(series):
        return False
    sample = series.dropna().head(DATE_SAMPLE_SIZE)
    if sample.empty:
        return False
    # Text like "100" or "2026" is a number, not a date
    if pd.to_numeric(sample, errors="coerce").notna().mean() >= 0.5:
        return False
    return to_datetime_safe(sample).notna().mean() >= DATE_PARSE_MIN_RATIO


def find_date_columns(df: pd.DataFrame) -> list:
    """Columns that are dates, or text that can be read as dates."""
    return [column for column in df.columns if _looks_like_dates(df[column])]


def _type_ok(series: pd.Series, kind: str) -> bool:
    if kind == "any":
        return True
    if kind == "number":
        return _is_number(series)
    if kind == "text":
        return _is_text(series)
    if kind == "date":
        return _looks_like_dates(series)
    return False


def format_number(value) -> str:
    """Format a number for display, e.g. 1234567.8 -> '1,234,568'."""
    if pd.isna(value):
        return "—"
    value = float(value)
    if value.is_integer() or abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def _label(agg: str, value_col) -> str:
    return "Number of rows" if agg == "count" else f"{agg.capitalize()} of {value_col}"


def _check_agg(agg: str) -> None:
    if agg not in ALLOWED_AGGREGATIONS:
        raise ValueError(f"Unsupported calculation '{agg}'. Allowed: {', '.join(ALLOWED_AGGREGATIONS)}.")


def _check_numeric_column(df: pd.DataFrame, column) -> None:
    if column is None or column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in this dataset.")
    if not _is_number(df[column]):
        raise ValueError(
            f"Column '{column}' is not numeric, so it cannot be summed or averaged. "
            "If it holds numbers stored as text, convert it in Data Quality → Cleaning."
        )


# ---------------------------------------------------------------------------
# 1. Column role detection
# ---------------------------------------------------------------------------
def detect_roles(df: pd.DataFrame) -> dict:
    """
    Guess which column plays which role, using column names AND data types.
    Returns e.g. {"revenue": "Revenue", "region": "Region", "product": None, ...}
    A role is None when no suitable column was found.
    """
    roles = {role: None for role, _, _ in ROLE_RULES}
    used = set()
    for role, keywords, kind in ROLE_RULES:
        for keyword in keywords:
            match = None
            for column in df.columns:
                if column in used or keyword not in _normalize(column):
                    continue
                if _type_ok(df[column], kind):
                    match = column
                    break
            if match is not None:
                roles[role] = match
                used.add(match)
                break
    return roles


# ---------------------------------------------------------------------------
# 2. Safe calculation engine
# ---------------------------------------------------------------------------
def aggregate_by_group(df, group_col, value_col=None, agg="sum", top_n=10, ascending=False):
    """
    Group rows by one column and calculate one validated operation per group.
    Example: group_col="Region", value_col="Revenue", agg="sum"

    Raises ValueError for unknown columns, non-numeric value columns or
    unsupported operations. Returns a DataFrame with two columns.
    """
    _check_agg(agg)
    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' does not exist in this dataset.")
    if top_n is not None and top_n < 1:
        raise ValueError("top_n must be at least 1.")

    label = _label(agg, value_col)
    if agg == "count":
        result = df.groupby(group_col, dropna=True, observed=True).size().reset_index()
    else:
        _check_numeric_column(df, value_col)
        if value_col == group_col:
            raise ValueError("The group column and the value column must be different.")
        result = df.groupby(group_col, dropna=True, observed=True)[value_col].agg(agg).reset_index()
    result.columns = [group_col, label]

    result = result.sort_values(label, ascending=ascending, kind="mergesort")
    if top_n is not None:
        result = result.head(top_n)
    return result.reset_index(drop=True)


def monthly_trend(df, date_col, value_col=None, agg="sum"):
    """
    Calculate one value per calendar month.
    Returns (result, invalid_dates): result has columns "Month" and the value;
    invalid_dates is how many rows had a date that could not be read (left out).
    """
    _check_agg(agg)
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' does not exist in this dataset.")
    if agg != "count":
        _check_numeric_column(df, value_col)

    dates = to_datetime_safe(df[date_col])
    invalid = int((dates.isna() & df[date_col].notna()).sum())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        months = dates.dt.to_period("M").dt.to_timestamp()

    work = pd.DataFrame({"_month": months})
    label = _label(agg, value_col)
    if agg == "count":
        result = work.dropna(subset=["_month"]).groupby("_month").size().reset_index()
    else:
        work["_value"] = df[value_col]
        result = work.dropna(subset=["_month"]).groupby("_month")["_value"].agg(agg).reset_index()
    result.columns = ["Month", label]
    return result.sort_values("Month").reset_index(drop=True), invalid


# ---------------------------------------------------------------------------
# 3. KPI cards
# ---------------------------------------------------------------------------
def calculate_kpis(df: pd.DataFrame, roles: dict = None):
    """
    Returns (cards, skipped).
      cards   - list of {"label", "value", "detail"} for metrics that CAN be calculated
      skipped - list of sentences explaining metrics that could NOT be calculated
    """
    roles = roles or detect_roles(df)
    cards, skipped = [], []

    def add(label, value, detail):
        cards.append({"label": label, "value": value, "detail": detail})

    revenue, quantity, order_id = roles["revenue"], roles["quantity"], roles["order_id"]

    add("Total records", f"{len(df):,}", "Number of rows in the dataset")

    # Total revenue
    total_revenue = None
    if revenue:
        total_revenue = df[revenue].sum()
        missing = int(df[revenue].isna().sum())
        detail = f"Sum of '{revenue}'"
        if missing:
            detail += f" ({missing:,} missing values ignored)"
        add("Total revenue", format_number(total_revenue), detail)
    else:
        skipped.append(
            "Total revenue: no numeric revenue / sales / amount column was found. "
            "(If your revenue column is stored as text, convert it in Data Quality → Cleaning.)"
        )

    # Total orders and average order value
    orders = None
    if order_id:
        orders = int(df[order_id].nunique())
        add("Total orders", f"{orders:,}", f"Unique values in '{order_id}'")
    else:
        skipped.append("Total orders: no order ID / invoice column was found.")

    if revenue and orders:
        add("Average order value", format_number(total_revenue / orders),
            f"Total revenue ÷ {orders:,} unique orders")
    else:
        skipped.append("Average order value: needs both a revenue column and an order ID column.")

    # Quantity
    if quantity:
        add("Total quantity sold", format_number(df[quantity].sum()), f"Sum of '{quantity}'")
    else:
        skipped.append("Total quantity sold: no numeric quantity column was found.")

    # Customers
    if roles["customer"]:
        add("Unique customers", f"{int(df[roles['customer']].nunique()):,}",
            f"Unique values in '{roles['customer']}'")

    # Date range
    if roles["date"]:
        dates = to_datetime_safe(df[roles["date"]]).dropna()
        if dates.empty:
            skipped.append(f"Date range: column '{roles['date']}' contains no valid dates.")
        else:
            add("Date range", f"{dates.min():%d %b %Y} – {dates.max():%d %b %Y}",
                f"Earliest and latest valid date in '{roles['date']}'")

    # Top performers (ranked by total revenue)
    for role, label in (("product", "Top product"), ("category", "Top category"), ("region", "Best region")):
        column = roles[role]
        if column is None:
            skipped.append(f"{label}: no {role} column was found.")
        elif revenue is None:
            skipped.append(f"{label}: a revenue column is needed to rank '{column}'.")
        else:
            top = aggregate_by_group(df, column, revenue, "sum", top_n=1)
            if top.empty:
                skipped.append(f"{label}: '{column}' has no usable values.")
            else:
                add(label, str(top.iloc[0, 0]),
                    f"{format_number(top.iloc[0, 1])} total {revenue} (grouped by '{column}')")

    # Generic fallback for datasets that are not sales data
    if not revenue and not quantity:
        for column in df.select_dtypes(include="number").columns[:3]:
            add(f"Total {column}", format_number(df[column].sum()), f"Sum of '{column}'")
            add(f"Average {column}", format_number(df[column].mean()), f"Mean of '{column}'")

    return cards, skipped