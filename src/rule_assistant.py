"""
rule_assistant.py
A FREE question-answering assistant that needs no AI model and no API key.

How it works:
  1. Read the question with keyword/pattern rules (which columns, which
     calculation, which filters, top N, ascending or descending ...).
  2. Build a request for one of the three SAFE operations in analysis_engine.py.
  3. Pandas calculates the result. This file only writes the answer text from templates.

Because answers are built from calculated tables, they cannot contain invented numbers.
"""

import calendar
import re

import pandas as pd

from src.analysis_engine import get_column_kinds, run_operation
from src.kpi import detect_roles, to_datetime_safe

MAX_QUESTION_CHARS = 1000
ANSWER_TOP_N = 5            # how many ranked rows to show when the user gives no number
MAX_TOP_N = 50
MAX_VALUE_CHOICES = 500     # only look for category values in columns with this many distinct values or fewer
MAX_FILTERS_ALLOWED = 5


class AssistantError(Exception):
    """A friendly message that is safe to show to the user."""


# ---------------------------------------------------------------------------
# Vocabulary: words people use for the columns detect_roles() finds
# ---------------------------------------------------------------------------
ROLE_WORDS = {
    "revenue": ["revenue", "sales", "income", "turnover", "amount"],
    "quantity": ["quantity", "qty", "units", "units sold"],
    "unit_price": ["unit price", "price"],
    "product": ["product", "item"],
    "category": ["category", "categories"],
    "region": ["region", "state", "city", "area", "zone", "location"],
    "customer": ["customer", "client", "buyer"],
    "payment": ["payment", "payment method", "payment mode"],
    "order_id": ["order", "transaction", "invoice"],
    "date": ["date"],
}

NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_NW = "|".join(NUM_WORDS)

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

IGNORED_VALUES = {"all", "other", "others", "na", "n a", "none", "yes", "no", "unknown", "total", "nan"}

AGG_WORDS = {"sum": "total", "mean": "average", "median": "median", "min": "minimum", "max": "maximum"}

# ---------------------------------------------------------------------------
# Patterns (the question is lower-cased and stripped of punctuation first)
# ---------------------------------------------------------------------------
SUMMARY_RE = re.compile(r"\b(summary|summarize|summarise|overview|insights?|describe|tell me about|key findings|highlights)\b")
AOV_RE = re.compile(
    r"\b(average|avg|mean) (order|basket|ticket) (value|size)\b|\baov\b|"
    r"\b(average|avg|mean) (revenue|sales|amount) (per|for each|by) (order|transaction|invoice)\b"
)
DATE_RANGE_RE = re.compile(r"\b(date range|time period|period covered|earliest|latest|oldest|newest|most recent|first order|last order|first date|last date|start date|end date)\b")
TREND_RE = re.compile(r"\b(trend|trends|monthly|month by month|month wise|monthwise|per month|by month|each month|every month|over time|which month|best month|worst month|growth)\b")
RANK_RE = re.compile(r"\b(which|top|bottom|best|worst|highest|lowest|rank|ranking|ranked|compare|comparison|performing|performance|performed|perform|performs|by|per|each|across|breakdown|wise|most|least|leading|weak|weakest|strongest|share|contribution)\b")
LOW_RE = re.compile(r"\b(lowest|worst|least|poor|poorly|weak|weakest|bottom|underperform\w*|smallest|fewest|minimum)\b")
HIGH_RE = re.compile(r"\b(highest|best|top|most|largest|biggest|greatest|leading|strongest|maximum)\b")
LOW_EXT_RE = re.compile(r"\b(lowest|smallest|minimum)\b")
HIGH_EXT_RE = re.compile(r"\b(highest|largest|biggest|greatest)\b")
COUNTISH_RE = re.compile(r"\b(orders|transactions|purchases|popular|common|frequent|frequently)\b")
TOPN_RE = re.compile(
    rf"\b(?:top|bottom|best|worst|highest|lowest|first|last)\s+(\d+|{_NW})\b|\b(\d+|{_NW})\s+(?:best|worst|top|highest|lowest)\b"
)
FOLLOWUP_RE = re.compile(r"^(and|also|what about|how about|then|now|only|same|ok and|what if)\b")
HELP_RE = re.compile(r"^(hi|hello|hey|help|what can you do|how do you work|how can you help)\b")
PREDICT_RE = re.compile(r"\b(predict|prediction|forecast|will|next (month|year|quarter|week)|future|projection|projected|expected)\b")
CAUSE_RE = re.compile(r"\b(why|reason|reasons|caused|causes|because|due to)\b")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _normalize(text) -> str:
    """'Unit_Price?' -> 'unit price'. Lower-case, only letters/digits/single spaces."""
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _fmt(value) -> str:
    """1234567.5 -> '1,234,567.5'; 143000.0 -> '143,000'. Text is returned unchanged."""
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return "—"
    return f"{float(value):,.2f}".rstrip("0").rstrip(".")


def _nice(label: str) -> str:
    """'Sum of Revenue' -> 'total Revenue'."""
    for prefix, word in (("Sum of ", "total "), ("Mean of ", "average "), ("Median of ", "median "),
                         ("Min of ", "minimum "), ("Max of ", "maximum ")):
        if label.startswith(prefix):
            return word + label[len(prefix):]
    return "number of rows" if label == "Number of rows" else label


def _scalar(result):
    return result["table"]["Value"].iloc[0]


def _run(df, name, args, calcs):
    """Run one safe operation. Problems become friendly errors."""
    try:
        result = run_operation(df, name, args)
    except ValueError as error:
        raise AssistantError(f"I couldn't calculate that: {error}") from error
    calcs.append(result)
    return result


def _try(df, name, args, calcs):
    """Like _run, but returns None instead of failing (used by the summary)."""
    try:
        result = run_operation(df, name, args)
    except ValueError:
        return None
    calcs.append(result)
    return result


# ---------------------------------------------------------------------------
# Finding columns, values and dates mentioned in the question
# ---------------------------------------------------------------------------
def _column_terms(df, roles) -> dict:
    role_of = {col: role for role, col in roles.items() if col}
    terms = {}
    for column in df.columns:
        names = {_normalize(column)}
        role = role_of.get(column)
        if role:
            names.update(ROLE_WORDS.get(role, []))
        terms[column] = {n for n in names if n}
    return terms


def _find_mentions(text, terms_by_column) -> dict:
    """Return {column: position of first mention} for columns named in the question."""
    found = {}
    for column, terms in terms_by_column.items():
        for term in terms:
            match = re.search(rf"\b{re.escape(term)}(?:s|es)?\b", text)
            if match and (column not in found or match.start() < found[column]):
                found[column] = match.start()
    return found


def _pick_group_col(text, text_cols, mentions):
    """Prefer the column that follows 'by', 'per', 'each' or 'across'; else the first one mentioned."""
    for column in text_cols:
        before = text[:mentions[column]].split()
        if before and before[-1] in ("by", "per", "each", "across"):
            return column
    return text_cols[0]


def _pick_date_col(kinds, roles, mentions):
    date_cols = [c for c, k in kinds.items() if k == "date"]
    if not date_cols:
        return None
    for column in sorted(mentions, key=mentions.get):
        if column in date_cols:
            return column
    if roles["date"] in date_cols:
        return roles["date"]
    return date_cols[0]


def _value_filters(text, df, kinds) -> list:
    """Find category values (e.g. 'Bengaluru', 'Laptop') written in the question -> filters."""
    candidates = []
    for column, kind in kinds.items():
        if kind != "text":
            continue
        values = df[column].dropna().astype(str).str.strip().unique()
        if len(values) > MAX_VALUE_CHOICES:
            continue
        for value in values:
            key = _normalize(value)
            if len(key) < 2 or key in IGNORED_VALUES:
                continue
            for match in re.finditer(rf"\b{re.escape(key)}s?\b", text):
                candidates.append((column, value, match.start(), match.end()))

    candidates.sort(key=lambda c: c[3] - c[2], reverse=True)   # longest match first
    taken, chosen = [], {}
    for column, value, start, end in candidates:
        if any(start < t_end and t_start < end for t_start, t_end in taken):
            continue
        taken.append((start, end))
        chosen.setdefault(column, [])
        if value not in chosen[column]:
            chosen[column].append(value)

    filters = []
    for column, values in chosen.items():
        if len(values) == 1:
            filters.append({"column": column, "op": "==", "value": values[0]})
        else:
            filters.append({"column": column, "op": "in", "value": values})
    return filters


def _date_filters(text, df, date_col) -> list:
    """Turn 'in January 2026' / 'in 2026' into date filters."""
    if date_col is None:
        return []
    tokens = text.split()
    year = month = None
    for i, token in enumerate(tokens):
        if year is None and re.fullmatch(r"(19|20)\d\d", token):
            year = int(token)
        if month is None and token in MONTHS:
            if token == "may":  # 'may' is also an ordinary word
                before = tokens[i - 1] if i > 0 else ""
                after = tokens[i + 1] if i + 1 < len(tokens) else ""
                if before not in ("in", "of", "for", "during", "since") and not re.fullmatch(r"(19|20)\d\d", after):
                    continue
            month = MONTHS[token]
    if month is None and year is None:
        return []

    if month is not None and year is None:
        years = sorted(to_datetime_safe(df[date_col]).dt.year.dropna().astype(int).unique())
        if len(years) == 1:
            year = int(years[0])
        elif len(years) > 1:
            name = calendar.month_name[month]
            raise AssistantError(
                f"Your data covers several years ({', '.join(map(str, years))}). "
                f"Which year do you mean? For example: \"in {name} {years[-1]}\"."
            )
        else:
            return []

    if month is not None:
        start = pd.Timestamp(year, month, 1)
        end = start + pd.offsets.MonthEnd(0)
    else:
        start, end = pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31)
    return [
        {"column": date_col, "op": ">=", "value": start.strftime("%Y-%m-%d")},
        {"column": date_col, "op": "<=", "value": end.strftime("%Y-%m-%d")},
    ]


# ---------------------------------------------------------------------------
# Understanding the question
# ---------------------------------------------------------------------------
def _detect_agg(text):
    if re.search(r"\b(average|avg|mean)\b", text):
        return "mean"
    if re.search(r"\bmedian\b", text):
        return "median"
    if re.search(r"\b(how many|number of|count|popular|common|frequent|frequently)\b", text):
        return "count"
    if re.search(r"\b(unique|distinct|different)\b", text):
        return "nunique"
    if re.search(r"\b(maximum|max)\b", text):
        return "max"
    if re.search(r"\b(minimum|min)\b", text):
        return "min"
    if re.search(r"\b(total|sum|overall|altogether)\b", text):
        return "sum"
    return None


def _parse(text, df, kinds, roles) -> dict:
    """
    Turn the question into a partial 'spec'. Anything the question does not say stays None
    (so follow-up questions can inherit it from the previous question).
    """
    mentions = _find_mentions(text, _column_terms(df, roles))
    ordered = sorted(mentions, key=mentions.get)
    text_cols = [c for c in ordered if kinds[c] == "text"]
    num_cols = [c for c in ordered if kinds[c] == "number"]
    agg = _detect_agg(text)

    if SUMMARY_RE.search(text):
        intent = "summary"
    elif AOV_RE.search(text):
        intent = "aov"
    elif DATE_RANGE_RE.search(text):
        intent = "date_range"
    elif TREND_RE.search(text):
        intent = "trend"
    elif text_cols and RANK_RE.search(text):
        intent = "group"
    elif agg or num_cols:
        intent = "metric"
    else:
        intent = None

    spec = {"intent": intent, "agg": None, "value_col": None, "group_col": None,
            "order": None, "top_n": None, "filters": None, "date_col": None}

    if LOW_RE.search(text):
        spec["order"] = "lowest"
    elif HIGH_RE.search(text):
        spec["order"] = "highest"

    match = TOPN_RE.search(text)
    if match:
        raw = next(group for group in match.groups() if group)
        number = int(raw) if raw.isdigit() else NUM_WORDS[raw]
        spec["top_n"] = max(1, min(number, MAX_TOP_N))

    if intent == "group":
        spec["group_col"] = _pick_group_col(text, text_cols, mentions)
        spec["value_col"] = num_cols[0] if num_cols else None
        spec["agg"] = agg
        if agg is None and not num_cols and COUNTISH_RE.search(text):
            spec["agg"] = "count"
    elif intent == "trend":
        spec["value_col"] = num_cols[0] if num_cols else None
        spec["agg"] = agg
    elif intent == "metric":
        if agg is None:
            agg = "min" if LOW_EXT_RE.search(text) else "max" if HIGH_EXT_RE.search(text) else "sum"
        if agg == "count" and text_cols:
            agg, spec["value_col"] = "nunique", text_cols[0]       # 'how many customers'
        elif agg == "nunique":
            spec["value_col"] = text_cols[0] if text_cols else None
        else:
            spec["value_col"] = num_cols[0] if num_cols else None
        spec["agg"] = agg

    date_col = _pick_date_col(kinds, roles, mentions)
    spec["date_col"] = date_col
    if intent != "summary":
        filters = _value_filters(text, df, kinds) + _date_filters(text, df, date_col)
        spec["filters"] = filters or None
    return spec


def _merge(context: dict, partial: dict) -> dict:
    """Follow-up question: keep the previous request, change only what the new question says."""
    merged = dict(context)
    for key in ("intent", "agg", "value_col", "group_col", "order", "top_n", "date_col"):
        if partial[key] is not None:
            merged[key] = partial[key]
    if partial["filters"]:
        new_columns = {f["column"] for f in partial["filters"]}
        kept = [f for f in (context.get("filters") or []) if f["column"] not in new_columns]
        merged["filters"] = kept + partial["filters"]
    return merged


def _fill_defaults(spec, df, kinds, roles) -> dict:
    """Fill in anything still missing with sensible defaults, or ask the user to be clearer."""
    spec = dict(spec)
    intent = spec["intent"]
    spec["filters"] = (spec.get("filters") or [])[:MAX_FILTERS_ALLOWED]
    spec["defaulted"] = False
    numeric = [c for c, k in kinds.items() if k == "number"]

    if intent in ("group", "trend", "metric") and spec["agg"] is None:
        spec["agg"] = "sum"
    if intent in ("group", "trend") and spec["agg"] == "nunique":
        spec["agg"] = "count"

    if intent == "group":
        if not spec["group_col"]:
            raise AssistantError("Which column should I group by? For example: Product, Region or Category.")
        spec["order"] = spec["order"] or "highest"
        spec["top_n"] = spec["top_n"] or ANSWER_TOP_N

    if intent in ("trend", "date_range") and not spec["date_col"]:
        raise AssistantError("This dataset has no date column, so I can't work with months or date ranges.")

    if intent == "metric" and spec["agg"] == "nunique":
        if not spec["value_col"]:
            raise AssistantError("Which column should I count unique values of? For example: \"How many customers?\"")
    elif intent in ("group", "trend", "metric") and spec["agg"] != "count":
        if spec["value_col"] not in numeric:
            spec["value_col"] = None
            if roles["revenue"] in numeric:
                spec["value_col"], spec["defaulted"] = roles["revenue"], True
            elif len(numeric) == 1:
                spec["value_col"], spec["defaulted"] = numeric[0], True
            else:
                raise AssistantError(
                    "Which number should I use? Please name a numeric column. "
                    f"Available numeric columns: {', '.join(numeric) or 'none'}."
                )
    return spec


# ---------------------------------------------------------------------------
# Describing what was understood
# ---------------------------------------------------------------------------
def _filters_text(filters) -> str:
    parts = []
    for f in filters:
        value = " / ".join(map(str, f["value"])) if isinstance(f["value"], list) else f["value"]
        op = {"==": "is", "in": "is", "!=": "is not", ">=": "from", ">": "after",
              "<=": "up to", "<": "before"}.get(f["op"], f["op"])
        parts.append(f"{f['column']} {op} {value}")
    return ", ".join(parts)


def _metric_phrase(spec) -> str:
    agg, col = spec["agg"], spec["value_col"]
    if agg == "count":
        return "number of rows"
    if agg == "nunique":
        return f"number of unique {col} values"
    return f"{AGG_WORDS[agg]} {col}"


def _describe(spec) -> str:
    intent = spec["intent"]
    measure = "number of rows" if spec.get("agg") == "count" else f"{AGG_WORDS.get(spec.get('agg'), '')} {spec.get('value_col')}"
    if intent == "group":
        text = f"{measure}, grouped by {spec['group_col']}, {spec['order']} first, top {spec['top_n']}"
    elif intent == "trend":
        text = f"{measure} per month (using {spec['date_col']})"
    elif intent == "metric":
        text = _metric_phrase(spec)
    elif intent == "aov":
        text = "average order value = total revenue ÷ unique orders"
    elif intent == "date_range":
        text = f"earliest and latest date in {spec['date_col']}"
    else:
        text = "overview of the dataset"
    if spec.get("defaulted") and spec.get("value_col"):
        text += f" (you did not name a measure, so I used {spec['value_col']})"
    if spec.get("filters"):
        text += f"; filtered to {_filters_text(spec['filters'])}"
    return text


# ---------------------------------------------------------------------------
# Answer templates (every number below comes from a calculated table)
# ---------------------------------------------------------------------------
def _group_answer(result, order) -> str:
    table = result["table"]
    group, label = table.columns[0], table.columns[1]
    share = "Share of total (%)"
    has_share = share in table.columns
    word = "lowest" if order == "lowest" else "highest"
    first = table.iloc[0]

    text = f"**{first[group]}** has the {word} {_nice(label)}: **{_fmt(first[label])}**"
    if has_share and pd.notna(first[share]):
        text += f" ({first[share]:.1f}% of the total)"
    text += "."
    if len(table) > 1 and table[label].iloc[1] == first[label]:
        text += f" It is tied with **{table[group].iloc[1]}**."
    if len(table) == 1:
        return text

    lines = [text, "", f"Ranking ({word} first):"]
    for position, row in table.iterrows():
        line = f"{position + 1}. {row[group]} — {_fmt(row[label])}"
        if has_share and pd.notna(row[share]):
            line += f" ({row[share]:.1f}%)"
        lines.append(line)
    return "\n".join(lines)


def _trend_answer(result) -> str:
    table = result["table"]
    label = table.columns[1]
    change_col = "Change vs previous month (%)"
    lines = [f"Here is the {_nice(label)} for each month:", ""]
    for _, row in table.iterrows():
        line = f"- **{row['Month']:%b %Y}**: {_fmt(row[label])}"
        if pd.notna(row[change_col]):
            line += f" ({row[change_col]:+.1f}% vs previous month)"
        lines.append(line)
    if len(table) > 1 and table[label].notna().any():
        high, low = table.loc[table[label].idxmax()], table.loc[table[label].idxmin()]
        lines += ["", f"Highest month: **{high['Month']:%b %Y}** ({_fmt(high[label])}). "
                      f"Lowest month: **{low['Month']:%b %Y}** ({_fmt(low[label])})."]
    return "\n".join(lines)


def _aov_result(total, orders, revenue_col, order_col) -> dict:
    """A small table showing how average order value was calculated."""
    table = pd.DataFrame({
        "Metric": [f"Total {revenue_col}", f"Unique orders ({order_col})", "Average order value (total ÷ unique orders)"],
        "Value": [float(total), float(orders), float(total) / orders],
    })
    return {"kind": "scalar", "title": "Average order value", "table": table, "notes": [],
            "tool": "derived", "inputs": {"formula": "total revenue / number of unique orders"}}


def _aov_answer(df, roles, spec, calcs) -> str:
    revenue, order_id = roles["revenue"], roles["order_id"]
    if not revenue or not order_id:
        raise AssistantError(
            "Average order value needs a numeric revenue column and an order ID column, "
            "and at least one of them was not found in this dataset."
        )
    filters = {"filters": spec["filters"]} if spec["filters"] else {}
    total = _scalar(_run(df, "calculate_metric", {"agg": "sum", "column": revenue, **filters}, calcs))
    orders = int(_scalar(_run(df, "calculate_metric", {"agg": "nunique", "column": order_id, **filters}, calcs)))
    if orders == 0:
        raise AssistantError("There are no orders to average.")
    calcs.append(_aov_result(total, orders, revenue, order_id))
    return (f"The average order value is **{_fmt(total / orders)}** "
            f"(total {revenue} of {_fmt(total)} ÷ {orders:,} unique orders).")


def _summary_answer(df, roles, kinds, calcs) -> str:
    lines = ["**Calculated facts**", ""]
    missing, duplicates = int(df.isna().sum().sum()), int(df.duplicated().sum())
    lines.append(f"- The dataset has **{len(df):,} rows** and **{len(df.columns)} columns**; "
                 f"{missing:,} cells are empty and {duplicates:,} rows are exact duplicates.")

    numeric = [c for c, k in kinds.items() if k == "number"]
    revenue = roles["revenue"] if roles["revenue"] in numeric else None

    if revenue:
        total_result = _try(df, "calculate_metric", {"agg": "sum", "column": revenue}, calcs)
        total = _scalar(total_result) if total_result else None
        if total is not None:
            lines.append(f"- Total {revenue}: **{_fmt(total)}**.")
        order_id = roles["order_id"]
        if order_id and total is not None:
            orders_result = _try(df, "calculate_metric", {"agg": "nunique", "column": order_id}, calcs)
            if orders_result:
                orders = int(_scalar(orders_result))
                lines.append(f"- Unique orders ({order_id}): **{orders:,}**.")
                if orders > 0:
                    calcs.append(_aov_result(total, orders, revenue, order_id))
                    lines.append(f"- Average order value: **{_fmt(total / orders)}** (total {revenue} ÷ unique orders).")

        for role in ("product", "category", "region"):
            column = roles[role]
            if not column or kinds.get(column) != "text":
                continue
            top = _try(df, "group_and_aggregate",
                       {"group_column": column, "agg": "sum", "value_column": revenue, "top_n": 3, "order": "highest"}, calcs)
            if not top:
                continue
            table = top["table"]
            first = table.iloc[0]
            share = f", {first['Share of total (%)']:.1f}% of the total" if "Share of total (%)" in table.columns else ""
            lines.append(f"- Highest {revenue} by {column}: **{first[column]}** ({_fmt(first.iloc[1])}{share}).")
            low = _try(df, "group_and_aggregate",
                       {"group_column": column, "agg": "sum", "value_column": revenue, "top_n": 1, "order": "lowest"}, calcs)
            if low and low["table"].iloc[0, 0] != first[column]:
                low_row = low["table"].iloc[0]
                lines.append(f"- Lowest {revenue} by {column}: **{low_row[column]}** ({_fmt(low_row.iloc[1])}).")

        date_col = roles["date"] if kinds.get(roles["date"]) == "date" else None
        if date_col:
            trend = _try(df, "monthly_trend", {"date_column": date_col, "agg": "sum", "value_column": revenue}, calcs)
            if trend and len(trend["table"]) > 0:
                table = trend["table"]
                label = table.columns[1]
                lines.append(f"- Monthly {revenue} runs from **{table['Month'].iloc[0]:%b %Y}** to "
                             f"**{table['Month'].iloc[-1]:%b %Y}** ({len(table)} months).")
                if len(table) > 1 and table[label].notna().any():
                    high, low = table.loc[table[label].idxmax()], table.loc[table[label].idxmin()]
                    lines.append(f"- Highest month: **{high['Month']:%b %Y}** ({_fmt(high[label])}); "
                                 f"lowest month: **{low['Month']:%b %Y}** ({_fmt(low[label])}).")
    else:
        for column in numeric[:3]:
            total = _try(df, "calculate_metric", {"agg": "sum", "column": column}, calcs)
            mean = _try(df, "calculate_metric", {"agg": "mean", "column": column}, calcs)
            if total and mean:
                lines.append(f"- {column}: total **{_fmt(_scalar(total))}**, average **{_fmt(_scalar(mean))}**.")

    lines += ["", "**Interpretation**", "",
              "This free assistant works with rules, not an AI model. It reports calculated facts only and does "
              "not interpret them or explain causes. Review the facts above to draw your own conclusions."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Help, refusals and examples
# ---------------------------------------------------------------------------
def example_questions(df) -> list:
    """Example questions written with THIS dataset's own column names."""
    kinds, roles = get_column_kinds(df), detect_roles(df)
    revenue, product, region, date = roles["revenue"], roles["product"], roles["region"], roles["date"]
    examples = ["Give me a summary of this dataset"]
    if revenue:
        examples.append(f"What is the total {revenue}?")
    if revenue and product:
        examples.append(f"What are the top 5 {product} by {revenue}?")
    if revenue and region:
        examples.append(f"Which {region} has the lowest {revenue}?")
    if revenue and date and kinds.get(date) == "date":
        examples.append(f"Show monthly {revenue}")
    if not revenue:
        text_cols = [c for c, k in kinds.items() if k == "text"]
        number_cols = [c for c, k in kinds.items() if k == "number"]
        if text_cols and number_cols:
            examples.append(f"Which {text_cols[0]} has the highest average {number_cols[0]}?")
            examples.append(f"What is the total {number_cols[0]}?")
    return examples


def _help_text(df) -> str:
    lines = [
        "I answer questions by running safe calculations on your data. Everything is calculated by Python, "
        "and I don't need an internet connection or an API key. I can do:",
        "",
        "- **Totals and averages**: total, average, median, minimum, maximum, counts, unique counts",
        "- **Rankings**: top or bottom N, best or worst, for any category column",
        "- **Monthly trends**, with the change versus the previous month",
        "- **Filters** by a value or date: \"for Bengaluru\", \"in January 2026\", \"in 2026\"",
        "- **Dataset summary**, average order value, date range",
        "",
        "Try, for example:",
    ]
    lines += [f"- {question}" for question in example_questions(df)]
    return "\n".join(lines)


def _check_unsupported(text, df):
    """Politely refuse questions that rules (and honest analysis) cannot answer."""
    column_text = " ".join(_normalize(c) for c in df.columns)
    for word in ("profit", "margin", "cost"):
        if re.search(rf"\b{word}\b", text) and word not in column_text:
            return (f"I can't calculate **{word}** because this dataset has no column related to it. "
                    "To analyse it, add a cost or profit column to your data and upload it again.")
    if PREDICT_RE.search(text):
        return ("I don't predict or forecast future values. I can only report what is in your data. "
                "I can show the monthly trend so far, for example by asking \"Show monthly revenue\".")
    if CAUSE_RE.search(text):
        return ("I can describe what the numbers show, but I can't tell you *why* something happened, because "
                "the data alone doesn't prove causes. Try asking for a breakdown instead, for example by region, "
                "product or month.")
    return None


# ---------------------------------------------------------------------------
# The main function used by the app
# ---------------------------------------------------------------------------
def _plain(text, context) -> dict:
    return {"answer": text, "calculations": [], "ungrounded": [], "context": context}


def _dispatch(df, spec, roles, kinds, calcs) -> str:
    intent, filters = spec["intent"], spec["filters"]
    extra = {"filters": filters} if filters else {}

    if intent == "summary":
        return _summary_answer(df, roles, kinds, calcs)
    if intent == "aov":
        return _aov_answer(df, roles, spec, calcs)

    if intent == "date_range":
        low = _run(df, "calculate_metric", {"agg": "min", "column": spec["date_col"], **extra}, calcs)
        high = _run(df, "calculate_metric", {"agg": "max", "column": spec["date_col"], **extra}, calcs)
        return (f"The dates in **{spec['date_col']}** run from **{_scalar(low)}** to **{_scalar(high)}**.")

    if intent == "metric":
        args = {"agg": spec["agg"], **extra}
        if spec["agg"] != "count":
            args["column"] = spec["value_col"]
        result = _run(df, "calculate_metric", args, calcs)
        return f"The {_metric_phrase(spec)} is **{_fmt(_scalar(result))}**."

    if intent == "group":
        args = {"group_column": spec["group_col"], "agg": spec["agg"], "top_n": spec["top_n"],
                "order": spec["order"], **extra}
        if spec["agg"] != "count":
            args["value_column"] = spec["value_col"]
        return _group_answer(_run(df, "group_and_aggregate", args, calcs), spec["order"])

    if intent == "trend":
        args = {"date_column": spec["date_col"], "agg": spec["agg"], **extra}
        if spec["agg"] != "count":
            args["value_column"] = spec["value_col"]
        return _trend_answer(_run(df, "monthly_trend", args, calcs))

    raise AssistantError("I'm not sure what to calculate. Try rephrasing your question.")


def ask_assistant(df, question: str, context=None) -> dict:
    """
    Answer one question about df.

    context: the 'context' returned by the previous call (used for follow-up questions).
    Returns {"answer": str, "calculations": [results], "ungrounded": [], "context": dict}
    Raises AssistantError with a friendly message when the question cannot be handled.
    """
    if df is None or df.empty:
        raise AssistantError("There is no data to analyse.")
    question = (question or "").strip()
    if not question:
        raise AssistantError("Please type a question.")
    if len(question) > MAX_QUESTION_CHARS:
        raise AssistantError(f"Please keep questions under {MAX_QUESTION_CHARS} characters.")
    text = _normalize(question)
    if not text:
        raise AssistantError("I couldn't find any words in that question. Please try again.")

    if HELP_RE.match(text) and len(text.split()) <= 6:
        return _plain(_help_text(df), context)
    refusal = _check_unsupported(text, df)
    if refusal:
        return _plain(refusal, context)

    kinds, roles = get_column_kinds(df), detect_roles(df)
    partial = _parse(text, df, kinds, roles)

    lacks_core = partial["intent"] is None and partial["group_col"] is None and partial["value_col"] is None
    is_followup = context is not None and (FOLLOWUP_RE.match(text) is not None or lacks_core)
    spec = _merge(context, partial) if is_followup else partial

    if spec["intent"] is None:
        examples = "\n".join(f"- {q}" for q in example_questions(df))
        return _plain("I didn't understand that question. Please mention a column name and what to calculate "
                      f"(total, average, top 5, monthly trend ...). For example:\n\n{examples}", context)

    spec = _fill_defaults(spec, df, kinds, roles)
    calcs = []
    answer = _dispatch(df, spec, roles, kinds, calcs)

    notes = []
    for result in calcs:
        for note in result["notes"]:
            if note not in notes:
                notes.append(note)
    if notes:
        answer += "\n\n" + "\n".join(f"*Note: {note}*" for note in notes)
    answer += f"\n\n*How I read your question: {_describe(spec)}.*"

    return {"answer": answer, "calculations": calcs, "ungrounded": [], "context": spec}