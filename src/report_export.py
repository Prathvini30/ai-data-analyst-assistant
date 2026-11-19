"""
report_export.py
Turns data and results into downloadable files (CSV, Excel, Markdown text).
Functions return bytes, which Streamlit's download button sends to the user.
"""

import io
import re
from datetime import datetime

import pandas as pd
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

INVALID_SHEET_CHARS = re.compile(r"[\[\]:*?/\\]")


# ---------------------------------------------------------------------------
# Safety and formatting helpers
# ---------------------------------------------------------------------------
def safe_stem(file_name: str) -> str:
    """'My Sales (2026).csv' -> 'My_Sales_2026' (safe to use inside a download file name)."""
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", str(file_name))
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "dataset"


def _safe_text(value):
    """Text starting with = + @ could run as a spreadsheet formula, so add a leading apostrophe."""
    if isinstance(value, str) and value[:1] in ("=", "+", "@"):
        return "'" + value
    return value


def make_export_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy where risky text cells are neutralised. The original DataFrame is untouched."""
    safe = df.copy()
    for column in safe.columns:
        if pd.api.types.is_object_dtype(safe[column]) or pd.api.types.is_string_dtype(safe[column]):
            safe[column] = safe[column].map(_safe_text)
    return safe


def prepare_table(table: pd.DataFrame) -> pd.DataFrame:
    """Show 'Month' columns as 2026-01 instead of 2026-01-01 00:00:00."""
    shown = table.copy()
    if "Month" in shown.columns and pd.api.types.is_datetime64_any_dtype(shown["Month"]):
        shown["Month"] = shown["Month"].dt.strftime("%Y-%m")
    return shown


def _sheet_name(name: str, used: set) -> str:
    """Excel sheet names: max 31 characters, no []:*?/\\ and must be unique."""
    base = INVALID_SHEET_CHARS.sub("", str(name)).strip()[:31] or "Sheet"
    candidate, number = base, 2
    while candidate.lower() in used:
        suffix = f"_{number}"
        candidate = base[: 31 - len(suffix)] + suffix
        number += 1
    used.add(candidate.lower())
    return candidate


def _style_sheet(sheet, table: pd.DataFrame) -> None:
    """Bold header, frozen first row, sensible column widths."""
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    sheet.freeze_panes = "A2"
    sample = table.head(500)
    for position, column in enumerate(table.columns, start=1):
        longest = max([len(str(column))] + [len(str(v)) for v in sample[column].tolist()])
        sheet.column_dimensions[get_column_letter(position)].width = min(60, longest + 2)


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """CSV that opens correctly in Excel (utf-8-sig keeps special characters intact)."""
    return make_export_safe(df).to_csv(index=False).encode("utf-8-sig")


def build_excel(sheets: dict) -> bytes:
    """Create an .xlsx file. sheets = {"Sheet name": DataFrame, ...}"""
    if not sheets:
        raise ValueError("There is nothing to export.")
    buffer = io.BytesIO()
    used = set()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, table in sheets.items():
            sheet_name = _sheet_name(name, used)
            table = make_export_safe(prepare_table(table))
            table.to_excel(writer, sheet_name=sheet_name, index=False)
            _style_sheet(writer.sheets[sheet_name], table)
    return buffer.getvalue()


def build_text_report(file_name: str, data_label: str, insights: dict) -> bytes:
    """A readable Markdown report that keeps facts and observations in separate sections."""
    lines = [
        "# Business Insights Report", "",
        f"- File: {file_name}",
        f"- Data analysed: {data_label}",
        f"- Generated: {datetime.now():%Y-%m-%d %H:%M}", "",
        "## Key metrics", "",
    ]
    lines += [f"- **{c['label']}**: {c['value']} ({c['detail']})" for c in insights["cards"]]
    lines += ["", "## Calculated facts", "",
              "These numbers were calculated from the dataset with Pandas.", ""]
    lines += [f"- {text}" for text in insights["facts"]]
    lines += ["", "## Rule-based observations", "",
              "Simple patterns found with fixed thresholds. They describe the numbers only and do not "
              "explain causes or predict the future.", ""]
    lines += [f"- {text}" for text in insights["observations"]]
    lines += ["", "## Areas needing attention", ""]
    lines += [f"- {text}" for text in insights["attention"]]
    return ("\n".join(lines) + "\n").encode("utf-8")