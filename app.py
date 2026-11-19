"""
app.py
Main entry point of the Streamlit application.
Run with:  streamlit run app.py
"""
from src.config import EXCEL_EXPORT_MAX_ROWS
from src.insights import build_insights
from src.report_export import build_excel, build_text_report, df_to_csv_bytes, safe_stem
import json

import pandas as pd
import streamlit as st

from src.config import (
    APP_ICON,
    APP_TITLE,
    DEFAULT_TOP_N,
    MAX_HISTOGRAM_BINS,
    PIE_MAX_SLICES,
    PREVIEW_ROWS,
)
from src.data_cleaning import apply_cleaning, detect_issues
from src.data_exploration import (
    get_column_summary,
    get_numeric_summary,
    get_overview,
    get_unique_values,
)
from src.file_handler import FileLoadError, get_extension, get_sheet_names, load_dataset
from src.kpi import (
    aggregate_by_group,
    calculate_kpis,
    detect_roles,
    find_date_columns,
    monthly_trend,
)
from src.rule_assistant import (
    MAX_QUESTION_CHARS,
    AssistantError,
    ask_assistant,
    example_questions,
)
from src.visualizations import (
    bar_chart,
    histogram,
    line_chart,
    pie_chart,
    sample_for_scatter,
    scatter_plot,
)

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")


# ---------- Sidebar navigation ----------
PAGES = [
    "📁 Upload & Preview",
    "🧹 Data Quality",
    "📈 Dashboard",
    "🤖 AI Analyst",
    "📊 Visualizations",
    "💡 Business Insights",
    "⬇️ Export",
]

st.sidebar.title(f"{APP_ICON} Data Analyst")
page = st.sidebar.radio("Navigate", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption("Upload a CSV or Excel file, then explore it.")

AGG_LABELS = {
    "sum": "Total (sum)",
    "mean": "Average (mean)",
    "median": "Median",
    "min": "Minimum",
    "max": "Maximum",
    "count": "Number of rows",
}


# ---------- Helpers ----------
def require_data():
    """Return the uploaded (raw) dataset, or show a warning if nothing is uploaded yet."""
    if "raw_df" not in st.session_state:
        st.warning("Please upload a dataset first on the **📁 Upload & Preview** page.")
        return None
    return st.session_state["raw_df"]


def get_active_data(key: str):
    """
    Choose which dataset to analyse: the cleaned copy (if one exists) or the raw data.
    Returns (dataframe, label) or (None, None) when nothing usable is available.
    """
    raw_df = require_data()
    if raw_df is None:
        return None, None

    df, label = raw_df, "original (raw)"
    if "clean_df" in st.session_state:
        if st.session_state["clean_df"].empty:
            st.warning("The cleaned dataset has no rows left, so the original data is being used.")
        else:
            choice = st.radio(
                "Which data should be analysed?",
                ["Cleaned dataset", "Original (raw) dataset"],
                horizontal=True,
                key=f"source_{key}",
            )
            if choice == "Cleaned dataset":
                df, label = st.session_state["clean_df"], "cleaned"

    if df.empty:
        st.warning("This dataset has no rows to analyse.")
        return None, None

    st.caption(f"Analysing the **{label}** dataset: {len(df):,} rows × {len(df.columns):,} columns.")
    return df, label


# ---------- Page: Upload & Preview ----------
def show_upload_page() -> None:
    """Upload a file, pick a worksheet (Excel), and preview the data."""
    st.title(APP_TITLE)
    st.write(
        "Upload a dataset to analyse it without writing any SQL or Python. "
        "All figures are calculated from your data."
    )

    uploaded_file = st.file_uploader("Upload a CSV or Excel (.xlsx) file", type=["csv", "xlsx"])

    if uploaded_file is None:
        if "raw_df" in st.session_state:
            st.info(f"Currently loaded: **{st.session_state['file_name']}**")
        else:
            st.info("👆 Upload a file to get started.")
        return

    # For Excel files, let the user choose a worksheet
    sheet_name = None
    try:
        if get_extension(uploaded_file.name) == ".xlsx":
            sheet_names = get_sheet_names(uploaded_file)
            if len(sheet_names) > 1:
                sheet_name = st.selectbox("Select worksheet", sheet_names)
            else:
                sheet_name = sheet_names[0]

        df = load_dataset(uploaded_file, sheet_name)
    except FileLoadError as error:
        st.error(f"❌ {error}")
        return

    # If this is a different file/sheet than before, forget any old cleaned data
    dataset_key = f"{uploaded_file.name}|{sheet_name}|{uploaded_file.size}"
    if st.session_state.get("dataset_key") != dataset_key:
        st.session_state.pop("clean_df", None)
        st.session_state.pop("cleaning_log", None)
        st.session_state["dataset_key"] = dataset_key

    # Save the RAW data. It is never modified; cleaned data is stored separately.
    st.session_state["raw_df"] = df
    st.session_state["file_name"] = uploaded_file.name

    st.success(f"✅ Loaded **{uploaded_file.name}** successfully.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", f"{len(df.columns):,}")
    col3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

    st.subheader("Data preview")
    st.dataframe(df.head(PREVIEW_ROWS), use_container_width=True)

    st.subheader("Columns and data types")
    column_info = pd.DataFrame(
        {
            "Column": df.columns,
            "Data type": df.dtypes.astype(str).values,
            "Example value": [
                df[column].dropna().iloc[0] if df[column].notna().any() else "—"
                for column in df.columns
            ],
        }
    )
    st.dataframe(column_info.astype(str), use_container_width=True, hide_index=True)


# ---------- Page: Data Quality ----------
def show_exploration_tab(df: pd.DataFrame) -> None:
    overview = get_overview(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{overview['rows']:,}")
    c2.metric("Columns", f"{overview['columns']:,}")
    c3.metric("Missing cells", f"{overview['missing_cells']:,}")
    c4.metric("Duplicate rows", f"{overview['duplicate_rows']:,}")

    st.subheader("Column summary")
    st.dataframe(get_column_summary(df), use_container_width=True, hide_index=True)

    st.subheader("Statistical summary (numeric columns)")
    stats = get_numeric_summary(df)
    if stats.empty:
        st.info("This dataset has no numeric columns.")
    else:
        st.dataframe(stats, use_container_width=True)

    st.subheader("Unique values")
    column = st.selectbox("Choose a column", df.columns)
    st.caption(f"'{column}' has {df[column].nunique():,} unique values. Showing the most common ones.")
    st.dataframe(get_unique_values(df, column), use_container_width=True, hide_index=True)


def show_issues_tab(issues: list) -> None:
    if not issues:
        st.success("✅ No data quality issues were detected.")
        return

    high = sum(1 for i in issues if i["severity"] == "High")
    medium = sum(1 for i in issues if i["severity"] == "Medium")
    low = sum(1 for i in issues if i["severity"] == "Low")
    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 High priority", high)
    c2.metric("🟠 Medium priority", medium)
    c3.metric("🟡 Low priority", low)

    st.subheader("All issues found")
    table = pd.DataFrame(issues)[["severity", "category", "column", "count"]]
    table.columns = ["Severity", "Issue", "Column", "Affected values"]
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("Details and recommendations")
    icons = {"High": "🔴", "Medium": "🟠", "Low": "🟡"}
    for issue in issues:
        with st.expander(f"{icons[issue['severity']]} {issue['category']} — {issue['column']}"):
            st.write(f"**What we found:** {issue['explanation']}")
            st.write(f"**Suggested action:** {issue['recommendation']}")


def show_cleaning_tab(raw_df: pd.DataFrame, issues: list) -> None:
    st.write(
        "Choose which cleaning steps to apply. Your **original data is never changed**: "
        "the result is saved as a separate *cleaned dataset*."
    )

    numeric_candidates = [i["column"] for i in issues if i["category"] == "Numbers stored as text"]
    date_candidates = [i["column"] for i in issues if i["category"] == "Dates stored as text"]
    has_duplicates = any(i["category"] == "Duplicate rows" for i in issues)

    with st.form("cleaning_form"):
        remove_duplicates = st.checkbox("Remove duplicate rows", value=has_duplicates)
        trim_whitespace = st.checkbox("Trim extra spaces and treat blank text as missing", value=True)
        unify_text = st.checkbox("Unify inconsistent spelling (e.g. 'bengaluru' → 'Bengaluru')", value=True)
        numeric_columns = st.multiselect(
            "Convert these text columns to numbers", numeric_candidates, default=numeric_candidates
        )
        date_columns = st.multiselect(
            "Convert these text columns to dates", date_candidates, default=date_candidates
        )
        missing_numeric = st.radio(
            "Missing values in numeric columns",
            ["leave", "median", "mean", "drop"],
            format_func=lambda x: {
                "leave": "Leave as they are (recommended)",
                "median": "Fill with the median",
                "mean": "Fill with the mean",
                "drop": "Drop those rows",
            }[x],
        )
        missing_text = st.radio(
            "Missing values in text columns",
            ["leave", "unknown", "drop"],
            format_func=lambda x: {
                "leave": "Leave as they are (recommended)",
                "unknown": "Fill with 'Unknown'",
                "drop": "Drop those rows",
            }[x],
        )
        submitted = st.form_submit_button("✨ Apply cleaning")

    if submitted:
        options = {
            "remove_duplicates": remove_duplicates,
            "trim_whitespace": trim_whitespace,
            "unify_text": unify_text,
            "numeric_columns": numeric_columns,
            "date_columns": date_columns,
            "missing_numeric": missing_numeric,
            "missing_text": missing_text,
        }
        cleaned_df, log = apply_cleaning(raw_df, options)
        st.session_state["clean_df"] = cleaned_df
        st.session_state["cleaning_log"] = log

    if "clean_df" not in st.session_state:
        st.info("No cleaned dataset yet. Choose your options and click **Apply cleaning**.")
        return

    cleaned_df = st.session_state["clean_df"]
    st.success("✅ Cleaned dataset created. The original data is untouched.")

    st.subheader("What was changed")
    for line in st.session_state["cleaning_log"]:
        st.write(f"- {line}")

    st.subheader("Before vs after")
    before, after = get_overview(raw_df), get_overview(cleaned_df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{after['rows']:,}", f"{after['rows'] - before['rows']:,}")
    c2.metric("Missing cells", f"{after['missing_cells']:,}", f"{after['missing_cells'] - before['missing_cells']:,}", delta_color="inverse")
    c3.metric("Duplicate rows", f"{after['duplicate_rows']:,}", f"{after['duplicate_rows'] - before['duplicate_rows']:,}", delta_color="inverse")

    st.subheader("Cleaned data preview")
    st.dataframe(cleaned_df.head(PREVIEW_ROWS), use_container_width=True)
    st.dataframe(
        pd.DataFrame({"Column": cleaned_df.columns, "Data type": cleaned_df.dtypes.astype(str).values}),
        use_container_width=True,
        hide_index=True,
    )


def show_quality_page() -> None:
    st.title("🧹 Data Quality")
    raw_df = require_data()
    if raw_df is None:
        return

    issues = detect_issues(raw_df)  # calculated once, shared by two tabs

    tab_explore, tab_issues, tab_clean = st.tabs(["🔎 Exploration", "⚠️ Quality issues", "🧼 Cleaning"])
    with tab_explore:
        show_exploration_tab(raw_df)
    with tab_issues:
        show_issues_tab(issues)
    with tab_clean:
        show_cleaning_tab(raw_df, issues)


# ---------- Page: Dashboard ----------
def show_dashboard_page() -> None:
    st.title("📈 KPI Dashboard")
    df, _ = get_active_data("dashboard")
    if df is None:
        return

    roles = detect_roles(df)
    cards, skipped = calculate_kpis(df, roles)

    duplicates = int(df.duplicated().sum())
    if duplicates:
        st.warning(
            f"This dataset contains {duplicates:,} duplicate rows. Totals such as revenue may be "
            "inflated. Remove them in **Data Quality → Cleaning** and choose the cleaned dataset above."
        )

    # Be transparent about how columns were interpreted
    with st.expander("🔍 How the app understood your columns"):
        found = [{"Meaning": role.replace("_", " ").title(), "Column": column}
                 for role, column in roles.items() if column]
        if found:
            st.dataframe(pd.DataFrame(found), use_container_width=True, hide_index=True)
        else:
            st.write("No standard business columns (revenue, product, region ...) were recognised.")
        st.caption("Detection uses column names and data types. Only metrics that can be calculated "
                   "reliably are shown below.")

    # KPI cards, three per row
    st.subheader("Key metrics")
    for start in range(0, len(cards), 3):
        columns = st.columns(3)
        for column, card in zip(columns, cards[start:start + 3]):
            with column:
                with st.container(border=True):
                    st.metric(card["label"], card["value"])
                    st.caption(card["detail"])

    if skipped:
        with st.expander(f"ℹ️ {len(skipped)} metrics could not be calculated"):
            for message in skipped:
                st.write(f"- {message}")

    # Charts
    st.subheader("Charts")
    revenue = roles["revenue"]
    figures = []

    def add_chart(key, builder, note=""):
        try:
            figures.append((key, builder(), note))
        except ValueError as error:
            st.warning(f"Could not draw the {key} chart: {error}")

    if revenue and roles["date"]:
        try:
            trend, invalid = monthly_trend(df, roles["date"], revenue, "sum")
            if not trend.empty:
                note = f"{invalid:,} rows with invalid dates are not included in this chart." if invalid else ""
                add_chart("monthly trend",
                          lambda: line_chart(trend, "Month", trend.columns[1], f"Monthly {revenue}"), note)
        except ValueError as error:
            st.warning(f"Could not draw the monthly trend chart: {error}")

    if revenue and roles["product"]:
        data = aggregate_by_group(df, roles["product"], revenue, "sum", top_n=DEFAULT_TOP_N)
        add_chart("top products",
                  lambda: bar_chart(data, roles["product"], data.columns[1],
                                    f"Top {len(data)} {roles['product']} by {revenue}", horizontal=True))

    if revenue and roles["category"]:
        data = aggregate_by_group(df, roles["category"], revenue, "sum", top_n=None)
        add_chart("category share",
                  lambda: pie_chart(data, roles["category"], data.columns[1],
                                    f"Share of {revenue} by {roles['category']}"))

    if revenue and roles["region"]:
        data = aggregate_by_group(df, roles["region"], revenue, "sum", top_n=None)
        add_chart("regions",
                  lambda: bar_chart(data, roles["region"], data.columns[1],
                                    f"{revenue} by {roles['region']}"))

    if not figures:
        st.info("Charts need a revenue column plus a date, product, category or region column. "
                "You can still build your own charts on the **📊 Visualizations** page.")
        return

    for start in range(0, len(figures), 2):
        columns = st.columns(2)
        for column, (key, fig, note) in zip(columns, figures[start:start + 2]):
            with column:
                st.plotly_chart(fig, use_container_width=True, key=f"dash_{key}")
                if note:
                    st.caption(note)

# ---------- Page: AI Analyst (chat) ----------
def chart_for_calculation(calc: dict):
    """Pick a chart for a calculation result. Returns a Plotly figure or None."""
    table = calc["table"]
    if len(table) < 2:
        return None
    if calc["kind"] == "group":
        return bar_chart(table, table.columns[0], table.columns[1], calc["title"], horizontal=True)
    if calc["kind"] == "trend":
        return line_chart(table, "Month", table.columns[1], calc["title"])
    return None


def table_for_display(table: pd.DataFrame) -> pd.DataFrame:
    """Show month columns as 2026-01 instead of 2026-01-01 00:00:00."""
    shown = table.copy()
    for column in shown.columns:
        if pd.api.types.is_datetime64_any_dtype(shown[column]):
            shown[column] = shown[column].dt.strftime("%Y-%m")
    return shown


def render_assistant_details(message: dict, index: int) -> None:
    """Show the chart and calculation details that belong under an answer."""
    calculations = message.get("calculations", [])
    charts_shown = 0
    for i, calc in enumerate(calculations):
        if charts_shown >= 2:
            break
        fig = chart_for_calculation(calc)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, key=f"chat_chart_{index}_{i}")
            charts_shown += 1

    if calculations:
        with st.expander(f"🔬 Calculations behind this answer ({len(calculations)})"):
            st.caption("Python (Pandas) calculated these tables. The answer text was written from them.")
            for calc in calculations:
                st.markdown(f"**{calc['title']}**")
                for note in calc["notes"]:
                    st.caption(note)
                st.dataframe(table_for_display(calc["table"]), use_container_width=True, hide_index=True)
                st.caption("Request sent to the calculation engine:")
                st.code(json.dumps(calc["inputs"], indent=2, default=str), language="json")
                st.divider()


def show_chat_page() -> None:
    st.title("🤖 Data Analyst Assistant")
    df, label = get_active_data("chat")
    if df is None:
        return

    st.caption("Free rule-based assistant: no API key, no internet. Python calculates every number.")
    with st.expander("ℹ️ What can I ask?"):
        st.markdown(
            "- **Totals / averages / counts**: *What is the total revenue? How many customers?*\n"
            "- **Rankings**: *Top 5 products by revenue. Which region has the lowest sales?*\n"
            "- **Trends**: *Show monthly revenue.*\n"
            "- **Filters**: *Total revenue for Bengaluru. Monthly revenue in 2026.*\n"
            "- **Follow-ups**: *And which was the worst?*\n"
            "- **Summary**: *Give me a summary of this dataset.*\n\n"
            "Use your own column names and values. I can't predict the future, explain causes, "
            "or calculate columns that don't exist (such as profit without a cost column)."
        )

    # Switching dataset starts a fresh conversation
    chat_id = f"{st.session_state.get('dataset_key')}|{label}|{len(df)}x{len(df.columns)}"
    if st.session_state.get("chat_id") != chat_id:
        st.session_state["chat_id"] = chat_id
        st.session_state["chat_display"] = []
        st.session_state["chat_context"] = None

    display = st.session_state["chat_display"]

    if display and st.button("🗑️ Clear conversation"):
        st.session_state["chat_display"] = []
        st.session_state["chat_context"] = None
        st.rerun()

    for index, message in enumerate(display):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_assistant_details(message, index)

    if not display:
        st.write("Ask a question about your data, or try one of these:")
        columns = st.columns(2)
        for i, example in enumerate(example_questions(df)):
            if columns[i % 2].button(example, key=f"example_{i}"):
                st.session_state["pending_question"] = example
                st.rerun()

    question = st.chat_input("Ask a question about your data...")
    if question is None and "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")
    if not question:
        return

    question = question.strip()
    if len(question) > MAX_QUESTION_CHARS:
        st.warning(f"Please keep your question under {MAX_QUESTION_CHARS} characters.")
        return

    display.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    assistant_index = len(display)
    with st.chat_message("assistant"):
        try:
            result = ask_assistant(df, question, st.session_state["chat_context"])
        except AssistantError as error:
            message = {"role": "assistant", "content": f"⚠️ {error}"}
        except Exception as error:  # last-resort safety net so the app never crashes
            message = {"role": "assistant",
                       "content": f"⚠️ Unexpected error ({type(error).__name__}): {error}"}
        else:
            st.session_state["chat_context"] = result["context"]
            message = {
                "role": "assistant",
                "content": result["answer"],
                "calculations": result["calculations"],
            }
        st.markdown(message["content"])
        render_assistant_details(message, assistant_index)
    display.append(message)

# ---------- Page: Visualizations (chart builder) ----------
def show_visualizations_page() -> None:
    st.title("📊 Visualizations")
    df, _ = get_active_data("visuals")
    if df is None:
        return

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    date_cols = find_date_columns(df)
    category_cols = [c for c in df.columns if c not in numeric_cols and c not in date_cols]

    chart_type = st.selectbox(
        "Chart type",
        ["Bar chart", "Line chart (monthly trend)", "Pie chart", "Histogram", "Scatter plot"],
    )

    try:
        if chart_type == "Bar chart":
            build_bar_chart(df, category_cols, numeric_cols)
        elif chart_type == "Line chart (monthly trend)":
            build_line_chart(df, date_cols, numeric_cols)
        elif chart_type == "Pie chart":
            build_pie_chart(df, category_cols, numeric_cols)
        elif chart_type == "Histogram":
            build_histogram(df, numeric_cols)
        else:
            build_scatter_chart(df, category_cols, numeric_cols)
    except ValueError as error:
        st.warning(f"⚠️ {error}")


def pick_value_column(numeric_cols: list, key: str, agg_options: list):
    """Ask for a calculation and (when needed) a numeric column. Returns (agg, value_col)."""
    agg = st.selectbox("Calculation", agg_options, format_func=AGG_LABELS.get, key=f"{key}_agg")
    value_col = None
    if agg != "count":
        if not numeric_cols:
            raise ValueError("This dataset has no numeric columns for that calculation.")
        value_col = st.selectbox("Value column", numeric_cols, key=f"{key}_value")
    return agg, value_col


def build_bar_chart(df, category_cols, numeric_cols) -> None:
    if not category_cols:
        raise ValueError("A bar chart needs at least one text / category column.")
    group_col = st.selectbox("Group by", category_cols, key="bar_group")
    agg, value_col = pick_value_column(numeric_cols, "bar", list(AGG_LABELS))
    top_n = st.slider("How many groups to show", 3, 30, DEFAULT_TOP_N, key="bar_top")
    order = st.radio("Show", ["Highest first", "Lowest first"], horizontal=True, key="bar_order")

    data = aggregate_by_group(df, group_col, value_col, agg, top_n, ascending=(order == "Lowest first"))
    if data.empty:
        raise ValueError("There is no data to chart for this selection.")
    fig = bar_chart(data, group_col, data.columns[1], f"{data.columns[1]} by {group_col}", horizontal=True)
    st.plotly_chart(fig, use_container_width=True, key="viz_bar")
    st.caption(f"{df[group_col].nunique():,} groups in total; showing {len(data)}.")
    with st.expander("📋 Data behind this chart"):
        st.dataframe(data, use_container_width=True, hide_index=True)


def build_line_chart(df, date_cols, numeric_cols) -> None:
    if not date_cols:
        raise ValueError("A trend chart needs a date column. None was found in this dataset.")
    date_col = st.selectbox("Date column", date_cols, key="line_date")
    agg, value_col = pick_value_column(numeric_cols, "line", list(AGG_LABELS))

    data, invalid = monthly_trend(df, date_col, value_col, agg)
    if data.empty:
        raise ValueError("No valid dates were found, so no trend can be drawn.")
    fig = line_chart(data, "Month", data.columns[1], f"{data.columns[1]} per month")
    st.plotly_chart(fig, use_container_width=True, key="viz_line")
    if invalid:
        st.caption(f"{invalid:,} rows with invalid dates are not included.")
    with st.expander("📋 Data behind this chart"):
        st.dataframe(data, use_container_width=True, hide_index=True)


def build_pie_chart(df, category_cols, numeric_cols) -> None:
    if not category_cols:
        raise ValueError("A pie chart needs at least one text / category column.")
    group_col = st.selectbox("Slices (categories)", category_cols, key="pie_group")
    agg, value_col = pick_value_column(numeric_cols, "pie", ["sum", "count"])

    data = aggregate_by_group(df, group_col, value_col, agg, top_n=None)
    fig = pie_chart(data, group_col, data.columns[1], f"{data.columns[1]} by {group_col}")
    st.plotly_chart(fig, use_container_width=True, key="viz_pie")
    st.caption(f"Only positive values are shown. Groups beyond the top {PIE_MAX_SLICES - 1} are combined as 'Other'.")
    with st.expander("📋 Data behind this chart"):
        st.dataframe(data, use_container_width=True, hide_index=True)


def build_histogram(df, numeric_cols) -> None:
    if not numeric_cols:
        raise ValueError("A histogram needs a numeric column. None was found in this dataset.")
    column = st.selectbox("Numeric column", numeric_cols, key="hist_col")
    bins = st.slider("Number of bins", 5, MAX_HISTOGRAM_BINS, 30, key="hist_bins")

    values = df[[column]].dropna()
    if values.empty:
        raise ValueError(f"Column '{column}' has no values to chart.")
    fig = histogram(values, column, f"Distribution of {column}", bins)
    st.plotly_chart(fig, use_container_width=True, key="viz_hist")
    st.caption(f"Based on {len(values):,} non-missing values.")


def build_scatter_chart(df, category_cols, numeric_cols) -> None:
    if len(numeric_cols) < 2:
        raise ValueError("A scatter plot needs at least two numeric columns.")
    x = st.selectbox("X axis", numeric_cols, key="sc_x")
    y = st.selectbox("Y axis", numeric_cols, index=1, key="sc_y")
    color_options = ["None"] + [c for c in category_cols if df[c].nunique() <= 10]
    color = st.selectbox("Colour by (optional)", color_options, key="sc_color")
    color = None if color == "None" else color

    columns = [x, y] + ([color] if color else [])
    plot_df = df[list(dict.fromkeys(columns))].dropna(subset=[x, y])
    if plot_df.empty:
        raise ValueError("There are no rows with values in both selected columns.")

    plot_df, was_sampled = sample_for_scatter(plot_df)
    fig = scatter_plot(plot_df, x, y, f"{y} vs {x}", color)
    st.plotly_chart(fig, use_container_width=True, key="viz_scatter")
    if was_sampled:
        st.caption(f"Large dataset: a random sample of {len(plot_df):,} points is shown.")
    else:
        st.caption(f"Showing all {len(plot_df):,} points.")

# ---------- Page: Business Insights ----------
def show_insights_page() -> None:
    st.title("💡 Business Insights")
    df, _ = get_active_data("insights")
    if df is None:
        return

    insights = build_insights(df)
    st.caption(
        "**Calculated facts** are numbers computed from your data. **Observations** are simple "
        "rule-based patterns. Neither explains causes or predicts the future."
    )

    st.subheader("✅ Calculated facts")
    for text in insights["facts"]:
        st.write(f"- {text}")

    figures = []
    for i, group in enumerate(insights["groups"]):
        table = group["table"].head(10)
        figures.append((f"ins_group_{i}", bar_chart(
            table, group["column"], group["label"],
            f"{group['label']} by {group['column']}", horizontal=True)))
    if insights["trend"]:
        trend = insights["trend"]
        figures.append(("ins_trend", line_chart(
            trend["table"], "Month", trend["label"], f"Monthly {trend['column']}")))
    for start in range(0, len(figures), 2):
        columns = st.columns(2)
        for column, (key, fig) in zip(columns, figures[start:start + 2]):
            with column:
                st.plotly_chart(fig, use_container_width=True, key=key)

    st.subheader("🧭 Observations (rule-based)")
    for text in insights["observations"]:
        st.write(f"- {text}")

    st.subheader("⚠️ Areas needing attention")
    for text in insights["attention"]:
        st.warning(text)


# ---------- Page: Export ----------
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def collect_chat_results() -> dict:
    """Gather every calculation table from the AI Analyst chat, with the question that caused it."""
    sheets, index_rows = {}, []
    question, number = "", 0
    for message in st.session_state.get("chat_display", []):
        if message["role"] == "user":
            question = message["content"]
            continue
        calculations = message.get("calculations", [])
        if not calculations:
            continue
        number += 1
        for i, calc in enumerate(calculations, start=1):
            name = f"Answer {number}.{i}"
            sheets[name] = calc["table"]
            index_rows.append({
                "Sheet": name,
                "Question": question,
                "Calculation": calc["title"],
                "Notes": " | ".join(calc["notes"]),
            })
    if index_rows:
        return {"Index": pd.DataFrame(index_rows), **sheets}
    return {}


def show_export_page() -> None:
    st.title("⬇️ Export")
    df, label = get_active_data("export")
    if df is None:
        return

    stem = safe_stem(st.session_state.get("file_name", "dataset"))
    tag = "cleaned" if label == "cleaned" else "original"
    insights = build_insights(df)
    st.caption("Text cells that start with = + or @ get a leading apostrophe in exported files, "
               "so they cannot run as spreadsheet formulas.")

    # 1. Dataset
    st.subheader("1. Dataset")
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Download as CSV", data=df_to_csv_bytes(df),
                       file_name=f"{stem}_{tag}.csv", mime="text/csv", key="dl_data_csv")
    if len(df) <= EXCEL_EXPORT_MAX_ROWS:
        c2.download_button("⬇️ Download as Excel", data=build_excel({"Data": df}),
                           file_name=f"{stem}_{tag}.xlsx", mime=XLSX_MIME, key="dl_data_xlsx")
    else:
        c2.info(f"Excel export is limited to {EXCEL_EXPORT_MAX_ROWS:,} rows. Use CSV for this dataset.")

    # 2. KPI summary
    st.subheader("2. KPI summary")
    kpi_table = pd.DataFrame(insights["cards"]).rename(
        columns={"label": "Metric", "value": "Value", "detail": "How it was calculated"})
    st.dataframe(kpi_table, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download KPI summary (CSV)", data=df_to_csv_bytes(kpi_table),
                       file_name=f"{stem}_kpi_summary.csv", mime="text/csv", key="dl_kpi")

    # 3. Analysis results from the chat
    st.subheader("3. Analysis results")
    chat_sheets = collect_chat_results()
    if chat_sheets:
        st.write(f"{len(chat_sheets) - 1} calculation table(s) from your AI Analyst conversation.")
        st.download_button("⬇️ Download analysis results (Excel)", data=build_excel(chat_sheets),
                           file_name=f"{stem}_analysis_results.xlsx", mime=XLSX_MIME, key="dl_chat")
    else:
        st.info("No analysis results yet. Ask questions on the **🤖 AI Analyst** page first.")

    # 4. Business insights
    st.subheader("4. Business insights report")
    report_sheets = {
        "KPIs": kpi_table,
        "Insights": pd.DataFrame(
            [{"Type": "Calculated fact", "Statement": t} for t in insights["facts"]]
            + [{"Type": "Rule-based observation", "Statement": t} for t in insights["observations"]]
            + [{"Type": "Needs attention", "Statement": t} for t in insights["attention"]]
        ),
    }
    for group in insights["groups"]:
        report_sheets[f"By {group['column']}"] = group["table"]
    if insights["trend"]:
        report_sheets["Monthly trend"] = insights["trend"]["table"]
    if insights["issues"]:
        report_sheets["Data quality"] = pd.DataFrame(insights["issues"]).rename(columns={
            "severity": "Severity", "category": "Issue", "column": "Column", "count": "Affected values",
            "explanation": "What we found", "recommendation": "Suggested action"})
    if label == "cleaned" and st.session_state.get("cleaning_log"):
        report_sheets["Cleaning log"] = pd.DataFrame({"What was changed": st.session_state["cleaning_log"]})

    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Download full report (Excel)", data=build_excel(report_sheets),
                       file_name=f"{stem}_insights_report.xlsx", mime=XLSX_MIME, key="dl_report_xlsx")
    c2.download_button(
        "⬇️ Download insights (Markdown text)",
        data=build_text_report(st.session_state.get("file_name", "dataset"), label, insights),
        file_name=f"{stem}_insights.md", mime="text/markdown", key="dl_report_md")

def show_coming_soon(page_name: str) -> None:
    st.title(page_name)
    st.info("🚧 This page will be built in a later stage.")


# ---------- Page routing ----------
if page == PAGES[0]:
    show_upload_page()
elif page == PAGES[1]:
    show_quality_page()
elif page == PAGES[2]:
    show_dashboard_page()
elif page == PAGES[3]:
    show_chat_page()
elif page == PAGES[4]:
    show_visualizations_page()
elif page == PAGES[5]:
    show_insights_page()
elif page == PAGES[6]:
    show_export_page()
else:
    show_coming_soon(page)