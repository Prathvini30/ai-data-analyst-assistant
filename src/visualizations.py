"""
visualizations.py
Builds interactive Plotly charts from data that Python has already calculated.
Every function takes a DataFrame and returns a Plotly figure. Charts only ever
show real values from the dataset.
"""

import pandas as pd
import plotly.express as px

from src.config import MAX_SCATTER_POINTS, PIE_MAX_SLICES


def _style(fig, title: str):
    """One consistent look for every chart."""
    fig.update_layout(
        title=title,
        title_font_size=18,
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def bar_chart(data: pd.DataFrame, x: str, y: str, title: str, horizontal: bool = False):
    """Bar chart. x = categories, y = values. Horizontal puts the first row at the top."""
    if horizontal:
        fig = px.bar(data, x=y, y=x, orientation="h", text_auto=".3s")
        fig.update_yaxes(type="category", autorange="reversed")
    else:
        fig = px.bar(data, x=x, y=y, text_auto=".3s")
        fig.update_xaxes(type="category")
    return _style(fig, title)


def line_chart(data: pd.DataFrame, x: str, y: str, title: str):
    """Line chart with markers. Monthly dates get month labels."""
    fig = px.line(data, x=x, y=y, markers=True)
    if pd.api.types.is_datetime64_any_dtype(data[x]):
        fig.update_xaxes(tickformat="%b %Y")
        if len(data) <= 24:
            fig.update_xaxes(dtick="M1")
    return _style(fig, title)


def pie_chart(data: pd.DataFrame, names: str, values: str, title: str):
    """Pie (donut) chart. Only positive values are shown; small slices become 'Other'."""
    positive = data[data[values] > 0].sort_values(values, ascending=False)
    if positive.empty:
        raise ValueError("There are no positive values to show in a pie chart.")
    if len(positive) > PIE_MAX_SLICES:
        top = positive.head(PIE_MAX_SLICES - 1)
        other_total = positive.iloc[PIE_MAX_SLICES - 1:][values].sum()
        other = pd.DataFrame({names: ["Other"], values: [other_total]})
        positive = pd.concat([top, other], ignore_index=True)
    fig = px.pie(positive, names=names, values=values, hole=0.4)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return _style(fig, title)


def histogram(data: pd.DataFrame, column: str, title: str, bins: int = 30):
    """Histogram: how many rows fall into each range of values."""
    fig = px.histogram(data, x=column, nbins=bins)
    fig.update_layout(bargap=0.05, yaxis_title="Number of rows")
    return _style(fig, title)


def sample_for_scatter(data: pd.DataFrame):
    """Large datasets are sampled for scatter plots so the browser stays fast."""
    if len(data) > MAX_SCATTER_POINTS:
        return data.sample(MAX_SCATTER_POINTS, random_state=0), True
    return data, False


def scatter_plot(data: pd.DataFrame, x: str, y: str, title: str, color: str = None):
    """Scatter plot of two numeric columns, optionally coloured by a category."""
    fig = px.scatter(data, x=x, y=y, color=color, opacity=0.7)
    return _style(fig, title)