"""
data_exploration.py
Functions that describe a dataset: size, column types, missing values,
statistics and unique values. These only READ the data; they never change it.
"""

import pandas as pd

from src.config import UNIQUE_VALUES_SHOWN


def get_overview(df: pd.DataFrame) -> dict:
    """Return the headline numbers about a dataset."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def get_column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: data type, missing values and number of unique values."""
    total_rows = len(df)
    missing = df.isna().sum()
    return pd.DataFrame(
        {
            "Column": df.columns,
            "Data type": df.dtypes.astype(str).values,
            "Missing": missing.values,
            "Missing %": (missing / total_rows * 100).round(1).values if total_rows else 0,
            "Unique values": [int(df[column].nunique()) for column in df.columns],
        }
    )


def get_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Statistical summary (count, mean, min, max ...) of numeric columns."""
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()
    return numeric.describe().T.round(2)


def get_unique_values(df: pd.DataFrame, column: str, limit: int = UNIQUE_VALUES_SHOWN) -> pd.DataFrame:
    """The most common values in one column, with how often each appears."""
    counts = df[column].value_counts(dropna=False).head(limit)
    return pd.DataFrame({"Value": counts.index.astype(str), "Count": counts.values})