"""
conftest.py
Shared test datasets (pytest "fixtures"). Every test that lists a fixture name
as an argument automatically receives a fresh copy of that dataset.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def kpi_df():
    """The clean 6-row dataset from data/test_kpi.csv (totals are easy to check by hand)."""
    return pd.DataFrame(
        {
            "Order_ID": ["ORD001", "ORD002", "ORD003", "ORD004", "ORD005", "ORD006"],
            "Order_Date": ["2026-01-05", "2026-01-18", "2026-02-07", "2026-02-19", "2026-03-09", "2026-03-21"],
            "Customer": ["Customer A", "Customer B", "Customer C", "Customer D", "Customer E", "Customer F"],
            "Product": ["Laptop", "Mouse", "Phone", "Keyboard", "Monitor", "Mouse"],
            "Category": ["Electronics", "Accessories", "Electronics", "Accessories", "Electronics", "Accessories"],
            "Quantity": [2, 5, 1, 3, 2, 4],
            "Unit_Price": [45000, 500, 20000, 1500, 12000, 500],
            "Revenue": [90000, 2500, 20000, 4500, 24000, 2000],
            "Region": ["Bengaluru", "Mysuru", "Bengaluru", "Hubballi", "Mysuru", "Bengaluru"],
            "Payment_Method": ["UPI", "Card", "UPI", "Cash", "Card", "UPI"],
        }
    )


@pytest.fixture
def messy_df():
    """The 12-row dataset from data/test_messy.csv, with deliberate problems."""
    return pd.DataFrame(
        {
            "Order_ID": ["ORD001", "ORD002", "ORD003", "ORD004", "ORD005", "ORD006",
                         "ORD007", "ORD008", "ORD009", "ORD010", "ORD010", "ORD011"],
            "Order_Date": ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-10",
                           "not a date", "2026-01-12", "2026-01-13", "2026-01-14", "2026-01-14", "2026-01-15"],
            "Customer": ["Customer A", "Customer B", "Customer C", "Customer D", "Customer E", "Customer F",
                         "Customer G", "Customer H", "Customer I", "Customer J", "Customer J", "Customer K"],
            "Product": ["Laptop", "Mouse", "Phone", "Keyboard", "Monitor", "Mouse",
                        "Laptop", "Phone", "Keyboard", "Monitor", "Monitor", "Mouse"],
            "Category": ["Electronics", "Accessories", "Electronics", "Accessories", "Electronics", "Accessories",
                         "Electronics", "Electronics", "Accessories", "Electronics", "Electronics", "Accessories"],
            "Quantity": [2, 5, 1, 3, 2, 4, 1, -2, 2, 1, 1, 1000],
            "Unit_Price": [45000, 500, 20000, 1500, 12000, 500, 45000, 20000, 1500, 12000, 12000, 500],
            "Revenue": [90000, 2500, 20000, 4500, 24000, 2000, 45000, -40000, np.nan, 12000, 12000, 500000],
            "Region": ["Bengaluru", "Mysuru", "bengaluru", "Hubballi ", "Mysuru", "Bengaluru",
                       "Hubballi", "Mysuru", "Bengaluru", "Bengaluru", "Bengaluru", "Mysuru"],
            "Payment_Method": ["UPI", "Card", "UPI", "Cash", "Card", "UPI",
                               "Card", "UPI", "Cash", "UPI", "UPI", "Card"],
        }
    )


@pytest.fixture
def generic_df():
    """A dataset that is NOT sales data."""
    return pd.DataFrame(
        {
            "Student": ["Asha", "Ravi", "Meena", "Kiran"],
            "Subject": ["Maths", "Maths", "Science", "Science"],
            "Marks": [82, 67, 91, 55],
            "Hours_Studied": [10, 6, 12, 4],
        }
    )