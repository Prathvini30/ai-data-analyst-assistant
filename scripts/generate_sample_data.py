"""
generate_sample_data.py
Creates a realistic sample sales dataset WITH deliberate data problems,
so you can test the data-quality checks, cleaning and dashboards.

Run from the project folder:
    python scripts\\generate_sample_data.py

Creates:  data/sample_sales.csv  and  data/sample_sales.xlsx
The random seed is fixed, so every run produces the same file.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ORDERS = 6000
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "sample_sales.csv"
XLSX_PATH = DATA_DIR / "sample_sales.xlsx"

# (product, category, typical price in rupees)
PRODUCTS = [
    ("Laptop", "Electronics", 45000),
    ("Phone", "Electronics", 20000),
    ("Tablet", "Electronics", 25000),
    ("Monitor", "Electronics", 12000),
    ("Headphones", "Accessories", 2500),
    ("Keyboard", "Accessories", 1500),
    ("Mouse", "Accessories", 500),
    ("Printer", "Office", 9000),
    ("Office Chair", "Office", 7500),
    ("Desk Lamp", "Office", 1200),
]
PRODUCT_WEIGHTS = [0.06, 0.10, 0.06, 0.08, 0.14, 0.14, 0.20, 0.06, 0.04, 0.12]  # sums to 1.0

REGIONS = ["Bengaluru", "Mysuru", "Hubballi", "Belagavi", "Mangaluru", "Dharwad"]
REGION_WEIGHTS = [0.35, 0.15, 0.15, 0.12, 0.13, 0.10]

PAYMENTS = ["UPI", "Card", "Cash", "Net Banking"]
PAYMENT_WEIGHTS = [0.45, 0.30, 0.15, 0.10]

# Sales grow slowly from January to October 2026
_MONTH_RAW = np.array([0.70, 0.75, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25])
MONTH_PROBABILITIES = _MONTH_RAW / _MONTH_RAW.sum()


def build_clean_data(rng) -> pd.DataFrame:
    """Step 1: a perfectly clean dataset where Revenue = Quantity x Unit_Price."""
    n = N_ORDERS
    picks = rng.choice(len(PRODUCTS), size=n, p=PRODUCT_WEIGHTS)
    names = np.array([p[0] for p in PRODUCTS])[picks]
    categories = np.array([p[1] for p in PRODUCTS])[picks]
    base_prices = np.array([p[2] for p in PRODUCTS])[picks]

    months = rng.choice(np.arange(1, 11), size=n, p=MONTH_PROBABILITIES)
    days = rng.integers(1, 29, size=n)
    dates = pd.to_datetime(pd.DataFrame({"year": 2026, "month": months, "day": days}))

    quantity = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.40, 0.25, 0.18, 0.10, 0.07])
    unit_price = (np.round(base_prices * rng.uniform(0.95, 1.05, size=n) / 10) * 10).astype(int)
    customers = [f"Customer {number:04d}" for number in rng.integers(1, 1201, size=n)]

    return pd.DataFrame(
        {
            "Order_ID": [f"ORD{number:05d}" for number in range(1, n + 1)],
            "Order_Date": dates.dt.strftime("%Y-%m-%d"),
            "Customer": customers,
            "Product": names,
            "Category": categories,
            "Quantity": quantity,
            "Unit_Price": unit_price,
            "Revenue": quantity * unit_price,
            "Region": rng.choice(REGIONS, size=n, p=REGION_WEIGHTS),
            "Payment_Method": rng.choice(PAYMENTS, size=n, p=PAYMENT_WEIGHTS),
        }
    )


def add_problems(df: pd.DataFrame, rng) -> pd.DataFrame:
    """Step 2: plant deliberate problems. Each problem uses different rows (no overlap)."""
    df = df.copy()
    shuffled_rows = rng.permutation(len(df))
    cursor = 0

    def take(count):
        nonlocal cursor
        chosen = shuffled_rows[cursor:cursor + count]
        cursor += count
        return chosen

    # 5 returns: negative quantity and negative revenue
    returns = take(5)
    df.loc[returns, "Quantity"] = -1
    df.loc[returns, "Revenue"] = -df.loc[returns, "Unit_Price"]

    # 3 huge orders (outliers)
    outliers = take(3)
    df.loc[outliers, "Quantity"] = 200
    df.loc[outliers, "Revenue"] = 200 * df.loc[outliers, "Unit_Price"]

    # Revenue becomes a type that can hold blanks without turning into 90000.0
    df["Revenue"] = df["Revenue"].astype("Int64")

    # Inconsistent spelling and extra spaces in Region
    lower = take(30)
    df.loc[lower, "Region"] = df.loc[lower, "Region"].str.lower()
    spaced = take(20)
    df.loc[spaced, "Region"] = df.loc[spaced, "Region"] + " "

    # Missing values
    df.loc[take(60), "Revenue"] = pd.NA
    df.loc[take(25), "Region"] = np.nan
    df.loc[take(15), "Customer"] = np.nan

    # Invalid dates
    df.loc[take(8), "Order_Date"] = "not available"

    # 40 exact duplicate rows
    duplicates = df.sample(40, random_state=SEED)
    df = pd.concat([df, duplicates]).sort_values("Order_ID", kind="stable").reset_index(drop=True)
    return df


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = add_problems(build_clean_data(rng), rng)

    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    excel_ready = df.astype(object).where(df.notna(), None)   # blanks stay blank in Excel
    excel_ready.to_excel(XLSX_PATH, index=False)

    deduplicated = df.drop_duplicates()
    print(f"Created {CSV_PATH.name} and {XLSX_PATH.name} in the data folder.\n")
    print("Numbers to check against the app")
    print("--------------------------------")
    print(f"Rows in the file:                    {len(df):,}")
    print(f"Unique orders (Order_ID):            {df['Order_ID'].nunique():,}")
    print(f"Exact duplicate rows:                {int(df.duplicated().sum()):,}")
    print(f"Blank Revenue cells:                 {int(df['Revenue'].isna().sum()):,}")
    print(f"Invalid dates ('not available'):     {int((df['Order_Date'] == 'not available').sum()):,}")
    print(f"Total Revenue, raw (with duplicates):    {int(df['Revenue'].sum()):,}")
    print(f"Total Revenue, after removing duplicates: {int(deduplicated['Revenue'].sum()):,}")


if __name__ == "__main__":
    main()