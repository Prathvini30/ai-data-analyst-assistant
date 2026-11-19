"""
file_handler.py
Handles everything about getting a file into Pandas:
validating it, reading CSV/Excel, and returning a clean DataFrame.

A DataFrame is Pandas' version of a spreadsheet table.
"""

import os

import pandas as pd

from src.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, MAX_ROWS


class FileLoadError(Exception):
    """Raised when a file cannot be loaded. The message is safe to show to users."""


def get_extension(filename: str) -> str:
    """Return the lowercase file extension, e.g. '.csv'."""
    return os.path.splitext(filename)[1].lower()


def validate_file(uploaded_file) -> None:
    """
    Check the uploaded file before reading it.
    Raises FileLoadError with a friendly message if something is wrong.
    """
    extension = get_extension(uploaded_file.name)

    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(ALLOWED_EXTENSIONS)
        raise FileLoadError(
            f"Unsupported file type '{extension}'. Please upload one of: {allowed}."
        )

    if uploaded_file.size == 0:
        raise FileLoadError("The uploaded file is empty (0 bytes).")

    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise FileLoadError(
            f"File is {size_mb:.1f} MB. The maximum allowed size is {MAX_FILE_SIZE_MB} MB."
        )


def get_sheet_names(uploaded_file) -> list:
    """Return the list of worksheet names in an Excel file."""
    try:
        uploaded_file.seek(0)  # go back to the start of the file
        with pd.ExcelFile(uploaded_file, engine="openpyxl") as excel_file:
            return excel_file.sheet_names
    except Exception as error:
        raise FileLoadError(
            "Could not read this Excel file. It may be corrupted or password-protected."
        ) from error


def _read_csv(uploaded_file) -> pd.DataFrame:
    """Read a CSV file, trying common text encodings."""
    # 'utf-8-sig' handles normal UTF-8 and files saved by Excel (which add a hidden marker).
    # 'latin-1' is a fallback for older files; it can read any byte sequence.
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=encoding)
        except UnicodeDecodeError:
            continue  # try the next encoding
        except pd.errors.EmptyDataError as error:
            raise FileLoadError("The CSV file contains no data.") from error
        except pd.errors.ParserError as error:
            raise FileLoadError(
                "The CSV file is malformed and could not be parsed. "
                "Please check that it is a properly formatted CSV."
            ) from error
    raise FileLoadError("Could not decode the CSV file. Try saving it as UTF-8.")


def _read_excel(uploaded_file, sheet_name) -> pd.DataFrame:
    """Read one worksheet from an Excel file."""
    try:
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file, sheet_name=sheet_name, engine="openpyxl")
    except Exception as error:
        raise FileLoadError(
            f"Could not read worksheet '{sheet_name}'. The file may be corrupted."
        ) from error


def load_dataset(uploaded_file, sheet_name=None) -> pd.DataFrame:
    """
    Validate and load an uploaded CSV or Excel file into a DataFrame.

    uploaded_file: the object returned by Streamlit's file_uploader
    sheet_name:    worksheet to read (only used for Excel files)
    """
    validate_file(uploaded_file)

    if get_extension(uploaded_file.name) == ".csv":
        df = _read_csv(uploaded_file)
    else:
        df = _read_excel(uploaded_file, sheet_name or 0)

    # Safeguards
    if df.empty or len(df.columns) == 0:
        raise FileLoadError("The dataset is empty. Please upload a file that contains data.")

    if len(df) > MAX_ROWS:
        raise FileLoadError(
            f"The dataset has {len(df):,} rows. The maximum supported is {MAX_ROWS:,}."
        )

    # Tidy column names: remove accidental spaces at the start/end
    df.columns = [str(column).strip() for column in df.columns]

    return df