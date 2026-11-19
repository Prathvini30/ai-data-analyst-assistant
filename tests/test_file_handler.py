"""Tests for file upload validation and loading."""

import io

import pandas as pd
import pytest

import src.file_handler as file_handler
from src.file_handler import FileLoadError, get_sheet_names, load_dataset, validate_file


class FakeUpload(io.BytesIO):
    """Pretends to be the object Streamlit gives us after a file upload."""

    def __init__(self, content: bytes, name: str):
        super().__init__(content)
        self.name = name
        self.size = len(content)


def make_excel(sheets: dict) -> FakeUpload:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, table in sheets.items():
            table.to_excel(writer, sheet_name=name, index=False)
    return FakeUpload(buffer.getvalue(), "book.xlsx")


def test_loads_csv():
    df = load_dataset(FakeUpload(b"Region,Revenue\nA,10\nB,20\n", "sales.csv"))
    assert df.shape == (2, 2)
    assert df["Revenue"].sum() == 30


def test_column_names_are_trimmed():
    df = load_dataset(FakeUpload(b" Region , Revenue \nA,10\n", "x.csv"))
    assert list(df.columns) == ["Region", "Revenue"]


def test_reads_latin1_csv():
    df = load_dataset(FakeUpload("Name\nJosé\n".encode("latin-1"), "names.csv"))
    assert df["Name"].iloc[0] == "José"


@pytest.mark.parametrize("name", ["notes.txt", "report.pdf", "old.xls", "noextension"])
def test_rejects_unsupported_file_types(name):
    with pytest.raises(FileLoadError, match="Unsupported"):
        validate_file(FakeUpload(b"abc", name))


def test_rejects_zero_byte_file():
    with pytest.raises(FileLoadError, match="empty"):
        load_dataset(FakeUpload(b"", "empty.csv"))


def test_rejects_csv_with_only_a_header():
    with pytest.raises(FileLoadError, match="empty"):
        load_dataset(FakeUpload(b"a,b\n", "header_only.csv"))


def test_rejects_csv_with_only_blank_lines():
    with pytest.raises(FileLoadError, match="no data"):
        load_dataset(FakeUpload(b"\n\n", "blank.csv"))


def test_rejects_too_many_rows(monkeypatch):
    monkeypatch.setattr(file_handler, "MAX_ROWS", 3)
    with pytest.raises(FileLoadError, match="maximum"):
        load_dataset(FakeUpload(b"a\n1\n2\n3\n4\n5\n", "big.csv"))


def test_rejects_too_large_file(monkeypatch):
    monkeypatch.setattr(file_handler, "MAX_FILE_SIZE_MB", 0)
    with pytest.raises(FileLoadError, match="maximum"):
        load_dataset(FakeUpload(b"a\n1\n", "large.csv"))


def test_excel_sheet_names_and_selection():
    upload = make_excel({
        "Sales": pd.DataFrame({"x": [1, 2, 3]}),
        "Other": pd.DataFrame({"y": [1, 2], "z": [3, 4]}),
    })
    assert get_sheet_names(upload) == ["Sales", "Other"]
    assert load_dataset(upload, "Sales").shape == (3, 1)
    assert load_dataset(upload, "Other").shape == (2, 2)


def test_corrupted_excel_gives_friendly_error():
    with pytest.raises(FileLoadError):
        get_sheet_names(FakeUpload(b"this is not an excel file", "bad.xlsx"))
    with pytest.raises(FileLoadError):
        load_dataset(FakeUpload(b"this is not an excel file", "bad.xlsx"), "Sheet1")