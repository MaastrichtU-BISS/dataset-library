"""Hostile and oversized files (#122, fixes 10 to 12): every limit is checked before its memory is allocated,
and a refusal says which limit it hit. A zip bomb, a sheet with a cell far out, a deep or wide JSON, and bytes
that are not text are each Unreadable, never a crash or a server that runs on."""
import datetime
import decimal
import io
import json
import time
import zipfile

import openpyxl
import pytest

from dataset_library import tables


def as_rows(table):
    return table.column_names, table.to_pylist()


def refused(data, name):
    with pytest.raises(tables.Unreadable) as caught:
        tables.read(data, name)
    return caught.value.message


def test_a_file_over_the_row_limit_says_which_limit(monkeypatch):
    monkeypatch.setattr(tables, "MAX_ROWS", 3)

    assert "more than 3 rows" in refused(b"text\na\nb\nc\nd\n", "data.csv")


def test_a_file_over_the_column_limit_says_which_limit(monkeypatch):
    monkeypatch.setattr(tables, "MAX_COLUMNS", 2)

    assert "more than 2 columns" in refused(b"a,b,c\n1,2,3\n", "data.csv")


def test_a_file_over_the_cell_limit_says_which_limit(monkeypatch):
    monkeypatch.setattr(tables, "MAX_CELLS", 3)  # two rows of two columns is four cells

    assert "cells" in refused(b"a,b\n1,2\n3,4\n", "data.csv")


def test_a_file_that_expands_past_the_limit_says_which_limit(monkeypatch):
    monkeypatch.setattr(tables, "MAX_EXPANDED_BYTES", 10)

    assert "once it is read" in refused(b"text\nsomething longer than ten bytes\n", "data.csv")


def test_a_json_array_nested_past_the_depth_limit_is_refused_before_it_is_built(monkeypatch):
    monkeypatch.setattr(tables, "MAX_JSON_DEPTH", 2)

    assert "levels deep" in refused(b'[{"a": {"b": {"c": 1}}}]', "data.json")


def test_absurdly_deep_json_is_unreadable_and_does_not_crash():
    data = b"[" * 50_000 + b"]" * 50_000

    assert "levels deep" in refused(data, "data.json")


def test_a_json_value_that_cannot_be_encoded_as_text_is_unreadable():
    # A lone surrogate parses as JSON but has no UTF-8 encoding: refused as unreadable, never a 500.
    with pytest.raises(tables.Unreadable):
        tables.read(b'[{"text": "\\ud800"}]', "data.json")


def test_a_wide_sparse_json_is_bounded_by_the_column_limit(monkeypatch):
    monkeypatch.setattr(tables, "MAX_COLUMNS", 3)
    data = json.dumps([{f"k{index}": 1} for index in range(10)]).encode()

    assert "more than 3 columns" in refused(data, "data.json")


def test_json_bytes_that_are_not_utf8_or_cp1252_are_unreadable():
    with pytest.raises(tables.Unreadable):
        tables.read(b'[{"text": "\x81\x8d\x8f\x90"}]', "data.json")


def test_a_zip_that_expands_far_past_its_size_is_refused_as_a_zip_bomb():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/worksheets/sheet1.xml", b"0" * 5_000_000)

    assert "expands more than 100 times" in refused(buffer.getvalue(), "data.xlsx")


def test_a_zip_whose_declared_size_is_past_the_limit_is_refused_before_it_is_read(monkeypatch):
    monkeypatch.setattr(tables, "MAX_EXPANDED_BYTES", 1_000)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("xl/worksheets/sheet1.xml", b"0" * 5_000)

    assert "once it is read" in refused(buffer.getvalue(), "data.xlsx")


def test_an_xlsx_with_a_cell_far_out_is_read_quickly_not_walked_row_by_row():
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(["text", "label"])
    sheet.append(["A", "x"])
    sheet["XFD1048576"] = "far away"
    buffer = io.BytesIO()
    book.save(buffer)

    started = time.monotonic()
    message = refused(buffer.getvalue(), "data.xlsx")

    assert time.monotonic() - started < 10
    assert "columns" in message  # the row runs to column XFD, which is past the column limit


def test_an_xlsx_dates_and_times_are_written_as_iso_text_with_no_quotes():
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(["when", "at"])
    sheet.append([datetime.date(2026, 11, 19), datetime.datetime(2026, 11, 19, 9, 30)])
    sheet.append(["later", datetime.time(9, 30)])
    buffer = io.BytesIO()
    book.save(buffer)

    # openpyxl reads a date back as a datetime at midnight: its ISO text, with no JSON quotes around it.
    assert as_rows(tables.read(buffer.getvalue(), "data.xlsx")) == (
        ["when", "at"],
        [{"when": "2026-11-19T00:00:00", "at": "2026-11-19T09:30:00"}, {"when": "later", "at": "09:30:00"}],
    )


def test_a_decimal_is_written_as_its_text_and_a_date_as_iso_without_json_quotes():
    assert tables._text_value(decimal.Decimal("1.50")) == "1.50"
    assert tables._text_value(datetime.date(2026, 1, 2)) == "2026-01-02"
    assert tables._text_value(datetime.datetime(2026, 1, 2, 3, 4, 5)) == "2026-01-02T03:04:05"
    assert tables._text_value(True) == "true"
    assert tables._text_value(float("nan")) is None
