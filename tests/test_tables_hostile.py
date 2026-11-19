"""Hostile files are refused in bounded time and memory (#122, fixes 1 to 7 of the second rework). Each refusal is
Unreadable and says which limit it hit, and each one is refused quickly: a JSON string that never closes, and a
sheet with a row far down it, take well under the time a request or a worker can spare."""
import io
import pathlib
import time

import openpyxl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dataset_library import tables
from dataset_library.providers.community import UNREADABLE

HOSTILE = pathlib.Path(__file__).parent / "fixtures" / "hostile"
# What publishing_host._dataset_bytes writes for a labelled Dataset, named data.json (generated with pandas 2.3.3).
BAIGLE_LABELLED = (
    b'text,label,label_source\n"The Supplier may terminate, on notice.",Terminations,file\n'
    b'"A paragraph.\n\nAnother paragraph.",01,file\n"Unlabelled text, with a comma.",,\n'
)


def refused(data, name):
    with pytest.raises(tables.Unreadable) as caught:
        tables.read(data, name)
    return caught.value.message


def seconds_of(function):
    started = time.monotonic()
    function()
    return time.monotonic() - started


# --- Fix 1: the JSON nesting scan is linear -------------------------------------------------------

@pytest.mark.parametrize("pairs", [20_000, 100_000])
def test_a_string_of_escaped_quotes_that_never_closes_is_refused_in_under_a_second(pairs):
    data = b'["' + b'\\"' * pairs

    seconds = seconds_of(lambda: refused(data, "data.json"))

    assert seconds < 1, f"the scan took {seconds:.2f}s on {len(data)} bytes"


def test_a_string_of_escaped_quotes_under_a_name_less_file_is_read_in_under_a_second():
    data = b'["' + b'\\"' * 20_000

    def read():
        try:
            tables.read(data, "")
        except tables.Unreadable:
            pass  # read as CSV, its header is one field, so it is a table with no rows, or refused

    assert seconds_of(read) < 1


# --- Fix 2: XLSX rows are counted as they are scanned, blank ones included ---------------------------

def test_an_xlsx_with_a_row_far_down_the_sheet_is_refused_at_the_sheet_row_limit_and_quickly():
    data = (HOSTILE / "row-gap.xlsx").read_bytes()
    assert len(data) < 2_000

    message = refused(data, "data.xlsx")

    assert "1,048,576 rows" in message
    assert seconds_of(lambda: refused(data, "data.xlsx")) < 10


def test_blank_rows_count_toward_the_sheet_row_limit(monkeypatch):
    monkeypatch.setattr(tables, "MAX_SHEET_ROWS", 3)
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(["text"])
    sheet.append(["A"])
    sheet.append([None])
    sheet.append([None])
    sheet.append(["B"])
    buffer = io.BytesIO()
    book.save(buffer)

    assert "more than 3 rows" in refused(buffer.getvalue(), "data.xlsx")


def test_a_sheet_within_the_row_limit_still_reads_its_rows_and_skips_blank_ones(monkeypatch):
    monkeypatch.setattr(tables, "MAX_SHEET_ROWS", 5)
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(["text", "label"])
    sheet.append(["A", "x"])
    sheet.append([None, None])
    sheet.append(["B", "y"])
    buffer = io.BytesIO()
    book.save(buffer)

    table = tables.read(buffer.getvalue(), "data.xlsx")

    assert table.column_names == ["text", "label"]
    assert table.to_pylist() == [{"text": "A", "label": "x"}, {"text": "B", "label": "y"}]


# --- Fix 3: JSON is refused at its own ceiling, and every row and cell is counted as it is built -----

def test_a_json_file_over_the_json_ceiling_is_refused_before_it_is_parsed(monkeypatch):
    monkeypatch.setattr(tables, "MAX_JSON_BYTES", 40)

    def parsed(*args, **kwargs):
        raise AssertionError("a JSON file over the ceiling was parsed")

    monkeypatch.setattr(tables.json, "loads", parsed)
    data = b'[{"text": "a", "label": "x"}, {"text": "b", "label": "y"}]'

    assert "This JSON file is more than" in refused(data, "data.json")


def test_a_name_less_file_that_opens_like_json_over_the_ceiling_is_refused_not_read_as_csv(monkeypatch):
    monkeypatch.setattr(tables, "MAX_JSON_BYTES", 40)
    data = b"[" + b'"x",' * 30 + b'"x"]'

    assert "This JSON file is more than" in refused(data, "")


def test_a_labelled_csv_over_the_json_ceiling_under_a_json_name_is_read_as_csv(monkeypatch):
    monkeypatch.setattr(tables, "MAX_JSON_BYTES", 40)
    data = BAIGLE_LABELLED + b'"More text, again",Terminations,file\n' * 3

    table = tables.read(data, "data.json")

    assert table.column_names == ["text", "label", "label_source"]
    assert table.num_rows == 6


def test_a_json_record_wider_than_the_cell_limit_is_refused_as_its_rows_are_counted(monkeypatch):
    monkeypatch.setattr(tables, "MAX_CELLS", 10)
    data = b"[" + b",".join(b'{"c%d": "x"}' % index for index in range(12)) + b"]"

    assert "cells" in refused(data, "data.json")


# --- Fix 4: Parquet is refused from its metadata, before any column is read ----------------------------

def test_a_parquet_file_over_the_cell_limit_is_refused_before_any_column_is_read(monkeypatch, tmp_path):
    path = tmp_path / "wide.parquet"
    pq.write_table(pa.table({"a": ["x"] * 10, "b": ["y"] * 10, "c": ["z"] * 10}), path)
    data = path.read_bytes()
    monkeypatch.setattr(tables, "MAX_CELLS", 20)

    def read(self, *args, **kwargs):
        raise AssertionError("the Parquet columns were read")

    monkeypatch.setattr(pq.ParquetFile, "read", read)

    assert "cells" in refused(data, "data.parquet")


def test_a_parquet_file_whose_row_groups_decode_past_the_limit_is_refused_before_it_is_read(monkeypatch, tmp_path):
    path = tmp_path / "big.parquet"
    # Unique strings, so the column is not dictionary-encoded down to a few bytes: its decoded size is what the metadata says.
    pq.write_table(pa.table({"text": [f"{index:04d}" + "x" * 1000 for index in range(50)]}), path)
    data = path.read_bytes()
    monkeypatch.setattr(tables, "MAX_EXPANDED_BYTES", 10_000)

    def read(self, *args, **kwargs):
        raise AssertionError("the Parquet columns were read")

    monkeypatch.setattr(pq.ParquetFile, "read", read)

    assert "more than 0 MB once it is read" in refused(data, "data.parquet")


# --- Fix 5: a name that cannot be written is unreadable, never an encoding error ----------------------

@pytest.mark.parametrize("data", [b'[{"\\ud800": "a"}]', b'[{"text": "a", "\\udfff": "b"}]'])
def test_a_lone_surrogate_in_a_json_column_name_is_unreadable_under_a_json_name(data):
    assert refused(data, "data.json") == UNREADABLE


def test_a_lone_surrogate_in_a_json_column_name_is_unreadable_with_no_name_either():
    assert refused(b'[{"\\ud800": "a"}]', "") == UNREADABLE


def test_a_json_value_that_is_a_lone_surrogate_is_unreadable_as_a_cell():
    assert refused(b'[{"text": "\\ud800"}]', "data.json") == UNREADABLE


# --- Fix 6: CSV line endings and empty header cells ----------------------------------------------------

def test_a_csv_whose_lines_end_in_a_bare_carriage_return_reads_every_row():
    data = b"text,label\rA,x\rB,y\r"

    assert tables.read(data, "data.csv").to_pylist() == [{"text": "A", "label": "x"}, {"text": "B", "label": "y"}]
    assert tables.read(data, "").column_names == ["text", "label"]


def test_an_empty_header_cell_is_named_for_its_column_as_pandas_names_it():
    table = tables.read(b"text,,label\nA,x,y\n", "data.csv")

    assert table.column_names == ["text", "Unnamed: 1", "label"]


def test_an_empty_header_cell_on_a_file_with_bare_carriage_returns_is_named_too():
    assert tables.read(b"text,\rA,x\r", "data.csv").column_names == ["text", "Unnamed: 1"]


# --- Fix 7: a labelled CSV under a JSON name --------------------------------------------------------

def test_the_bytes_the_publisher_writes_under_a_json_name_read_as_its_csv():
    table = tables.read(BAIGLE_LABELLED, "data.json")

    assert table.column_names == ["text", "label", "label_source"]
    assert table.column("label").to_pylist() == ["Terminations", "01", ""]


def test_a_broken_json_document_under_a_json_name_is_not_read_as_csv():
    assert refused(b'[{"text": "a", "label": "x"},', "data.json") == UNREADABLE
    assert refused(b'{"text": "a"', "data.jsonl") == UNREADABLE


def test_a_file_of_numbers_named_json_is_read_as_csv_with_its_header_rules():
    # Not JSON (one value per line, not an object), so the CSV rules read it: its header is "1" and it has two rows.
    table = tables.read(b"1\n2\n3\n", "data.json")

    assert table.column_names == ["1"]
    assert table.num_rows == 2


def test_the_publisher_bytes_round_trip_through_the_sampler_format(tmp_path):
    path = tmp_path / "sample.parquet"

    tables.write_parquet(BAIGLE_LABELLED, "data.json", str(path))

    assert pq.read_table(path).column("label").to_pylist() == ["Terminations", "01", ""]
