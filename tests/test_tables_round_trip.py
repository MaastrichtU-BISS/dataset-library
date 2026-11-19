"""Reading the files BAIGLE accepts and publishes (#122, fixes 11 to 15), and writing them as the sampler reads them.

BAIGLE uploads CSV, XLSX and JSON, which its upload reads with pandas, and publishes a labelled one as the CSV
pandas writes (`to_csv(index=False)`), with `label_source` beside the label. The fixtures here are written the
same way, so a round trip through the reader is the round trip a student's file takes."""
import csv
import datetime
import io
import json

import openpyxl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dataset_library import tables


def as_rows(table):
    return table.column_names, table.to_pylist()


def pandas_csv(rows, columns):
    """`to_csv(index=False)`: standard quoting, a quoted field where a value has a comma, quote or line break,
    and an empty field for a missing value."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow(["" if value is None else value for value in row])
    return buffer.getvalue().encode()


# --- CSV (fix 13) --------------------------------------------------------------------------------

def test_a_csv_with_quoted_paragraph_breaks_reads_every_row():
    rows = [[f"Paragraph one of {index}.\n\nParagraph two of {index}.", "A" if index % 2 else "B"] for index in range(400)]

    table = tables.read(pandas_csv(rows, ["text", "label"]), "data.csv")

    assert table.num_rows == 400
    assert table.column("text")[399].as_py() == "Paragraph one of 399.\n\nParagraph two of 399."
    assert table.column("label")[0].as_py() == "B"


def test_the_header_is_read_by_a_csv_parser_so_a_quoted_comma_or_line_break_is_one_name():
    assert tables.read(b'"text, with a comma",label\nA,x\n', "data.csv").column_names == ["text, with a comma", "label"]
    assert tables.read(b'"text\nlines",label\nA,x\n', "data.csv").column_names == ["text\nlines", "label"]


def test_a_csv_whose_header_starts_with_a_bracket_is_csv_when_its_name_says_so():
    assert as_rows(tables.read(b"[text],label\nA,x\n", "data.csv")) == (["[text]", "label"], [{"[text]": "A", "label": "x"}])


def test_a_csv_whose_header_starts_with_a_bracket_is_csv_even_with_no_name():
    assert as_rows(tables.read(b"[text],label\nA,x\n", "")) == (["[text]", "label"], [{"[text]": "A", "label": "x"}])


def test_a_header_with_no_line_end_is_a_table_with_no_rows():
    assert as_rows(tables.read(b"text,label", "data.csv")) == (["text", "label"], [])


def test_a_windows_1252_csv_keeps_its_accents_and_a_utf8_byte_order_mark_is_dropped():
    assert as_rows(tables.read("text\ncafé\n".encode("cp1252"), "data.csv")) == (["text"], [{"text": "café"}])
    assert as_rows(tables.read(b"\xef\xbb\xbftext\nA\n", "data.csv")) == (["text"], [{"text": "A"}])


def test_a_json_name_is_never_read_as_a_csv():
    with pytest.raises(tables.Unreadable):
        tables.read(b"text,label\nA,x\n", "data.json")


# --- JSON layouts BAIGLE's upload reads (fix 12) --------------------------------------------------

def test_pandas_column_layout_is_read_as_columns():
    data = json.dumps({"text": {"0": "a", "1": "b"}, "label": {"0": "x", "1": "y"}}).encode()

    assert as_rows(tables.read(data, "data.json")) == (
        ["text", "label"],
        [{"text": "a", "label": "x"}, {"text": "b", "label": "y"}],
    )


def test_a_column_layout_with_sparse_row_keys_keeps_each_value_in_its_row():
    data = json.dumps({"text": {"0": "a", "2": "c"}, "label": {"0": "x", "1": "y"}}).encode()

    assert as_rows(tables.read(data, "data.json")) == (
        ["text", "label"],
        [{"text": "a", "label": "x"}, {"text": "c", "label": None}, {"text": None, "label": "y"}],
    )


def test_a_list_of_records_is_read_one_row_per_object():
    data = json.dumps([{"text": "a", "label": "Notices"}, {"text": "b", "label": "01"}]).encode()

    assert as_rows(tables.read(data, "data.json")) == (
        ["text", "label"],
        [{"text": "a", "label": "Notices"}, {"text": "b", "label": "01"}],
    )


def test_a_single_object_is_one_row():
    assert as_rows(tables.read(b'{"text": "a", "label": 1}', "data.json")) == (["text", "label"], [{"text": "a", "label": "1"}])


def test_a_json_lines_file_named_json_is_read_one_object_per_line():
    assert as_rows(tables.read(b'{"text": "a"}\n{"text": "b"}\n', "data.json")) == (["text"], [{"text": "a"}, {"text": "b"}])


# --- Parquet (fix 14) ----------------------------------------------------------------------------

def test_parquet_columns_are_cast_to_text_in_arrow_and_nested_values_are_written_as_json():
    buffer = io.BytesIO()
    pq.write_table(pa.table({"text": ["a"], "n": [3], "tags": [["x", "y"]]}), buffer)

    assert as_rows(tables.read(buffer.getvalue(), "data.parquet")) == (
        ["text", "n", "tags"],
        [{"text": "a", "n": "3", "tags": '["x", "y"]'}],
    )


def test_a_parquet_file_past_the_row_limit_is_refused_from_its_metadata(monkeypatch):
    monkeypatch.setattr(tables, "MAX_ROWS", 2)
    buffer = io.BytesIO()
    pq.write_table(pa.table({"text": ["a", "b", "c"]}), buffer)

    with pytest.raises(tables.Unreadable) as caught:
        tables.read(buffer.getvalue(), "data.parquet")

    assert "more than 2 rows" in caught.value.message


# --- Round trips, as BAIGLE publishes and uploads them -------------------------------------------

def test_a_labelled_dataset_as_baigle_publishes_it_reads_back_with_its_labels_as_written(tmp_path):
    published = pandas_csv(
        [
            ["The Supplier may terminate, on notice.", "Terminations", "file"],
            ["A paragraph.\n\nAnother paragraph.", "01", "file"],
            ["Unlabelled text.", None, None],
        ],
        ["text", "label", "label_source"],
    )
    path = tmp_path / "sample.parquet"

    tables.write_parquet(published, "data.csv", str(path))

    back = pq.read_table(path).to_pylist()
    assert back == [
        {"text": "The Supplier may terminate, on notice.", "label": "Terminations", "label_source": "file"},
        {"text": "A paragraph.\n\nAnother paragraph.", "label": "01", "label_source": "file"},
        {"text": "Unlabelled text.", "label": "", "label_source": ""},
    ]


def test_a_labelled_json_upload_round_trips_through_the_sampler_format(tmp_path):
    data = json.dumps([{"text": "a", "label": "Notices"}, {"text": "b", "label": "01"}]).encode()
    path = tmp_path / "sample.parquet"

    tables.write_parquet(data, "data.json", str(path))

    assert pq.read_table(path).to_pylist() == [{"text": "a", "label": "Notices"}, {"text": "b", "label": "01"}]


def test_an_xlsx_with_a_date_column_round_trips_its_dates_as_iso_text(tmp_path):
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(["text", "decided", "label"])
    sheet.append(["A", datetime.date(2026, 11, 19), "Granted"])
    buffer = io.BytesIO()
    book.save(buffer)
    path = tmp_path / "sample.parquet"

    tables.write_parquet(buffer.getvalue(), "data.xlsx", str(path))

    # openpyxl reads a date cell back as a datetime at midnight, and its ISO text says so, with no JSON quotes.
    assert pq.read_table(path).to_pylist() == [{"text": "A", "decided": "2026-11-19T00:00:00", "label": "Granted"}]
