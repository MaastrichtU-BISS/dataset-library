"""Reading a Community record's data file as a table (#122): by its bytes, every cell as text."""
import io
import json
import pathlib

import openpyxl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dataset_library import tables

XLSX_FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "zenodo" / "files" / "102" / "data.xlsx"


def as_rows(table):
    return table.column_names, table.to_pylist()


def test_csv_keeps_every_cell_as_the_text_it_was_written_as():
    table = tables.read(b'text,label\n"Keep 01, a comma",01\nSecond,1.0\n', "data.csv")

    assert as_rows(table) == (
        ["text", "label"],
        [{"text": "Keep 01, a comma", "label": "01"}, {"text": "Second", "label": "1.0"}],
    )


def test_a_byte_order_mark_and_windows_line_ends_are_read_like_any_csv():
    table = tables.read(b"\xef\xbb\xbftext,label\r\nA,x\r\n", "data.csv")

    assert as_rows(table) == (["text", "label"], [{"text": "A", "label": "x"}])


def test_a_repeated_column_name_is_numbered_as_pandas_numbers_it():
    table = tables.read(b"text,text,label\na,b,c\n", "data.csv")

    assert table.column_names == ["text", "text.1", "label"]


def test_a_tsv_name_reads_tabs_and_a_comma_is_just_text():
    table = tables.read(b"text\tlabel\nA, b\tx\n", "data.tsv")

    assert as_rows(table) == (["text", "label"], [{"text": "A, b", "label": "x"}])


def test_a_labelled_file_named_xlsx_is_read_by_its_bytes_as_csv():
    # BAIGLE writes a labelled dataset as CSV whatever the upload was called (publishing_host._dataset_bytes).
    table = tables.read(b"text,label,label_source\nA,Notices,file\n", "data.xlsx")

    assert as_rows(table) == (
        ["text", "label", "label_source"],
        [{"text": "A", "label": "Notices", "label_source": "file"}],
    )


def test_a_json_array_of_objects_reads_each_value_as_text():
    data = json.dumps(
        [{"text": "a", "label": 1}, {"text": "b", "label": True}, {"text": "c", "label": None, "extra": [1, 2]}]
    ).encode()

    assert as_rows(tables.read(data, "data.json")) == (
        ["text", "label", "extra"],
        [
            {"text": "a", "label": "1", "extra": None},
            {"text": "b", "label": "true", "extra": None},
            {"text": "c", "label": None, "extra": "[1, 2]"},
        ],
    )


def test_json_lines_are_read_one_object_per_line():
    table = tables.read(b'{"text": "a", "label": "x"}\n\n{"text": "b", "label": "y"}\n', "data.jsonl")

    assert as_rows(table) == (["text", "label"], [{"text": "a", "label": "x"}, {"text": "b", "label": "y"}])


def test_the_recorded_xlsx_reads_its_first_sheet_as_text():
    table = tables.read(XLSX_FIXTURE.read_bytes(), "data.xlsx")

    assert table.column_names == ["text", "label"]
    assert table.num_rows == 6
    assert table.to_pylist()[0] == {
        "text": "The appeal is allowed and the judgment below is set aside.",
        "label": "Granted",
    }


def test_an_xlsx_skips_blank_rows_and_names_an_unnamed_column():
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(["text", None])
    sheet.append(["A", "x"])
    sheet.append([None, None])
    sheet.append(["B", "y"])
    buffer = io.BytesIO()
    book.save(buffer)

    assert as_rows(tables.read(buffer.getvalue(), "data.xlsx")) == (
        ["text", "Unnamed: 1"],
        [{"text": "A", "Unnamed: 1": "x"}, {"text": "B", "Unnamed: 1": "y"}],
    )


def test_a_parquet_file_is_read_with_its_values_as_text():
    buffer = io.BytesIO()
    pq.write_table(pa.table({"text": ["a"], "n": [3]}), buffer)

    assert as_rows(tables.read(buffer.getvalue(), "data.parquet")) == (["text", "n"], [{"text": "a", "n": "3"}])


def test_a_header_with_no_rows_is_a_table_with_no_rows():
    table = tables.read(b"text,label\n", "data.csv")

    assert as_rows(table) == (["text", "label"], [])


@pytest.mark.parametrize(
    "data, name",
    [
        (b"", "data.csv"),
        (b"\xff\xfe\xfa\xfb", "data.csv"),
        (b"PK\x03\x04 not really a zip", "data.xlsx"),
        (b"PAR1 not really parquet", "data.parquet"),
        (b"[1, 2, 3]", "data.json"),
    ],
)
def test_bytes_that_are_not_a_table_are_unreadable_with_a_plain_reason(data, name):
    with pytest.raises(tables.Unreadable) as refused:
        tables.read(data, name)

    assert refused.value.message.startswith("We can't read this record's data file as a table")


def test_write_parquet_makes_row_groups_the_sampler_reads_a_group_at_a_time(tmp_path):
    lines = "text,label\n" + "".join(f"row {i},{'A' if i % 2 else 'B'}\n" for i in range(2500))
    path = tmp_path / "sample.parquet"

    tables.write_parquet(lines.encode(), "data.csv", str(path))

    metadata = pq.ParquetFile(str(path)).metadata
    assert metadata.num_rows == 2500
    assert metadata.num_row_groups == 3


def test_head_gives_the_first_rows_and_the_column_names():
    table = tables.read(b"text,label\na,x\nb,y\nc,z\n", "data.csv")

    assert tables.head(table, 2) == (["text", "label"], [["a", "x"], ["b", "y"]])
