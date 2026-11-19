"""Read a Community dataset's data file as a table (#122).

BAIGLE publishes an uploaded dataset as `data<ext>`: CSV, XLSX or JSON. A labelled
one is published as CSV whatever its name says, so a file is read by what its
bytes are, and only a `.tsv` name asks for tabs. Every column is read as text, so
a label such as "01" stays "01" as it was written.
"""
import csv
import io
import json
import math

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from .providers.community import UNREADABLE

ROW_GROUP_ROWS = 1000
PARQUET_MAGIC = b"PAR1"
# An XLSX file is a zip archive.
ZIP_MAGIC = b"PK\x03\x04"
UTF8_BOM = b"\xef\xbb\xbf"


class Unreadable(Exception):
    """The bytes are not a table this module can read; `message` is for the student."""

    def __init__(self, message=UNREADABLE):
        super().__init__(message)
        self.message = message


def read(data, name=""):
    """The file's rows as a pyarrow table whose columns are all text."""
    if data.startswith(PARQUET_MAGIC):
        return _parquet(data)
    if data.startswith(ZIP_MAGIC):
        return _xlsx(data)
    body = data.removeprefix(UTF8_BOM)
    first = body.lstrip()[:1]
    if first == b"[":
        return _records(_json_array(_decode(body)))
    if first == b"{":
        return _records(_json_lines(_decode(body)))
    return _delimited(body, "\t" if name.lower().endswith(".tsv") else ",")


def write_parquet(data, name, path):
    """Write the table as a Parquet file the sampler reads, in row groups."""
    pq.write_table(read(data, name), path, row_group_size=ROW_GROUP_ROWS)


def head(table, count):
    """The column names, and the first `count` rows as lists of text."""
    names = table.column_names
    rows = table.slice(0, count).to_pylist()
    return names, [[row[name] for name in names] for row in rows]


def _decode(data):
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Unreadable() from exc


def _delimited(body, delimiter):
    # Only the header line is read here; pyarrow reads the rest, with every column as text.
    header_line = _decode(body.split(b"\n", 1)[0])
    try:
        header = next(csv.reader(io.StringIO(header_line), delimiter=delimiter), None)
    except csv.Error as exc:
        raise Unreadable() from exc
    if not header:
        raise Unreadable()
    try:
        table = pacsv.read_csv(
            io.BytesIO(body),
            parse_options=pacsv.ParseOptions(delimiter=delimiter),
            convert_options=pacsv.ConvertOptions(column_types={name: pa.string() for name in header}),
        )
    except (pa.ArrowException, ValueError) as exc:
        raise Unreadable() from exc
    return table.rename_columns(_unique(table.column_names))


def _json_array(text):
    try:
        rows = json.loads(text)
    except ValueError as exc:
        raise Unreadable() from exc
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise Unreadable()
    return rows


def _json_lines(text):
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise Unreadable() from exc
        if not isinstance(row, dict):
            raise Unreadable()
        rows.append(row)
    return rows


def _records(rows):
    """JSON objects as a table: the columns are the keys, in the order first seen."""
    names = []
    seen = set()
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                names.append(name)
    return _from_rows(names, [[row.get(name) for name in names] for row in rows])


def _xlsx(data):
    try:
        import openpyxl  # the [import] extra; the other formats need nothing extra
    except ImportError as exc:
        raise Unreadable("XLSX files can't be read on this platform, so this record cannot be imported.") from exc
    try:
        workbook = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        rows = workbook.worksheets[0].iter_rows(values_only=True)
        header = next(rows, None)
        body = [list(row) for row in rows if any(value not in (None, "") for value in row)]
    except Exception as exc:  # noqa: BLE001 - any failure to read corrupt bytes means unreadable
        raise Unreadable() from exc
    if header is None:
        return _from_rows([], [])
    names = [f"Unnamed: {index}" if value is None else str(value) for index, value in enumerate(header)]
    return _from_rows(names, body)


def _parquet(data):
    try:
        table = pq.read_table(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001 - corrupt bytes of any kind are unreadable
        raise Unreadable() from exc
    columns = [table.column(index).to_pylist() for index in range(table.num_columns)]
    return _from_rows(table.column_names, [list(row) for row in zip(*columns)])


def _from_rows(names, rows):
    """A text table from header names and rows of values; a short row reads as blanks."""
    names = _unique(names)
    columns = [[] for _ in names]
    for row in rows:
        for index in range(len(names)):
            columns[index].append(_text_value(row[index] if index < len(row) else None))
    return pa.Table.from_arrays([pa.array(column, type=pa.string()) for column in columns], names=names)


def _text_value(value):
    """A cell as text: strings as they are, numbers and booleans as JSON writes them."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, float) and math.isnan(value):
        return None
    return json.dumps(value, ensure_ascii=False, default=str)


def _unique(names):
    """Column names made distinct, as pandas does: a repeated name gets ".1", ".2"..."""
    taken = set()
    out = []
    for name in names:
        candidate, number = name, 0
        while candidate in taken:
            number += 1
            candidate = f"{name}.{number}"
        taken.add(candidate)
        out.append(candidate)
    return out
