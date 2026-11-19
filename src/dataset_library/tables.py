"""Read a Community dataset's data file as a table (#122).

BAIGLE publishes an uploaded dataset as `data<ext>`: CSV, XLSX or JSON. A labelled one is
published as CSV whatever its name says, so a file is read by what its bytes are. A name
says only what the bytes cannot: `.tsv` asks for tabs, `.csv` and `.tsv` are never read as
JSON, and a JSON name is read as JSON. Every column is read as text, so a label such as "01"
stays "01" as it was written.

Each reader checks the file's size, rows, columns, cells, JSON nesting and XLSX zip ratio
before it allocates them, and raises `Unreadable` with the limit it hit. pyarrow and openpyxl
are imported only when this module loads, which the Community paths do lazily, so a host without
the `[import]` extra can still serve its routes.
"""
import csv
import datetime
import decimal
import io
import json
import math
import re
import zipfile

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from .providers.community import UNREADABLE

ROW_GROUP_ROWS = 1000
PARQUET_MAGIC = b"PAR1"
# An XLSX file is a zip archive.
ZIP_MAGIC = b"PK\x03\x04"
UTF8_BOM = b"\xef\xbb\xbf"

# The limits a file is read under (#122). Each is checked before the memory it would take is allocated.
MAX_ROWS = 200_000
MAX_COLUMNS = 1_000
MAX_CELLS = 2_000_000
# The most text a file may expand to once read: a CSV's or JSON's bytes, an XLSX's or Parquet's declared size.
MAX_EXPANDED_BYTES = 500_000_000
MAX_JSON_DEPTH = 8
MAX_ZIP_ENTRIES = 5_000
# An XLSX that expands more than this many times over its compressed size is refused as a zip bomb.
XLSX_MAX_RATIO = 100
# How much of a CSV's text the header is read from; a header longer than this is refused.
HEADER_CHARACTERS = 1_000_000
# Strings of JSON, one token at a time: a quoted string, or a bracket that opens or closes a level.
_JSON_TOKEN = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"|[\[\]{}]')


class Unreadable(Exception):
    """The bytes are not a table this module can read; `message` is for the student."""

    def __init__(self, message=UNREADABLE):
        super().__init__(message)
        self.message = message


class _NotJson(Exception):
    """The text does not parse as JSON. A name-less file that opens like JSON is then read as CSV."""


def read(data, name=""):
    """The file's rows as a pyarrow table whose columns are all text. Raises Unreadable."""
    lowered = name.lower()
    if data.startswith(PARQUET_MAGIC):
        return _parquet(data)
    if data.startswith(ZIP_MAGIC):
        return _xlsx(data)
    if lowered.endswith((".csv", ".tsv")):
        return _delimited(data, "\t" if lowered.endswith(".tsv") else ",")
    if lowered.endswith((".json", ".jsonl")):
        try:
            return _json(data, lines=lowered.endswith(".jsonl"))
        except _NotJson as exc:
            raise Unreadable() from exc
    if data.removeprefix(UTF8_BOM).lstrip()[:1] in (b"[", b"{"):
        try:
            return _json(data, lines=False)
        except (_NotJson, Unreadable):
            pass  # A CSV whose first header starts with a bracket is CSV, not JSON.
    return _delimited(data, ",")


def write_parquet(data, name, path):
    """Write the table as a Parquet file the sampler reads, in row groups."""
    pq.write_table(read(data, name), path, row_group_size=ROW_GROUP_ROWS)


def head(table, count):
    """The column names, and the first `count` rows as lists of text."""
    names = table.column_names
    rows = table.slice(0, count).to_pylist()
    return names, [[row[name] for name in names] for row in rows]


def _refuse(what, limit):
    return Unreadable(f"This file has more than {limit:,} {what}, which is more than this platform reads.")


def _expanded():
    return Unreadable(
        f"This file is more than {MAX_EXPANDED_BYTES // 1_000_000} MB once it is read, which is more than this platform reads."
    )


def _check_cells(rows, columns):
    if rows * columns > MAX_CELLS:
        raise _refuse("cells (rows times columns)", MAX_CELLS)


def _text(body):
    """The bytes as text: UTF-8, else Windows-1252, as spreadsheets save a CSV that is not UTF-8."""
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        return body.decode("cp1252")
    except UnicodeDecodeError as exc:
        raise Unreadable() from exc


def _delimited(data, delimiter):
    if len(data) > MAX_EXPANDED_BYTES:
        raise _expanded()
    body = data.removeprefix(UTF8_BOM)
    try:
        body.decode("utf-8")
        encoded = body  # UTF-8 already: pyarrow reads the bytes as they are
        prefix = body[:HEADER_CHARACTERS].decode("utf-8", errors="ignore")
    except UnicodeDecodeError:
        text = _text(body)
        encoded = text.encode("utf-8")
        prefix = text[:HEADER_CHARACTERS]
    # The header is one CSV record, which may span lines inside quotes, so a real CSV parser reads it.
    try:
        header = next(csv.reader(io.StringIO(prefix), delimiter=delimiter), None)
    except csv.Error as exc:
        raise Unreadable() from exc
    if not header:
        raise Unreadable()
    if len(header) > MAX_COLUMNS:
        raise _refuse("columns", MAX_COLUMNS)
    if b"\n" not in encoded and b"\r" not in encoded:
        encoded += b"\n"  # a header with no line end, and so no rows, is one line pyarrow cannot infer columns from
    try:
        reader = pacsv.open_csv(
            io.BytesIO(encoded),
            parse_options=pacsv.ParseOptions(delimiter=delimiter, newlines_in_values=True),
            convert_options=pacsv.ConvertOptions(column_types={name: pa.string() for name in header}),
        )
    except (pa.ArrowException, ValueError) as exc:
        raise Unreadable() from exc
    # The file is read a batch at a time, so the limits stop it before it is all in memory.
    batches, rows, expanded = [], 0, 0
    try:
        for batch in reader:
            rows += batch.num_rows
            if rows > MAX_ROWS:
                raise _refuse("rows", MAX_ROWS)
            _check_cells(rows, len(header))
            expanded += batch.nbytes
            if expanded > MAX_EXPANDED_BYTES:
                raise _expanded()
            batches.append(batch)
        table = pa.Table.from_batches(batches, schema=reader.schema)
    except (pa.ArrowException, ValueError) as exc:
        raise Unreadable() from exc
    return table.rename_columns(_unique(table.column_names))


def _json(data, lines):
    """A JSON file as a table: an array of objects, one object per line, or pandas' column layout.

    The layouts BAIGLE's upload reads with pandas' read_json are an array of objects and an object of
    columns. Those are read as they are, and so is one object on its own, as a single row."""
    if len(data) > MAX_EXPANDED_BYTES:
        raise _expanded()
    text = _text(data.removeprefix(UTF8_BOM))
    _check_depth(text)
    if lines:
        return _records(_json_lines(text))
    if text.lstrip()[:1] == "[":
        rows = _parse(text)
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise Unreadable()
        return _records(rows)
    try:
        document = _parse(text)
    except _NotJson:
        return _records(_json_lines(text))
    if not isinstance(document, dict):
        raise Unreadable()
    values = list(document.values())
    if values and (all(isinstance(value, dict) for value in values) or all(isinstance(value, list) for value in values)):
        return _columns(document)
    return _records([document])


def _parse(text):
    try:
        return json.loads(text)
    except RecursionError as exc:
        raise _refuse_depth() from exc
    except ValueError as exc:
        raise _NotJson() from exc


def _refuse_depth():
    return Unreadable(f"This JSON nests more than {MAX_JSON_DEPTH} levels deep, which is more than this platform reads.")


def _check_depth(text):
    """Refuse JSON that nests too deep before it is parsed, so nothing deep is ever built."""
    depth = 0
    for match in _JSON_TOKEN.finditer(text):
        token = match.group()
        if token in ("[", "{"):
            depth += 1
            if depth > MAX_JSON_DEPTH:
                raise _refuse_depth()
        elif token in ("]", "}"):
            depth -= 1


def _json_lines(text):
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        row = _parse(line)
        if not isinstance(row, dict):
            raise Unreadable()
        rows.append(row)
        if len(rows) > MAX_ROWS:
            raise _refuse("rows", MAX_ROWS)
    return rows


def _columns(document):
    """pandas' column layout: each key is a column, and each column maps row keys to values, or is a list."""
    names = list(document)
    if len(names) > MAX_COLUMNS:
        raise _refuse("columns", MAX_COLUMNS)
    if all(isinstance(document[name], dict) for name in names):
        index, seen = [], set()
        for name in names:
            for key in document[name]:
                if key not in seen:
                    seen.add(key)
                    index.append(key)
                    if len(index) > MAX_ROWS:
                        raise _refuse("rows", MAX_ROWS)
        _check_cells(len(index), len(names))
        columns = [[_text_value(document[name].get(key)) for key in index] for name in names]
    else:
        lists = [document[name] for name in names]
        if not all(isinstance(values, list) for values in lists):
            raise Unreadable()
        rows = max((len(values) for values in lists), default=0)
        if rows > MAX_ROWS:
            raise _refuse("rows", MAX_ROWS)
        _check_cells(rows, len(names))
        columns = [[_text_value(values[i]) if i < len(values) else None for i in range(rows)] for values in lists]
    return _table(names, columns)


def _records(rows):
    """JSON objects as a table: the columns are the keys, in the order first seen. Keys are bounded as they are found."""
    if len(rows) > MAX_ROWS:
        raise _refuse("rows", MAX_ROWS)
    names, seen = [], set()
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                names.append(name)
                if len(names) > MAX_COLUMNS:
                    raise _refuse("columns", MAX_COLUMNS)
    _check_cells(len(rows), len(names))
    return _table(names, [[_text_value(row.get(name)) for row in rows] for name in names])


def _xlsx(data):
    try:
        import openpyxl  # the [import] extra; the other formats need nothing extra
    except ImportError as exc:
        raise Unreadable("XLSX files can't be read on this platform, so this record cannot be imported.") from exc
    _check_archive(data)
    try:
        workbook = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except Exception as exc:  # noqa: BLE001 - any failure to read corrupt bytes means unreadable
        raise Unreadable() from exc
    try:
        return _sheet(workbook)
    except Unreadable:
        raise
    except Exception as exc:  # noqa: BLE001 - a sheet that fails to read is unreadable, whatever broke
        raise Unreadable() from exc
    finally:
        workbook.close()


def _check_archive(data):
    """An XLSX's zip central directory, read before anything is decompressed. A zip bomb declares a huge
    size or a huge ratio there, and is refused without being expanded (#122)."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
    except (zipfile.BadZipFile, ValueError, OSError) as exc:
        raise Unreadable() from exc
    if len(entries) > MAX_ZIP_ENTRIES:
        raise Unreadable(f"This XLSX file has more than {MAX_ZIP_ENTRIES:,} parts, which is more than this platform reads.")
    expanded = sum(entry.file_size for entry in entries)
    packed = sum(entry.compress_size for entry in entries)
    if expanded > MAX_EXPANDED_BYTES:
        raise _expanded()
    if expanded > XLSX_MAX_RATIO * max(packed, 1):
        raise Unreadable(
            f"This XLSX file expands more than {XLSX_MAX_RATIO} times over its size, which is more than this platform reads."
        )


def _sheet(workbook):
    """The first sheet's first row as names and the rest as rows, blank rows skipped. Limits are checked
    as the rows are read, so a sheet cannot run on past them."""
    if not workbook.worksheets:
        raise Unreadable()
    sheet = workbook.worksheets[0]
    # The sheet may declare a range far bigger than its cells (A1:XFD1048576): read the cells there are.
    sheet.reset_dimensions()
    rows = sheet.iter_rows(values_only=True)
    header = next(rows, None)
    if header is None:
        return _table([], [])
    if len(header) > MAX_COLUMNS:
        raise _refuse("columns", MAX_COLUMNS)
    # A row reads only as far as its last cell that was written, so the width is the widest row's, not the header's.
    width, body = len(header), []
    for row in rows:
        if len(row) > MAX_COLUMNS:
            raise _refuse("columns", MAX_COLUMNS)
        if not any(value not in (None, "") for value in row):
            continue
        body.append(row)
        if len(body) > MAX_ROWS:
            raise _refuse("rows", MAX_ROWS)
        width = max(width, len(row))
        _check_cells(len(body), width)
    names = [
        f"Unnamed: {index}" if index >= len(header) or header[index] is None else str(header[index])
        for index in range(width)
    ]
    columns = [[_text_value(row[index]) if index < len(row) else None for row in body] for index in range(width)]
    return _table(names, columns)


def _parquet(data):
    try:
        reader = pq.ParquetFile(io.BytesIO(data))
        metadata = reader.metadata
    except Exception as exc:  # noqa: BLE001 - corrupt bytes of any kind are unreadable
        raise Unreadable() from exc
    if metadata.num_rows > MAX_ROWS:
        raise _refuse("rows", MAX_ROWS)
    if metadata.num_columns > MAX_COLUMNS:
        raise _refuse("columns", MAX_COLUMNS)
    _check_cells(metadata.num_rows, metadata.num_columns)
    if sum(metadata.row_group(group).total_byte_size for group in range(metadata.num_row_groups)) > MAX_EXPANDED_BYTES:
        raise _expanded()
    try:
        table = reader.read()
    except Exception as exc:  # noqa: BLE001 - corrupt bytes of any kind are unreadable
        raise Unreadable() from exc
    columns = [_as_text(table.column(index)) for index in range(table.num_columns)]
    try:
        return pa.Table.from_arrays(columns, names=_unique(table.column_names))
    except (pa.ArrowException, ValueError) as exc:
        raise Unreadable() from exc


def _as_text(column):
    """A Parquet column as Arrow strings. Nested values are written as JSON, cell by cell, within the cell limit."""
    try:
        return column.cast(pa.string())
    except (pa.ArrowException, ValueError):
        return pa.chunked_array([pa.array([_text_value(value) for value in column.to_pylist()], type=pa.string())])


def _table(names, columns):
    """A text table from header names and columns of text (or None). Refused when it expands past the limit."""
    expanded = sum(len(value) for values in columns for value in values if isinstance(value, str))
    if expanded > MAX_EXPANDED_BYTES:
        raise _expanded()
    try:
        arrays = [pa.array(values, type=pa.string()) for values in columns]
    except (UnicodeEncodeError, UnicodeDecodeError, pa.ArrowException) as exc:
        raise Unreadable() from exc
    return pa.Table.from_arrays(arrays, names=_unique(names))


def _text_value(value):
    """A cell as text: strings as they are, dates and times as ISO text, decimals as written, and numbers,
    booleans and nested values as JSON writes them."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return None if math.isnan(value) else json.dumps(value)
    if isinstance(value, (datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return str(value)
    if isinstance(value, (list, dict)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


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
