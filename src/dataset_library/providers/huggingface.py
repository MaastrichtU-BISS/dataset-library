"""Search the Hugging Face Hub's dataset API.

Only https://huggingface.co is ever called, with parameters built here, never
a user-supplied URL."""
import re
import unicodedata

import requests

from ..errors import ProviderRateLimited, ProviderUnavailable
from ..licenses import resolve_license
from ..tasks import hugging_face_filter, hugging_face_tasks

API = "https://huggingface.co/api/datasets"
DATASET_URL = "https://huggingface.co/datasets/{}"
README_URL = "https://huggingface.co/datasets/{}/resolve/{}/README.md"
TIMEOUT = 10
# A dataset's README is read for its citation only: a short timeout and a cap (#120).
README_TIMEOUT = 5
README_BYTES = 512 * 1024
LIMIT = 30
# Organisation, or organisation/dataset.
ID_PATTERN = re.compile(r"^[\w.-]+(/[\w.-]+)?$")
# A dataset: organisation/dataset, no dots alone, nothing that walks a path.
DATASET_ID_PATTERN = re.compile(r"^(?!\.)[\w.-]+/(?!\.)[\w.-]+$")
# A git revision as the Hub names one (a README is fetched at one).
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SIZE = {"n<1K": (0, 1_000)}
_SUFFIX = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}


class PrivateDataset(Exception):
    """The Hub knows the dataset, but its owner keeps it private: it can still
    be told apart from one nobody has, so the import can say why."""


def search(query, task, license_id, authors, task_types):
    """Matching datasets as library cards. `authors` limits the search to those
    organisations (one Hub call each); empty searches the whole Hub."""
    params = {"limit": LIMIT, "full": "true", "sort": "downloads"}
    if query:
        params["search"] = query
    filters = []
    if task:
        tag = hugging_face_filter(task)
        if tag is None:
            return []
        filters.append(tag)
    if license_id:
        filters.append(f"license:{license_id}")
    if filters:
        params["filter"] = filters
    records = []
    for author in authors or [None]:
        records.extend(_get_with_retry({**params, "author": author} if author else params))
    return [_card(record, task_types) for record in records if not record.get("private") and not record.get("disabled")]


# A source: one organisation's name.
ORGANISATION_PATTERN = re.compile(r"^(?!\.)[\w.-]{1,96}$")


def organisation_has_datasets(name):
    """Whether the Hub lists any public dataset from this organisation."""
    return bool(_get_with_retry({"author": name, "limit": 1}))


def dataset(external_id, task_types):
    """One dataset's card, or None when the Hub does not know it. This single-
    dataset read is the only one that may fetch the README for a citation (#120)."""
    try:
        response = requests.get(f"{API}/{external_id}", timeout=TIMEOUT)
    except requests.RequestException as exc:
        raise ProviderUnavailable(str(exc)) from exc
    if response.status_code in (401, 404):
        return None
    if response.status_code == 429:
        raise ProviderRateLimited("Hugging Face is limiting requests")
    if response.status_code != 200:
        raise ProviderUnavailable(f"Hugging Face answered {response.status_code}")
    try:
        record = response.json()
    except ValueError as exc:
        raise ProviderUnavailable("Unreadable answer") from exc
    if not isinstance(record, dict):
        return None
    if record.get("private"):
        raise PrivateDataset()
    if record.get("disabled"):
        return None
    return _card(record, task_types, readme=lambda: _readme_bibtex(external_id, record.get("sha")))


def _get_with_retry(params):
    """One retry when the Hub is down; none when it is limiting requests."""
    try:
        return _get(params)
    except ProviderRateLimited:
        raise
    except ProviderUnavailable:
        return _get(params)


def _get(params):
    try:
        response = requests.get(API, params=params, timeout=TIMEOUT)
    except requests.RequestException as exc:
        raise ProviderUnavailable(str(exc)) from exc
    if response.status_code == 429:
        raise ProviderRateLimited("Hugging Face is limiting requests")
    if response.status_code != 200:
        raise ProviderUnavailable(f"Hugging Face answered {response.status_code}")
    try:
        body = response.json()
    except ValueError as exc:
        raise ProviderUnavailable(f"Unreadable answer: {response.text[:200]}") from exc
    if not isinstance(body, list):
        raise ProviderUnavailable(f"Unexpected answer: {response.text[:200]}")
    return body


def _card(record, task_types, readme=None):
    """A library card from a Hub record. `readme` is a function giving the
    README's BibTeX; it is called only when the record has no BibTeX of its own."""
    external_id = record["id"]
    tags = record.get("tags") or []
    card_data = record.get("cardData") or {}
    license_tag = next((t for t in tags if t.startswith("license:")), None) or _first(card_data.get("license"))
    return {
        "id": f"huggingface:{external_id}",
        "provider": "huggingface",
        "externalId": external_id,
        "author": external_id.split("/")[0],
        "title": _title(card_data.get("pretty_name"), external_id),
        "url": DATASET_URL.format(external_id),
        "tasks": hugging_face_tasks(tags, task_types),
        "license": resolve_license(license_tag).id,
        "size": _size(tags),
        "gated": bool(record.get("gated")),
        "description": _description(record.get("description")),
        "updated": (record.get("lastModified") or "")[:10] or None,
        "revision": record.get("sha"),
        "downloads": record.get("downloads") or 0,
        "citation": _citation(record, external_id, card_data, readme),
    }


def _first(value):
    return value[0] if isinstance(value, list) and value else value if isinstance(value, str) else None


def _title(pretty_name, external_id):
    if isinstance(pretty_name, str) and pretty_name.strip():
        return pretty_name.strip()
    return external_id.split("/")[-1]


# --- Citations (#120) ---------------------------------------------------------
# What a dataset is cited from: its title, authors, year and BibTeX. The BibTeX
# is read by a small scanner, so a field is only read where it sits at the top
# level of the first entry, and names and values come out as plain text.

# An entry's start; @string, @preamble and @comment are not entries.
_ENTRY_START = re.compile(r"@(?!(?:string|preamble|comment)\s*\{)[A-Za-z]+\s*\{", re.IGNORECASE)
_FENCE = re.compile(r"^```[^\n]*\n(.*?)^```", re.MULTILINE | re.DOTALL)
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*$", re.MULTILINE)
_FIELD = re.compile(r"[A-Za-z][\w:.+-]*")
_WORD = re.compile(r"[\w.:+/-]+")
_AND = re.compile(r"\s+and\s+", re.IGNORECASE)
_SPACE = re.compile(r"\s+")
_COMMA = re.compile(r",")
_YEAR = re.compile(r"\d{4}")
# LaTeX accents: a mark then a letter, as \"u, \"{u}, {\"u}, and \c{c} for letter marks.
_ACCENT_MARKS = {
    '"': "\u0308", "'": "\u0301", "`": "\u0300", "^": "\u0302", "~": "\u0303", "=": "\u0304",
    ".": "\u0307", "u": "\u0306", "v": "\u030c", "H": "\u030b", "c": "\u0327", "k": "\u0328",
    "r": "\u030a", "d": "\u0323", "b": "\u0331",
}
_ACCENT = re.compile(r"\\(?:([\"'`^~=.])|([uvHckrdb])(?![A-Za-z]))\s*(?:\{([A-Za-z])\}|([A-Za-z]))")
# LaTeX letters with no accent: \ss, \o, \ae, and so on.
_LETTER_MARKS = {
    "ss": "ß", "oe": "œ", "OE": "Œ", "ae": "æ", "AE": "Æ", "aa": "å", "AA": "Å",
    "o": "ø", "O": "Ø", "l": "ł", "L": "Ł", "i": "ı",
}
_LETTER = re.compile(r"\\(" + "|".join(_LETTER_MARKS) + r")(?![A-Za-z])(?:\{\}| )?")
_ESCAPED = re.compile(r"\\([&%$#_])")


def _citation(record, external_id, card_data, readme=None):
    """The fields a Dataset is cited from (#120). `accessedAt` stays null here:
    looking at a card is not reading the dataset; an import sets it."""
    bibtex = _bibtex(record, card_data, readme)
    return {
        "title": _title(card_data.get("pretty_name"), external_id),
        "authors": _authors(card_data, bibtex, external_id),
        "year": _year(card_data, bibtex),
        "url": DATASET_URL.format(external_id),
        "revision": record.get("sha"),
        "accessedAt": None,
        "bibtex": bibtex,
    }


def _bibtex(record, card_data, readme=None):
    """The dataset's own BibTeX, exactly as written, or None. Read in order: the
    Hub's top-level `citation`, the card's `citation_bibtex` and `citation`, then
    the README. A plain-text citation is not an entry, so it is skipped."""
    for value in (record.get("citation"), card_data.get("citation_bibtex"), card_data.get("citation")):
        if isinstance(value, str) and value.lstrip().startswith("@"):
            return value
    return readme() if readme is not None else None


def _readme_bibtex(external_id, revision):
    """The BibTeX a dataset's README gives at `revision`, or None. The first 512 KB
    are read, and any failure is no BibTeX."""
    if not (isinstance(revision, str) and REVISION_PATTERN.match(revision) and DATASET_ID_PATTERN.match(external_id)):
        return None
    try:
        response = requests.get(README_URL.format(external_id, revision), timeout=README_TIMEOUT, stream=True)
    except requests.RequestException:
        return None
    try:
        if response.status_code != 200:
            return None
        body = bytearray()
        for chunk in response.iter_content(chunk_size=16_384):
            body.extend(chunk[: README_BYTES - len(body)])
            if len(body) >= README_BYTES:
                break
    except Exception:  # noqa: BLE001 - whatever failed, this card has no BibTeX from its README
        return None
    finally:
        response.close()
    return _readme_entry(bytes(body).decode("utf-8", errors="replace"))


def _readme_entry(text):
    """The BibTeX in a README (#120): an entry in a section whose heading says
    "Citation" (a fenced block there, else a bare entry), else the first fenced
    block anywhere holding one. The Citation section comes first because a README
    often quotes an upstream dataset's BibTeX elsewhere."""
    fenced = [(match.span(), match.group(1)) for match in _FENCE.finditer(text)]
    # A "#" line inside a code block is not a heading.
    headings = [h for h in _HEADING.finditer(text) if not any(a <= h.start() < b for (a, b), _ in fenced)]
    for index, heading in enumerate(headings):
        if "citation" not in heading.group(2).lower():
            continue
        level = len(heading.group(1))
        end = next((h.start() for h in headings[index + 1:] if len(h.group(1)) <= level), len(text))
        for (a, b), body in fenced:
            if heading.end() <= a < end and _ENTRY_START.search(body):
                return body.strip()
        section = text[heading.end():end]
        found = _ENTRY_START.search(section)
        close = _close(section, found.end() - 1) if found else None
        if close is not None:
            return section[found.start():close + 1].strip()
    for _, body in fenced:
        if _ENTRY_START.search(body):
            return body.strip()
    return None


def _authors(card_data, bibtex, external_id):
    """The card's authors ("Given Family" strings), else the BibTeX entry's, else
    the organisation or person who owns the dataset, as a literal (#120)."""
    listed = card_data.get("authors")
    names = []
    for value in listed if isinstance(listed, list) else [listed]:
        if isinstance(value, str):
            names.extend(_names(value))
    if names:
        return names
    written = _entry(bibtex).get("author")
    if written:
        names = _names(written)
        if names:
            return names
    return [{"literal": external_id.split("/")[0]}]


def _names(written):
    """An author list: split at `and` outside braces, in any case, each name read
    by `_name`."""
    return [name for name in (_name(part) for part in _split_top(written, _AND)) if name]


def _name(text):
    """One name as a citation holds it (#120): `{"family", "given"}` for a person,
    `{"literal"}` for an organisation, a single word, or a whole braced group.
    Reads "Last, First", "Last, Jr, First" and "First von Last"."""
    text = text.strip()
    if not text:
        return None
    if text.startswith("{") and _close(text, 0) == len(text) - 1:
        return _literal(text[1:-1])
    # A stray trailing comma ("Last, First,") is no part of its own.
    parts = [part for part in _split_top(text, _COMMA) if part.strip()]
    if not parts:
        return None
    if len(parts) == 1:
        text = parts[0].strip()
    if len(parts) > 1:
        # The middle part of "Last, Jr, First" is a suffix, which a citation leaves out.
        family, given = _plain(parts[0]), _plain(parts[-1])
    else:
        words = [word for word in _split_top(text, _SPACE) if word]
        if len(words) < 2:
            return _literal(text)
        # The lower-case words before the last name ("von", "de la") belong to the family name.
        start = len(words) - 1
        while start > 1 and _plain(words[start - 1])[:1].islower():
            start -= 1
        family, given = _plain(" ".join(words[start:])), _plain(" ".join(words[:start]))
    if not family:
        return _literal(text)
    return {"family": family, "given": given}


def _literal(text):
    plain = _plain(text)
    return {"literal": plain} if plain else None


def _year(card_data, bibtex):
    """The publication year (#120): the BibTeX entry's, else the card's. Never the
    repository's creation date, which the Hub stamps on older datasets as 2022-03-02."""
    for value in (_bibtex_field(bibtex, "year"), card_data.get("year")):
        text = str(value).strip() if value is not None else ""
        if _YEAR.fullmatch(text):
            return int(text)
    return None


def _bibtex_field(bibtex, name):
    """One top-level field of the first BibTeX entry as plain text, or None."""
    value = _entry(bibtex).get(name)
    return (_plain(value) or None) if value is not None else None


def _entry(bibtex):
    """The first BibTeX entry's top-level fields: {name: raw value}, names in lower
    case, values as written. A field inside another value's braces or quotes is part
    of that value, so `note = {year=1999}, year = {2025}` reads 2025. {} when there
    is no entry, or the entry stops making sense."""
    if not isinstance(bibtex, str):
        return {}
    start = _ENTRY_START.search(bibtex)
    if start is None:
        return {}
    fields, index = {}, start.end()
    # The citation key runs to its first comma.
    while index < len(bibtex) and bibtex[index] not in ",}":
        index += 1
    if index >= len(bibtex) or bibtex[index] == "}":
        return fields
    index += 1
    while True:
        while index < len(bibtex) and (bibtex[index] == "," or bibtex[index].isspace()):
            index += 1
        if index >= len(bibtex) or bibtex[index] == "}":
            return fields
        name = _FIELD.match(bibtex, index)
        if name is None:
            return fields
        index = _skip_space(bibtex, name.end())
        if index >= len(bibtex) or bibtex[index] != "=":
            return fields
        value, index = _value(bibtex, _skip_space(bibtex, index + 1))
        if value is None:
            return fields
        fields.setdefault(name.group(0).lower(), value)


def _value(text, index):
    """The value at `index`: `{...}`, `"..."` or a bare word, joined by `#` as BibTeX
    joins them. Returns (raw text, index after it), or (None, index) when malformed."""
    parts = []
    while True:
        if index >= len(text):
            return None, index
        if text[index] == "{":
            close = _close(text, index)
            if close is None:
                return None, index
            parts.append(text[index + 1:close])
            index = close + 1
        elif text[index] == '"':
            close = _quote_close(text, index)
            if close is None:
                return None, index
            parts.append(text[index + 1:close])
            index = close + 1
        else:
            word = _WORD.match(text, index)
            if word is None:
                return None, index
            parts.append(word.group(0))
            index = word.end()
        index = _skip_space(text, index)
        if index < len(text) and text[index] == "#":
            index = _skip_space(text, index + 1)
            continue
        return "".join(parts), index


def _close(text, start):
    """The index of the brace that closes the one at `start`, or None."""
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _quote_close(text, start):
    """The index of the quote that closes the one at `start`. A quote inside braces,
    or one a backslash escapes, does not close it; None when it never closes."""
    depth, index = 0, start + 1
    while index < len(text):
        character = text[index]
        if character == "\\":
            index += 2
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
        elif character == '"' and depth <= 0:
            return index
        index += 1
    return None


def _skip_space(text, index):
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _split_top(text, pattern):
    """`text` split where `pattern` matches outside braces, so a braced group stays whole."""
    parts, depth, start, index = [], 0, 0, 0
    while index < len(text):
        character = text[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth = max(depth - 1, 0)
        elif depth == 0:
            found = pattern.match(text, index)
            if found is not None and found.end() > index:
                parts.append(text[start:index])
                start = index = found.end()
                continue
        index += 1
    parts.append(text[start:])
    return parts


def _plain(raw):
    """A BibTeX value as plain text: LaTeX accents and letters decoded, escaped
    specials unescaped, `~` a space, braces and extra spaces removed."""
    text = _ACCENT.sub(_accented, raw)
    text = _LETTER.sub(lambda match: _LETTER_MARKS[match.group(1)], text)
    text = _ESCAPED.sub(r"\1", text).replace("~", " ")
    text = text.replace("{", "").replace("}", "")
    return unicodedata.normalize("NFC", " ".join(text.split()))


def _accented(match):
    letter = match.group(3) or match.group(4)
    mark = match.group(1) or match.group(2)
    return unicodedata.normalize("NFC", letter + _ACCENT_MARKS[mark])


def _number(text):
    text = text.strip()
    if text and text[-1] in _SUFFIX:
        return int(float(text[:-1]) * _SUFFIX[text[-1]])
    return int(text)


def _size(tags):
    """`size_categories:100K<n<1M` as {"label": "100K–1M rows", "min", "max"}."""
    tag = next((t.removeprefix("size_categories:") for t in tags if t.startswith("size_categories:")), None)
    if not tag:
        return None
    if tag == "n<1K":
        return {"label": "under 1K rows", "min": 0, "max": 1_000}
    if tag.startswith("n>"):
        return {"label": f"over {tag[2:]} rows", "min": _number(tag[2:]), "max": None}
    low, _, high = tag.partition("<n<")
    try:
        return {"label": f"{low}–{high} rows", "min": _number(low), "max": _number(high)}
    except ValueError:
        return None


def _description(text, limit=300):
    """The card's summary as plain text: whitespace collapsed, the Hub's
    "Dataset Card for X / Dataset Summary" headings dropped, cut at a word."""
    text = " ".join((text or "").split())
    text = re.sub(r"^Dataset Card for .*? Dataset Summary ", "", text)
    text = re.sub(r"^Dataset Card for \S+\s*", "", text)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0] + "…"
