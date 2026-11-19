"""One dataset, opened from the library: its facts, configs and first rows."""
import json
import re

from . import labels as labels_module
from . import policy as policy_module
from .licenses import LICENSES, UNKNOWN
from .providers import huggingface, viewer

CACHE_SECONDS = 600
# Set on a preview built while the viewer did not answer: the card stands, but its rows
# are not known, so it is not kept, and a copy of the part (#121) may stand in for them.
VIEWER_DOWN = "viewerDown"
ROWS = 8
CELL_CHARACTERS = 400
# Names that say which column is which, before falling back to the types.
NER_TEXT_COLUMNS = ("tokens", "words")
# Tag columns in the order a label is chosen. The first two say NER by name;
# the generic ones must show it (see `_entity_tags`).
NER_LABEL_COLUMNS = ("ner_tags", "ner", "tags", "labels", "label")
NER_NAMED_COLUMNS = ("ner_tags", "ner")
GENERIC_TAG_COLUMNS = ("tags", "labels", "label")
# Tag columns that are BIO but not entities: chunks, parts of speech.
NOT_ENTITY_WORDS = ("chunk", "pos")
# Source columns by name only: guessing by length once took metadata (a
# `split` column, a title) as the document.
SUMMARY_TEXT_COLUMNS = (
    "document", "article", "text", "dialogue", "source", "input",
    "report", "description", "content", "body",
)
# Only names that say "summary": generic `target`/`output` also hold code,
# answers and translations.
SUMMARY_LABEL_COLUMNS = ("summary", "highlights", "abstract")
LIST_TYPES = ("Sequence", "List", "LargeList")
# A BIO-style tag: a position letter, a separator, then the entity type.
_BIO = re.compile(r"^[BIESLU][-_].")
INTEGER_DTYPES = ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64")


class NotInLibrary(Exception):
    pass


class LicenseHidden(NotInLibrary):
    """The admin set this license to hide, so the dataset may not enter the
    library at all; the import route says so rather than "not found"."""


class UnknownConfig(Exception):
    pass


def preview(host, external_id, config=None, split=None):
    """One dataset's preview. The library's own rules come first: a dataset
    outside the enabled sources never reaches the Hub or the cache, so its
    cards and viewer answers are not fetched and no refusal reveals it."""
    from .search import enabled_sources

    sources = {source.ref for source in enabled_sources(host.db)}
    if sources and external_id.split("/", 1)[0] not in sources:
        raise NotInLibrary()
    key = f"library:preview:{external_id}:{config or ''}:{split or ''}"
    body = host.cache.get(key)
    # A body cached before citations (#120) has no citation: rebuild it, so an
    # import started right after the deploy still gets one.
    if not isinstance(body, dict) or "citation" not in body:
        body = _build(host, external_id, config, split)
        # A preview built while the viewer was down is not kept: the next open asks again.
        if not body.get(VIEWER_DOWN):
            host.cache.set(key, body, CACHE_SECONDS)
    return _apply_library_rules(host, body)


def _apply_library_rules(host, body):
    """Hidden licenses and datasets outside the enabled sources are not in the library."""
    from .search import enabled_sources

    sources = {source.ref for source in enabled_sources(host.db)}
    if sources and body["author"] not in sources:
        raise NotInLibrary()
    policy = policy_module.overrides(host.db)
    license = LICENSES.get(body["license"], UNKNOWN)
    level = policy_module.level(license, policy)
    if level == "hide":
        raise LicenseHidden()
    return {**body, "license": license.at_level(level)}


def _build(host, external_id, config, split):
    card = huggingface.dataset(external_id, host.task_types)
    if card is None:
        raise NotInLibrary()
    # The Hub may answer with a renamed dataset under another owner: check the
    # owner it names, then the license, before asking the viewer or checking
    # configs, so neither leaks a dataset outside the library.
    from .search import enabled_sources

    sources = {source.ref for source in enabled_sources(host.db)}
    if sources and card["author"] not in sources:
        raise NotInLibrary()
    policy = policy_module.overrides(host.db)
    if policy_module.level(LICENSES.get(card["license"], UNKNOWN), policy) == "hide":
        raise LicenseHidden()
    body = {
        **card,
        "preview": "unavailable", "configs": [], "config": None, "split": None,
        "columns": [], "rows": [], "rowCount": None, "proposed": _no_proposal(),
    }
    body["labels"] = labels_module.for_ids(host.cache, [external_id])[external_id]
    try:
        configs = _configs(viewer.get("splits", dataset=external_id))
    except viewer.ViewerDown:
        # Not "no rows": the viewer did not answer. A configured part is not refused here either,
        # since a copy of that part may still stand in for the viewer (#121).
        body[VIEWER_DOWN] = True
        return body
    except viewer.ViewerUnavailable:
        # The viewer said no for good (it cannot read this dataset): no preview, and no outage.
        return body
    if not configs:
        return body
    names = [c["name"] for c in configs]
    if config and config not in names:
        raise UnknownConfig()
    chosen = next(c for c in configs if c["name"] == (config or names[0]))
    if split and split not in chosen["splits"]:
        raise UnknownConfig()
    split = split or ("train" if "train" in chosen["splits"] else chosen["splits"][0])
    body.update(configs=configs, config=chosen["name"], split=split)
    body["rowCount"] = _row_count(external_id, chosen["name"], split)
    try:
        rows = viewer.get("first-rows", dataset=external_id, config=chosen["name"], split=split)
    except viewer.ViewerDown:
        body[VIEWER_DOWN] = True
        return body
    except viewer.ViewerUnavailable:
        return body
    features = [f for f in rows.get("features") or [] if isinstance(f, dict) and isinstance(f.get("name"), str)]
    body["columns"] = [{"name": f["name"], "type": _type(f.get("type"))} for f in features]
    body["rows"] = [
        [_cell((r.get("row") or {}).get(f["name"]), f.get("type")) for f in features]
        for r in (rows.get("rows") or [])[:ROWS]
    ]
    body["preview"] = "ok" if body["rows"] else "unavailable"
    if body["preview"] == "ok":
        body["proposed"] = _propose(card["tasks"], features, rows.get("rows") or [])
    return body


def _no_proposal():
    return {"taskType": None, "textColumn": None, "labelColumn": None}


def _propose(tasks, features, rows):
    """What the import's column step starts with: the task's usual columns
    when the viewer shows a pair that fits, and nothing when it cannot tell."""
    named = [(f["name"], f.get("type")) for f in features if isinstance(f, dict) and isinstance(f.get("name"), str)]
    if "ner" in tasks:
        text, label = _ner_columns(named, rows)
        if text is not None and label is not None:
            return {"taskType": "ner", "textColumn": text, "labelColumn": label}
    if "summarization" in tasks:
        label = _first_text(named, SUMMARY_LABEL_COLUMNS)
        text = _first_text(named, SUMMARY_TEXT_COLUMNS, exclude=label)
        if text is not None and label is not None and text != label:
            return {"taskType": "summarization", "textColumn": text, "labelColumn": label}
    return _no_proposal()


def _ner_columns(named, rows):
    """The `tokens`/`words` column and an entity tag column, or (None, None).

    Hugging Face tags part-of-speech, chunking and parsing configs as token
    classification too, so a tag column has to show it holds entities (see
    `_entity_tags`), and the text is only ever a token column: guessing from
    other string lists picked annotations as the text."""
    lists = {name: _list_inner(type_) for name, type_ in named if _is_list(type_)}
    text = next((name for name in NER_TEXT_COLUMNS if _is_text(lists.get(name))), None)
    if text is None:
        return None, None
    tags = [
        name for name, inner in lists.items()
        if name not in NER_TEXT_COLUMNS and _is_tag(inner) and _entity_tags(name, inner, rows)
    ]
    order = [name for name in NER_LABEL_COLUMNS if name in tags]
    order += [name for name in tags if name.endswith("tags") and name not in order]
    order += [name for name in tags if name not in order]
    return (text, order[0]) if order else (None, None)


def _entity_tags(name, feature, rows):
    """Whether a flat tag column holds entity tags.

    `ner_tags` and `ner` say so by name. Otherwise class names, or the tags in
    the first rows, must all be `O` or BIO with at least one BIO tag; numbered
    tags without names (tner) count only under a generic name, so their import
    can say it cannot read the entity names. Chunk and part-of-speech columns
    never count."""
    lowered = name.lower()
    if any(word in lowered for word in NOT_ENTITY_WORDS):
        return False
    if name in NER_NAMED_COLUMNS:
        return True
    if _is_class_label(feature):
        return _all_bio(feature.get("names") or [])
    if _is_integer(feature):
        return name in GENERIC_TAG_COLUMNS
    values = [tag for row in rows if isinstance((cell := (row.get("row") or {}).get(name)), list) for tag in cell]
    return _all_bio(values)


def _all_bio(values):
    tags = [value for value in values if isinstance(value, str) and value.strip()]
    return any(_BIO.match(tag) for tag in tags) and all(tag == "O" or _BIO.match(tag) for tag in tags)


def _first_text(named, preferred, exclude=None):
    """The first plain text column among `preferred` names, in that order."""
    types = dict(named)
    for name in preferred:
        type_ = types.get(name)
        if name != exclude and not _is_list(type_) and _is_text(type_):
            return name
    return None


def _is_list(feature):
    return isinstance(feature, list) or (isinstance(feature, dict) and feature.get("_type") in LIST_TYPES)


def _list_inner(feature):
    """The element type of a list feature; a bare list is a list of one. A list
    of lists stays a list, which no text or tag check accepts."""
    if isinstance(feature, list):
        return feature[0] if feature else None
    return feature.get("feature") if isinstance(feature, dict) else None


def _is_text(feature):
    if isinstance(feature, str):
        return feature in ("string", "large_string")
    return (
        isinstance(feature, dict)
        and feature.get("_type") == "Value"
        and feature.get("dtype") in ("string", "large_string")
    )


def _is_integer(feature):
    return (
        isinstance(feature, dict)
        and feature.get("_type") == "Value"
        and feature.get("dtype") in INTEGER_DTYPES
    )


def _is_tag(feature):
    """A flat tag: a ClassLabel, a plain tag string, or a numbered tag."""
    return _is_text(feature) or _is_integer(feature) or _is_class_label(feature)


def _is_class_label(feature):
    """A ClassLabel itself, not one nested in further lists."""
    return isinstance(feature, dict) and feature.get("_type") == "ClassLabel"


def _row_count(external_id, config, split):
    """The split's size from the viewer's /info, or None when it does not say. Only the size:
    once the first rows answer, a missing /info is not an outage (#121)."""
    try:
        info = viewer.get("info", dataset=external_id, config=config).get("dataset_info") or {}
    except viewer.ViewerUnavailable:
        return None
    count = ((info.get("splits") or {}).get(split) or {}).get("num_examples")
    return count if isinstance(count, int) else None


def _configs(body):
    configs = {}
    for entry in body.get("splits") or []:
        configs.setdefault(entry["config"], []).append(entry["split"])
    return [{"name": name, "splits": splits} for name, splits in configs.items()]


def _type(feature):
    if isinstance(feature, list):
        return "list"
    if not isinstance(feature, dict):
        return "other"
    if feature.get("_type") == "ClassLabel":
        return f"class label, {len(feature.get('names') or [])} classes"
    if feature.get("_type") in ("Sequence", "List", "LargeList"):
        return "list"
    dtype = feature.get("dtype", "")
    if dtype in ("string", "large_string"):
        return "text"
    if dtype.startswith(("int", "uint", "float")):
        return "number"
    if dtype == "bool":
        return "yes/no"
    return dtype or "other"


def _cell(value, feature):
    """A value as the preview shows it: class labels by name, lists joined, text cut."""
    if isinstance(feature, dict) and feature.get("_type") == "ClassLabel" and isinstance(value, int):
        names = feature.get("names") or []
        value = names[value] if 0 <= value < len(names) else value
    elif isinstance(value, list):
        value = ", ".join(str(v) for v in value)
    elif isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=False)
    if isinstance(value, str) and len(value) > CELL_CHARACTERS:
        value = value[: CELL_CHARACTERS - 1] + "…"
    return value
