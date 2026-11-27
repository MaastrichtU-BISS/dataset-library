"""One dataset, opened from the library: its facts, configs and first rows."""
import json
import re

from . import labels as labels_module
from . import ner as ner_module
from . import policy as policy_module
from .licenses import LICENSES, UNKNOWN
from .providers import huggingface, viewer

CACHE_SECONDS = 600
# Set on a preview built while the viewer did not answer: the card stands, but its rows
# are not known, so it is not kept, and a copy of the part (#121) may stand in for them.
VIEWER_DOWN = "viewerDown"
ROWS = 8
# The most rows the viewer's first-rows gives; numbered tags are read from all of them.
FIRST_ROWS = 100
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
# A readable copy of a row's entities without positions, as WikiANN's `spans`
# column holds them: "PER: Ibrahim". The type is checked against the dataset's own.
_TYPED_TEXT = re.compile(r"^([^\s:]+): \S")
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


def preview(host, external_id, config=None, split=None, tag_names=None):
    """One dataset's preview. The library's own rules come first: a dataset
    outside the enabled sources never reaches the Hub or the cache, so its
    cards and viewer answers are not fetched and no refusal reveals it.

    `tag_names`, a checked list (``ner.tag_names_problem``), names an NER
    dataset's numbered tags as the student named them (#180): the `ner` part is
    computed again with them from the raw rows the cached body keeps. The cached
    body itself never holds them, so a preview without them is unchanged."""
    from .search import enabled_sources

    sources = {source.ref for source in enabled_sources(host.db)}
    if sources and external_id.split("/", 1)[0] not in sources:
        raise NotInLibrary()
    key = f"library:preview:{external_id}:{config or ''}:{split or ''}"
    body = host.cache.get(key)
    # A body cached before citations (#120) has no citation, and one cached before
    # the NER preview (#180) no `ner`: rebuild it, so an import started right after
    # the deploy still gets both.
    # An NER body with unnamed numbered tags cached without its raw rows cannot be named either.
    if not isinstance(body, dict) or "citation" not in body or "ner" not in body or _lacks_ner_source(body):
        body = _build(host, external_id, config, split)
        # A preview built while the viewer was down is not kept: the next open asks again.
        if not body.get(VIEWER_DOWN):
            host.cache.set(key, body, CACHE_SECONDS)
    return with_tag_names(_apply_library_rules(host, body), tag_names)


# The raw first rows and features an NER preview with unnamed numbered tags
# keeps in its cached body, so tag names can be applied to them (#180). Never
# served: `with_tag_names` takes it out.
NER_SOURCE = "_nerSource"


def _lacks_ner_source(body):
    """An NER body with unnamed tags whose source is missing, or was kept before it held
    every first row's tags (round 2): the named preview reads `missingTags` from them all."""
    ner = body.get("ner")
    return bool(ner) and not ner.get("namesKnown") and "tags" not in (body.get(NER_SOURCE) or {})


def _ner_source(features, every, tag_column):
    """What a cached body keeps to name its tags later: the features, the shown rows,
    and every first row's tag cell."""
    return {
        "features": features,
        "rows": [{"row": r.get("row") or {}} for r in every[:ROWS]],
        "tags": [(r.get("row") or {}).get(tag_column) for r in every],
    }


def with_tag_names(body, tag_names):
    """The preview as served: `ner` computed with the student's `tag_names` when
    its tags are numbers without names, and `ner.tagNames` saying which names
    were applied (null when none were). A copy: the cached body is not touched."""
    source = body.get(NER_SOURCE)
    body = {k: v for k, v in body.items() if k != NER_SOURCE}
    ner = body.get("ner")
    if not ner:
        return body
    if tag_names and not ner.get("namesKnown") and source:
        tag_column = ner["tagColumn"]
        # The shown rows whole, then the rest of the first rows' tags, so numbers past the
        # eighth row still count; and the base preview's "unnamed" reading is kept.
        rows = source["rows"] + [{"row": {tag_column: tags}} for tags in source["tags"][ROWS:]]
        body["ner"] = _ner_preview(
            source["features"], rows, ner["textColumn"], tag_column, tag_names, unnamed=True,
        )
    else:
        body["ner"] = {**ner, "tagNames": None, "missingTags": []}
    return body


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
        "ner": None,
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
        every = rows.get("rows") or []
        body["ner"] = ner_part(body["proposed"], features, every)
        if body["ner"] and not body["ner"]["namesKnown"]:
            body[NER_SOURCE] = _ner_source(features, every, body["ner"]["tagColumn"])
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
    if _is_class_label(feature) and feature.get("names"):
        return _all_bio(feature["names"])
    # A ClassLabel that names nothing is numbers without names, as an integer column is.
    if _is_integer(feature) or _is_class_label(feature):
        return name in GENERIC_TAG_COLUMNS
    values = [tag for row in rows if isinstance((cell := (row.get("row") or {}).get(name)), list) for tag in cell]
    return _all_bio(values)


def _all_bio(values):
    tags = [value for value in values if isinstance(value, str) and value.strip()]
    return any(_BIO.match(tag) for tag in tags) and all(tag == "O" or _BIO.match(tag) for tag in tags)


def ner_part(proposed, features, rows, tag_names=None):
    """The preview's `ner`: what the proposed NER columns hold, or None when
    the proposal is not NER. `rows` are the viewer's `{"row": {...}}` entries,
    every first row it gave: the first ROWS are shown."""
    if proposed.get("taskType") != "ner":
        return None
    return _ner_preview(features, rows, proposed["textColumn"], proposed["labelColumn"], tag_names)


def _ner_preview(features, rows, text_column, tag_column, tag_names=None, unnamed=False):
    """What an NER dataset holds, read from its first rows (#180).

    `rows` are every first row the viewer gave (up to FIRST_ROWS); the first
    ROWS of them are shown and counted, and numbered tags are read from all.

    `labels` are the tag names: a ClassLabel's names in order (an empty one
    names nothing), or for string tags the distinct tags in the shown rows,
    `O` first and the rest as first seen, or for numbers without names every
    number from ``"0"`` to the highest non-negative tag in any of `rows`, as
    strings, at most MAX_TAG_NAMES, with `namesKnown` false: position i is tag
    number i, gaps included. Negative numbers (``-100`` padding) are outside
    any entity and never a label. `entityTypes` are the types in `labels` with
    their position prefix stripped and `O` left out, in that order, each with
    the number of spans of it across the shown rows (0 when none shows).
    Numbers without names have no types: `ner.spans` cannot read a number it
    has no name for and leaves it outside, as the import refuses such tags, so
    `entityTypes` is empty and every row has no entities.

    `rows` in the result are the shown rows as the import writes them: a row
    with a tag for every token (`ner.complete_tags`) is converted by
    `ner.spans` into its text and entities; a row of tokens whose tags are
    missing, short or partly empty is shown with no entities and counted in
    `blankRows`, as it is imported with a blank entities cell; a row whose
    tokens are not a list, or whose text is empty, is left out and counted in
    `skippedRows`. `typesOnlyColumns` are the other list columns whose values
    all read `TYPE: text`, at least one of them: entity types without positions
    (WikiANN's `spans`), which cannot be trained on. With the types known (or
    named), TYPE must be one of `entityTypes`; with numbers without names any
    TYPE counts, so such a column is flagged before the student names the tags.

    `unnamedNumberedColumns` are the other list columns, in feature order, that
    hold numbers without names: their element type is an integer or a ClassLabel
    with no names, or every value of theirs in the shown rows is a list of
    integers (at least one), and the feature names none. tner's `pos_tags` is
    one. The served rows join lists into text, so this says what they hold.

    `tag_names`, when the tags are numbers without names, are the student's
    names for them (#180): item i names number i. The rows are converted with
    them and `entityTypes` are their types, as the import will read them;
    `labels` stay the numbers and `namesKnown` false, and `tagNames` echoes
    them. `missingTags` are then the non-negative tag numbers in any of `rows`
    that the names do not reach (number >= len(tag_names)), sorted and each
    once: a number not in the rows is never listed. Otherwise `tagNames` is
    null, `missingTags` is ``[]`` and the names change nothing.

    `unnamed` keeps a reading made before (the base preview read every first
    row): the tags are numbers without names whatever these rows show."""
    types = {f["name"]: f.get("type") for f in features}
    # An empty ClassLabel names nothing: on the NER paths it is numbers without names.
    names = ner_module.class_label_names(types.get(tag_column)) or None
    inner = _list_inner(types.get(tag_column))
    every = [(r.get("row") or {}) for r in rows]
    cells = every[:ROWS]
    if names is not None:
        labels, names_known = [str(name) for name in names], True
    else:
        seen = [tag for cell in cells if isinstance(cell.get(tag_column), list) for tag in cell.get(tag_column)]
        numbers = [
            tag for cell in every if isinstance(cell.get(tag_column), list) for tag in cell.get(tag_column)
            if isinstance(tag, int) and not isinstance(tag, bool)
        ]
        strings = list(dict.fromkeys(tag for tag in seen if isinstance(tag, str)))
        names_known = not (unnamed or _is_integer(inner) or _is_class_label(inner) or (numbers and not strings))
        if names_known:
            labels = (["O"] if "O" in strings else []) + [tag for tag in strings if tag != "O"]
        else:
            highest = max((number for number in numbers if number >= 0), default=-1)
            labels = [str(number) for number in range(min(highest + 1, ner_module.MAX_TAG_NAMES))]
    applied, missing = None, []
    if names is None and not names_known and tag_names:
        names = applied = list(tag_names)
        missing = sorted({number for number in numbers if number >= len(applied)})
    converted, skipped, blank = [], 0, 0
    for cell in cells:
        tokens, tags = cell.get(text_column), cell.get(tag_column)
        if not isinstance(tokens, list) or not ner_module.tokens_to_text(tokens).strip():
            skipped += 1
            continue
        if not ner_module.complete_tags(tokens, tags):
            blank += 1
            converted.append({"text": ner_module.tokens_to_text(tokens), "entities": []})
            continue
        text, entities = ner_module.spans(tokens, tags, names)
        converted.append({"text": text, "entities": entities})
    counts = {}
    if names_known or applied:
        for label in applied or labels:
            type_ = ner_module.entity_type(label)
            if type_ is not None:
                counts.setdefault(type_, 0)
    for row in converted:
        for entity in row["entities"]:
            counts[entity["type"]] = counts.get(entity["type"], 0) + 1
    entity_types = [{"type": type_, "count": count} for type_, count in counts.items()]
    others = [name for name in types if name not in (text_column, tag_column)]
    # Before the types are known, any TYPE may be one.
    known_types = set(counts) if names_known or applied else None
    return {
        "textColumn": text_column,
        "tagColumn": tag_column,
        "labels": labels,
        "namesKnown": names_known,
        "entityTypes": entity_types,
        "rows": converted,
        "skippedRows": skipped,
        "blankRows": blank,
        "typesOnlyColumns": [name for name in others if _types_only(name, cells, known_types)],
        "unnamedNumberedColumns": [name for name in others if _unnamed_numbers(types[name], name, cells)],
        "tagNames": applied,
        "missingTags": missing,
    }


def _types_only(name, cells, entity_types):
    """Whether `name` is a list column whose every value in these rows is a list
    of strings reading `TYPE: text` with TYPE one of `entity_types` (any TYPE
    when it is None), at least one of them. A plain text column is never one:
    a summary may well start "ORG: "."""
    values = []
    for cell in cells:
        value = cell.get(name)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            values.extend(value)
        elif value is not None:
            return False
    values = [value for value in values if value.strip()]
    if not values or entity_types is not None and not entity_types:
        return False
    for value in values:
        match = _TYPED_TEXT.match(value.strip())
        if not match or (entity_types is not None and match.group(1) not in entity_types):
            return False
    return True


def _unnamed_numbers(feature, name, cells):
    """Whether a column is a list of numbers without names (see `_ner_preview`)."""
    if not _is_list(feature) or ner_module.class_label_names(feature):
        return False
    inner = _list_inner(feature)
    if _is_integer(inner) or _is_class_label(inner):
        return True
    values = [cell.get(name) for cell in cells if cell.get(name) is not None]
    return bool(values) and any(values) and all(
        isinstance(value, list) and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
        for value in values
    )


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
