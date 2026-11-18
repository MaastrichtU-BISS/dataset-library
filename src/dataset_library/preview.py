"""One dataset, opened from the library: its facts, configs and first rows."""
import json

from . import labels as labels_module
from . import policy as policy_module
from .licenses import LICENSES, UNKNOWN
from .providers import huggingface, viewer

CACHE_SECONDS = 600
ROWS = 8
CELL_CHARACTERS = 400
# Names that say which column is which, before falling back to the types.
NER_TEXT_COLUMNS = ("tokens", "words")
NER_LABEL_COLUMNS = ("ner_tags", "tags")
SUMMARY_TEXT_COLUMNS = ("document", "article", "text", "dialogue", "source", "input")
SUMMARY_LABEL_COLUMNS = ("summary", "highlights", "abstract", "target", "output")
LIST_TYPES = ("Sequence", "List", "LargeList")


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
    if body is None:
        body = _build(host, external_id, config, split)
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
    except viewer.ViewerUnavailable:
        if config:
            raise UnknownConfig()
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
    except viewer.ViewerUnavailable:
        return body
    features = rows.get("features") or []
    body["columns"] = [{"name": f["name"], "type": _type(f.get("type"))} for f in features]
    body["rows"] = [
        [_cell(r.get("row", {}).get(f["name"]), f.get("type")) for f in features]
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
    when the viewer shows them, and nothing when it cannot tell."""
    if "ner" in tasks:
        task = "ner"
    elif "summarization" in tasks:
        task = "summarization"
    else:
        return _no_proposal()
    named = [(f.get("name"), f.get("type")) for f in features if isinstance(f, dict)]
    if task == "ner":
        text = _pick(
            [name for name, type_ in named if _is_list(type_) and _is_text(_list_inner(type_))],
            NER_TEXT_COLUMNS,
        )
        label = _pick(
            [name for name, type_ in named if _is_list(type_) and (
                _is_text(_list_inner(type_)) or labels_module.is_class_label(_list_inner(type_))
            )],
            NER_LABEL_COLUMNS, suffix="tags",
        )
    else:
        label = _first_text(named, SUMMARY_LABEL_COLUMNS)
        text = _first_text(named, SUMMARY_TEXT_COLUMNS, exclude=label)
        if text is None:
            text = _longest_text(named, rows, exclude=label)
    return {"taskType": task, "textColumn": text, "labelColumn": label}


def _pick(candidates, preferred, suffix=None):
    for name in preferred:
        if name in candidates:
            return name
    if suffix:
        for name in candidates:
            if name.endswith(suffix):
                return name
    return candidates[0] if candidates else None


def _first_text(named, preferred, exclude=None):
    """The first plain text column among `preferred` names, in that order."""
    types = dict(named)
    for name in preferred:
        type_ = types.get(name)
        if name != exclude and not _is_list(type_) and _is_text(type_):
            return name
    return None


def _longest_text(named, rows, exclude=None):
    """The plain text column whose sample rows read longest."""
    best, best_length = None, -1.0
    for name, type_ in named:
        if name == exclude or _is_list(type_) or not _is_text(type_):
            continue
        lengths = [len(value) for row in rows if isinstance((value := (row.get("row") or {}).get(name)), str)]
        length = sum(lengths) / len(lengths) if lengths else 0
        if length > best_length:
            best, best_length = name, length
    return best


def _is_list(feature):
    return isinstance(feature, list) or (isinstance(feature, dict) and feature.get("_type") in LIST_TYPES)


def _list_inner(feature):
    """The element type of a list feature; a bare list is a list of one."""
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


def _row_count(external_id, config, split):
    """The split's size from the viewer's /info, or None when it does not say."""
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
