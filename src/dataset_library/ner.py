"""Named-entity rows as legal-ml-core trains on them (#6, BAIGLE #117).

Hugging Face NER datasets hold a list of tokens and a list of BIO tags per
row; the host wants the tokens as one text and the row's entities as character
spans in that text, the format legal-ml-core's annotation queue stores. These
are pure functions so the conversion can be tested without a Parquet file or a
database.
"""

# BIO and the neighbouring schemes all put the position before the type.
POSITIONS = frozenset("BIESLU")
PREFIX_SEPARATORS = ("-", "_")


def tokens_to_text(tokens):
    """A row's text: tokens joined by single spaces, a string kept as it is.

    Character-level datasets (klue/klue's ner) keep their spaces as tokens of
    their own; those are joined with nothing between them, so the text reads
    as it was written."""
    return _joined(tokens)[0]


def _joined(tokens):
    """A row's text and each kept token's ``(start, end)`` in it.

    Character-level rows are joined as they were written, everything else by
    single spaces; None tokens are skipped. The offsets are where each token
    sits in the returned text, so a span can be cut from it."""
    if tokens is None:
        return "", []
    if not isinstance(tokens, (list, tuple)):
        word = str(tokens)
        return word, [(0, len(word))]
    words = [str(token) for token in tokens if token is not None]
    joiner = "" if _character_level(words) else " "
    parts, offsets = [], []
    at = 0
    for index, word in enumerate(words):
        if index:
            parts.append(joiner)
            at += len(joiner)
        offsets.append((at, at + len(word)))
        parts.append(word)
        at += len(word)
    return "".join(parts), offsets


def _character_level(words):
    """A row of single characters that keeps its spaces as tokens: a word-level
    row with a stray newline or space token is not one."""
    spaces = [word for word in words if word and not word.strip()]
    letters = [word for word in words if word.strip()]
    return bool(spaces) and bool(letters) and all(len(word) == 1 for word in spaces + letters)


def spans(tokens, tags, names=None):
    """A row's text and its entity spans, in the format NER Datasets store.

    Each span is ``{"text", "start", "end", "type"}``, with ``start`` and
    ``end`` character offsets into the returned text, end exclusive, in text
    order and never overlapping. Numbers are names from the dataset's
    ClassLabel; strings are tags already. ``B-`` starts a span, ``I-``
    continues one of its type or starts a new one, ``E-``/``L-`` continue or
    start then close, ``S-``/``U-`` is a one-token span, a tag with no prefix
    continues an open span of its type or starts one, and ``O`` or an empty
    tag is outside. A space kept as a token never starts a span; inside one it
    is included only when the span carries on after it."""
    text, offsets = _joined(tokens)
    tag_list = _as_list(tags)
    if isinstance(tokens, (list, tuple)):
        words = [str(token) for token in tokens if token is not None]
        aligned = [tag for token, tag in zip(tokens, tag_list) if token is not None]
    else:
        words = [] if tokens is None else [str(tokens)]
        aligned = tag_list[:1]
    found = []
    open_type = open_start = open_end = None

    def close():
        nonlocal open_type, open_start, open_end
        if open_type is not None:
            found.append(
                {"text": text[open_start:open_end], "start": open_start, "end": open_end, "type": open_type}
            )
        open_type = open_start = open_end = None

    def open_span(type_, start, end):
        nonlocal open_type, open_start, open_end
        open_type, open_start, open_end = type_, start, end

    for word, (start_at, end_at), tag in zip(words, offsets, aligned):
        if not word.strip():
            continue
        parsed = _parse(tag, names)
        if parsed is None:
            close()
            continue
        position, type_ = parsed
        if position in ("S", "U"):
            close()
            found.append({"text": text[start_at:end_at], "start": start_at, "end": end_at, "type": type_})
            continue
        if position == "B" or open_type != type_:
            close()
            open_span(type_, start_at, end_at)
        else:
            open_end = end_at
        if position in ("E", "L"):
            close()
    close()
    return text, found


def _as_list(tags):
    if tags is None:
        return []
    if isinstance(tags, (list, tuple)):
        return list(tags)
    return [tags]


def _parse(tag, names):
    """A tag as ``(position, type)``, position None for a tag with no prefix,
    or None when it is outside (or a number the names cannot read)."""
    if isinstance(tag, int) and not isinstance(tag, bool):
        if not names or not 0 <= tag < len(names):
            return None
        tag = names[tag]
    if not isinstance(tag, str):
        return None
    tag = tag.strip()
    if not tag or tag == "O":
        return None
    if len(tag) > 2 and tag[0] in POSITIONS and tag[1] in PREFIX_SEPARATORS:
        return tag[0], tag[2:]
    return None, tag
