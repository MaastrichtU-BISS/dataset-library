"""Named-entity rows as legal-ml-core trains on them (#117).

Hugging Face NER datasets hold a list of tokens and a list of BIO tags per
row; the host wants the tokens as one text and the row's distinct entity
types in its ``entities`` column. These are pure functions so the conversion
can be tested without a Parquet file or a database.
"""

# BIO and the neighbouring schemes all put the position before the type.
POSITIONS = frozenset("BIESLU")
PREFIX_SEPARATORS = ("-", "_")


def tokens_to_text(tokens):
    """A row's text: tokens joined by single spaces, a string kept as it is."""
    if tokens is None:
        return ""
    if isinstance(tokens, (list, tuple)):
        return " ".join(str(token) for token in tokens if token is not None)
    return str(tokens)


def entity_types(tags, names=None):
    """The distinct entity types in one row, first seen first, joined by ", ".

    Numbers are names from the dataset's ClassLabel; strings are tags already.
    The BIO position is dropped and ``O`` says "no entity", so neither is a
    type of its own."""
    if tags is None:
        return ""
    if isinstance(tags, (str, int)):
        tags = [tags]
    found = []
    for tag in tags:
        if isinstance(tag, int) and not isinstance(tag, bool):
            if not names or not 0 <= tag < len(names):
                continue
            tag = names[tag]
        if not isinstance(tag, str):
            continue
        tag = tag.strip()
        if len(tag) > 2 and tag[0] in POSITIONS and tag[1] in PREFIX_SEPARATORS:
            tag = tag[2:]
        if tag and tag != "O" and tag not in found:
            found.append(tag)
    return ", ".join(found)
