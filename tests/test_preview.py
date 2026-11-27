from dataset_library.preview import _list_inner, _ner_preview, _propose
from tests.conftest import ALICE

NOTHING = {"taskType": None, "textColumn": None, "labelColumn": None}


def preview(client, external_id, status=200, **params):
    resp = client.get(f"/api/library/datasets/huggingface/{external_id}", query_string=params, headers=ALICE)
    assert resp.status_code == status, resp.get_json()
    return resp.get_json()


def test_the_first_config_and_its_train_split_preview_by_default(client):
    body = preview(client, "AI-team-UoA/greek_legal_code")

    assert body["id"] == "huggingface:AI-team-UoA/greek_legal_code"
    assert body["title"] == "Greek Legal Code"
    assert body["preview"] == "ok"
    assert body["config"] == "volume" and body["split"] == "train"
    assert [c["name"] for c in body["configs"]] == ["volume", "chapter", "subject"]
    assert body["configs"][0]["splits"] == ["train", "test", "validation"]
    assert [c["name"] for c in body["columns"]] == ["text", "label"]
    assert body["columns"][0]["type"] == "text"
    assert body["columns"][1]["type"] == "class label, 47 classes"
    assert len(body["rows"]) == 8
    assert body["rowCount"] == 28536  # the train split, from the viewer's /info
    # Class labels arrive as numbers and are shown by name.
    assert all(isinstance(row[1], str) for row in body["rows"])
    assert body["labels"] == "yes"
    assert body["license"]["id"] == "cc-by-4.0"
    assert body["tasks"] == ["classification"]


def test_another_config_and_split_can_be_previewed(client):
    body = preview(client, "nguha/legalbench", config="hearsay", split="test")

    assert body["config"] == "hearsay" and body["split"] == "test"
    assert [c["name"] for c in body["columns"]] == ["index", "answer", "text", "slice"]
    assert body["columns"][0]["type"] == "number"
    assert body["rows"][0][1] == "No"
    assert body["labels"] == "unclear"


def test_the_description_is_its_first_paragraph_as_plain_text(client):
    body = preview(client, "coastalcph/lex_glue", config="ledgar")

    assert body["description"].startswith("Inspired by the recent widespread use of the GLUE")
    assert "\n" not in body["description"] and len(body["description"]) <= 300


def test_a_dataset_the_viewer_cannot_read_still_shows_its_facts(client):
    body = preview(client, "jfrenz/legalglue")

    assert body["preview"] == "unavailable"
    assert body["rows"] == [] and body["columns"] == []
    assert body["rowCount"] is None
    assert body["title"] == "LegalGLUE"
    assert body["tasks"] == ["classification", "ner"]


def test_a_gated_dataset_has_no_preview_and_says_it_is_gated(client):
    body = preview(client, "free-law/Caselaw_Access_Project")

    assert body["gated"] is True
    assert body["preview"] == "unavailable"


def test_an_unknown_config_is_refused(client):
    preview(client, "AI-team-UoA/greek_legal_code", status=400, config="nonsense")


def test_a_dataset_outside_the_library_is_not_found(app, client):
    from dataset_library.models import LibrarySettings, LibrarySource

    preview(client, "nobody/nothing", status=404)
    with app.app_context():
        app.db.session.add(LibrarySource(provider="huggingface", kind="org", ref="coastalcph"))
        app.db.session.add(LibrarySettings(id=1, license_policy={"mit": "hide"}))
        app.db.session.commit()
    preview(client, "nguha/legalbench", status=404)  # not an enabled source
    preview(client, "coastalcph/lex_glue", config="ledgar")


def test_a_hidden_license_is_not_found(app, client):
    from dataset_library.models import LibrarySettings

    with app.app_context():
        app.db.session.add(LibrarySettings(id=1, license_policy={"cc-by-4.0": "hide"}))
        app.db.session.commit()
    preview(client, "AI-team-UoA/greek_legal_code", status=404)


def test_a_path_like_id_never_leaves_the_server(client, hub):
    client.get("/api/library/datasets/huggingface/../../etc/passwd", headers=ALICE)
    resp = client.get("/api/library/datasets/huggingface/a/b/c", headers=ALICE)

    assert resp.status_code in (400, 404)
    assert hub.calls == []


def test_previews_are_cached_for_ten_minutes(client, hub):
    preview(client, "AI-team-UoA/greek_legal_code")
    calls = len(hub.calls)
    preview(client, "AI-team-UoA/greek_legal_code")

    assert len(hub.calls) == calls


def test_a_preview_cached_before_citations_is_rebuilt_with_one(client, hub, cache):
    # A body cached by 0.10.x has no citation: it is a miss, so an import started now still gets one (#120).
    preview(client, "AI-team-UoA/greek_legal_code")
    key = "library:preview:AI-team-UoA/greek_legal_code::"
    cache.set(key, {k: v for k, v in cache.get(key).items() if k != "citation"}, 600)
    calls = len(hub.calls)

    body = preview(client, "AI-team-UoA/greek_legal_code")

    assert len(hub.calls) > calls
    assert body["citation"]["revision"] == "de0fdb34424f07d1ac6f0ede23ee0ed44bd9f5d1"
    assert cache.get(key)["citation"]["revision"] == "de0fdb34424f07d1ac6f0ede23ee0ed44bd9f5d1"


def test_an_ner_card_proposes_its_tokens_and_tag_columns(client):
    body = preview(client, "coastalcph/legal_ner")

    assert body["tasks"] == ["ner"]
    assert body["proposed"] == {"taskType": "ner", "textColumn": "tokens", "labelColumn": "ner_tags"}


def test_a_summarization_card_proposes_its_document_and_summary_columns(client):
    body = preview(client, "coastalcph/legal_summaries")

    assert body["tasks"] == ["summarization"]
    assert body["proposed"] == {"taskType": "summarization", "textColumn": "document", "labelColumn": "summary"}


def test_a_classification_card_proposes_nothing(client):
    body = preview(client, "AI-team-UoA/greek_legal_code")

    assert body["proposed"] == NOTHING


def test_an_unavailable_preview_proposes_nothing(client):
    body = preview(client, "jfrenz/legalglue")

    assert body["preview"] == "unavailable"
    assert body["proposed"] == NOTHING


def _string():
    return {"dtype": "string", "_type": "Value"}


def _integer(dtype="int64"):
    return {"dtype": dtype, "_type": "Value"}


def _sequence(feature):
    return {"_type": "Sequence", "feature": feature}


def _class_label(names):
    return {"_type": "ClassLabel", "names": names}


def test_ner_prefers_tokens_and_ner_tags_over_the_other_names():
    features = [
        {"name": "words", "type": _sequence(_string())},
        {"name": "tokens", "type": _sequence(_string())},
        {"name": "tags", "type": _sequence(_class_label(["O"]))},
        {"name": "ner_tags", "type": _sequence(_class_label(["O"]))},
    ]

    assert _propose(["ner"], features, []) == {"taskType": "ner", "textColumn": "tokens", "labelColumn": "ner_tags"}


def test_ner_takes_a_string_tag_sequence_and_a_name_ending_in_tags():
    features = [
        {"name": "words", "type": _sequence(_string())},
        {"name": "pos", "type": _sequence(_class_label(["NN"]))},
        {"name": "entity_tags", "type": _sequence(_string())},
    ]
    rows = [{"row": {"words": ["Ada", "Lovelace"], "pos": [0, 0], "entity_tags": ["B-PER", "I-PER"]}}]

    assert _propose(["ner"], features, rows) == {"taskType": "ner", "textColumn": "words", "labelColumn": "entity_tags"}


def test_ner_covers_the_viewers_bare_list_type():
    features = [
        {"name": "tokens", "type": [_string()]},
        {"name": "ner_tags", "type": [_class_label(["O", "B-PER"])]},
    ]

    assert _propose(["ner"], features, []) == {"taskType": "ner", "textColumn": "tokens", "labelColumn": "ner_tags"}


def test_ner_reads_numbered_tags_and_keeps_the_text_column_distinct():
    features = [
        {"name": "tokens", "type": _sequence(_string())},
        {"name": "tags", "type": _sequence(_integer())},
    ]

    assert _propose(["ner"], features, []) == {"taskType": "ner", "textColumn": "tokens", "labelColumn": "tags"}


def test_ner_reads_a_class_label_sequence_next_to_the_text():
    features = [
        {"name": "tokens", "type": _sequence(_string())},
        {"name": "labels", "type": _sequence(_class_label(["O", "B-PER"]))},
    ]

    assert _propose(["ner"], features, []) == {"taskType": "ner", "textColumn": "tokens", "labelColumn": "labels"}


def test_ner_prefers_words_when_the_tag_column_is_called_ner():
    features = [
        {"name": "words", "type": _sequence(_string())},
        {"name": "ner", "type": _sequence(_string())},
    ]

    assert _propose(["ner"], features, []) == {"taskType": "ner", "textColumn": "words", "labelColumn": "ner"}


def test_ner_never_guesses_the_text_from_other_string_lists():
    # Without a tokens or words column the student picks; a guess once took a
    # part-of-speech list (DFKI-SLT/wikitext_linked's tok_upos) as the text.
    sentences = [
        {"name": "ner_tags", "type": _sequence(_string())},
        {"name": "sentences", "type": _sequence(_string())},
    ]
    wikitext = [
        {"name": "text", "type": _string()},
        {"name": "tok_upos", "type": _sequence(_string())},
        {"name": "tok_ner", "type": _sequence(_string())},
    ]

    assert _propose(["ner"], sentences, []) == NOTHING
    assert _propose(["ner"], wikitext, []) == NOTHING


def test_a_hyphenated_word_does_not_make_the_tokens_look_like_tags():
    features = [
        {"name": "tokens", "type": _sequence(_string())},
        {"name": "ner_tags", "type": _sequence(_class_label(["O", "B-PER", "I-PER"]))},
    ]
    rows = [{"row": {"tokens": ["Send", "an", "E-mail", "about", "B-cell", "research"], "ner_tags": [0, 0, 0, 0, 0, 0]}}]

    assert _propose(["ner"], features, rows) == {"taskType": "ner", "textColumn": "tokens", "labelColumn": "ner_tags"}


def test_a_generic_tag_column_must_hold_entity_tags():
    # batterydata/pos_tagging: words and part-of-speech tags called labels.
    pos_strings = [
        {"name": "words", "type": _sequence(_string())},
        {"name": "labels", "type": _sequence(_string())},
    ]
    pos_rows = [{"row": {"words": ["It", "runs"], "labels": ["PRP", "VBZ"]}}]
    pos_names = [
        {"name": "tokens", "type": _sequence(_string())},
        {"name": "labels", "type": _sequence(_class_label(["NN", "VB"]))},
    ]
    stray = [
        {"name": "tokens", "type": _sequence(_string())},
        {"name": "tags", "type": _sequence(_string())},
    ]
    stray_rows = [{"row": {"tokens": ["a", "b"], "tags": ["B-PER", "NN"]}}]

    assert _propose(["ner"], pos_strings, pos_rows) == NOTHING
    assert _propose(["ner"], pos_names, []) == NOTHING
    assert _propose(["ner"], stray, stray_rows) == NOTHING


def test_chunk_tags_are_bio_but_not_entities():
    # conll2000: tokens, pos_tags and B-NP/I-VP chunk tags.
    features = [
        {"name": "tokens", "type": _sequence(_string())},
        {"name": "pos_tags", "type": _sequence(_class_label(["NN", "VB"]))},
        {"name": "chunk_tags", "type": _sequence(_class_label(["O", "B-NP", "I-NP", "B-VP"]))},
    ]

    assert _propose(["ner"], features, []) == NOTHING


def test_a_nested_tag_column_next_to_flat_tokens_is_not_proposed():
    features = [
        {"name": "tokens", "type": _sequence(_string())},
        {"name": "ner_tags", "type": _sequence([_class_label(["O", "B-PER"])])},
    ]

    assert _propose(["ner"], features, []) == NOTHING


def test_ner_with_only_a_text_like_column_proposes_nothing():
    features = [{"name": "tokens", "type": _sequence(_string())}]

    assert _propose(["ner"], features, []) == NOTHING


def test_ner_without_the_expected_columns_proposes_nothing():
    features = [{"name": "text", "type": _string()}]

    assert _propose(["ner"], features, []) == NOTHING


def test_ner_with_a_scalar_text_and_label_proposes_nothing():
    # klue/klue's ynat config is tagged ner but has one text and one label.
    features = [
        {"name": "text", "type": _string()},
        {"name": "label", "type": _class_label(["yes", "no"])},
    ]

    assert _propose(["ner"], features, []) == NOTHING


def test_a_card_tagged_both_tasks_falls_back_when_ner_has_no_pair():
    features = [
        {"name": "article", "type": _string()},
        {"name": "summary", "type": _string()},
    ]

    assert _propose(["ner", "summarization"], features, []) == {
        "taskType": "summarization", "textColumn": "article", "labelColumn": "summary",
    }


def test_list_inner_keeps_a_list_of_lists_a_list():
    # In Hugging Face's features a list inside a Sequence is a list of lists.
    feature = {"_type": "Sequence", "feature": [{"_type": "Value", "dtype": "string"}]}

    assert _list_inner(feature) == [{"_type": "Value", "dtype": "string"}]


def test_ner_declines_nested_token_and_tag_lists():
    features = [
        {"name": "tokens", "type": _sequence([_string()])},
        {"name": "ner_tags", "type": _sequence([_class_label(["O", "B-PER"])])},
    ]

    assert _propose(["ner"], features, []) == NOTHING


def test_a_dependency_parsing_config_on_an_ner_card_proposes_nothing():
    # klue/klue's dp config: tagged token classification, but its numbered
    # lists are word positions and heads, not entity tags.
    features = [
        {"name": "sentence", "type": _string()},
        {"name": "index", "type": _sequence(_integer("int32"))},
        {"name": "word_form", "type": _sequence(_string())},
        {"name": "lemma", "type": _sequence(_string())},
        {"name": "pos", "type": _sequence(_string())},
        {"name": "head", "type": _sequence(_integer("int32"))},
        {"name": "deprel", "type": _sequence(_string())},
    ]
    rows = [{"row": {"index": [1, 2], "word_form": ["a", "b"], "lemma": ["a", "b"], "pos": ["NNG", "VV"],
                     "head": [2, 0], "deprel": ["NP_SBJ", "VP"]}}]

    assert _propose(["ner"], features, rows) == NOTHING


def test_part_of_speech_tags_are_not_entity_types():
    # A Universal Dependencies shape, and a tokens + pos_tags one.
    upos = [
        {"name": "tokens", "type": _sequence(_string())},
        {"name": "upos", "type": _sequence(_class_label(["NOUN", "VERB", "ADJ"]))},
    ]
    pos_tags = [
        {"name": "tokens", "type": _sequence(_string())},
        {"name": "pos_tags", "type": _sequence(_class_label(["NN", "VB", "IN"]))},
    ]

    assert _propose(["ner"], upos, []) == NOTHING
    assert _propose(["ner"], pos_tags, []) == NOTHING


def test_an_unusual_tag_column_with_bio_class_names_is_the_label():
    features = [
        {"name": "tokens", "type": _sequence(_string())},
        {"name": "pos_tags", "type": _sequence(_class_label(["NN", "VB"]))},
        {"name": "entities_bio", "type": _sequence(_class_label(["O", "B-ORG", "I-ORG"]))},
    ]

    assert _propose(["ner"], features, []) == {"taskType": "ner", "textColumn": "tokens", "labelColumn": "entities_bio"}


def test_summarization_never_picks_an_id_column_as_the_text():
    features = [
        {"name": "id", "type": _string()},
        {"name": "doc_id", "type": _string()},
        {"name": "summary", "type": _string()},
    ]
    rows = [{"row": {"id": "a-very-long-identifier-0001", "doc_id": "another-long-id", "summary": "s"}}]

    assert _propose(["summarization"], features, rows) == NOTHING


def test_a_feature_without_a_name_is_ignored():
    features = [
        {"type": _sequence(_string())},
        {"name": "tokens", "type": _sequence(_string())},
        {"name": "ner_tags", "type": _sequence(_class_label(["O", "B-PER"]))},
    ]

    assert _propose(["ner"], features, []) == {"taskType": "ner", "textColumn": "tokens", "labelColumn": "ner_tags"}


def test_summarization_never_guesses_the_source_from_other_text_columns():
    # ai-sage/TimeGround-1M's `sum` config once gave `split` as the source,
    # and arxiv-abstracts-2021 its `title`.
    split = [
        {"name": "split", "type": _string()},
        {"name": "summary", "type": _string()},
    ]
    title = [
        {"name": "id", "type": _string()},
        {"name": "title", "type": _string()},
        {"name": "abstract", "type": _string()},
    ]
    rows = [{"row": {"split": "train", "title": "A long title about many things", "summary": "s"}}]

    assert _propose(["summarization"], split, rows) == NOTHING
    assert _propose(["summarization"], title, rows) == NOTHING


def test_summarization_reads_report_and_description_sources():
    # govreport-style and big_patent-style columns.
    report = [{"name": "report", "type": _string()}, {"name": "summary", "type": _string()}]
    patent = [{"name": "description", "type": _string()}, {"name": "abstract", "type": _string()}]

    assert _propose(["summarization"], report, []) == {
        "taskType": "summarization", "textColumn": "report", "labelColumn": "summary",
    }
    assert _propose(["summarization"], patent, []) == {
        "taskType": "summarization", "textColumn": "description", "labelColumn": "abstract",
    }


def test_summarization_uses_the_usual_names_in_order():
    features = [
        {"name": "source", "type": _string()},
        {"name": "article", "type": _string()},
        {"name": "highlights", "type": _string()},
        {"name": "summary", "type": _string()},
    ]

    assert _propose(["summarization"], features, []) == {
        "taskType": "summarization", "textColumn": "article", "labelColumn": "summary",
    }


def test_ner_wins_when_the_card_lists_both_tasks():
    features = [
        {"name": "tokens", "type": _sequence(_string())},
        {"name": "ner_tags", "type": _sequence(_class_label(["O"]))},
    ]

    assert _propose(["ner", "summarization"], features, [])["taskType"] == "ner"


def test_a_generic_target_or_output_is_not_a_summary():
    # flytech/python-codes-25k: an instruction, its input and code as output.
    features = [
        {"name": "instruction", "type": _string()},
        {"name": "text", "type": _string()},
        {"name": "output", "type": _string()},
    ]

    assert _propose(["summarization"], features, []) == NOTHING


# The NER preview (#180): what an NER dataset's tags mean, before columns are picked.
WIKIANN_NAMES = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


def _features(**types):
    return [{"name": name, "type": type_} for name, type_ in types.items()]


def _rows(*cells):
    return [{"row_idx": index, "row": cell} for index, cell in enumerate(cells)]


def _wikiann():
    features = _features(
        tokens=_sequence(_string()), ner_tags=_sequence(_class_label(WIKIANN_NAMES)),
        langs=_sequence(_string()), spans=_sequence(_string()), note=_string(),
    )
    rows = _rows(
        {"tokens": ["Angela", "Merkel", "visited", "Paris", "."], "ner_tags": [1, 2, 0, 5, 0],
         "langs": ["en"] * 5, "spans": ["PER: Angela Merkel", "LOC: Paris"], "note": "Note: a state visit"},
        {"tokens": ["Ibrahim", "joined", "Acme", "Corp", "in", "Berlin"], "ner_tags": [1, 0, 3, 4, 0, 5],
         "langs": ["en"] * 6, "spans": ["PER: Ibrahim", "ORG: Acme Corp", "LOC: Berlin"], "note": "plain text"},
        {"tokens": ["Rome", "and", "Bob"], "ner_tags": [5, 0, 1],
         "langs": ["en"] * 3, "spans": ["LOC: Rome", "PER: Bob"], "note": ""},
    )
    return features, rows


def test_a_class_label_ner_preview_names_its_tags_types_and_rows():
    features, rows = _wikiann()

    ner = _ner_preview(features, rows, "tokens", "ner_tags")

    assert ner["textColumn"] == "tokens" and ner["tagColumn"] == "ner_tags"
    assert ner["labels"] == WIKIANN_NAMES
    assert ner["namesKnown"] is True
    assert ner["entityTypes"] == [{"type": "PER", "count": 3}, {"type": "ORG", "count": 1}, {"type": "LOC", "count": 3}]
    assert ner["rows"][0] == {
        "text": "Angela Merkel visited Paris .",
        "entities": [
            {"text": "Angela Merkel", "start": 0, "end": 13, "type": "PER"},
            {"text": "Paris", "start": 22, "end": 27, "type": "LOC"},
        ],
    }
    assert ner["rows"][1]["entities"][1] == {"text": "Acme Corp", "start": 15, "end": 24, "type": "ORG"}
    assert len(ner["rows"]) == 3 and ner["skippedRows"] == 0
    # `spans` repeats the entities without positions; `langs` and a "Note: …" text column do not.
    assert ner["typesOnlyColumns"] == ["spans"]


def test_a_type_with_no_span_in_the_preview_still_shows_with_none():
    features, rows = _wikiann()

    ner = _ner_preview(features, rows[:1], "tokens", "ner_tags")

    assert ner["entityTypes"] == [{"type": "PER", "count": 1}, {"type": "ORG", "count": 0}, {"type": "LOC", "count": 1}]


def test_string_tags_are_their_own_labels_with_o_first():
    features = _features(tokens=_sequence(_string()), ner_tags=_sequence(_string()))
    rows = _rows(
        {"tokens": ["The", "Court", "of", "Justice", "ruled"], "ner_tags": ["B-ORG", "I-ORG", "I-ORG", "I-ORG", "O"]},
        {"tokens": ["Judge", "Smith", "said"], "ner_tags": ["O", "S-PER", "O"]},
    )

    ner = _ner_preview(features, rows, "tokens", "ner_tags")

    assert ner["labels"] == ["O", "B-ORG", "I-ORG", "S-PER"]
    assert ner["namesKnown"] is True
    assert ner["entityTypes"] == [{"type": "ORG", "count": 1}, {"type": "PER", "count": 1}]
    assert ner["rows"][0]["entities"] == [{"text": "The Court of Justice", "start": 0, "end": 20, "type": "ORG"}]
    assert ner["rows"][1]["entities"] == [{"text": "Smith", "start": 6, "end": 11, "type": "PER"}]
    assert ner["typesOnlyColumns"] == []


def test_numbered_tags_without_names_are_not_known_and_have_no_types():
    features = _features(tokens=_sequence(_string()), tags=_sequence(_integer()), spans=_sequence(_string()))
    rows = _rows(
        {"tokens": ["Angela", "Merkel", "visited", "Paris"], "tags": [3, 4, 0, 1], "spans": ["PER: Angela Merkel"]},
        {"tokens": ["Berlin"], "tags": [1], "spans": ["LOC: Berlin"]},
    )

    ner = _ner_preview(features, rows, "tokens", "tags")

    assert ner["labels"] == ["0", "1", "3", "4"]
    assert ner["namesKnown"] is False
    # ner.spans cannot read a number without a name, so nothing becomes an entity type.
    assert ner["entityTypes"] == []
    assert ner["rows"] == [
        {"text": "Angela Merkel visited Paris", "entities": []},
        {"text": "Berlin", "entities": []},
    ]
    # With no known types, nothing reads as a types-only column either.
    assert ner["typesOnlyColumns"] == []


def test_a_row_that_cannot_be_converted_is_left_out_and_counted():
    features, rows = _wikiann()
    rows[1]["row"]["ner_tags"] = [1, 0]  # fewer tags than tokens
    rows.append({"row_idx": 3, "row": {"tokens": None, "ner_tags": [0]}})  # a missing value

    ner = _ner_preview(features, rows, "tokens", "ner_tags")

    assert [row["text"] for row in ner["rows"]] == ["Angela Merkel visited Paris .", "Rome and Bob"]
    assert ner["skippedRows"] == 2
    assert ner["entityTypes"] == [{"type": "PER", "count": 2}, {"type": "ORG", "count": 0}, {"type": "LOC", "count": 2}]


def test_a_types_only_column_needs_every_value_to_name_a_known_type():
    features, rows = _wikiann()
    rows[2]["row"]["spans"] = ["LOC: Rome", "Plain words"]

    ner = _ner_preview(features, rows, "tokens", "ner_tags")

    assert ner["typesOnlyColumns"] == []


def test_a_types_only_string_column_is_flagged_too():
    features, rows = _wikiann()
    features.append({"name": "summary", "type": _string()})
    for row in rows:
        row["row"]["summary"] = row["row"]["spans"][0]

    ner = _ner_preview(features, rows, "tokens", "ner_tags")

    assert ner["typesOnlyColumns"] == ["spans", "summary"]


def test_an_ner_preview_carries_its_tags_through_the_route(client):
    body = preview(client, "coastalcph/legal_ner")

    ner = body["ner"]
    assert ner["labels"] == ["O", "B-PER", "I-PER", "B-ORG", "I-ORG"]
    assert ner["namesKnown"] is True
    assert ner["entityTypes"] == [{"type": "PER", "count": 2}, {"type": "ORG", "count": 2}]
    assert ner["rows"][0] == {
        "text": "John Smith works at ACME",
        "entities": [
            # The recorded tags are [O, B-PER, I-PER, O, B-ORG]: the span is what they say.
            {"text": "Smith works", "start": 5, "end": 16, "type": "PER"},
            {"text": "ACME", "start": 20, "end": 24, "type": "ORG"},
        ],
    }
    assert ner["skippedRows"] == 0 and ner["typesOnlyColumns"] == []


def test_a_preview_that_is_not_ner_has_no_ner_part(client):
    assert preview(client, "AI-team-UoA/greek_legal_code")["ner"] is None
    assert preview(client, "coastalcph/legal_summaries")["ner"] is None
    assert preview(client, "jfrenz/legalglue")["ner"] is None


def test_a_preview_cached_before_the_ner_part_is_rebuilt_with_it(client, hub, cache):
    preview(client, "coastalcph/legal_ner")
    key = "library:preview:coastalcph/legal_ner::"
    cache.set(key, {k: v for k, v in cache.get(key).items() if k != "ner"}, 600)

    assert preview(client, "coastalcph/legal_ner")["ner"]["labels"][0] == "O"
