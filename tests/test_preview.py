from dataset_library.preview import _list_inner, _propose
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

    assert _propose(["ner"], features, []) == {"taskType": "ner", "textColumn": "words", "labelColumn": "entity_tags"}


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


def test_ner_keeps_a_string_tag_column_from_being_its_own_text():
    features = [
        {"name": "ner_tags", "type": _sequence(_string())},
        {"name": "sentences", "type": _sequence(_string())},
    ]

    assert _propose(["ner"], features, []) == {"taskType": "ner", "textColumn": "sentences", "labelColumn": "ner_tags"}


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


def test_list_inner_unwraps_a_sequence_whose_feature_is_a_one_element_list():
    feature = {"_type": "Sequence", "feature": [{"_type": "Value", "dtype": "string"}]}

    assert _list_inner(feature) == {"_type": "Value", "dtype": "string"}


def test_summarization_falls_back_to_the_longest_text_column_for_the_source():
    features = [
        {"name": "id", "type": {"dtype": "int64", "_type": "Value"}},
        {"name": "body", "type": _string()},
        {"name": "notes", "type": _string()},
        {"name": "summary", "type": _string()},
    ]
    rows = [
        {"row": {"body": "short", "notes": "a much longer note", "summary": "one"}},
        {"row": {"body": "tiny", "notes": "another long note here", "summary": "two"}},
    ]

    assert _propose(["summarization"], features, rows) == {
        "taskType": "summarization", "textColumn": "notes", "labelColumn": "summary",
    }


def test_summarization_uses_the_usual_names_before_the_length_fallback():
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
