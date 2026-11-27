"""Starting an import and following it (#113): POST /imports, GET /imports/<ref>."""
import pytest

from conftest import ALICE

BOB = {"X-User": "bob"}
LEXGLUE = {"dataset": "coastalcph/lex_glue", "config": "ledgar", "split": "train", "textColumn": "text", "labelColumn": "label", "size": 200}


class Queue:
    def __init__(self):
        self.refs = []
        self.down = False

    def __call__(self, import_ref):
        if self.down:
            raise ConnectionError("Redis is down")
        self.refs.append(import_ref)


@pytest.fixture
def queue():
    return Queue()


@pytest.fixture
def client(make_app, queue):
    return make_app(enqueue=queue, max_rows=lambda: 300).test_client()


def start(client, status=202, headers=ALICE, **body):
    response = client.post("/api/library/imports", json={**LEXGLUE, **body}, headers=headers)
    assert response.status_code == status, response.get_json()
    return response.get_json()


def test_an_import_is_queued_then_followed_by_its_owner_only(client, queue):
    started = start(client)

    assert started["status"] == "queued"
    assert started["name"] == "LexGLUE"
    assert started["license"] == "cc-by-4.0"
    assert started["revision"] == "c23fdff1a6bf74e0e1a71cb86f1e781d37da888c"
    assert started["url"] == "https://huggingface.co/datasets/coastalcph/lex_glue"
    assert queue.refs == [started["id"]]
    followed = client.get(f"/api/library/imports/{started['id']}", headers=ALICE)
    assert followed.status_code == 200
    assert followed.get_json() == started
    assert client.get(f"/api/library/imports/{started['id']}", headers=BOB).status_code == 404


def test_the_sample_is_cut_to_the_row_cap_read_when_queued(client):
    assert start(client, size=500)["size"] == 300
    assert start(client, size=50)["size"] == 50


def finish_all(client):
    """The worker finished every import so far."""
    from dataset_library.models import LibraryImport

    with client.application.app_context():
        session = client.application.db.session
        session.query(LibraryImport).update({"status": "completed"})
        session.commit()


def test_importing_the_same_dataset_again_is_named_with_its_number(client):
    start(client)
    finish_all(client)
    assert start(client)["name"] == "LexGLUE (2)"
    finish_all(client)
    assert start(client)["name"] == "LexGLUE (3)"
    assert start(client, headers=BOB)["name"] == "LexGLUE"


def test_a_license_set_to_warn_needs_the_students_confirmation(make_app, queue):
    app = make_app(enqueue=queue, max_rows=lambda: 300)
    from dataset_library.models import LibrarySettings

    with app.app_context():
        app.db.session.add(LibrarySettings(id=1, license_policy={"cc-by-4.0": "warn"}))
        app.db.session.commit()
    client = app.test_client()

    refused = start(client, status=409)
    assert refused["code"] == "needs_ack"
    assert queue.refs == []
    started = start(client, licenseAck=True)
    assert started["status"] == "queued"
    from dataset_library.models import LibraryImport

    with app.app_context():
        assert app.db.session.get(LibraryImport, started["id"]).license_ack_at is not None


def test_a_hidden_license_is_refused_but_the_preview_still_says_not_found(make_app, queue):
    app = make_app(enqueue=queue, max_rows=lambda: 300)
    from dataset_library.models import LibrarySettings

    with app.app_context():
        app.db.session.add(LibrarySettings(id=1, license_policy={"cc-by-4.0": "hide"}))
        app.db.session.commit()
    client = app.test_client()

    refused = start(client, status=403)
    assert refused["code"] == "license_hidden"
    assert refused["error"] == "This dataset's license is not allowed on this platform, so it cannot be imported."
    assert queue.refs == []
    assert client.get("/api/library/datasets/huggingface/coastalcph/lex_glue", headers=ALICE).status_code == 404


def test_a_hidden_license_is_refused_before_the_config_is_checked(make_app, queue):
    app = make_app(enqueue=queue, max_rows=lambda: 300)
    from dataset_library.models import LibrarySettings

    with app.app_context():
        app.db.session.add(LibrarySettings(id=1, license_policy={"cc-by-4.0": "hide"}))
        app.db.session.commit()
    client = app.test_client()

    refused = start(client, status=403, config="no-such-config")
    assert refused["code"] == "license_hidden"
    assert queue.refs == []

    preview = client.get(
        "/api/library/datasets/huggingface/coastalcph/lex_glue?config=no-such-config", headers=ALICE
    )
    assert preview.status_code == 404
    assert preview.get_json()["code"] == "not_found"


def test_an_allow_license_stores_no_confirmation_time_even_when_acknowledged(client):
    from dataset_library.models import LibraryImport

    started = start(client, licenseAck=True)

    with client.application.app_context():
        assert client.application.db.session.get(LibraryImport, started["id"]).license_ack_at is None


@pytest.mark.parametrize("ending", ["failed", "cancelled"])
def test_a_retry_keeps_when_the_student_first_confirmed_the_license(make_app, queue, ending):
    from datetime import datetime

    from dataset_library.models import LibraryImport, LibrarySettings

    app = make_app(enqueue=queue, max_rows=lambda: 300)
    with app.app_context():
        app.db.session.add(LibrarySettings(id=1, license_policy={"cc-by-4.0": "warn"}))
        app.db.session.commit()
    client = app.test_client()
    started = start(client, licenseAck=True)
    confirmed = datetime(2026, 1, 2, 3, 4, 5)
    with app.app_context():
        session = app.db.session
        record = session.get(LibraryImport, started["id"])
        record.license_ack_at = confirmed
        record.status = ending
        session.commit()

    retried = client.post(f"/api/library/imports/{started['id']}/retry", headers=ALICE)

    assert retried.status_code == 202, retried.get_json()
    with app.app_context():
        again = app.db.session.get(LibraryImport, retried.get_json()["id"])
        assert again.license_ack_at == confirmed


def test_try_again_after_a_license_turns_to_warn_asks_for_confirmation(make_app, queue):
    from dataset_library.models import LibraryImport, LibrarySettings

    app = make_app(enqueue=queue, max_rows=lambda: 300)
    client = app.test_client()
    started = start(client)
    with app.app_context():
        session = app.db.session
        session.get(LibraryImport, started["id"]).status = "failed"
        session.add(LibrarySettings(id=1, license_policy={"cc-by-4.0": "warn"}))
        session.commit()

    refused = client.post(f"/api/library/imports/{started['id']}/retry", headers=ALICE)

    assert refused.status_code == 409
    assert refused.get_json()["code"] == "needs_ack"
    assert queue.refs == [started["id"]]


# The Hub only says `private: true` to an authenticated request. This fixture is
# that answer; an anonymous request for a private dataset gets 401/404, the same
# as a missing one, and is reported as `not_found`.
def test_a_private_dataset_is_refused_with_its_reason(client, queue):
    refused = start(client, status=409, dataset="private-law/restricted_corpus", config=None, split=None)

    assert refused["code"] == "private"
    assert refused["error"] == "Its owner keeps this dataset private on Hugging Face, so it cannot be imported."
    assert queue.refs == []
    assert client.get("/api/library/datasets/huggingface/private-law/restricted_corpus", headers=ALICE).status_code == 404


def test_a_private_dataset_outside_the_enabled_sources_is_just_not_found(make_app, queue):
    from dataset_library.models import LibrarySource

    app = make_app(enqueue=queue, max_rows=lambda: 300)
    with app.app_context():
        app.db.session.add(LibrarySource(provider="huggingface", kind="org", ref="coastalcph"))
        app.db.session.commit()
    client = app.test_client()

    refused = start(client, status=404, dataset="private-law/restricted_corpus", config=None, split=None)

    assert refused["code"] == "not_found"
    assert queue.refs == []


def test_a_hub_401_for_a_dataset_is_reported_as_not_found(client, queue, hub):
    hub.fail = 401

    refused = start(client, status=404)

    assert refused["code"] == "not_found"
    assert queue.refs == []


@pytest.mark.parametrize(
    "body, status, code",
    [
        ({"dataset": "not an id"}, 400, "bad_id"),
        ({"textColumn": "nope"}, 400, "bad_mapping"),
        ({"labelColumn": "nope"}, 400, "bad_mapping"),
        ({"size": 0}, 400, "bad_size"),
        ({"size": "lots"}, 400, "bad_size"),
        ({"balance": "yes"}, 400, "bad_balance"),
        ({"taskType": "astrology"}, 400, "bad_task"),
        ({"dataset": "free-law/Caselaw_Access_Project", "config": None, "split": None}, 409, "gated"),
        ({"dataset": "jfrenz/legalglue", "config": None, "split": None}, 409, "not_importable"),
        ({"dataset": "nobody/nothing"}, 404, "not_found"),
    ],
)
def test_what_cannot_be_imported_is_refused_before_anything_is_queued(client, queue, body, status, code):
    assert start(client, status=status, **body)["code"] == code
    assert queue.refs == []


def test_with_the_queue_down_the_import_is_failed_and_the_student_told_to_try_again(client, queue):
    queue.down = True

    refused = start(client, status=503)

    assert refused["error"] == "Import could not start. Try again."
    [record] = client.get("/api/library/imports", headers=ALICE).get_json()["imports"]
    assert record["status"] == "failed"


def test_without_a_queue_the_host_offers_no_import(make_app):
    client = make_app().test_client()

    assert client.post("/api/library/imports", json=LEXGLUE, headers=ALICE).status_code == 404
    assert client.get("/api/library/options", headers=ALICE).get_json()["importable"] is False


def test_the_student_may_name_the_dataset(client):
    assert start(client, name="  Contract clauses  ")["name"] == "Contract clauses"
    finish_all(client)
    assert start(client, name="")["name"] == "LexGLUE (2)"
    assert start(client, status=400, name="x" * 256)["code"] == "bad_name"


def test_an_admins_row_cap_bounds_the_sample(client):
    from conftest import ADMIN

    saved = client.put("/api/library/admin/settings", headers=ADMIN,
                       json={"sources": [], "license_policy": {}, "limits": {"maxRows": 40}})
    assert saved.status_code == 200, saved.get_json()

    assert start(client, size=500)["size"] == 40


def set_import(client, ref, **fields):
    from dataset_library.models import LibraryImport

    with client.application.app_context():
        session = client.application.db.session
        session.query(LibraryImport).filter_by(id=ref).update(fields)
        session.commit()


def test_starting_the_same_import_again_while_it_runs_returns_it(client, queue):
    first = start(client)
    again = start(client, status=200)

    assert again["id"] == first["id"]
    assert queue.refs == [first["id"]]
    # A different size, or another student, is another import.
    assert start(client, size=20)["id"] != first["id"]
    assert start(client, headers=BOB)["id"] != first["id"]


def test_a_balance_that_is_not_a_boolean_is_refused_with_its_reason(client, queue):
    refused = start(client, status=400, balance="yes")

    assert refused["code"] == "bad_balance"
    assert refused["error"] == "Say whether to keep the label balance."
    assert queue.refs == []


def test_a_label_column_balances_the_import_by_default_and_can_turn_it_off(client, queue):
    balanced = start(client)

    assert balanced["balance"] is True
    assert balanced["sampleMethod"] is None
    assert balanced["sampleNote"] is None
    turned_off = start(client, balance=False)
    assert turned_off["id"] != balanced["id"]
    assert turned_off["balance"] is False
    assert queue.refs == [balanced["id"], turned_off["id"]]


def test_without_a_label_column_the_import_is_never_balanced(client, queue):
    started = start(client, labelColumn=None, balance=True)

    assert started["balance"] is False
    assert started["sampleMethod"] is None


@pytest.mark.parametrize("task_type", ["ner", "summarization"])
def test_a_task_type_with_a_label_is_stored_and_never_balanced(client, queue, task_type):
    started = start(client, taskType=task_type)

    assert started["taskType"] == task_type
    assert started["balance"] is False
    assert queue.refs == [started["id"]]


def test_the_text_and_the_label_cannot_be_the_same_column(client, queue):
    refused = start(client, status=400, labelColumn="text")

    assert refused["code"] == "bad_mapping"
    assert refused["error"] == "Pick different columns for the text and the label."
    assert queue.refs == []


@pytest.mark.parametrize("balance", [True, False])
def test_a_retry_keeps_whether_to_keep_the_balance(client, balance):
    started = start(client, balance=balance)
    set_import(client, started["id"], status="failed")

    retried = client.post(f"/api/library/imports/{started['id']}/retry", headers=ALICE)

    assert retried.status_code == 202, retried.get_json()
    assert retried.get_json()["balance"] is balance


def test_the_student_cancels_their_own_running_import(client):
    started = start(client)
    path = f"/api/library/imports/{started['id']}/cancel"

    assert client.post(path, headers=BOB).status_code == 404
    cancelled = client.post(path, headers=ALICE)
    assert cancelled.status_code == 200
    assert cancelled.get_json()["status"] == "cancelled"


def test_a_finished_import_cannot_be_cancelled(client):
    started = start(client)
    set_import(client, started["id"], status="completed")

    refused = client.post(f"/api/library/imports/{started['id']}/cancel", headers=ALICE)

    assert refused.status_code == 409
    assert refused.get_json()["code"] == "too_late"


def test_try_again_starts_a_fresh_import_with_the_same_settings(client, queue):
    started = start(client, size=50, name="Clauses", labelColumn=None)
    set_import(client, started["id"], status="failed", error_code="provider_busy")

    retried = client.post(f"/api/library/imports/{started['id']}/retry", headers=ALICE)

    assert retried.status_code == 202, retried.get_json()
    again = retried.get_json()
    assert again["id"] != started["id"]
    assert again["status"] == "queued"
    for key in ("externalId", "config", "split", "revision", "textColumn", "labelColumn", "taskType", "size", "name"):
        assert again[key] == started[key], key
    assert queue.refs == [started["id"], again["id"]]
    current = client.get("/api/library/imports?current=true", headers=ALICE).get_json()["imports"]
    assert [i["id"] for i in current] == [again["id"]]


def test_only_a_failed_or_cancelled_import_can_be_tried_again(client):
    started = start(client)

    refused = client.post(f"/api/library/imports/{started['id']}/retry", headers=ALICE)

    assert refused.status_code == 409
    assert refused.get_json()["code"] == "not_retryable"
    assert client.post(f"/api/library/imports/{started['id']}/retry", headers=BOB).status_code == 404


def test_try_again_after_the_dataset_changed_does_not_import_the_new_version_unseen(client):
    started = start(client)
    set_import(client, started["id"], status="failed", revision="an-older-revision")

    refused = client.post(f"/api/library/imports/{started['id']}/retry", headers=ALICE)

    assert refused.status_code == 409
    assert refused.get_json()["code"] == "changed"


def test_the_current_list_holds_running_imports_and_recent_failures_not_put_away(client):
    from datetime import timedelta

    from dataset_library.imports import _now

    running = start(client)
    failed = start(client, size=11)
    old_failure = start(client, size=12)
    put_away = start(client, size=13)
    done = start(client, size=14)
    cancelled = start(client, size=15)
    set_import(client, failed["id"], status="failed")
    set_import(client, old_failure["id"], status="failed", updated_at=_now() - timedelta(days=2))
    set_import(client, put_away["id"], status="failed")
    set_import(client, done["id"], status="completed")
    set_import(client, cancelled["id"], status="cancelled")
    assert client.post(f"/api/library/imports/{put_away['id']}/dismiss", headers=ALICE).status_code == 200

    current = client.get("/api/library/imports?current=true", headers=ALICE).get_json()["imports"]

    assert [i["id"] for i in current] == [failed["id"], running["id"]]
    assert client.get("/api/library/imports?current=true", headers=BOB).get_json()["imports"] == []
    assert len(client.get("/api/library/imports", headers=ALICE).get_json()["imports"]) == 6


def test_a_running_import_is_cancelled_before_it_is_put_away(client):
    started = start(client)

    refused = client.post(f"/api/library/imports/{started['id']}/dismiss", headers=ALICE)

    assert refused.status_code == 409
    assert refused.get_json()["code"] == "running"


def test_a_retry_keeps_the_confirmation_after_the_license_turns_to_allow(make_app, queue):
    from datetime import datetime

    from dataset_library.models import LibraryImport, LibrarySettings

    app = make_app(enqueue=queue, max_rows=lambda: 300)
    with app.app_context():
        app.db.session.add(LibrarySettings(id=1, license_policy={"cc-by-4.0": "warn"}))
        app.db.session.commit()
    client = app.test_client()
    started = start(client, licenseAck=True)
    confirmed = datetime(2026, 1, 2, 3, 4, 5)
    with app.app_context():
        session = app.db.session
        record = session.get(LibraryImport, started["id"])
        record.license_ack_at, record.status = confirmed, "failed"
        session.get(LibrarySettings, 1).license_policy = {}
        session.commit()

    retried = client.post(f"/api/library/imports/{started['id']}/retry", headers=ALICE)

    assert retried.status_code == 202, retried.get_json()
    with app.app_context():
        assert app.db.session.get(LibraryImport, retried.get_json()["id"]).license_ack_at == confirmed


def test_a_dataset_the_hub_moved_to_an_owner_outside_the_library_is_not_found(make_app, queue):
    from dataset_library.models import LibrarySettings, LibrarySource

    app = make_app(enqueue=queue, max_rows=lambda: 300)
    with app.app_context():
        app.db.session.add(LibrarySource(provider="huggingface", kind="org", ref="coastalcph"))
        app.db.session.add(LibrarySettings(id=1, license_policy={"cc-by-4.0": "hide"}))
        app.db.session.commit()
    client = app.test_client()

    refused = start(client, status=404, dataset="coastalcph/moved_corpus", config=None, split=None)

    assert refused["code"] == "not_found"
    assert queue.refs == []


# Naming numbered tags (BAIGLE #180): the student's names travel with the import.
TNER_NAMES = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG"]


def tner_body(tner, **body):
    return {"dataset": tner, "config": None, "split": None, "textColumn": "tokens", "labelColumn": "tags",
            "taskType": "ner", "size": 2, **body}


def stored_citation(client, ref):
    from dataset_library.models import LibraryImport

    with client.application.app_context():
        return client.application.db.session.get(LibraryImport, ref).citation


def test_an_ner_import_keeps_the_students_tag_names_on_its_citation(client, queue, tner):
    started = start(client, **tner_body(tner, tagNames=[" O", "B-PER ", "I-PER", "B-ORG", "I-ORG"]))

    citation = stored_citation(client, started["id"])
    assert citation["tagNames"] == TNER_NAMES  # trimmed
    assert citation["title"] == "Legal NER, numbered"
    assert queue.refs == [started["id"]]


def test_other_tag_names_are_another_import_and_a_retry_keeps_them(client, queue, tner):
    first = start(client, **tner_body(tner, tagNames=TNER_NAMES))
    assert start(client, status=200, **tner_body(tner, tagNames=TNER_NAMES))["id"] == first["id"]
    other = start(client, **tner_body(tner, tagNames=["O", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]))
    assert other["id"] != first["id"]
    assert start(client, **tner_body(tner))["id"] not in (first["id"], other["id"])

    set_import(client, first["id"], status="failed", error_code="provider_busy")
    retried = client.post(f"/api/library/imports/{first['id']}/retry", headers=ALICE).get_json()

    assert stored_citation(client, retried["id"])["tagNames"] == TNER_NAMES


@pytest.mark.parametrize("tag_names", [
    [""],  # an empty name
    ["O", "PER"],  # not a tag: no position prefix
    ["O", None],  # not a string
    "O, B-PER",  # not a list
    [],  # nothing named
    ["O"] * 201,  # too many
    ["O", "S-PER"],  # only B- and I- name a position
    ["O", "B_PER"],
    ["O", "B-"],
    ["O", "B- PER"],
    ["O", "B-\u0085"],
])
def test_tag_names_that_are_not_a_list_of_tags_refuse_the_import(client, queue, tner, tag_names):
    refused = start(client, status=400, **tner_body(tner, tagNames=tag_names))

    assert refused["code"] == "bad_tag_names"
    assert refused["error"]
    assert queue.refs == []


def test_a_retry_after_a_missing_name_says_to_name_the_tags_again(client, queue, tner):
    """Round 1 (#180): a retry would send the same names, which fail the same way."""
    first = start(client, **tner_body(tner, tagNames=["O", "B-PER", "I-PER"]))
    set_import(client, first["id"], status="failed", error_code="no_entity_names")

    resp = client.post(f"/api/library/imports/{first['id']}/retry", headers=ALICE)

    assert resp.status_code == 409
    assert resp.get_json()["code"] == "name_tags_again"
    assert resp.get_json()["error"] == "Name every tag number in the import window, then import it again."
    assert queue.refs == [first["id"]]


@pytest.mark.parametrize("label_column", [None, ""])
def test_tag_names_need_a_tag_column(client, queue, tner, label_column):
    refused = start(client, status=400, **tner_body(tner, labelColumn=label_column, tagNames=TNER_NAMES))

    assert refused["code"] == "bad_tag_names"
    assert queue.refs == []


def test_tag_names_need_a_tag_column_even_when_none_is_sent(client, queue, tner):
    body = {k: v for k, v in tner_body(tner, tagNames=TNER_NAMES).items() if k != "labelColumn"}

    resp = client.post("/api/library/imports", json=body, headers=ALICE)
    refused = resp.get_json()

    assert resp.status_code == 400
    assert refused["code"] == "bad_tag_names"
    assert queue.refs == []


def test_a_lowercase_o_is_kept_as_o(client, queue, tner):
    started = start(client, **tner_body(tner, tagNames=["o", "B-PER", "I-PER", "B-ORG", "I-ORG"]))

    assert stored_citation(client, started["id"])["tagNames"] == TNER_NAMES


def test_tag_names_are_only_for_an_ner_import(client, queue):
    refused = start(client, status=400, tagNames=TNER_NAMES)

    assert refused["code"] == "bad_tag_names"
    assert queue.refs == []
