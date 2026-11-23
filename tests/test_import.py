"""Importing a sample of a Hugging Face dataset (#113): the worker's side.

Shards are small Parquet files written here; `create_dataset` is a fake that
keeps what it was given, as the host's would store it.
"""
import csv
import json
import random
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dataset_library import imports
from dataset_library.models import LibraryImport

LABELS = ["Terminations", "Governing Laws", "Notices"]
NER_NAMES = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG"]


def entity_spans(rows):
    """The rows' entities cells as the span lists they hold, keyed by text."""
    return {row["text"]: json.loads(row["entities"]) for row in rows}


def check_entity_offsets(rows):
    for row in rows:
        for span in json.loads(row["entities"]):
            assert row["text"][span["start"]:span["end"]] == span["text"]


def write_shard(path, texts, labels, row_group_size=None):
    """A Parquet shard as Hugging Face writes one: features in the schema metadata."""
    features = {
        "text": {"dtype": "string", "_type": "Value"},
        "label": {"names": LABELS, "_type": "ClassLabel"},
        "extra": {"dtype": "string", "_type": "Value"},
    }
    table = pa.table({"text": texts, "label": labels, "extra": ["x"] * len(texts)})
    table = table.replace_schema_metadata({"huggingface": json.dumps({"info": {"features": features}})})
    pq.write_table(table, path, row_group_size=row_group_size)
    return path


def write_unlabelled_shard(path, texts, row_group_size=None):
    """A shard that does not hold the mapped label column at all."""
    features = {"text": {"dtype": "string", "_type": "Value"}}
    table = pa.table({"text": texts, "extra": ["x"] * len(texts)})
    table = table.replace_schema_metadata({"huggingface": json.dumps({"info": {"features": features}})})
    pq.write_table(table, path, row_group_size=row_group_size)
    return path


def write_labels_shard(path, texts, labels, label_feature=None, row_group_size=None):
    """A shard whose label column is whatever the test wants it to be."""
    features = {
        "text": {"dtype": "string", "_type": "Value"},
        "label": label_feature or {"dtype": "string", "_type": "Value"},
        "extra": {"dtype": "string", "_type": "Value"},
    }
    table = pa.table({"text": texts, "label": labels, "extra": ["x"] * len(texts)})
    table = table.replace_schema_metadata({"huggingface": json.dumps({"info": {"features": features}})})
    pq.write_table(table, path, row_group_size=row_group_size)
    return path


def write_ner_shard(path, tokens, tags, tag_feature=None, row_group_size=None):
    """A shard as a Hugging Face NER dataset writes one: tokens and BIO tags."""
    features = {
        "tokens": {"feature": {"dtype": "string", "_type": "Value"}, "_type": "Sequence"},
        "ner_tags": tag_feature or {"feature": {"names": NER_NAMES, "_type": "ClassLabel"}, "_type": "Sequence"},
    }
    table = pa.table({"tokens": tokens, "ner_tags": tags})
    table = table.replace_schema_metadata({"huggingface": json.dumps({"info": {"features": features}})})
    pq.write_table(table, path, row_group_size=row_group_size)
    return path


def write_tner_shard(path, tokens, tags, row_group_size=None):
    """A shard as tner/conll2003 writes one: tokens and numbered tags with no names."""
    features = {
        "tokens": {"feature": {"dtype": "string", "_type": "Value"}, "_type": "Sequence"},
        "tags": {"feature": {"dtype": "int64", "_type": "Value"}, "_type": "Sequence"},
    }
    table = pa.table({"tokens": tokens, "tags": tags})
    table = table.replace_schema_metadata({"huggingface": json.dumps({"info": {"features": features}})})
    pq.write_table(table, path, row_group_size=row_group_size)
    return path


def write_summarization_shard(path, documents, summaries, row_group_size=None):
    """A shard as Hugging Face writes a summarization dataset: document, summary."""
    features = {
        "document": {"dtype": "string", "_type": "Value"},
        "summary": {"dtype": "string", "_type": "Value"},
    }
    table = pa.table({"document": documents, "summary": summaries})
    table = table.replace_schema_metadata({"huggingface": json.dumps({"info": {"features": features}})})
    pq.write_table(table, path, row_group_size=row_group_size)
    return path


class FakeCreateDataset:
    def __init__(self, refuse=None, while_saving=None):
        self.calls = []
        self.refuse = refuse
        self.while_saving = while_saving

    def __call__(self, owner_id, file_path, name, description, task_type):
        if self.while_saving:
            self.while_saving()
        self.last_path = file_path
        with open(file_path, newline="") as f:
            rows = list(csv.DictReader(f))
        self.calls.append({"owner": owner_id, "name": name, "description": description, "task_type": task_type, "rows": rows})
        if self.refuse:
            raise ValueError(self.refuse)
        return "dataset-1"


@pytest.fixture
def run(app, tmp_path):
    """Queue an import directly and run it, as the host's worker would."""

    def go(shards, *, size=4, fail_fetch=None, refuse=None, open_parquet=None, while_fetching=None,
           while_saving=None, heartbeat_seconds=None, snapshot_dir=None, **fields):
        create = FakeCreateDataset(refuse, while_saving and (lambda: while_saving(app)))

        def fetch_shards(external_id, config, split, max_bytes, dest_dir):
            create.fetched_with = (external_id, config, split, max_bytes)
            if while_fetching:
                while_fetching(app.db.engine)
            if fail_fetch:
                raise fail_fetch
            return imports.Fetched(paths=[str(p) for p in shards], revision="abc123")

        with app.app_context():
            session = app.db.session
            record = LibraryImport(
                owner_id="alice", provider="huggingface",
                external_id=fields.pop("external_id", "coastalcph/lex_glue"),
                config=fields.pop("config", "ledgar"), split=fields.pop("split", "train"),
                text_column=fields.pop("text_column", "text"),
                label_column=fields.pop("label_column", "label"),
                task_type=fields.pop("task_type", "classification"),
                sample_size=size, max_rows=1000, name="LexGLUE",
                license="cc-by-4.0", url="https://huggingface.co/datasets/coastalcph/lex_glue",
                **fields,
            )
            session.add(record)
            session.commit()
            ref = record.id
            imports.run_import(
                ref, session=session, fetch_shards=fetch_shards,
                create_dataset=create, data_dir=str(tmp_path / "data"), seed=7, open_parquet=open_parquet,
                heartbeat_seconds=heartbeat_seconds, snapshot_dir=snapshot_dir,
            )
            session.expunge_all()
            found = session.get(LibraryImport, ref)
            return (found.to_dict() if found else None), create

    return go


def test_an_import_samples_the_asked_number_of_rows_with_label_names(run, tmp_path):
    texts = [f"Clause {i}" for i in range(10)]
    shard = write_shard(tmp_path / "0000.parquet", texts, [i % 3 for i in range(10)])

    result, create = run([shard], size=4)

    assert result["status"] == "completed", result
    assert result["datasetRef"] == "dataset-1"
    assert result["rows"] == 4
    assert result["revision"] == "abc123"
    assert create.fetched_with == ("coastalcph/lex_glue", "ledgar", "train", 200_000_000)
    [call] = create.calls
    assert call["task_type"] == "classification"
    assert call["name"] == "LexGLUE"
    assert len(call["rows"]) == 4
    assert list(call["rows"][0]) == ["text", "label"]
    by_text = {t: LABELS[i % 3] for i, t in enumerate(texts)}
    for row in call["rows"]:
        assert row["label"] == by_text[row["text"]]


class SpyParquet:
    """A ParquetFile that records what is read: metadata, or which row groups and columns."""

    reads = []

    def __init__(self, path):
        self._file = pq.ParquetFile(path)
        self.path = str(path)

    @property
    def metadata(self):
        return self._file.metadata

    @property
    def schema_arrow(self):
        return self._file.schema_arrow

    def read_row_group(self, i, columns=None):
        SpyParquet.reads.append((self.path, i, tuple(columns or ())))
        return self._file.read_row_group(i, columns=columns)

    def __getattr__(self, name):
        raise AssertionError(f"read more than row counts and chosen row groups: {name}")


def test_pass_one_reads_row_counts_and_pass_two_only_the_mapped_columns_of_chosen_row_groups(run, tmp_path):
    shards = [
        write_shard(tmp_path / f"{n:04}.parquet", [f"Shard {n} clause {i}" for i in range(100)], [i % 3 for i in range(100)], row_group_size=10)
        for n in range(2)
    ]
    SpyParquet.reads = []

    result, create = run(shards, size=3, open_parquet=SpyParquet)

    assert result["status"] == "completed", result
    assert len(create.calls[0]["rows"]) == 3
    assert 1 <= len(SpyParquet.reads) <= 3
    assert all(columns == ("text", "label") for _, _, columns in SpyParquet.reads)
    # Each row group read holds one of the chosen rows.
    chosen = {row["text"] for row in create.calls[0]["rows"]}
    for path, group, _ in SpyParquet.reads:
        shard = path.rsplit("/", 1)[1][:4]
        assert any(t.startswith(f"Shard {int(shard)} clause ") and int(t.rsplit(" ", 1)[1]) // 10 == group for t in chosen)


def test_a_balanced_import_keeps_uneven_labels_across_many_row_groups(run, tmp_path):
    labels = [0] * 700 + [1] * 250 + [2] * 50
    random.Random(1).shuffle(labels)
    texts = [f"Clause {i}" for i in range(len(labels))]
    shard = write_shard(tmp_path / "0000.parquet", texts, labels, row_group_size=50)

    result, create = run([shard], size=200, balance=True)

    assert result["status"] == "completed", result
    assert result["balance"] is True
    assert result["sampleMethod"] == "stratified"
    assert result["sampleNote"] is None
    rows = create.calls[0]["rows"]
    assert len(rows) == 200
    assert len({row["text"] for row in rows}) == 200
    counts = Counter(row["label"] for row in rows)
    assert abs(counts[LABELS[0]] - 140) <= 1
    assert abs(counts[LABELS[1]] - 50) <= 1
    assert abs(counts[LABELS[2]] - 10) <= 1
    assert create.calls[0]["description"].startswith("A sample that keeps the label balance, from Hugging Face:")


def test_a_balanced_import_reads_the_label_column_first_then_only_chosen_row_groups(run, tmp_path):
    shards = [
        write_shard(tmp_path / f"{n:04}.parquet", [f"Shard {n} clause {i}" for i in range(100)], [i % 3 for i in range(100)], row_group_size=10)
        for n in range(2)
    ]
    SpyParquet.reads = []

    result, create = run(shards, size=3, balance=True, open_parquet=SpyParquet)

    assert result["status"] == "completed", result
    label_reads = [(path, group) for path, group, columns in SpyParquet.reads if columns == ("label",)]
    mapped_reads = [(path, group) for path, group, columns in SpyParquet.reads if columns == ("text", "label")]
    assert all(columns in (("label",), ("text", "label")) for _, _, columns in SpyParquet.reads)
    # Pass 1 reads every row group, and only its label column.
    assert len(label_reads) == 20
    # Pass 2 reads only the row groups that hold a chosen row.
    chosen = {row["text"] for row in create.calls[0]["rows"]}
    expected = set()
    for text in chosen:
        where, clause = text.split(" clause ")
        n = int(where.removeprefix("Shard "))
        expected.add((str(tmp_path / f"{n:04}.parquet"), int(clause) // 10))
    assert {tuple(read) for read in mapped_reads} == expected


def test_a_balanced_import_replaces_empty_text_from_the_same_label(run, tmp_path):
    texts = ["", "", "Keep A", "Keep B", "Keep C", "Keep D", *[f"Other {i}" for i in range(6)]]
    labels = [0] * 6 + [1] * 6
    shard = write_shard(tmp_path / "0000.parquet", texts, labels)

    result, create = run([shard], size=6, balance=True)

    assert result["status"] == "completed", result
    rows = create.calls[0]["rows"]
    assert len(rows) == 6
    assert Counter(row["label"] for row in rows) == {LABELS[0]: 3, LABELS[1]: 3}
    assert all(row["text"].strip() for row in rows)


def test_an_empty_label_column_falls_back_to_a_random_sample_with_a_note(run, tmp_path):
    texts = [f"Clause {i}" for i in range(8)]
    shard = write_shard(tmp_path / "0000.parquet", texts, [None, "", "  ", None, "", "  ", None, ""])

    result, create = run([shard], size=3, balance=True)

    assert result["status"] == "completed", result
    assert result["sampleMethod"] == "random"
    assert result["sampleNote"] == imports.LABEL_EMPTY
    assert len(create.calls[0]["rows"]) == 3


def test_a_label_column_missing_from_the_files_falls_back_with_its_own_note(run, tmp_path):
    shard = write_unlabelled_shard(tmp_path / "0000.parquet", [f"Clause {i}" for i in range(8)])

    result, create = run([shard], size=3, balance=True)

    assert result["status"] == "completed", result
    assert result["sampleMethod"] == "random"
    assert result["sampleNote"] == imports.LABEL_MISSING
    assert len(create.calls[0]["rows"]) == 3


def test_with_the_balance_turned_off_a_labelled_dataset_is_sampled_at_random(run, tmp_path):
    shard = write_shard(tmp_path / "0000.parquet", [f"Clause {i}" for i in range(30)], [i % 3 for i in range(30)])

    result, create = run([shard], size=6, balance=False)

    assert result["status"] == "completed", result
    assert result["sampleMethod"] == "random"
    assert result["sampleNote"] is None
    assert create.calls[0]["description"].startswith("A random sample from Hugging Face:")


def test_rows_with_empty_text_are_dropped_and_the_sample_stays_exact(run, tmp_path):
    texts = ["", "  ", None, "Kept one", "Kept two", "Kept three", "", "Kept four"]
    shard = write_shard(tmp_path / "0000.parquet", texts, [0] * len(texts))

    result, create = run([shard], size=4)

    assert result["status"] == "completed", result
    assert sorted(r["text"] for r in create.calls[0]["rows"]) == ["Kept four", "Kept one", "Kept three", "Kept two"]


def test_with_no_text_left_the_import_fails_and_no_dataset_is_made(run, tmp_path):
    shard = write_shard(tmp_path / "0000.parquet", ["", " ", None], [0, 1, 2])

    result, create = run([shard], size=2)

    assert result["status"] == "failed"
    assert result["code"] == "no_usable_rows"
    assert "nothing to import" in result["error"]
    assert create.calls == []


def test_a_refusal_from_create_dataset_is_shown_as_the_upload_would_show_it(run, tmp_path):
    shard = write_shard(tmp_path / "0000.parquet", ["A clause", "Another"], [0, 1])

    result, _ = run([shard], size=2, refuse="Classification needs at least 2 examples of each label")

    assert result["status"] == "failed"
    assert result["error"] == "Classification needs at least 2 examples of each label"
    assert not list((tmp_path / "data").glob("library-*.csv"))


def test_hugging_face_busy_after_retries_fails_with_a_plain_reason(run, tmp_path):
    from dataset_library.errors import ProviderUnavailable

    result, create = run([], fail_fetch=ProviderUnavailable("503 three times"))

    assert result["status"] == "failed"
    assert result["code"] == "provider_busy"
    assert result["error"] == "Hugging Face is busy. Try again in a few minutes."
    assert create.calls == []


def test_a_finished_import_is_not_run_again(run, tmp_path, app):
    shard = write_shard(tmp_path / "0000.parquet", ["A clause", "Another"], [0, 1])
    result, create = run([shard], size=2, status="failed")

    assert result["status"] == "failed"
    assert create.calls == []


@pytest.mark.parametrize(
    "head, code, message",
    [
        # Still at the revision the student saw: the conversion has not caught up.
        ("f5a645c5f00fe5acfe939d15e00c2faa3cb85375", "conversion_behind", imports.BEHIND),
        # Moved on since it was opened: opening it again shows the new version.
        ("2" * 40, "changed", imports.CHANGED),
        # The head could not be asked: the dataset may have changed.
        (None, "changed", imports.CHANGED),
    ],
)
def test_a_conversion_at_another_revision_than_the_queued_one_is_refused(app, tmp_path, head, code, message):
    # A conversion recorded while it lagged (turhancan97/doppeldwarf-306,
    # 2026-11-23): the import was queued at the Hub's head, its rows would be
    # read from the older commit the conversion was built from.
    shard = write_shard(tmp_path / "0000.parquet", ["A clause", "Another"], [0, 1])
    with app.app_context():
        session = app.db.session
        record = LibraryImport(
            owner_id="alice", provider="huggingface", external_id="turhancan97/doppeldwarf-306",
            config="classes", split="reference", revision="f5a645c5f00fe5acfe939d15e00c2faa3cb85375",
            text_column="text", label_column="label", task_type="classification",
            sample_size=2, max_rows=1000, name="Doppeldwarf", license="cc-by-4.0",
            url="https://huggingface.co/datasets/turhancan97/doppeldwarf-306", status="queued",
        )
        session.add(record)
        session.commit()
        ref = record.id

        def fetch_shards(external_id, config, split, max_bytes, dest_dir):
            return imports.Fetched(paths=[str(shard)], revision="18d016c41417e9beb85fb83a94b4fbf866805cce", head=head)

        imports.run_import(
            ref, session=session, fetch_shards=fetch_shards,
            create_dataset=FakeCreateDataset(), data_dir=str(tmp_path / "data"), seed=7,
        )
        found = session.get(LibraryImport, ref).to_dict()

    assert found["status"] == "failed"
    assert found["code"] == code
    assert found["error"] == message
    assert found["revision"] == "f5a645c5f00fe5acfe939d15e00c2faa3cb85375", "a refused import keeps the revision it was queued at"


def test_an_import_of_a_caught_up_conversion_records_the_revision_its_rows_came_from(run, tmp_path):
    # The conversion sits at the revision the import was queued at, so the import
    # completes and records it — that revision, not the Hub's head at fetch time.
    shard = write_shard(tmp_path / "0000.parquet", ["A clause", "Another"], [0, 1])

    result, create = run([shard], size=2, revision="abc123")

    assert result["status"] == "completed", result
    assert result["revision"] == "abc123"
    assert len(create.calls) == 1


def test_an_import_erased_while_it_runs_stops_without_making_a_dataset(run, tmp_path):
    """Account deletion removes the user's import rows; a running import then just ends."""
    import sqlalchemy as sa

    shard = write_shard(tmp_path / "0000.parquet", ["A clause", "Another"], [0, 1])

    def erase(engine):
        with engine.begin() as connection:
            connection.execute(sa.text("DELETE FROM library_imports"))

    result, create = run([shard], size=2, while_fetching=erase)

    assert result is None
    assert create.calls == []
    assert not list((tmp_path / "data").glob("library-*.csv"))


def test_a_cancelled_import_is_not_finished_by_the_worker(run, tmp_path):
    import sqlalchemy as sa

    shard = write_shard(tmp_path / "0000.parquet", ["A clause", "Another"], [0, 1])

    def cancel(engine):
        with engine.begin() as connection:
            connection.execute(sa.text("UPDATE library_imports SET status = 'cancelled'"))

    result, create = run([shard], size=2, while_fetching=cancel)

    assert result["status"] == "cancelled"
    assert create.calls == []


def test_an_admins_download_cap_replaces_the_workers_default(run, app, tmp_path):
    from dataset_library.models import LibrarySettings

    with app.app_context():
        app.db.session.add(LibrarySettings(id=1, license_policy={}, max_download_mb=50))
        app.db.session.commit()
    shard = write_shard(tmp_path / "0000.parquet", [f"Clause {i}" for i in range(6)], [0] * 6)

    result, create = run([shard], size=2)

    assert result["status"] == "completed", result
    assert create.fetched_with[3] == 50_000_000


def test_list_valued_labels_fall_back_to_a_random_sample_without_crashing(run, tmp_path):
    texts = [f"Clause {i}" for i in range(8)]
    labels = [["Terminations", "Notices"] for _ in texts]
    shard = write_labels_shard(
        tmp_path / "0000.parquet", texts, labels,
        label_feature={"feature": {"dtype": "string", "_type": "Value"}, "_type": "Sequence"},
    )

    result, create = run([shard], size=3, balance=True)

    assert result["status"] == "completed", result
    assert result["sampleMethod"] == "random"
    assert result["sampleNote"] == imports.LABEL_LISTS
    assert len(create.calls[0]["rows"]) == 3


def test_a_non_classification_task_type_never_stores_a_balance(app):
    host = app.extensions["dataset_library"]
    host.enqueue = lambda ref: None

    with app.app_context():
        started, _ = imports.start(
            host, "alice",
            {
                "dataset": "coastalcph/lex_glue", "config": "ledgar", "split": "train",
                "textColumn": "text", "labelColumn": "label", "taskType": "ner",
                "size": 10, "balance": True,
            },
        )

    assert started["balance"] is False


def test_more_distinct_labels_than_rows_wanted_falls_back_to_random(run, tmp_path):
    texts = [f"Clause {i}" for i in range(50)]
    labels = [f"Topic {i}" for i in range(50)]
    shard = write_labels_shard(tmp_path / "0000.parquet", texts, labels)

    result, create = run([shard], size=10, balance=True)

    assert result["status"] == "completed", result
    assert result["sampleMethod"] == "random"
    assert result["sampleNote"] == imports.LABEL_TOO_MANY
    assert len(create.calls[0]["rows"]) == 10


def test_equal_remainders_are_broken_at_random_for_different_seeds():
    counts = {f"Label {i}": 1 for i in range(6)}

    first = imports._allocate(counts, 3, random.Random(1))
    second = imports._allocate(counts, 3, random.Random(4))

    assert first != second


def test_class_label_minus_one_rows_are_left_out_and_all_of_them_mean_no_labels(run, tmp_path):
    texts = [f"Clause {i}" for i in range(20)]
    labels = [-1] * 4 + [i % 3 for i in range(16)]
    shard = write_shard(tmp_path / "0000.parquet", texts, labels)

    result, create = run([shard], size=9, balance=True)

    assert result["status"] == "completed", result
    assert len(create.calls[0]["rows"]) == 9
    assert "Clause 0" not in {row["text"] for row in create.calls[0]["rows"]}

    all_unlabelled = write_shard(tmp_path / "0001.parquet", [f"Other {i}" for i in range(8)], [-1] * 8)

    empty, _ = run([all_unlabelled], size=3, balance=True)

    assert empty["sampleMethod"] == "random"
    assert empty["sampleNote"] == imports.LABEL_EMPTY


def test_nan_labels_are_left_out_of_a_balanced_sample(run, tmp_path):
    texts = [f"Clause {i}" for i in range(20)]
    labels = [float("nan")] * 4 + [float(i % 3) for i in range(16)]
    shard = write_labels_shard(
        tmp_path / "0000.parquet", texts, labels,
        label_feature={"names": LABELS, "_type": "ClassLabel"},
    )

    result, create = run([shard], size=9, balance=True)

    assert result["status"] == "completed", result
    assert len(create.calls[0]["rows"]) == 9
    assert "Clause 0" not in {row["text"] for row in create.calls[0]["rows"]}


def test_a_label_with_no_text_is_topped_up_from_the_others_and_says_so(run, tmp_path):
    texts = ["", "", "", "", ""] + [f"Other {i}" for i in range(20)]
    labels = [0] * 5 + [1] * 20
    shard = write_shard(tmp_path / "0000.parquet", texts, labels)

    result, create = run([shard], size=10, balance=True)

    assert result["status"] == "completed", result
    assert result["sampleMethod"] == "stratified"
    assert result["sampleNote"] == imports.LABEL_ROUGH
    rows = create.calls[0]["rows"]
    assert len(rows) == 10
    assert all(row["text"].strip() for row in rows)


def test_a_label_sorted_source_does_not_produce_a_label_sorted_csv(run, tmp_path):
    texts = [f"Clause {i}" for i in range(90)]
    labels = [0] * 30 + [1] * 30 + [2] * 30
    shard = write_shard(tmp_path / "0000.parquet", texts, labels)

    result, create = run([shard], size=30, balance=True)

    assert result["status"] == "completed", result
    order = [row["label"] for row in create.calls[0]["rows"]]
    assert order != sorted(order)
    assert set(order) == set(LABELS)


def test_a_label_with_no_place_of_its_own_still_fills_the_sample(run, tmp_path):
    # 1000 rows of label 0 have no text; the 2 rows of label 1 get no place in a
    # 2-row sample by share, but hold the only text. The top-up finds them.
    texts = [""] * 1000 + ["Clause A", "Clause B"]
    labels = [0] * 1000 + [1, 1]
    shard = write_shard(tmp_path / "0000.parquet", texts, labels)

    result, create = run([shard], size=2, balance=True)

    assert result["status"] == "completed", result
    assert len(create.calls[0]["rows"]) == 2
    assert result["sampleNote"] == imports.LABEL_ROUGH


def test_a_balanced_sample_short_of_its_size_says_so(run, tmp_path):
    # One label, 3 rows with text: nothing to top up from, the sample cannot
    # reach 10, and the note still says so.
    texts = ["One", "", "Two", "", "Three"] + [""] * 20
    labels = [0] * 25
    shard = write_shard(tmp_path / "0000.parquet", texts, labels)

    result, create = run([shard], size=10, balance=True)

    assert result["status"] == "completed", result
    assert len(create.calls[0]["rows"]) == 3
    assert result["sampleNote"] == imports.LABEL_ROUGH


def test_an_ner_import_writes_joined_text_and_json_entity_spans(run, tmp_path):
    tokens = [["John", "Smith", "works", "at", "ACME"], ["ACME", "sued", "Mary", "Jones"]]
    tags = [[1, 2, 0, 0, 3], [3, 0, 1, 2]]
    shard = write_ner_shard(tmp_path / "0000.parquet", tokens, tags)

    result, create = run([shard], size=2, task_type="ner", text_column="tokens", label_column="ner_tags")

    assert result["status"] == "completed", result
    [call] = create.calls
    assert call["task_type"] == "ner"
    assert list(call["rows"][0]) == ["text", "entities"]
    assert entity_spans(call["rows"]) == {
        "John Smith works at ACME": [
            {"text": "John Smith", "start": 0, "end": 10, "type": "PER"},
            {"text": "ACME", "start": 20, "end": 24, "type": "ORG"},
        ],
        "ACME sued Mary Jones": [
            {"text": "ACME", "start": 0, "end": 4, "type": "ORG"},
            {"text": "Mary Jones", "start": 10, "end": 20, "type": "PER"},
        ],
    }
    check_entity_offsets(call["rows"])


def test_an_ner_import_accepts_string_tags(run, tmp_path):
    shard = write_ner_shard(
        tmp_path / "0000.parquet",
        [["Alice", "runs", "ACME"], ["Bob", "left"]],
        [["B-PER", "O", "B-ORG"], ["B-PER", "O"]],
        tag_feature={"feature": {"dtype": "string", "_type": "Value"}, "_type": "Sequence"},
    )

    result, create = run([shard], size=2, task_type="ner", text_column="tokens", label_column="ner_tags")

    assert result["status"] == "completed", result
    assert entity_spans(create.calls[0]["rows"]) == {
        "Alice runs ACME": [
            {"text": "Alice", "start": 0, "end": 5, "type": "PER"},
            {"text": "ACME", "start": 11, "end": 15, "type": "ORG"},
        ],
        "Bob left": [{"text": "Bob", "start": 0, "end": 3, "type": "PER"}],
    }
    check_entity_offsets(create.calls[0]["rows"])


def test_an_ner_row_with_only_outside_tags_keeps_an_empty_entities_cell(run, tmp_path):
    shard = write_ner_shard(
        tmp_path / "0000.parquet",
        [["just", "words"], ["John", "Smith"]],
        [[0, 0], [1, 2]],
    )

    result, create = run([shard], size=2, task_type="ner", text_column="tokens", label_column="ner_tags")

    assert result["status"] == "completed", result
    assert entity_spans(create.calls[0]["rows"]) == {
        "just words": [],
        "John Smith": [{"text": "John Smith", "start": 0, "end": 10, "type": "PER"}],
    }
    check_entity_offsets(create.calls[0]["rows"])


def test_an_empty_token_list_is_empty_text_and_the_sample_stays_exact(run, tmp_path):
    shard = write_ner_shard(
        tmp_path / "0000.parquet",
        [[], ["Kept", "one"], [], ["Kept", "two"]],
        [[], [0, 0], [], [0, 0]],
    )

    result, create = run([shard], size=2, task_type="ner", text_column="tokens", label_column="ner_tags")

    assert result["status"] == "completed", result
    assert sorted(row["text"] for row in create.calls[0]["rows"]) == ["Kept one", "Kept two"]
    assert all(row["entities"] == "[]" for row in create.calls[0]["rows"])


def test_numbered_tags_without_names_fail_with_a_plain_reason(run, tmp_path):
    shard = write_ner_shard(
        tmp_path / "0000.parquet",
        [["John", "Smith"]],
        [[1, 2]],
        tag_feature={"feature": {"dtype": "int64", "_type": "Value"}, "_type": "Sequence"},
    )

    result, create = run([shard], size=1, task_type="ner", text_column="tokens", label_column="ner_tags")

    assert result["status"] == "failed"
    assert result["code"] == "no_entity_names"
    assert "can't read entity names" in result["error"]
    assert create.calls == []
    assert not list((tmp_path / "data").glob("library-*.csv"))


def test_tags_without_names_fail_before_any_row_is_read(run, tmp_path):
    """The tner shape, mapped as a preview proposes it: no names, no rows read."""
    shard = write_tner_shard(tmp_path / "0000.parquet", [["John", "Smith"], ["ACME"]], [[0, 1], [2]])
    SpyParquet.reads = []

    result, create = run(
        [shard], size=2, task_type="ner", text_column="tokens", label_column="tags",
        open_parquet=SpyParquet,
    )

    assert result["status"] == "failed"
    assert result["code"] == "no_entity_names"
    assert create.calls == []
    assert SpyParquet.reads == []
    assert not list((tmp_path / "data").glob("library-*.csv"))


def test_tags_with_an_empty_name_list_also_fail_before_any_row_is_read(run, tmp_path):
    shard = write_ner_shard(
        tmp_path / "0000.parquet", [["John", "Smith"]], [[0, 1]],
        tag_feature={"feature": {"names": [], "_type": "ClassLabel"}, "_type": "Sequence"},
    )
    SpyParquet.reads = []

    result, create = run(
        [shard], size=1, task_type="ner", text_column="tokens", label_column="ner_tags",
        open_parquet=SpyParquet,
    )

    assert result["status"] == "failed"
    assert result["code"] == "no_entity_names"
    assert create.calls == []
    assert SpyParquet.reads == []


def test_nested_ner_columns_fail_before_any_row_is_read(run, tmp_path):
    path = tmp_path / "0000.parquet"
    pq.write_table(pa.table({"tokens": [[["John"]]], "ner_tags": [[["B-PER"]]]}), path)
    SpyParquet.reads = []

    result, create = run(
        [path], size=1, task_type="ner", text_column="tokens", label_column="ner_tags",
        open_parquet=SpyParquet,
    )

    assert result["status"] == "failed"
    assert result["code"] == "nested_columns"
    assert create.calls == []
    assert SpyParquet.reads == []


def test_a_nested_tag_column_next_to_flat_tokens_fails_before_any_row_is_read(run, tmp_path):
    path = tmp_path / "0000.parquet"
    pq.write_table(pa.table({"tokens": [["John", "Smith"]], "ner_tags": [[["B-PER"], ["I-PER"]]]}), path)
    SpyParquet.reads = []

    result, create = run(
        [path], size=1, task_type="ner", text_column="tokens", label_column="ner_tags",
        open_parquet=SpyParquet,
    )

    assert result["code"] == "nested_columns"
    assert SpyParquet.reads == []


def test_span_records_as_the_tag_column_fail_before_any_row_is_read(run, tmp_path):
    path = tmp_path / "0000.parquet"
    spans = [[{"start": 0, "end": 10, "type": "PER"}]]
    pq.write_table(pa.table({"tokens": [["John", "Smith"]], "entities": spans}), path)
    SpyParquet.reads = []

    result, create = run(
        [path], size=1, task_type="ner", text_column="tokens", label_column="entities",
        open_parquet=SpyParquet,
    )

    assert result["code"] == "nested_columns"
    assert create.calls == []
    assert SpyParquet.reads == []


def test_a_record_of_span_lists_as_the_tag_column_fails_before_any_row_is_read(run, tmp_path):
    # Hugging Face stores Sequence({start, end, type}) as one record of lists.
    path = tmp_path / "0000.parquet"
    spans = [{"start": [0], "end": [10], "type": ["PER"]}]
    pq.write_table(pa.table({"tokens": [["John", "Smith"]], "entities": spans}), path)
    SpyParquet.reads = []

    result, create = run(
        [path], size=1, task_type="ner", text_column="tokens", label_column="entities",
        open_parquet=SpyParquet,
    )

    assert result["code"] == "nested_columns"
    assert create.calls == []
    assert SpyParquet.reads == []


def test_a_missing_tag_value_or_a_string_text_leaves_the_entities_blank():
    # A blank cell keeps the row waiting to be annotated in legal-ml-core;
    # "[]" would count as annotated with no entities.
    from dataset_library.imports import _mapped

    assert _mapped([["John", "Smith"], None], None, "ner") == ["John Smith", ""]
    assert _mapped(["John Smith works", ["B-PER", "I-PER", "O"]], None, "ner") == ["John Smith works", ""]
    assert _mapped(["John Smith works", "PER, ORG"], None, "ner") == ["John Smith works", ""]
    assert _mapped([["John", "works"], ["O", "O"]], None, "ner") == ["John works", "[]"]
    # Tags that do not cover every token say nothing about the rest of the row.
    for tags in ([], [None, None], ["O"], ["B-PER"], "O B-PER", ["B-PER", None]):
        assert _mapped([["John", "works"], tags], None, "ner") == ["John works", ""], tags


def test_a_string_text_with_numbered_unnamed_tags_imports_blank_rather_than_failing():
    from dataset_library.imports import _mapped

    assert _mapped(["John works", [1, 0]], None, "ner") == ["John works", ""]


def test_a_string_text_shard_with_unnamed_numbered_tags_imports_with_blank_entities(run, tmp_path):
    # The file-level no_entity_names check applies only to token lists.
    path = tmp_path / "0000.parquet"
    features = {
        "text": {"dtype": "string", "_type": "Value"},
        "tags": {"feature": {"dtype": "int64", "_type": "Value"}, "_type": "Sequence"},
    }
    table = pa.table({"text": ["John works", "Ada writes"], "tags": [[1, 0], [1, 0]]})
    table = table.replace_schema_metadata({"huggingface": json.dumps({"info": {"features": features}})})
    pq.write_table(table, path)

    result, create = run([path], size=2, task_type="ner", text_column="text", label_column="tags")

    assert result["status"] == "completed", result
    assert all(row["entities"] == "" for row in create.calls[0]["rows"])


def test_an_ner_import_with_no_fully_tagged_row_says_why_its_entities_are_blank(run, tmp_path):
    # Sub-word tags: more tags than words in every row.
    shard = write_ner_shard(tmp_path / "0000.parquet", [["John", "Smith"], ["Ada"]], [[1, 2, 0], [1, 0]])

    result, create = run([shard], size=2, task_type="ner", text_column="tokens", label_column="ner_tags")

    assert result["status"] == "completed", result
    assert all(row["entities"] == "" for row in create.calls[0]["rows"])
    assert "tag for every word" in (result.get("sampleNote") or "")


def test_a_summarization_import_writes_text_and_summary(run, tmp_path):
    shard = write_summarization_shard(
        tmp_path / "0000.parquet",
        ["The court held that the clause was void.", "The appeal was dismissed with costs."],
        ["The clause was void.", "Appeal dismissed."],
    )

    result, create = run(
        [shard], size=2, task_type="summarization", text_column="document", label_column="summary",
    )

    assert result["status"] == "completed", result
    [call] = create.calls
    assert call["task_type"] == "summarization"
    assert list(call["rows"][0]) == ["text", "summary"]
    assert {row["text"]: row["summary"] for row in call["rows"]} == {
        "The court held that the clause was void.": "The clause was void.",
        "The appeal was dismissed with costs.": "Appeal dismissed.",
    }


def test_summarization_rows_without_a_summary_are_replaced_and_the_sample_stays_exact(run, tmp_path):
    shard = write_summarization_shard(
        tmp_path / "0000.parquet",
        [f"Doc {i}" for i in range(6)],
        ["Summary one", "", None, "Summary four", "   ", "Summary six"],
    )

    result, create = run(
        [shard], size=3, task_type="summarization", text_column="document", label_column="summary",
    )

    assert result["status"] == "completed", result
    assert sorted(row["summary"] for row in create.calls[0]["rows"]) == [
        "Summary four", "Summary one", "Summary six",
    ]


def test_summarization_with_no_summary_at_all_fails_and_no_dataset_is_made(run, tmp_path):
    shard = write_summarization_shard(
        tmp_path / "0000.parquet",
        ["Doc one", "Doc two", "Doc three"],
        ["", "  ", None],
    )

    result, create = run(
        [shard], size=2, task_type="summarization", text_column="document", label_column="summary",
    )

    assert result["status"] == "failed"
    assert result["code"] == "no_usable_rows"
    assert "both a text and a summary" in result["error"]
    assert create.calls == []
    assert not list((tmp_path / "data").glob("library-*.csv"))


def test_the_label_header_follows_the_task_type(run, tmp_path):
    shard = write_ner_shard(tmp_path / "0000.parquet", [["John"], ["ACME"]], [[1], [3]])

    result, create = run(
        [shard], size=2, task_type="named_entity_recognition", text_column="tokens", label_column="ner_tags",
    )

    assert result["status"] == "completed", result
    assert list(create.calls[0]["rows"][0]) == ["text", "entities"]


def test_class_label_names_are_read_from_a_sequence_and_the_older_list_form(tmp_path):
    sequence = write_ner_shard(tmp_path / "0000.parquet", [["a"]], [[1]])
    older = write_ner_shard(
        tmp_path / "0001.parquet", [["a"]], [[1]],
        tag_feature={"feature": [{"_type": "ClassLabel", "names": NER_NAMES}], "_type": "Sequence"},
    )

    assert imports._label_names(pq.ParquetFile(sequence), "ner_tags") == NER_NAMES
    assert imports._label_names(pq.ParquetFile(older), "ner_tags") == NER_NAMES


@pytest.mark.parametrize("features", [None, ["not", "a", "dict"]])
def test_label_names_without_a_feature_map_is_none(tmp_path, features):
    path = tmp_path / "0000.parquet"
    table = pa.table({"tokens": [["a"]], "ner_tags": [[1]]})
    table = table.replace_schema_metadata({"huggingface": json.dumps({"info": {"features": features}})})
    pq.write_table(table, path)

    assert imports._label_names(pq.ParquetFile(path), "ner_tags") is None


def test_a_conll_style_fixture_imports_spans_legal_ml_core_reads(run, tmp_path):
    annotation_format = pytest.importorskip("legal_ml_core.annotation_format")

    conll = write_ner_shard(
        tmp_path / "conll.parquet",
        [
            ["EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", "."],
            ["Peter", "Blackburn"],
        ],
        [
            ["B-ORG", "O", "B-MISC", "O", "O", "O", "B-MISC", "O", "O"],
            ["B-PER", "I-PER"],
        ],
        tag_feature={"feature": {"dtype": "string", "_type": "Value"}, "_type": "Sequence"},
    )

    result, create = run([conll], size=2, task_type="ner", text_column="tokens", label_column="ner_tags")

    assert result["status"] == "completed", result
    assert entity_spans(create.calls[0]["rows"]) == {
        "EU rejects German call to boycott British lamb .": [
            {"text": "EU", "start": 0, "end": 2, "type": "ORG"},
            {"text": "German", "start": 11, "end": 17, "type": "MISC"},
            {"text": "British", "start": 34, "end": 41, "type": "MISC"},
        ],
        "Peter Blackburn": [{"text": "Peter Blackburn", "start": 0, "end": 15, "type": "PER"}],
    }
    check_entity_offsets(create.calls[0]["rows"])
    for row in create.calls[0]["rows"]:
        assert annotation_format._stored_entities(row["entities"]) == json.loads(row["entities"])


def test_ner_and_summarization_csvs_pass_legal_ml_cores_validation(run, tmp_path):
    datasets = pytest.importorskip("legal_ml_core.routes.datasets")
    from legal_ml_core.annotation_format import _stored_entities

    ner = write_ner_shard(
        tmp_path / "ner.parquet",
        [["John", "Smith", "works", "at", "ACME"], ["ACME", "sued", "Mary", "Jones"]],
        [[1, 2, 0, 0, 3], [3, 0, 1, 2]],
    )
    result, create = run([ner], size=2, task_type="ner", text_column="tokens", label_column="ner_tags")

    assert result["status"] == "completed", result
    _, error = datasets.validate_dataset_file(create.last_path, "ner")
    assert error is None, error
    check_entity_offsets(create.calls[0]["rows"])
    for row in create.calls[0]["rows"]:
        assert _stored_entities(row["entities"]) == json.loads(row["entities"])

    summaries = write_summarization_shard(
        tmp_path / "summaries.parquet",
        ["The court held that the clause was void.", "The appeal was dismissed with costs."],
        ["The clause was void.", "Appeal dismissed."],
    )
    result, create = run(
        [summaries], size=2, task_type="summarization", text_column="document", label_column="summary",
    )

    assert result["status"] == "completed", result
    _, error = datasets.validate_dataset_file(create.last_path, "summarization")
    assert error is None, error


def test_imported_csvs_become_datasets_through_legal_ml_cores_real_create_dataset(run, tmp_path):
    pytest.importorskip("legal_ml_core")
    from flask import Flask
    from flask_sqlalchemy import SQLAlchemy
    from legal_ml_core import create_dataset, init_ml_feature
    from legal_ml_core.annotation_format import _stored_entities
    from legal_ml_core.db import MLBase

    app = Flask(__name__)
    app.config.update(
        TESTING=True,
        SQLALCHEMY_DATABASE_URI="sqlite:///:memory:",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        UPLOAD_FOLDER=str(tmp_path / "uploads"),
        MODEL_FOLDER=str(tmp_path / "models"),
    )
    db = SQLAlchemy(model_class=MLBase)
    db.init_app(app)
    init_ml_feature(app, None, app.config["MODEL_FOLDER"], db=db)

    ner = write_ner_shard(
        tmp_path / "real_ner.parquet",
        [["John", "Smith", "works", "at", "ACME"], ["ACME", "sued", "Mary", "Jones"]],
        [[1, 2, 0, 0, 3], [3, 0, 1, 2]],
    )
    result, create = run([ner], size=2, task_type="ner", text_column="tokens", label_column="ner_tags")

    assert result["status"] == "completed", result
    conll = write_ner_shard(
        tmp_path / "real_conll.parquet",
        [["Peter", "Blackburn", "met", "EU"], ["EU", "rejects", "it"]],
        [["B-PER", "I-PER", "O", "B-ORG"], ["B-ORG", "O", "O"]],
        tag_feature={"feature": {"dtype": "string", "_type": "Value"}, "_type": "Sequence"},
    )
    conll_result, conll_create = run(
        [conll], size=2, task_type="ner", text_column="tokens", label_column="ner_tags",
    )

    assert conll_result["status"] == "completed", conll_result
    for row in conll_create.calls[0]["rows"]:
        assert _stored_entities(row["entities"]) == json.loads(row["entities"])
    summaries = write_summarization_shard(
        tmp_path / "real_summaries.parquet",
        ["The court held that the clause was void.", "The appeal was dismissed with costs."],
        ["The clause was void.", "Appeal dismissed."],
    )
    summary_result, summary_create = run(
        [summaries], size=2, task_type="summarization", text_column="document", label_column="summary",
    )

    assert summary_result["status"] == "completed", summary_result
    with app.app_context():
        db.create_all()
        ner_dataset = create_dataset("alice", create.last_path, "Imported NER", "A sample", "ner")
        conll_dataset = create_dataset("alice", conll_create.last_path, "Imported conll NER", "A sample", "ner")
        summary_dataset = create_dataset(
            "alice", summary_create.last_path, "Imported summaries", "A sample", "summarization",
        )

        assert (ner_dataset.task_type, ner_dataset.row_count) == ("ner", 2)
        assert (conll_dataset.task_type, conll_dataset.row_count) == ("ner", 2)
        assert (summary_dataset.task_type, summary_dataset.row_count) == ("summarization", 2)
