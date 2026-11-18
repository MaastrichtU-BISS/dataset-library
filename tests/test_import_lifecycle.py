"""Imports you can leave, cancel and retry (#114): cancel against the worker
finishing, the heartbeat, and the sweep."""
import time
from datetime import timedelta

import pytest
import sqlalchemy as sa

from dataset_library import imports
from dataset_library.models import LibraryImport
from test_import import run, write_shard  # noqa: F401 - `run` is a fixture


def status_of(app):
    with app.app_context():
        return app.db.session.execute(sa.text("SELECT status FROM library_imports")).scalar_one()


def cancel_as_alice(app):
    with app.app_context():
        ref = app.db.session.execute(sa.text("SELECT id FROM library_imports")).scalar_one()
        return imports.cancel(app.extensions["dataset_library"], "alice", ref)


def test_a_cancel_before_saving_wins_and_no_dataset_is_made(run, app, tmp_path, monkeypatch):
    """Cancel lands after the sample is written but before the worker claims `saving`."""
    shard = write_shard(tmp_path / "0000.parquet", ["A clause", "Another"], [0, 1])
    write_csv = imports._write_csv

    def write_then_cancel(path, record, rows):
        write_csv(path, record, rows)
        assert cancel_as_alice(app)["status"] == "cancelled"

    monkeypatch.setattr(imports, "_write_csv", write_then_cancel)

    result, create = run([shard], size=2)

    assert result["status"] == "cancelled"
    assert create.calls == []
    assert not list((tmp_path / "data").glob("library-*.csv"))


def test_a_cancel_while_saving_is_refused_and_the_import_completes(run, app, tmp_path):
    shard = write_shard(tmp_path / "0000.parquet", ["A clause", "Another"], [0, 1])
    refusals = []

    def try_cancel(app):
        with pytest.raises(imports.Refused) as refused:
            cancel_as_alice(app)
        refusals.append(refused.value.code)

    result, create = run([shard], size=2, while_saving=try_cancel)

    assert refusals == ["too_late"]
    assert result["status"] == "completed"
    assert len(create.calls) == 1


def test_a_cancel_after_completing_is_refused(run, app, tmp_path):
    shard = write_shard(tmp_path / "0000.parquet", ["A clause", "Another"], [0, 1])
    result, _ = run([shard], size=2)
    assert result["status"] == "completed"

    with pytest.raises(imports.Refused) as refused:
        cancel_as_alice(app)

    assert refused.value.code == "too_late"
    assert status_of(app) == "completed"


def test_cancelling_twice_or_after_failing_changes_nothing(run, app, tmp_path):
    from dataset_library.errors import ProviderError

    result, _ = run([], fail_fetch=ProviderError("busy"))
    assert result["status"] == "failed"

    assert cancel_as_alice(app)["status"] == "failed"


def add(app, **fields):
    defaults = dict(
        owner_id="alice", provider="huggingface", external_id="coastalcph/lex_glue", config="ledgar", split="train",
        text_column="text", label_column="label", task_type="classification", sample_size=10, max_rows=1000,
        name="LexGLUE", license="cc-by-4.0", status="fetching",
    )
    with app.app_context():
        record = LibraryImport(**{**defaults, **fields})
        app.db.session.add(record)
        app.db.session.commit()
        return record.id


def statuses(app):
    with app.app_context():
        return {r.id: (r.status, r.error_code, r.error_message) for r in app.db.session.query(LibraryImport)}


def test_the_sweep_fails_quiet_imports_and_queued_ones_nothing_picked_up(app):
    now = imports._now()
    quiet = add(app, status="sampling", heartbeat_at=now - timedelta(minutes=3))
    alive = add(app, status="fetching", heartbeat_at=now - timedelta(seconds=30))
    never_beat = add(app, status="saving", updated_at=now - timedelta(minutes=5))
    forgotten = add(app, status="queued", created_at=now - timedelta(hours=1))
    waiting = add(app, status="queued", created_at=now - timedelta(minutes=5))
    done = add(app, status="completed", heartbeat_at=now - timedelta(days=1))

    with app.app_context():
        swept = imports.sweep(app.db.session, now=now)

    assert sorted(swept) == sorted([quiet, never_beat, forgotten])
    found = statuses(app)
    assert found[quiet] == ("failed", "stopped", "Import stopped. Try again.")
    assert found[never_beat] == ("failed", "stopped", "Import stopped. Try again.")
    assert found[forgotten] == ("failed", "not_started", "Import could not start. Try again.")
    assert found[alive][0] == "fetching"
    assert found[waiting][0] == "queued"
    assert found[done][0] == "completed"


def test_a_worker_whose_import_was_swept_stops_without_making_a_dataset(run, app, tmp_path):
    shard = write_shard(tmp_path / "0000.parquet", ["A clause", "Another"], [0, 1])

    def swept_meanwhile(engine):
        with app.app_context():
            imports.sweep(app.db.session, stale_after=timedelta(seconds=-1))

    result, create = run([shard], size=2, while_fetching=swept_meanwhile)

    assert result["status"] == "failed"
    assert result["code"] == "stopped"
    assert create.calls == []


def test_a_running_import_beats_while_it_downloads(make_app, tmp_path):
    """On a file database, so the heartbeat's own connection sees the import."""
    from test_import import FakeCreateDataset

    app = make_app(config={"SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'db.sqlite'}"})
    ref = add(app, status="queued")
    shard = write_shard(tmp_path / "0000.parquet", ["A clause", "Another"], [0, 1])
    beats = []

    def slow_fetch(external_id, config, split, max_bytes, dest_dir):
        with app.db.engine.connect() as connection:
            first = connection.execute(sa.text("SELECT heartbeat_at FROM library_imports")).scalar_one()
        time.sleep(0.3)
        with app.db.engine.connect() as connection:
            beats.append((first, connection.execute(sa.text("SELECT heartbeat_at FROM library_imports")).scalar_one()))
        return imports.Fetched(paths=[str(shard)], revision=None)

    with app.app_context():
        imports.run_import(ref, session=app.db.session, fetch_shards=slow_fetch, create_dataset=FakeCreateDataset(),
                           data_dir=str(tmp_path / "data"), heartbeat_seconds=0.05)

    [(first, later)] = beats
    assert first is not None
    assert later > first
    assert statuses(app)[ref][0] == "completed"
