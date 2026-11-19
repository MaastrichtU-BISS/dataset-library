# dataset-library

Search public legal datasets by **license** and **task** from a Flask app, and
import a random sample as the host's own Dataset. Built for
[BAIGLE](https://github.com/shashankmc/reducing_digital_divide), where law
students start with no data.

v0.1.0 searches the [Hugging Face Hub](https://huggingface.co/datasets).

## Installing into a host

```python
from dataset_library import init_dataset_library

init_dataset_library(
    app,
    db=db,                          # the host's Flask-SQLAlchemy handle
    login_required=jwt_required(),  # the host's sign-in decorator
    current_owner=get_jwt_identity,
    task_types=[t["value"] for t in TASK_TYPES],  # legal-ml-core's task types
    cache=RedisJSONCache(redis),    # get(key) / set(key, value, ttl)
    is_admin=lambda: "baigle-admin" in roles(),   # who may change the settings
    audit=lambda event, **fields: log(event, fields),
    enqueue=lambda ref: redis.lpush("library_import_queue", ref),  # no import without it
    max_rows=lambda: 1000,           # the row cap when an admin set none
    hard_max_rows=5000,              # the most an admin may set (default 5000)
)
```

An admin's row cap is returned by `dataset_library.limits.row_cap(db.session)`
(`None` when unset). Feed it to the host's own row check so uploads meet the
same cap, e.g. legal-ml-core's
`app.config["LEGAL_ML_MAX_ROWS"] = lambda: row_cap(db.session) or 1000`.

The worker that takes refs off that queue installs `dataset-library[import]`
(pyarrow) and runs each one, inside the host's app context:

```python
from dataset_library.imports import run_import
from dataset_library.providers.parquet import fetch_shards

run_import(ref, session=db.session, fetch_shards=fetch_shards,
           create_dataset=create, data_dir="/app/data")
```

`create(owner_id, file_path, name, description, task_type)` returns the new
dataset's id, or raises `ValueError` with a message a student can read
(legal-ml-core's `DatasetInvalid` is one).

While it runs, `run_import` refreshes the import's `heartbeat_at` every
`heartbeat_seconds` (15) from a thread on its own connection. The worker also
calls `dataset_library.imports.sweep(db.session)` now and then (once a minute
will do; any number of workers may): it fails running imports quiet for two
minutes ("Import stopped. Try again.") and queued ones nothing picked up in 30
minutes ("Import could not start. Try again.").

Create the tables from `dataset_library.models.LibraryBase.metadata`, or add
`dataset_library/migrations/versions` to Alembic's `version_locations` (branch
label `dataset_library`).

## API

- `GET /api/library/search?q=&task=&license=&source=` →
  `{"status": "ok" | "stale" | "unavailable", "results": [...], "count": n}`.
  `stale`: the Hub is limiting requests and these are the last results, up to a
  day old. `unavailable`: the Hub is down or unreadable; nothing is raised.
  Every card carries `pinRef`: the ref of the pin for that dataset, or null.
  Every card also carries `citation`: `title`, `authors` (a list of
  `{"family", "given"}` or `{"literal"}`), `year`, `url`, `revision`,
  `accessedAt` (null until someone imports it) and `bibtex`, the dataset's own
  BibTeX entry exactly as written, or null when it has none. A search result
  does not read the dataset's README, so its `bibtex` can be null where the
  single-dataset card's is not.
- `GET /api/library/labels?ids=huggingface:org/name,...` → `{"labels": {id: "yes" | "no" | "unclear"}}`,
  from the dataset viewer's column types; up to 30 ids, cached an hour each.
- `GET /api/library/datasets/huggingface/<org>/<name>?config=&split=` → the
  card plus `configs`, `config`, `split`, `columns`, the first `rows`, `labels`
  `rowCount` (the split's size, when known) and `preview: "ok" | "unavailable"`.
- `GET /api/library/options` → the task, license and source filters, the
  license table (`level` is this platform's: `allow`, `warn` or `hide`),
  `importable` and `maxRows` (the admin's cap, else `max_rows()`).

- `GET /api/library/featured` (any signed-in user) → `{"featured": [...]}`:
  up to three featured pins, oldest pinned first, leaving out any whose
  license an admin has since hidden or whose owner is outside the enabled
  sources. Empty until an admin features one.
- Admins only (`is_admin()`) pin and feature:
  - `POST /api/library/pins` with `{"dataset": "org/name"}` → 201 and the pin;
    pinning the same provider and dataset id again → 200 and that pin (one pin
    per provider and dataset id, enforced by a unique constraint too). Refused
    like the preview route: 404 when the dataset is outside the enabled
    sources or private, 403 `license_hidden`, 400 `bad_id`, 503 when the Hub
    is down. The pin keeps the card as it was pinned and the labels verdict
    (`labelled`: `yes`, `no` or `unclear`), so its badge stays exact; the
    card's `license` is sent at the platform's current level.
  - `GET /api/library/pins` → `{"pins": [...]}`, oldest first.
  - `PATCH /api/library/pins/<pin_ref>` with `{"featured"?: bool, "title"?:
    string|null}` → 200 and the pin. At most three are featured at a time (409
    `too_many_featured`); a title over 120 characters → 400 `bad_title`; null
    or empty clears it back to the card's; a featured that is not a boolean →
    400 `bad_featured`; an unknown ref → 404.
  - `DELETE /api/library/pins/<pin_ref>` → 204.
  - Each change is audited: `library.pin_added`, `library.pin_updated`,
    `library.pin_removed` — ids and flags only, never personal data. Pins are
    course data; a host erasing a user should keep or null `pinned_by`, not
    delete their pins.

- `POST /api/library/imports` with `{"dataset", "config", "split", "textColumn",
  "labelColumn", "size", "name"?, "taskType"?, "licenseAck"?, "revision"?,
  "balance"?}` → 202 and the import,
  `queued`. The size is cut to the row cap (the admin's, else `max_rows()`).
  `balance` keeps the label proportions of the whole dataset; it defaults to
  `true` when a `labelColumn` is mapped and is always `false` without one. The
  same import (dataset, revision, mapping, size, balance) while one is queued
  or running → 200 and that one, queued once. Refused before anything is
  queued: 400 (`bad_id`, `bad_mapping`, `bad_size`, `bad_name`, `bad_task`,
  `bad_config`, `bad_balance`), 403
  (`license_hidden`), 404, 409
  (`gated`, `private` when the Hub's answer says the owner keeps it private
  (an anonymous 401/404 is `not_found`), `not_importable`, `needs_ack` for a license set to warn, `changed`
  when `revision` is not the dataset's now), and 503
  when the queue is down (the import is stored as failed).
- `GET /api/library/imports` and `GET /api/library/imports/<ref>` → the
  signed-in user's imports: `status` (`queued`, `fetching`, `sampling`,
  `validating`, `saving`, `completed`, `failed`, `cancelled`), `error`, `code`, `rows`, `datasetRef`,
  the mapping (`textColumn`, `labelColumn`, `taskType`, `size`) and the
  provenance (`externalId`, `revision`, `url`, `license`, `licenseAckAt`, `importedAt`).
  `balance`, `sampleMethod` (`"stratified"` or `"random"`) and `sampleNote`
  (why a balance fell back to a random sample, else `null`) say how it sampled.
  `?current=true` → only those queued or running, and those failed in the last
  day and not put away.
- `GET /api/library/provenance/<dataset_ref>` → the provenance of a Dataset the
  signed-in user imported: `provider`, `externalId`, `config`, `split`,
  `revision`, `url`, `importedAt`, `licenseAckAt` (when the license was
  confirmed), `license` (the license table entry, at the platform's current
  level, which an admin may have changed since the import), plus `balance`,
  `sampleMethod` and `sampleNote`. `citation` is what the Dataset is cited
  from: the card's fields as the import read them, with `revision` the
  revision it read and `accessedAt` the time it was imported; `null` for
  imports made before 0.11.0. 404 when this user has no completed import of
  it; works without an import queue.
- `POST /api/library/imports/<ref>/cancel` → the import, `cancelled`; once it is
  `saving` or `completed`, 409 `too_late` (cancel and finishing never both win).
  A failed or cancelled import comes back as it is.
- `POST /api/library/imports/<ref>/retry` → 202 and a new import with the same
  settings at the revision first imported (409 `changed` if the dataset moved
  on); the old one is put away. Only for `failed` or `cancelled` (else 409
  `not_retryable`).
- `POST /api/library/imports/<ref>/dismiss` → puts a finished import away;
  409 `running` while it runs.
- Admins only (`is_admin()`): `GET` / `PUT /api/library/admin/settings` (sources
  and license levels, saved whole; 400 with `fields` when anything is invalid).
  `limits` is `{"maxRows", "maxDownloadMb"}`, each a whole number from 1 to its
  hard maximum (`maxRowsHardMax`, `maxDownloadMbHardMax` = 1000) or `null` for
  the default; a `PUT` without `limits` keeps them. The download cap replaces
  the `max_bytes` the worker passes to `run_import`
  and `POST /api/library/admin/sources/check` (`{"ref"}` → `{"ok", "error"?}`).

## Licenses

`dataset_library.licenses` is the one vocabulary. Each license has flags
(`nc`, `nd`, `sa`, `unknown`), a default level (`warn` for non-commercial,
no-derivatives and unknown licenses, `allow` otherwise), and for `warn` the
condition and the confirmation a student reads before importing. A license
tag nobody recognises resolves to `unknown`.

## Security

Only `https://huggingface.co` (the API and the converted Parquet files) and `https://datasets-server.huggingface.co` are called, with parameters built by
the package. Descriptions are returned as plain text for the host to escape.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[test]"
.venv/bin/python -m pytest -q
```

Tests never call the Hub: `tests/fixtures/huggingface` holds recorded answers.

## License

MIT
