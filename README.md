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
    cache=RedisJSONCache(redis),    # get / set(key, value, ttl), add (SET NX) and delete
    is_admin=lambda: "baigle-admin" in roles(),   # who may change the settings
    audit=lambda event, **fields: log(event, fields),
    enqueue=lambda ref: redis.lpush("library_import_queue", ref),  # no import without it
    snapshot_dir="/app/library-snapshots",  # copies of Allow-licensed pins, shared with the worker
    enqueue_snapshot=lambda ref: redis.lpush("library_snapshot_queue", ref),  # both or neither
    max_rows=lambda: 1000,           # the row cap when an admin set none
    hard_max_rows=5000,              # the most an admin may set (default 5000)
    max_fetch_bytes=200_000_000,     # the host's download limit: an admin's cap never goes above it
    community_datasets=community_listing,    # Community source (#122): all five, or none
    community_record=lambda rid: Zenodo(zenodo_url()).record_files(rid),
    community_download=lambda rid, key, max_bytes: Zenodo(zenodo_url()).download(rid, key, max_bytes),
    community_base_url=zenodo_url,   # the Zenodo host this platform reads; other hosts' records are not listed
    community_enqueue_preview=lambda job: redis.lpush("library_preview_queue", job),  # the worker reads previews
    community_preview_max_bytes=16_000_000,  # optional: the most of a file a preview reads (default 16 MB)
    legal_data_hunter_key=lambda owner_id: legal_data_hunter_key(owner_id),  # #126: a user's key or None
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
           create_dataset=create, data_dir="/app/data",
           snapshot_dir="/app/library-snapshots", max_bytes=MAX_FETCH_BYTES,  # the host's download limit
           community_datasets=..., community_base_url=..., community_record=..., community_download=...,
           legal_data_hunter_key=legal_data_hunter_key)  # #126: the same seam as the API's
```

The same worker takes copy refs off `library_snapshot_queue` and runs
`dataset_library.snapshots.run_copy(ref, session=db.session, fetch_shards=fetch_shards,
snapshot_dir=..., max_bytes=...)`, which makes a pinned dataset's copy: the same
parts an import would download, up to the same byte cap, in one Parquet file.
The worker calls `snapshots.sweep(db.session, snapshot_dir)` with the import
sweep (failing copies a worker left queued or building for thirty minutes, and
deleting removed copies no import reads), and `snapshots.check_upstream(db.session)`
about once an hour: it asks the Hub whether each copy's revision still exists, and
flags the pin "Upstream changed" when it does not. The copy is kept either way.

`create(owner_id, file_path, name, description, task_type)` returns the new
dataset's id, or raises `ValueError` with a message a student can read
(legal-ml-core's `DatasetInvalid` is one).

`run_import` claims the import with one conditional update from `queued` to `fetching`; a
worker whose claim loses (another worker took the same ref) returns without doing anything.
While it runs, `run_import` refreshes the import's `heartbeat_at` every
`heartbeat_seconds` (15) from a thread on its own connection. The worker also
calls `dataset_library.imports.sweep(db.session)` now and then (once a minute
will do; any number of workers may): it fails running imports quiet for two
minutes ("Import stopped. Try again.") and queued ones nothing picked up in 30
minutes ("Import could not start. Try again.").

Create the tables from `dataset_library.models.LibraryBase.metadata`, or add
`dataset_library/migrations/versions` to Alembic's `version_locations` (branch
label `dataset_library`).

## Community datasets

The Community source lists open dataset Publications (datasets published to Zenodo
through BAIGLE, #122) with the license each one states. It is offered only when the
host gives all five seams above and an admin switches it on in the settings
(`PUT /api/library/admin/settings` with `"community": {"enabled": true}`). The library
imports none of the repository's client: the host's seams do the Zenodo calls, and
the host's Zenodo base URL is the only host a Community file is fetched from. A
Publication whose record is on another Zenodo host (the sandbox, say) is not listed,
previewed or imported, even when its record id matches: the origin comes from its
record link, or from its DOI prefix when the link is missing, and a link and a DOI
that name different hosts count as neither.

A record is importable only while Zenodo says it is open now: public records with
public files. A restricted or metadata-only record, a record Zenodo no longer
serves, and a record with no table file (CSV, TSV, JSON, JSONL, Parquet or XLSX) are
shown with their reason. A file is read by its bytes, not its name, so a labelled
file BAIGLE publishes as CSV under an `.xlsx` name still reads. An import reads the
record's own file, never a local Dataset, and fails with "This record is no longer
downloadable" when Zenodo has closed the record or removed the file. A Community
dataset carries no task in its publication, so a task filter does not find it.

A request never downloads or parses a Community file. It re-checks the record with
Zenodo (one metadata call, kept for a minute for display) and answers from the
preview's cache. A preview that is not cached yet is queued once per cache key on
`community_enqueue_preview`, and the answer is `preparing: true` until the worker has
read it: the worker calls `dataset_library.community.run_preview(job, cache=, download=)`,
which reads the file's head under the same parser limits as an import, checks its bytes
against the checksum Zenodo lists, and caches what it found. A file's preview is read only
within `community_preview_max_bytes` (16 MB); a larger one shows its card, files and size
with "Too large to preview here; it can still be imported" (`preview: "too_large"`) and
stays importable within the import cap. A refusal is cached for ten minutes, so a bad file
is not fetched again on every open; Zenodo's own failure is cached for a minute. The
preview lock is random and lasts 660 s, longer than Zenodo's read deadline, and only the
job that took it releases it. The record and the preview are bound to the Zenodo origin.

Start and retry decide eligibility on a live record, not the cached preview. A queued
import checks its Publication, the origin it started on, the source and the license
again before it reads: the license must be the Publication's current one, and the one
the import started on (otherwise it fails and asks for a fresh import). Then it checks
the file it started on first, and reads exactly that file, at the checksum it started
on. A retry keeps the Publication, origin and file of the import it repeats. The
import's download is capped at the admin's cap, never above the host's `max_fetch_bytes`.

## Legal Data Hunter entries

[Legal Data Hunter](https://legaldatahunter.com) has documents, not datasets (#126). A course
admin searches its case law, legislation or doctrine with their own key, previews the hits,
and saves up to 100 of them as an entry: a title, a description, a task type
(`classification` by default), the query, and the frozen document ids (`country`, `source`,
`sourceId`). Refresh reruns the query and replaces the ids; an import copies the ids when it
starts, so a Refresh never touches an import or a dataset already made. Entries are shown to
every signed-in user as featured cards labelled "Legal Data Hunter".

Each user imports with their **own** key. The host gives one seam,
`legal_data_hunter_key(owner_id)`, returning the user's key (`sk-…`) or None; the package
never stores, logs or returns a key. Without the seam, Legal Data Hunter is not offered; a seam
that raises is read as no key (only the exception's type is logged), so no page breaks. In the
worker the same holds: a seam that raises fails the import as `needs_key`, which Try again resumes
from, and only the exception's type is logged. The key
goes out through `requests`' `auth=`, so a `.netrc` entry for the host cannot replace it. Only
`https://legaldatahunter.com` is called: redirects are followed by hand, only on that
origin, and one to any other host is refused before anything is sent there.

Its terms forbid storing full text or training models unless the user's plan allows it, and
the free plan does not. So:

- every import needs `licenseAck: true` (the checkbox "My Legal Data Hunter plan allows
  storing these documents and training models on them."), and its time is the dataset's
  provenance (`licenseAckAt`, also `legalDataHunter.termsConfirmedAt`); a retry keeps it;
- the license table has an entry "Legal Data Hunter terms" (`legal-data-hunter-terms`,
  group "Provider terms") that an admin may set to Warn (the default) or Hide, never Allow:
  `PUT /admin/settings` refuses `allow` with "Choose Warn or Hide.", and an `allow` stored any
  other way is read as Warn. Every license now carries `levels`, the levels an admin may choose.
  Hide takes the entries out of the library, fails queued imports, and deletes the documents
  every import not running fetched (no Try again may resume while the terms are hidden). A
  running import checks the terms before it writes each document and once more before it saves
  the dataset: under Hide it fails as `license_hidden` and discards what it fetched. While the
  terms are Hide, `imports.sweep` deletes the documents of every import not running, whatever
  its failure.
  `GET /options` leaves the terms out of the student's license filter, and no Hugging Face
  dataset's tag ever resolves to them (`resolve_license` gives `unknown`);
- a dataset built from Legal Data Hunter, and any model trained on one, must not be published.
  Its provenance says `publishable: false` with `publishRefusal`; the host refuses with
  `legal_data_hunter.publish_refusal(session, dataset_ref, model=False|True, file_path=None)`,
  which returns `DATASET_PUBLISH_REFUSED` or `MODEL_PUBLISH_REFUSED`, or None for any other
  dataset. **Hosts should pass the Dataset's `file_path`** (the path `create_dataset` was given
  and stored). The worker saves that path on its import (`saved_path`) in the same commit as the
  move to saving, before `create_dataset` is called, and records the Dataset's ref, in its own
  commit, the moment `create_dataset` returns. `publish_refusal` matches any Legal Data Hunter
  import by `dataset_ref` or by `saved_path == file_path`, whatever its status, so a dataset
  whose worker died before its ref was recorded is still refused by its file
  (`from_legal_data_hunter(session, dataset_ref, file_path=None)` is the same check as a bool). Provenance's `publishable` and
  `publishRefusal` cover a provider's terms only, not licenses: a dataset under a share-alike or
  no-derivatives license says `publishable: true`, and the host still checks its license itself.

The worker fetches each document with `GET /v1/documents/{country}/{source}?source_id=…`
(the query form, as a source id may hold slashes), one credit each, and writes one row per
document with the columns `text, court, date, country, language, ecli, eli, source_url,
text_truncated`, unlabelled, in the entry's order. **The CSV is not `["text"]` alone**, so a
host that decides annotation mode from the header must treat a Legal Data Hunter import as an
annotation dataset (it has no label). A document with no text is skipped (`skipped`), one
Legal Data Hunter no longer has (404) or refuses as malformed (400, 422) too (`missing`); both
are counted on the import and in provenance. An import fetches at most the row cap's worth of
documents, and its documents' responses together may not pass the worker's download cap (the
`max_bytes` it passes to `run_import`, under the admin's cap). Each response body is read
streamed, and reading stops within one chunk of what is left of the cap; the bytes each
document's response measured (`nbytes`, also for one with no text or one gone) are kept with
it, and a resumed import starts its total from them. An import that would pass the cap fails as
`too_large`, keeping the documents fetched before it, and records the cap it stopped under
(`failed_cap`). Try again resumes past them once an admin has raised the download limit; until
then it is refused with 409 `cap_unchanged` ("This import stopped at the platform's download
limit, so trying again now would stop at the same place. An admin has to raise the download
limit first."), without calling Legal Data Hunter. The worker's `max_bytes` should be the
host's `max_fetch_bytes`, which the retry compares with. A country keeps the case Legal Data Hunter
gives it (`CoE`, `EU`, `INTL`); a search's country filter upper-cases two-letter codes only, and
a hit whose source prefix (`FR/…`) disagrees with its `country` is left out.

Each document is kept as it is fetched (`library_ldh_documents`). A 429 for Legal Data Hunter's
per-minute cap (`limit_type: "minute"`, on every plan) is waited out (its `Retry-After`, at least a
second and at most a minute), up to three times in a row, and the same document is asked for again;
if it does not lift, the import fails as `provider_busy`. A `Retry-After` is read as ASCII digits or
an HTTP date, and never as more than 31 days; more than ten digits is read as 31 days without
being parsed. When
the credits run out (a 429 for the day or the period) the import fails with code `out_of_credits` and "Fetched N of M — your Legal Data Hunter
credits ran out. Try again when they reset.", and `resumeAfter` from `Retry-After`. Such a pause
stays on the student's list (`GET /imports`), and resumable, until a week after it paused or a
day after its credits reset, whichever is later; any other failure Try again resumes from stays
listed for the week its documents are kept. Try again (`/imports/<ref>/retry`) moves the
fetched documents to the new import, which fetches only the rest, from the ids the first import
was started with, all of them, even when the entry has since been deleted or the row cap
lowered. It is refused with 409 `already_running` while another import of the same entry runs,
and with 409 `not_retryable` once the import was put away. A retry claims the paused import
(puts it away, with one conditional update from not put away) in the same transaction that
inserts the new import and moves the documents, so of two retries of one import exactly one
starts and the other gets 409 `not_retryable`. When the queue refuses the new import, one commit
fails it as `not_started` and puts it away, moves the documents back to the paused import and
takes the paused import out of put away, so a sweep in between finds nothing to delete.

At most one import of an entry per owner is queued or running: such an import holds
`active_key` (`"<owner>:<entry>"`), which has a unique index, and every move to completed,
failed or cancelled (a sweep, a cancel, a failure, a queue refusal) sets it to null in the same
statement. A start that loses a race to another start of the same entry gets 409
`already_running`. Hugging Face and Community imports leave it null.

Each document is written with one statement that writes it only while its import is running and
not put away (`INSERT … SELECT … WHERE EXISTS`), so a document that arrives after the import was
cancelled, put away or erased, even between the worker's last check and its write, is never
written.

Fetched documents are kept only for a failure Try again resumes from (`out_of_credits`,
`provider_busy`, `bad_key`, `needs_key`, `too_large`, and `stopped`, the sweep's code for an
import whose worker went quiet, as on a restart), for as long as above. They are deleted when the import
completes (best effort: a failed delete leaves them to the sweep, never the dataset's file), is
cancelled or put away, or fails any other way, and `imports.sweep` deletes any left over. A host
erasing a user calls `legal_data_hunter.erase(session, owner_id)`, before or after deleting the
user's imports: it deletes the documents of that user's imports and any whose import is gone,
and sets `created_by` to null on the entries the user made (entries are course data and stay).

### Legal Data Hunter routes

Any signed-in user:

- `GET /api/library/featured` → also `"legalDataHunter": [card, ...]`.
- `GET /api/library/legal-data-hunter/entries` → `{"entries": [card], "offered", "hasKey",
  "needsKey"}` (`needsKey` is the message, or null with a key).
- `GET /api/library/datasets/legal_data_hunter/<entry_ref>` → one card; 404 when it is not in
  the library.
- `GET /api/library/options` → also `"legalDataHunter": {"offered", "hasKey", "needsKey",
  "corpora": [{"value", "label"}], "maxDocuments": 100}`.
- A card: `id` (`legal_data_hunter:<ref>`), `provider` (`legal_data_hunter`), `providerLabel`
  ("Legal Data Hunter"), `externalId` (the entry ref), `title`, `description`, `taskType`,
  `featured: true`, `corpus`, `documentCount`, `credits` (documents an import would fetch,
  under the row cap), `creditCost` ("uses 100 of your Legal Data Hunter credits"),
  `refreshedAt`, `license` (the terms at the platform's level), `hasKey`, `importable`,
  `reasonCode`/`reason` (`needs_key`: "Needs your Legal Data Hunter key — add it in Settings";
  `no_documents`), `confirmation: {"text", "checkbox"}`, `publishable: false`,
  `publishRefusal`, `citation`.
- `POST /api/library/imports` with `{"provider": "legal_data_hunter", "dataset": <entry ref>,
  "licenseAck": true, "name"?}` → 202 and the import (`documentCount`, `skipped`, `missing`,
  `resumeAfter`). Refused: 404 `not_found` / `not_offered`, 403 `license_hidden`, 409
  `needs_key`, `needs_ack` (every time without the confirmation), `no_documents`, 400
  `bad_name`. The task type is the entry's. A queued or running import of the same entry →
  200 and that one; a start that races another start of it → 409 `already_running`. Failure
  codes from the worker: `out_of_credits`, `bad_key`, `needs_key`, `provider_busy`,
  `license_hidden`, `needs_ack`, `no_usable_rows`, `not_offered`, `too_large`, and the sweep's
  `stopped`.
- `POST /api/library/imports/<ref>/retry` of a Legal Data Hunter import → 202 and the new
  import. Refused: 409 `already_running` ("Another import of this entry is still running. Wait
  for it to finish, then try again."), 409 `not_retryable` (not failed or cancelled, put away,
  or claimed by another retry), 409 `cap_unchanged` (it stopped as `too_large` and the download
  limit is not higher yet), 409 `already_saved` (it stopped after its Dataset was made, so
  Try again would make a second one). The paused import is put away only once a new import stands in for
  it; this holds for every provider.

Admins only (404 `not_offered` without the seam):

- `POST /api/library/admin/legal-data-hunter/preview` with `{"q", "corpus"?: "case_law" |
  "legislation" | "doctrine" (case law only when missing or null; any other value must name
  one), "country"?: [..], "source"?: [..], "dateFrom"?, "dateTo"?
  (YYYY-MM-DD), "limit"? (1–100, default 100)}` → `{"query", "hits": [{"country", "source",
  "sourceId", "title", "court", "date", "ecli", "language", "url", "snippet"}], "total", "max"}`,
  searched with the admin's own key (one credit).
- `GET /api/library/admin/legal-data-hunter/entries` → `{"entries": [entry], "max": 100}`;
  `GET .../entries/<ref>` → one entry.
- `POST .../entries` with `{"title", "description"?, "taskType"?, "query", "documents"?}` →
  201 and the entry. `documents` are the hits the admin chose (at most 100: 400 `invalid` with
  `fields.documents` "An entry holds at most 100 documents."), and their number is stored as
  `query.limit`, so a Refresh keeps the entry's size; without them the query is run and its
  hits are kept.
- `PATCH .../entries/<ref>` with `{"title"?, "description"?, "taskType"?}`; `POST
  .../entries/<ref>/refresh` reruns the query and replaces the ids (a search that finds nothing
  is refused with 400 `invalid`, `fields.documents`, and the entry is left as it was); `DELETE .../entries/<ref>`
  → 204.
- An entry: `ref`, `title`, `description`, `taskType`, `query`, `documents`, `documentCount`,
  `createdAt`, `updatedAt`, `refreshedAt`.
- Errors: 400 `invalid` with `fields` (`query.q`, `query.corpus` (also for a corpus that is not
  text, such as `[]`, `{}`, `false`, `0` or `""`), `query.dateFrom`, …), 409
  `needs_key` (the admin has no key), 409 `bad_key`, 429 `out_of_credits` with `retryAfter` (or `slow_down` for the per-minute cap), 503
  `provider_unavailable` (also for a redirect off Legal Data Hunter's origin, or one with no
  `Location`), 404 `not_found`.
- Audited: `library.ldh_entry_created`, `_updated`, `_refreshed`, `_removed`, with counts and
  field names only.

## API

- `GET /api/library/search?q=&task=&license=&source=` (`license=legal-data-hunter-terms`
  asks Hugging Face nothing and answers `{"status": "ok", "results": [], "count": 0}`) →
  `{"status": "ok" | "stale" | "unavailable", "results": [...], "count": n}`.
  `stale`: the Hub is limiting requests and these are the last results, up to a
  day old. `unavailable`: the Hub is down or unreadable; nothing is raised, and
  the results are the pinned datasets (filtered the same way), each with a
  `snapshot` of `{"status": "ready" | "queued" | "building" | "failed" | "none"}`.
  A `ready` one can be imported from its copy while the Hub is down.
  Every card carries `pinRef`: the ref of the pin for that dataset, or null.
  Every card also carries `citation`: `title`, `authors` (a list of
  `{"family", "given"}` or `{"literal"}`), `year`, `url`, `revision`,
  `accessedAt` (null until someone imports it) and `bibtex`, the dataset's own
  BibTeX entry exactly as written, or null when it has none. A search result
  does not read the dataset's README, so its `bibtex` can be null where the
  single-dataset card's is not.
- `GET /api/library/datasets/community/<record_id>` (#122) → a Community dataset's card, `importable`
  and `reasonCode`/`reason` when it is not, `columns`, `rows` and `rowCount` read from its table file,
  `proposed` columns, `citation`, and `revision` (the file's checksum). 404 when it is not listed;
  503 while Zenodo cannot be asked.
- `GET /api/library/labels?ids=huggingface:org/name,...` → `{"labels": {id: "yes" | "no" | "unclear"}}`,
  from the dataset viewer's column types; up to 30 ids, cached an hour each.
- `GET /api/library/datasets/huggingface/<org>/<name>?config=&split=` → the
  card plus `configs`, `config`, `split`, `columns`, the first `rows`, `labels`
  `rowCount` (the split's size, when known) and `preview: "ok" | "unavailable"`.
- `GET /api/library/options` → the task, license and source filters, the
  license table (`level` is this platform's: `allow`, `warn` or `hide`; a
  provider's terms, such as `legal-data-hunter-terms`, and their group "Provider
  terms" are left out, since no dataset in search has them),
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
- `GET /api/library/datasets/huggingface/<org>/<name>` while Hugging Face cannot be asked: the copy the
  platform keeps of that part stands in (`offline: true`, its columns and first rows); 503 without one.
- Admins only: `POST /api/library/pins/<pin_ref>/snapshot` keeps a copy of a pin
  that has none, or whose copy failed (202; the worker makes it). 409 `not_allowed`
  unless the dataset's license is Allow; 404 `not_offered` when the host keeps no
  copies. Each pin in `GET /api/library/pins` carries `snapshot` (as above) and
  `upstreamChanged`, true when the Hub no longer has the copy's revision.
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
the package, and `https://legaldatahunter.com` for Legal Data Hunter entries, with the user's own key
(redirects only on that origin). Descriptions are returned as plain text for the host to escape.

Copies of pinned datasets are kept only while the dataset's license level is
`allow`: a pin made under another level gets none, and a change to another level
removes the copies it no longer allows. A copy is deleted when its pin is removed,
unless a queued or running import still reads it; then it goes once that import
has finished. Copies hold the dataset's rows and no personal data.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[test]"
.venv/bin/python -m pytest -q
```

Tests never call the Hub: `tests/fixtures/huggingface` holds recorded answers.

## License

MIT
