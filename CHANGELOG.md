# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version is 0.x, minor releases may contain breaking changes.

## [Unreleased]

## [0.9.0] - 2026-11-18

### Added

- NER and summarization dataset imports (BAIGLE #117).
  - A preview of a card tagged `ner` or `summarization` now carries
    `proposed: {taskType, textColumn, labelColumn}`, so the import's column
    step starts on the right columns. A task is proposed only when the
    selected config holds a distinct text and label pair for it. NER takes a
    tag list (ClassLabels, strings or integers) named `ner_tags`, `tags`,
    `ner`, `labels` or `label` as the label; any other tag list must show BIO
    tags (`B-PER`) in its class names or first rows, preferring names ending
    in `tags`, so part-of-speech and parsing configs on a token-classification
    card get no proposal. The text is a list of strings that is neither the
    label nor BIO-tagged (`tokens`, `words`, else the first). Lists of lists
    are not proposed. Summarization proposes the first of `document`,
    `article`, `text`, `dialogue`, `source`, `input` and of `summary`,
    `highlights`, `abstract`, `target`, `output`, falling back to the longest
    sampled text column that is not an id (`id`, `*_id`). All three are
    `null` when no pair fits the selected config (a `ner`-tagged
    classification or dependency-parsing config, for example) or the preview
    is unavailable.
  - NER imports join a row's tokens into `text` and write the row's distinct
    entity types into `entities` (BIO position stripped, `O` dropped, first
    seen first, missing tokens skipped). Numbered tags are read through the
    ClassLabel names in the Parquet's `huggingface` metadata, including a
    ClassLabel inside a Sequence and its older list form. A tag column that is
    a list of numbers with no names (or an empty name list) fails the import with `no_entity_names`
    before any row is sampled, with a per-row check as a backstop.
  - Summarization imports write the source document as `text` and its target
    as `summary`. A row whose summary is missing or blank is skipped and
    replaced like a row with empty text, and when no row has both a text and a
    summary the import fails with `no_usable_rows`.
  - `POST /api/library/imports` refuses a request whose `labelColumn` is the
    same as its `textColumn` with 400 `bad_mapping`
    ("Pick different columns for the text and the label."), for every task
    type.
  - A dataset whose summary column is `highlights` (cnn_dailymail, for
    example) now counts as labelled.
  - The import CSV's second header follows the task type: `entities` for
    `ner` and `named_entity_recognition`, `summary` for `summarization`, and
    `label` for classification-like tasks (unchanged). A dataset with no label
    column still writes `text` alone.
  - The conversion lives in `dataset_library.ner` as `tokens_to_text` and
    `entity_types`.
  - Tests cover NER with numbered and string tags, rows with no entities,
    empty token lists, tags without names, summarization, missing summaries,
    the preview proposals, and the written CSVs against legal-ml-core's real
    `validate_dataset_file` and `create_dataset` (skipped when legal-ml-core
    is not installed).

## [0.8.0] - 2026-11-18

### Added

- Keeping the label balance when sampling (BAIGLE #115).
  - `POST /api/library/imports` takes an optional `balance`. With a
    `labelColumn` it defaults to `true`; without one it is always `false`, and
    a value that is not a boolean is refused with 400 `bad_balance`. Only the
    single-label classification task types (`classification`,
    `document_classification`, `judgment_prediction`) keep a balance; any
    other task type stores `balance` as `false`.
  - A balanced import samples proportionally on the label column (largest
    remainder, so the size is exact), reading only that column first and then
    only the mapped columns of the chosen row groups. An empty or missing
    label column, a list-valued label column, or more distinct labels than
    rows wanted falls back to a random sample with a note saying why. A label
    whose rows all have empty text is topped up from the other labels so the
    size stays exact, with a note that the sample is only roughly balanced.
    The chosen rows are shuffled, so a label-sorted source does not produce a
    label-sorted CSV.
  - Imports carry `balance`, `sampleMethod` (`"stratified"` or `"random"`) and
    `sampleNote` (why a balance fell back, else `null`) in `to_dict()` and in
    `GET /api/library/provenance/<dataset_ref>`.
  - The Dataset description says "A sample that keeps the label balance, from
    Hugging Face: …" when it was stratified.
  - Migration `dl_0005_label_balance` adds `library_imports.balance`,
    `sample_method` and `sample_note`.
- Pins: a course admin curates a starting point (BAIGLE #119).
  - `POST /api/library/pins` with `{"dataset": "org/name"}` (admins) → 201 and
    the pin; the same provider and dataset id pins once (200 and the existing
    pin). The dataset must still be in the library: 404 outside the enabled
    sources, 403 `license_hidden`, 404 when private, 400 `bad_id`, 503 when
    the Hub is down. The pin stores the card as it was pinned and the labels
    verdict, so its badge is exact; the card's license carries the platform's
    current level.
  - `PATCH /api/library/pins/<pin_ref>` with `{"featured"?: bool, "title"?:
    string|null}` (admins) → 200 and the pin. Featuring a fourth while three
    are featured → 409 `too_many_featured`; a title over 120 characters → 400
    `bad_title` (null or empty clears it); a featured that is not a boolean →
    400 `bad_featured`; an unknown ref → 404.
  - `GET /api/library/pins` (admins) → `{"pins": [...]}`, oldest first; 403
    `not_admin` otherwise. `DELETE /api/library/pins/<pin_ref>` → 204.
  - `GET /api/library/featured` (any signed-in user) → `{"featured": [...]}`:
    up to three featured pins, oldest first, leaving out any whose license is
    now hidden or whose owner is outside the enabled sources.
  - Search results carry `pinRef` (the pin's ref, or null) on every card.
  - Every pin change is audited: `library.pin_added`, `library.pin_updated`,
    `library.pin_removed` (ids and flags only). A title change records
    `title_changed` or `title_cleared`, never the title itself, and an update
    that changes nothing is not audited.
  - Migration `dl_0006_pins` adds the `library_pins` table.

## [0.7.0] - 2026-11-18

### Added

- Provenance for an imported Dataset (BAIGLE #116).
  - `GET /api/library/provenance/<dataset_ref>` returns the signed-in owner's
    most recent completed import of that Dataset: `provider`, `externalId`,
    `config`, `split`, `revision`, `url`, `importedAt`, `licenseAckAt` and the
    stored `license` from the license table, with its current platform level
    (an admin may have changed it since the import); 404 when there is none. It
    reads the database only, so it works without an import queue.
  - `LibraryImport.to_dict()` carries `licenseAckAt`.

### Changed

- Importing a dataset whose license an admin set to `hide` is refused with
  403 `license_hidden`; the preview route still answers 404.
- Importing a dataset its owner keeps private on Hugging Face is refused with
  409 `private` when the Hub's answer says it is private; an anonymous
  401/404 is treated as `not_found`.
- A license only records a confirmation time when it is set to warn, and a
  retried import keeps the time the student first confirmed.
- The requested dataset's source is checked before the Hub is asked, and the
  owner and license the Hub returns are checked before the viewer is asked,
  so no answer reveals a dataset outside the library.
- A license an admin sets to warn that has no confirmation words of its own
  gets "I have read the <name> license and will follow its terms." wherever
  the license is sent with its level (`License.at_level`).

## [0.6.0] - 2026-11-18

### Added

- Imports you can leave, cancel and retry (BAIGLE #114).
  - `POST /api/library/imports/<ref>/cancel`, `/retry` and `/dismiss`, and
    `GET /api/library/imports?current=true` (running, plus failures from the
    last day not put away).
  - A heartbeat: `run_import(heartbeat_seconds=15)` refreshes
    `library_imports.heartbeat_at`; `imports.sweep(session)` fails imports
    quiet for two minutes and queued ones not picked up in 30.
  - Starting the same import (dataset, revision, mapping, size) while one is
    queued or running returns that one, with 200.
  - `POST /imports` takes `revision`: an import of a newer one is refused (`changed`).
  - Imports carry `textColumn` and `labelColumn`.
  - Migration `dl_0004_import_lifecycle` adds `heartbeat_at` and `dismissed_at`.

### Changed

- A new `saving` status between `validating` and `completed`, claimed before
  the Dataset is written. Every status change is a conditional update, so a
  cancel and the worker finishing never both win: a cancel before `saving`
  wins and no Dataset is made; one after is refused.
- `imports.start` returns `(import, created)`.

## [0.5.0] - 2026-11-17

### Added

- Admin-set limits (BAIGLE #118). `GET` / `PUT /api/library/admin/settings`
  carry `limits`: the row cap (`maxRows`, 1 to `hard_max_rows`, 5,000 by
  default) and the download cap (`maxDownloadMb`, 1 to 1,000). `null` returns to
  the default, and a `PUT` without `limits` keeps them. The audit names
  `limits_changed`.
  - The row cap bounds imports and `/options` `maxRows`; `dataset_library.limits.row_cap(session)`
    lets the host apply it to uploads too.
  - The download cap replaces the worker's `max_bytes` in `run_import`.
  - Migration `dl_0003_limits` adds `library_settings.max_rows` and `max_download_mb`.
- `init_dataset_library(hard_max_rows=)`.

## [0.4.0] - 2026-11-12

### Added

- Import a random sample of a Hugging Face dataset as the host's own Dataset
  (BAIGLE #113). `init_dataset_library` takes `enqueue(import_ref)` and
  `max_rows()`; `/options` reports `importable` and `maxRows`.
  - `POST /api/library/imports` (an optional `name`; without one, the title,
    numbered on a repeat import), `GET /api/library/imports[/<ref>]`.
  - `dataset_library.imports.run_import` for the host's worker: queued →
    fetching → sampling → validating → completed, or failed; the database is the
    source of truth and is checked before every step.
  - `dataset_library.providers.parquet.fetch_shards`: the split's converted
    Parquet shards, downloaded up to a byte cap (200 MB by default), 429/5xx
    retried three times.
  - Two-pass sampling: row counts first, then only the mapped columns of the
    row groups holding a chosen row. Empty text is dropped; class labels are
    written by name.
  - The Hub revision seen at preview is pinned; a dataset changed since then
    is not imported. An import whose row is erased (account deletion) stops
    without making a dataset.
  - Table `library_imports` (migration `dl_0002_imports`).
- Cards carry `revision` (the Hub's commit sha).
- Optional extra `dataset-library[import]` (pyarrow, below 26 while hosts keep NumPy 1).

## [0.3.1] - 2026-11-12

### Fixed

- The preview route's parameter is `dataset_ref`, not `external_id`: BAIGLE
  refuses any `*_id` URL parameter that is not a UUID, so every preview there
  answered 404.

## [0.3.0] - 2026-11-12

### Added

- Library settings for admins (BAIGLE #118). `init_dataset_library` takes
  `is_admin()` (nobody by default) and `audit(event, **fields)`.
  - `GET /api/library/admin/settings`: the sources and every license with its
    default and current level.
  - `PUT /api/library/admin/settings` with `{"sources": [{"ref", "enabled"}],
    "license_policy": {id: level}}` replaces both. New sources must be Hugging
    Face organisations with public datasets. Any invalid field gives 400 with
    `fields` naming it, and nothing is saved. Each save is audited as
    `library.settings_saved` with counts and the license ids whose level changed.
  - `POST /api/library/admin/sources/check` with `{"ref"}` checks a source before saving.
  - `/options` reports `admin`.

## [0.2.1] - 2026-11-12

### Added

- The dataset preview's `rowCount`: the chosen split's size from the viewer's
  `/info`, or `null` when it does not say.

## [0.2.0] - 2026-11-12

### Added

- `GET /api/library/labels?ids=huggingface:org/name,...` (at most 30): whether
  each dataset has labels, `yes`, `no` or `unclear`, from its column types in
  Hugging Face's dataset viewer. A class-label column, or a summary or judgment
  column, is `yes`; a label-like text column, or a dataset the viewer cannot
  read, is `unclear`. Cached for an hour per dataset (BAIGLE #112).
- `GET /api/library/datasets/huggingface/<org>/<name>?config=&split=`: one
  dataset's card, configs and splits, column types and first 8 rows (class
  labels by name). `preview` is `unavailable` when the viewer cannot read the
  dataset or it is gated; the facts are still returned. 404 for a dataset
  outside the enabled sources or under a hidden license. Cached 10 minutes.

## [0.1.0] - 2026-11-12

### Added

- `init_dataset_library(app, *, db, login_required, current_owner, task_types, cache)`
  mounts the API under `/api/library`.
- `GET /api/library/search?q&task&license&source`: Hugging Face datasets as
  cards (title, tasks, license, size, gated, description), limited to the
  enabled sources. Cached for 10 minutes; the last good results are served as
  `stale` while the Hub limits requests, and `unavailable` when it is down.
- `GET /api/library/options`: the task, license and source filters, and the
  license table with this platform's levels.
- `resolve_license(tag)`: one license vocabulary, with flags and the condition
  and confirmation a student reads before importing.
- Tables `library_settings` and `library_sources`, with an Alembic revision
  (branch `dataset_library`).
