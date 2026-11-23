# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version is 0.x, minor releases may contain breaking changes.

## [Unreleased]

## [0.15.0] - 2026-11-23

### Fixed

- An import whose conversion is at another revision than the one it was queued at is now refused
  before any shard is downloaded (#174, BAIGLE #174). `fetch_shards` knew the conversion's source
  revision (`X-Revision`) and the Hub's head before its first download, but `_fetch` compared them
  with the queued revision only after the fetch returned, so a refused import downloaded up to its
  byte cap (200 MB by default) on every try and threw the shards away — the `conversion_behind`
  message asks the student to try again later, so this repeated.
  - `fetch_shards` takes the queued revision as the keyword `expected_revision` and raises the same
    refusal once the revision and head are known: `conversion_behind` when the Hub's head is still
    the queued revision, `changed` when the dataset moved on. The refusal is decided by one shared
    helper, `imports.refusal_for`, which `_fetch` still uses after a fetch as a safety net for a
    host's own fetch, so the codes and messages cannot drift.
  - Copies (#121) pass no `expected_revision` and behave exactly as before.
  - With the built-in Hugging Face fetch, a refused import now always fails with its refusal. Before, one whose download then hit a
    `ProviderError` could fall back to a copy, or fail as `provider_busy`, without being refused.
- **Seam change for hosts:** `fetch_shards` implementations and test fakes that a worker passes to
  `run_import` must now accept an extra keyword-only `expected_revision=None` (e.g. `def
  fetch_shards(external_id, config, split, max_bytes, dest_dir, **kwargs)`); it is None for a copy.
- `dataset_library.__version__` is 0.15.0.

## [0.14.0] - 2026-11-23

### Fixed

- An import records the revision its rows were read from (#168, BAIGLE #168). `fetch_shards` used to
  record the Hub's current head (`GET /api/datasets/<id>`), while the shards came from the converted
  Parquet on `refs/convert/parquet`, which can lag a new commit or be rebuilt mid-import; provenance
  and the citation (#120) could then name a revision whose data was never read.
  - The revision is now the listing's `X-Revision` header: the dataset commit the conversion was built
    from. The queued-revision check in `_fetch` therefore catches a conversion at another revision.
    When the Hub's head is still the queued revision, the conversion has not caught up, and the import
    fails as `conversion_behind` ("Hugging Face hasn't converted this dataset's latest version yet…
    Try again later."), not as `changed`, whose "open it again" would never help. `Fetched.head`
    carries that head: it is never recorded, asked only when the listing names a revision, and a Hub
    that cannot answer gives None rather than failing the fetch.
  - Every shard is downloaded from the commit `refs/convert/parquet` points at (`GET
    /api/datasets/<id>/refs`), not from the moving branch. The branch is read on both sides of a second
    listing, and every chosen shard must be in the pinned build at the listed size (one `POST
    /api/datasets/<id>/paths-info/<commit>` for exactly those files, so no paging), which catches a
    listing cached while the branch moved.
    A commit or source revision that moves, a size that differs, a missing or malformed convert commit,
    or a shard URL that does not name the convert branch raises `ProviderUnavailable`; the import fails
    as `provider_busy` (unless a copy stands in) and can be retried.
  - A listing that names no revision records none, and the import no longer falls back to the revision
    it was queued at, which its rows may not have come from. Community (#122) imports are unchanged.
  - Copies (#121) use the same fetch, so a copy now records the conversion's source revision rather
    than the Hub's head, fails as `no_revision` when the listing names none, and fails as
    `provider_busy` during a rebuild.
- `dataset_library.__version__` is 0.14.0.

## [0.13.0] - 2026-11-19

### Added

- The Community source: datasets BAIGLE users published openly to Zenodo (#122, decision D8).
  - Host seams, all five or none: `init_dataset_library(community_datasets=, community_record=,
    community_download=, community_base_url=, community_enqueue_preview=)`. `community_datasets()` returns the open dataset
    Publications as fair-publishing 0.2.0's `open_datasets` does, each with the Zenodo origin its record
    is on under `origin` (`open_datasets` does not return the record link, so the host adds it);
    `community_record(record_id)` returns `Zenodo.record_files`; `community_download(record_id, key,
    max_bytes)` returns one file's bytes; `community_base_url()` returns the Zenodo base URL the platform
    reads. Only a Publication on that origin is listed, previewed or imported. A host maps the
    repository's `RecordNotDownloadable` to `providers.community.NotDownloadable`, its `FileTooLarge` to
    `TooLarge`, and any other failure to `ProviderError`. The package imports no repository client (ADR
    0005). `community_enqueue_preview(job)` hands a preview job (a JSON string) to the host's worker. Without
    all five seams the source is not offered, and a partial set is logged as a warning.
  - Host limits and the cache: `max_fetch_bytes` is the host's download limit, and an import downloads at
    most the admin's cap and never more than this. `community_preview_max_bytes` (16 MB) bounds a preview.
    A host's `Cache` also gives `add` (set only when unset: the preview lock), `delete`, and `delete_if`
    (removes a key only while it holds the given value, so a lock is released only by its job).
  - Off until an admin switches it on: `GET` / `PUT /api/library/admin/settings` carries
    `community: {available, enabled}`, and a save with `"community": {"enabled": bool}` sets it. A
    save that does not name it audits as before; one that does adds `community_changed`. The source is
    the `library_sources` row `provider="community"`, `kind="community"`, `ref="zenodo"`. An import row
    gains the Publication it was started from, the Zenodo origin and the file it reads (migration
    `dl_0009_community_file`).
  - Search: the Community cards that match the query, license filter and license policy are listed
    with the Hugging Face results, and `source=community` lists only them. A Hub outage keeps them. A
    Publication has no task, so a task filter finds none. `GET /api/library/options` offers
    `{"value": "community", "label": "Community"}` when the source is on.
  - Cards: `id` `community:<record id>`, `provider: "community"`, `externalId` the record id, `url` and
    `doi` the DOI URL, the license resolved through the license table, and a citation in #120's shape:
    creators as `{family, given}` (Zenodo's "Family, Given") or `{literal}`, the year from the
    publication date, the DOI URL, and `bibtex: null`.
  - Preview: `GET /api/library/datasets/community/<record_id>` re-checks the record with Zenodo (a metadata
    call, kept for a minute) and answers from the preview cache. It returns `importable`, and
    `reasonCode`/`reason` when not: `restricted`, `metadata_only`, `not_downloadable` ("This record is no
    longer downloadable"), `no_table`, `too_large`, `unreadable`, `no_rows`, `changed`. When importable it
    returns the table's `columns`, first `rows`, `rowCount`, `proposed` columns and the file's `revision`.
    A file the worker has not read yet is answered `preparing: true`, and the page asks again; the request
    downloads and parses nothing, and queues one job per cache key. The worker's read is cached for ten
    minutes by origin, Publication, file, checksum and cap, a refusal included. A file over the preview
    budget (16 MB) is not read: `preview` is `too_large` ("Too large to preview here; it can still be
    imported") and the card, files and size stand. A hidden license is refused before anything is queued.
    A start or retry whose preview is no longer cached (it expired while the student chose columns) is not
    refused: the record's live checks still apply, and the worker checks the columns against the file it reads.
    A preview job that finds its answer already cached reads nothing.
  - Import: `POST /api/library/imports` with `provider: "community"` and the record id as `dataset`.
    The same license confirmation as Hugging Face applies: a warn license needs `licenseAck`, a hide
    license is refused with 403. A record that is not importable is refused with 409 and its reason.
    Retries keep the provider.
  - The worker: `run_import(community_datasets=, community_base_url=, community_record=,
    community_download=)` checks the import again before it reads. The Publication the import started on
    is still listed, on the origin it started on and on this one (`not_in_library`); the source is still
    on (`source_off`); the license is the Publication's current one, the one it started on (`license_changed`,
    asking for a fresh import), not now hidden (`license_hidden`), and one that warns has its confirmation
    (`needs_ack`). Then it asks the record and looks for the file the import started on first: a file
    missing from the record, or a record closed or gone, or one left with only a README, fails with "This
    record is no longer downloadable", and no other table is picked. The download's bytes must match the
    checksum the import started on, or it fails with the Zenodo "changed" message. A file over the cap fails
    with "Too large to sample here (limit N MB)"; Zenodo busy fails with "Zenodo is busy. Try again in a few
    minutes."; a column the file lacks fails as `bad_mapping`. The local Dataset is never read.
  - Formats: CSV, TSV, JSON (an array of objects, one object per line, or pandas' column layout), XLSX
    and Parquet, read by their bytes rather than their names, so a labelled file BAIGLE publishes as CSV
    under an `.xlsx` name reads as CSV, and a CSV whose header starts with `[` is still CSV. A file named
    JSON whose bytes do not open like JSON (BAIGLE published labelled CSV under `data.json` before #171)
    reads as CSV, with the same header rules; bytes that open like JSON and are broken are unreadable.
    Every cell is read as text, so a label `01` stays `01`, and a date is ISO text. Columns named "text" and
    "label" are proposed, with no task. Each reader checks its limits before it allocates: 200,000 rows,
    1,000 columns, 2,000,000 cells, 500 MB of text, JSON nested 8 deep and JSON over 64 MB, and a refusal
    says which. Rows, columns and cells are counted as each JSON record is read, and a Parquet file is
    refused from its metadata before a column is read. A sheet's rows are counted as they are scanned,
    blank ones included, up to Excel's 1,048,576, so a cell far down a sheet is refused, not walked. A JSON
    string that never closes is scanned in linear time. A name that cannot be written (a lone surrogate)
    is unreadable, and an empty header name is `Unnamed: N`, as pandas names it. A CSV whose lines end in a
    bare carriage return is read. An XLSX whose zip expands past 100 times its size is refused as a zip
    bomb. A file that is not UTF-8 is read as Windows-1252. pyarrow and openpyxl load only when a Community
    file is read, so the base install serves the routes without `[import]`.
  - Provenance: `provider: "community"`, `externalId` the record id, `doi`, `url` the DOI URL, `revision`
    the file's checksum, `source: "zenodo"`, and a citation with the DOI and the import's day but no
    revision: Zenodo gives each version its own DOI. A Community dataset's description says "from Zenodo".
  - `dataset_library.tables` (the table reader) and `dataset_library.community` (the host-aware source).
    `dataset_library.providers.community` holds the card, citation, table choice and refusal rules.
- `openpyxl>=3.1,<4` in the `[import]` extra, for reading XLSX.

### Changed

- `search()` finds the Hugging Face results and the Community cards separately. A Hub outage with no
  pins still lists the Community cards, and the Hub's status is unchanged. With Community on and no
  organisation, an empty query asks the Hub for nothing, so no whole-Hub list sits beside the Community
  cards.
- An admin's download cap never goes above the host's `max_fetch_bytes`, in the API and the worker, for
  Hugging Face imports too.
- A Community preview is read by the worker, never by a request (see Added): a request re-checks the record
  with one metadata call and queues one preview job per cache key. A Community import's citation has no revision.
- An organisation named `community` is refused as a library source, since the search filter names the
  Community source that way.
- `run_import` skips the copy release for Community imports, as they have no copy.

## [0.12.0] - 2026-11-19

### Added

- Snapshots of pinned datasets (BAIGLE #121, decision D10, design DR14).
  - A pin whose license level is `allow` gets a copy: the parts of the dataset an
    import would download, up to the same byte cap, joined into one Parquet file
    in the host's `snapshot_dir`. A worker makes it from the `enqueue_snapshot`
    queue, through `dataset_library.snapshots.run_copy`, off the request path. The
    file is written beside its final name and renamed into place, so a reader never
    sees a part-written copy; a write that fails, a full disk included, records the
    copy as `failed`. A copy is made for no other license level, and never for a
    gated dataset.
  - `library_snapshots` (migration `dl_0008_snapshots`), one row per pin:
    `revision`, `row_count`, `license`, `columns` (names and types), `file_path`
    and `status` (`queued`, `building`, `ready`, `failed` or `removing`), with
    `error_code` and `error_message`, `upstream_changed` and `upstream_checked_at`.
  - While Hugging Face cannot be asked, an import of a pinned dataset with a ready
    copy of the same config and split is queued and read from the copy. The import
    records `source` (`huggingface` or `snapshot`, new in `library_imports`, with
    `snapshot_id`), and provenance returns it with the copy's `revision`. A running
    import whose Hub fetch fails falls back to the copy the same way.
  - Search: when the Hub is unavailable (after its one retry), `status` is
    `unavailable` and the results are the pinned datasets, filtered as the search
    is (query, task, license, source; a hidden license is left out), each with a
    `snapshot` of `{"status"}`: `ready`, `queued`, `building`, `failed` or `none`.
    With no pins, the results stay empty, as before.
  - A dataset opened while Hugging Face cannot be asked (`GET /api/library/datasets/huggingface/<id>`)
    shows its copy instead: the card, the copy's columns and its first rows, with `offline: true`.
    A copy whose license is now hidden is not shown. The import takes the same facts.
  - `GET /api/library/pins` carries each pin's `snapshot` (`status`, `error`,
    `revision`, `rows`) and `upstreamChanged`. `POST /api/library/pins/<pin_ref>/snapshot`
    (admins) keeps a copy for a pin that has none or whose copy failed: 202, 409
    `not_allowed` unless the license is Allow, 404 `not_offered` on a host that
    keeps no copies.
  - Upstream changed: the worker's `snapshots.check_upstream` asks the Hub at most
    once an hour about each ready copy's revision (`huggingface.revision_exists`,
    `GET /api/datasets/<id>/revision/<sha>`). A 404 flags the pin and keeps the copy;
    any other answer leaves the flag as it was and is tried again later. The
    settings page reads the flag, so it makes no Hub call.
  - Removal: unpinning deletes the pin's copy and its file at once, unless a queued
    or running import of that dataset may still read it. Then the copy is marked
    `removing` (not served) and is deleted when those imports finish; the worker's
    `snapshots.sweep` deletes any such copy no import reads any more. Changing a
    license level to anything but `allow` removes the copies it no longer allows.
  - Host seams: `init_dataset_library(snapshot_dir=, enqueue_snapshot=)`, and
    `run_import(snapshot_dir=)`. Without both seams, a pin keeps no copy.
- `dataset_library.providers.huggingface.revision_exists(external_id, revision)`:
  True; False on a 404 (revision gone) or a 401 (dataset gone or no longer public);
  `ProviderError` for any other answer.

### Changed

- `search()` returns the pinned datasets, not an empty list, when the Hub is
  unavailable. The status stays `unavailable`, which the page shows as its banner.
- Copies are kept for a viewer outage too (#121 rework). When the dataset viewer
  (datasets-server) does not answer but the Hub does, the preview says so with
  `viewerDown: true` and is not cached. The dataset opens from its copy of the same
  part (`offline: true`), and `POST /imports` reads the copy, as it does when the Hub
  is down. With no copy, the preview is unchanged and the import is refused as
  `not_importable`. A configured part is no longer refused while the viewer is down.
  Only a real outage counts: no connection, a server error other than 501, or an
  unreadable answer (`viewer.ViewerDown`). A viewer that says no for good (501, it
  cannot read the dataset; 404; 401) gives the usual unavailable preview, and a pin
  of it records `no_rows`. A missing `/info` alone is not an outage.
- A try at a copy has its own id (a failed copy is tried again as a new row), and its
  files are named by that id: `<id>.parquet`, and the work directory `.incoming/<id>`.
  A try still running after its copy was swept and tried again cannot finish the
  newer copy or delete its files. A building copy refreshes `updated_at` every 15
  seconds, so the sweep fails only builds that have really stopped.
- Unpinning deletes a copy's file and work area by the copy's id, whatever path its row
  holds, so a file left by a crashed build goes too. The sweep fails a stuck build and
  removes its work area and file; it also removes any work area or copy file that no
  copy still owns. Only names it made (`<uuid>.parquet`, `.incoming/<uuid>`) are
  touched. A queued copy waits behind imports on the same worker, so the sweep fails
  it only after 6 hours (a building copy, after 30 minutes without a heartbeat).
- A copy is kept only while its pin exists and its license is still Allow, checked when
  it is ready. A copy whose pin is removed, or whose license is no longer Allow, while
  it is made is dropped with its file. An import reads a copy, or an offline dataset
  shows one, only while the pin exists and the license is Allow; a hidden license is
  not in the library (404, or `license_hidden` for an import).
- A request that loses a race to another request for the same pin returns the copy
  that other request queued.
- `check_upstream` records each answer on its own, so a copy unpinned meanwhile is a
  no-op. A copy whose check fails goes to the back of the order with its flag
  unchanged, so one dataset the Hub always refuses cannot stop the others being
  checked. The round ends after two failed checks in a row (the Hub is down), or at
  once on a rate limit.
- An offline dataset's columns are typed from the copy's Hugging Face features, as the
  live preview types them (a ClassLabel is a class label), and its proposed task is the
  one the live preview would propose from the same columns (NER and summarization keep
  theirs). `rowCount` is the copy's rows, not the split's.
- An offline search result's `title` is the pin's title, else the card's, else the last
  part of the dataset id, as `pins.to_dict` names a pin.
- A copy file that cannot be read (corrupt, or missing) opens without rows, never a 500.

## [0.11.0] - 2026-11-19

### Added

- Citing an imported dataset (BAIGLE #120, design DR13).
  - Every Hugging Face card carries `citation`: `title` (the card's
    `pretty_name`, else the dataset name), `authors` (a list, each
    `{"family", "given"}` for a person or `{"literal"}` for an organisation or
    anything that is not a person's name: the card's `authors`, else the BibTeX
    `author` field, else the owning organisation), `year` (the BibTeX `year`,
    else the card's `year`, else null: never the repository's creation date),
    `url`, `revision` (the Hub's `sha`), `accessedAt` (null on a card: looking
    at it is not reading it) and `bibtex`.
  - `bibtex` is the dataset's own BibTeX, kept exactly as written. It is read
    from the Hub's top-level `citation`, else the card's `citation_bibtex`, else
    `citation` (each only when the value starts with `@`), else the dataset's
    README at the card's revision (512 KB at most, with a short timeout; any
    failure is no BibTeX). In the README, an entry in a section headed
    "Citation" comes before a fenced entry elsewhere, which is often an upstream
    dataset's. Only a single dataset reads its README, so a search
    result's `bibtex` can be null where the dataset's own cannot. A
    plain-text citation, or none at all, gives null.
  - The BibTeX is read by a small scanner: only a top-level field of the first
    entry counts, so a `year` inside another field is not the year; braces,
    quotes and `#` joins are read; authors split on `and` outside braces, and
    LaTeX accents decode to letters.
  - A completed import stores that citation in the new `library_imports.citation`
    column (JSON, nullable; migration `dl_0007_citation`), set when the import
    is created and pinned again when it completes: `revision` is the revision
    the import read and `accessedAt` the time it was imported. A fetch that
    reports no revision keeps the one the import was queued at. `provenance()`
    returns it as `citation`; imports made before 0.11.0 have none, so it is
    `null`.
  - A featured pin keeps its card's citation, and a preview cached before
    citations existed is read again, so an import started right after the
    upgrade is cited too.

## [0.10.0] - 2026-11-18

### Changed

- NER imports store each row's entities as JSON spans in `entities`, replacing
  0.9.0's `"PER, ORG"` type list (#6, BAIGLE #117). A cell holds a JSON list
  of `{"text", "start", "end", "type"}`, with character offsets into the
  row's `text`, end exclusive, in text order and never overlapping; a row
  whose tags are all outside gets `[]`. This is the format legal-ml-core's
  annotation queue stores, so imported NER data can be trained on.
  - BIO, IOB1, BIOES/BILOU, BMES and IO tags decode to spans for word-level
    and character-level rows alike; `O`, `B-O` and placeholder tags (`X`,
    `PAD`, `[CLS]`, `[SEP]`) are outside; prefixes and `O` are read in either
    case. Whitespace never starts or ends a span; an `O` on a
    whitespace token closes a span, except inside a prefixed entity of a
    character-level row (klue's `ner` tags those spaces `O`).
  - Spans are written only for a token list with a tag for every token. A
    row whose text column is already a string, or whose tags are missing,
    scalar, shorter than its tokens or partly empty, gets a blank `entities`
    cell, so it waits to be annotated instead of counting as a row with no
    entities or a partial annotation. For a string text column, numbered
    tags without names therefore no longer fail the import. When no sampled
    row has a tag for every word (sub-word tags, for example), the import
    completes with a `sampleNote` saying the entities are left to annotate.
  - The conversion lives in `dataset_library.ner.spans`; `tokens_to_text` is
    unchanged.

## [0.9.0] - 2026-11-18

### Added

- NER and summarization dataset imports (BAIGLE #117).
  - A preview of a card tagged `ner` or `summarization` now carries `proposed:
    {taskType, textColumn, labelColumn}`, so the import's column step starts
    on the right columns. A task is proposed only when the selected config
    holds a distinct text and label pair for it. NER needs a `tokens` or
    `words` list of strings as the text (it never guesses from other string
    lists) and a flat tag list holding entity tags: `ner_tags` or `ner` by
    name; otherwise ClassLabel names, or the tags in the first rows, that are
    all `O` or BIO (`B-PER`) with at least one BIO tag, or numbered tags
    without names under `tags`, `labels` or `label` (whose import then fails
    with `no_entity_names`). Columns named with `chunk` or `pos` never count,
    so part-of-speech, chunking and parsing configs on a token-classification
    card get no proposal. Lists of lists are not proposed. BIO tags are taken
    as entity tags whatever their types say, so a BIO-tagged column under a
    generic name that marks something else (acronym forms, semantic roles) is
    still proposed; Step 1 shows the tags before the import. Summarization
    proposes the first of `document`, `article`, `text`, `dialogue`, `source`,
    `input`, `report`, `description`, `content`, `body` and of `summary`,
    `highlights`, `abstract` (never a generic `target` or `output`), with no
    guessing from other columns. All three are `null` when no pair fits
    the selected config (a `ner`-tagged classification or dependency-parsing
    config, for example) or the preview is unavailable.
  - NER imports join a row's tokens into `text` and write the row's distinct
    entity types into `entities` (BIO position stripped, `O` dropped, first
    seen first, missing tokens skipped; a row of single characters that keeps
    its spaces as tokens, as in klue/klue's `ner`, is joined as written).
    Numbered tags are read through the ClassLabel names in the Parquet's
    `huggingface` metadata, including a ClassLabel inside a Sequence and its
    older list form. A tag column that is a list of numbers with no names, or
    an empty name list, fails the import with `no_entity_names` before any row
    is sampled, with a per-row check as a backstop. NER columns that are
    records, maps or lists of lists, records or maps (span annotations, in
    either of Hugging Face's forms) fail with `nested_columns` before any row
    is read.
  - Summarization imports write the source document as `text` and its target
    as `summary`. A row whose summary is missing or blank is skipped and
    replaced like a row with empty text, and when no row has both a text and a
    summary the import fails with `no_usable_rows`.
  - `POST /api/library/imports` refuses a request whose `labelColumn` is the
    same as its `textColumn` with 400 `bad_mapping` ("Pick different columns
    for the text and the label."), for every task type.
  - A dataset whose summary column is `highlights` (cnn_dailymail, for
    example) now counts as labelled.
  - The import CSV's second header follows the task type: `entities` for `ner`
    and `named_entity_recognition`, `summary` for `summarization`, and `label`
    for classification-like tasks (unchanged). A dataset with no label column
    still writes `text` alone.
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
