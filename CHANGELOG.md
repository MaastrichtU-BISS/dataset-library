# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version is 0.x, minor releases may contain breaking changes.

## [Unreleased]

### Added

- Legal Data Hunter search for every signed-in user (#15, BAIGLE #202):
  `POST /api/library/legal-data-hunter/search` searches with the caller's own key (admin or not)
  and asks for 50 hits with snippets, whatever `limit` the body gives. Refusals: 404
  `not_offered` without the key seam or while the "Legal Data Hunter terms" level is Hide, 409
  `needs_key` and `bad_key`, 429 `slow_down` and `out_of_credits` with `retryAfter` and a
  `Retry-After` header, 503 `provider_unavailable`, 400 `invalid` and `bad_request`.
  `legal_data_hunter.search(host, body)` and `legal_data_hunter.searchable(host)` back it. The
  admin preview is unchanged. A hit's `url` is kept only when it is an http(s) URL (else `null`),
  in the user search and the admin preview alike, so a `javascript:` link never reaches a page.

## [0.17.0] - 2026-11-24

### Fixed

- A dataset whose Parquet shards are each bigger than the import's byte cap can be imported (#178,
  BAIGLE #178). `fetch_shards` downloaded whole shards and refused with `too_large` when even the
  first was over the cap (200 MB by default), so FineFineWeb (`m-a-p/FineFineWeb`, ten shards of
  263–292 MB) could never be imported, although an import keeps at most 5000 rows.
  - `fetch_shards` now reads with HTTP range requests. Up to 16 shards, spread across the split, are
    read from. First every chosen shard's last 8 bytes are read, which are cheap and give its
    footer's length. Then the footers are read cheapest first, each only while it leaves room for
    the cheapest row group that fits of those read so far (before any is known, while it leaves any
    room), so a shard listed early with a dear footer cannot use the cap an affordable shard needs.
    Then only the row groups that fit under the cap are read, taken one shard at a time and spread
    across each shard, so a sample is not just the head of shard 0. Footers and row groups both
    count against the cap, and no byte is read twice. While fewer than two shards have rows, one
    shard does not take the room another shard's cheapest row group needs, so uneven row groups
    still give a sample from two.
  - `fetch_shards` takes an optional keyword `max_rows`, the rows an import keeps. With it, the
    reading stops once it has `parquet.row_target(max_rows)` rows, `max(3 * max_rows, max_rows +
    1000)`, headroom for the random and stratified samplers and for rows without text, instead of
    filling the byte cap. The target ends the reading only once two shards have rows, or once no
    other shard's row group fits, so a first row group past the target (one of blank rows, say) is
    not the whole sample. `imports` passes the import's sample size only to a `fetch_shards` that
    accepts the keyword, so a host's own fetch without it keeps working. Copies pass none and fill
    the byte cap as before.
  - What was read is written as one local Parquet file per shard, with the shard's schema (nested
    field names included, such as a list item named `custom`) and its `huggingface` metadata, so
    `Fetched.paths`, imports, label balance and copies (#121) read it as before. Each file is written
    under a temporary name and moved into place only when whole; a fetch that fails leaves no file.
  - Empty row groups are skipped. A split with no rows at all is refused as `not_importable`.
  - Files that parquet-mr 1.2.8 and older wrote are read with the 100 bytes pyarrow reads past each
    column chunk (PARQUET-816), and those bytes count against the cap. Which files those are follows
    Arrow's own parse of the footer's `created_by`, so a bare `parquet-mr`, or one whose version
    stops parsing early (`parquet-mr version 1.2`), is padded too. Padding that runs into the footer
    is served from the footer already read, neither read nor counted twice.
  - `too_large` is raised only when no row group of any shard whose footer was read fits under the
    cap; a shard whose footer or row groups do not fit is skipped and later shards are still read.
    Its message is now `parquet.TOO_LARGE` ("Too large to sample here: even one block of rows in this
    dataset is over the download limit."). Retrying cannot help, so a host should not offer Try
    again for `too_large`.
  - Ranges are asked for with `Accept-Encoding: identity`. The Hub answers a pinned `resolve/` URL
    with a 302 to its CDN. Each range request is sent straight to a `requests.Session`'s adapter,
    because the session's own `send` reads a redirect's body even with `allow_redirects=False` (to
    make `Response.next`). `fetch_shards` follows redirects itself: each is closed with its body
    unread, and the range asked again where it points with the same headers, at most five times and
    only over https (`parquet.BAD_REDIRECT` otherwise). A server that answers a range request
    with the whole file (200) is closed before its body is read (`parquet.NO_RANGES`). A 206 must
    name the range asked of a file of the size the build check confirmed (or `*`) in its
    `Content-Range`, and any `Content-Length` must be the range's; otherwise it is refused before
    its body is read. A body without a `Content-Length` is read one byte past the range, to tell one
    that sends more, only when the cap has that byte to spare; otherwise exactly the range is read,
    which the exact `Content-Range` names, so no byte past the cap is read. A body longer or shorter
    than the range is refused (`parquet.WRONG_RANGE`). A connection lost while a body is read raises
    `ProviderUnavailable` (`parquet.CUT_OFF`). All of these are `ProviderUnavailable`, so an import
    still falls back to a copy (#121).
  - A shard that is not a Parquet file we can read (a bad trailer, a footer of length 0, row groups
    outside the file's data, pages that cannot be decoded) raises `errors.NotParquet`, a
    `ProviderUnavailable` with the message `parquet.NOT_PARQUET`. An import refuses it as
    `not_importable` with that message rather than calling Hugging Face busy, as asking again
    cannot help, and reads no copy for it. A copy of such a shard fails as `no_rows` rather than
    `provider_busy`. Only what pyarrow raises decoding the bytes is `NotParquet`: an error reading
    or writing the local files (a full disk, a failing one) is raised as the `OSError` it is, so a
    copy still reports `storage_full` or `write_failed`, and a read of bytes that were never fetched
    raises `errors.RangeNotFetched`, which is neither, so it ends an import or a copy as
    `unexpected`, logged. The old "bigger than Hugging
    Face said" `too_large` is gone with whole-shard downloads.
  - #168 and #174 are kept: every range is read from the shard URL pinned to the convert commit, the
    build check covers every shard whose footer is read, and a refusal for the revision comes before
    any byte of a shard, footers included.
- `fetch_shards` gains only the optional keyword `max_rows`, and `Fetched` is unchanged; hosts need
  no code change.
- `dataset_library.__version__` is 0.17.0.
- Once two shards' footers have row groups that fit, a further footer is read only if it still leaves room
  for both shards' cheapest groups, so a late, dear footer never costs the sample its second shard.
- A listed shard URL that is not https is refused before it is asked for, so every range is read over https
  from the first hop. A range read to its stated length hands its connection back to the pool.

## [0.16.0] - 2026-11-24

### Added

- Legal Data Hunter entries (BAIGLE #126, decisions L1–L15). See the README's "Legal Data Hunter
  entries" for the routes and shapes.
  - Host seam: `init_dataset_library(legal_data_hunter_key=)` and `run_import(legal_data_hunter_key=)`,
    `legal_data_hunter_key(owner_id) -> str | None`. The package never stores, logs or returns a key.
    Without the seam, Legal Data Hunter is not offered.
  - Admin routes under `/api/library/admin/legal-data-hunter/`: `preview` (search with the admin's own
    key: query, corpus, country, source, date range), and `entries` (create, list, get, update,
    `refresh`, delete). An entry keeps its query and at most 100 frozen document ids; Refresh reruns the
    query and replaces them, and imports keep the ids they copied.
  - Entries are featured cards (`GET /featured` → `legalDataHunter`, `GET /legal-data-hunter/entries`,
    `GET /datasets/legal_data_hunter/<ref>`) labelled "Legal Data Hunter", with `documentCount` and
    `creditCost` ("uses 100 of your Legal Data Hunter credits"). Without a key, Import is disabled with
    "Needs your Legal Data Hunter key — add it in Settings".
  - Importing (`POST /imports` with `provider: "legal_data_hunter"`) always needs `licenseAck: true`; its
    time is the provenance's `licenseAckAt`. The worker fetches each document through
    `GET /v1/documents/{country}/{source}?source_id=` with the student's key (waiting out a 429 for the
    per-minute cap rather than pausing as out of credits), writes one unlabelled row
    per document (`text, court, date, country, language, ecli, eli, source_url, text_truncated`), and counts
    documents with no text (`skipped`) and documents gone or refused as malformed (404, 400, 422:
    `missing`). Its documents' responses together may not pass the worker's download cap: past it
    the import fails as `too_large`, keeping what it fetched. A country keeps Legal Data Hunter's
    case (`CoE`, `EU`, `INTL`); the search filter upper-cases two-letter codes only.
  - Out of credits (429): the import fails as `out_of_credits` with "Fetched N of M — your Legal Data
    Hunter credits ran out. Try again when they reset." and `resumeAfter`; the documents it fetched are
    kept, and Try again fetches only the rest, from the import's own documents even after the entry is
    deleted or the row cap lowered. The pause stays listed and resumable until a week after it paused or
    a day after the credits reset, whichever is later. Try again is refused with 409 `already_running`
    while another import of the entry runs, and with 409 `not_retryable` once the import was put away.
  - Fetched documents are kept only for a failure Try again resumes from (`out_of_credits`,
    `provider_busy`, `bad_key`, `needs_key`, `too_large`, the sweep's `stopped` and `not_started`, and
    `unexpected`) of an import that never reached saving (`saved_path` and `dataset_ref` null), and every
    such failure stays on the student's list while they are kept; a cancel, another failure, completion, Put away, or an
    admin setting the terms to Hide deletes them. A document answered after a cancel, Put away or
    erasure is never written.
  - The license table has "Legal Data Hunter terms" (`legal-data-hunter-terms`, new group "Provider
    terms"): Warn or Hide, never Allow. Every license dict now carries `levels`, the levels an admin may
    choose.
  - Provenance carries `publishable` and `publishRefusal` for every import, and `legalDataHunter` for an
    entry's. `legal_data_hunter.publish_refusal(session, dataset_ref, model=, file_path=)` gives the host the
    reason to refuse publishing a dataset built from Legal Data Hunter, or a model trained on one, to Zenodo.
    Hosts should pass the Dataset's `file_path`: the import saves it (`saved_path`) with the move to saving,
    before `create_dataset` is called, and records the Dataset's ref the moment it returns; the refusal
    matches any import by either, whatever its status. `publishable` covers a provider's terms only, not
    licenses.
  - The key is sent through `requests`' `auth=` on every hop, so a `.netrc` entry cannot replace it; a key
    seam that raises is read as no key (in the worker: the import fails as `needs_key`, resumable), and only
    the exception's type is logged. A `Retry-After` is ASCII digits or a date, at most 31 days; leading
    zeros are dropped, then more than ten digits is read as 31 days unparsed. A search's `corpus` is case law only when missing or null.
  - Each document's response (200, or a 404, 400 or 422) is read streamed and stops within one chunk of what
    is left of the download cap; its measured bytes are kept (`nbytes`), and a resumed import counts from
    them, failing as `too_large` before fetching anything when they already pass a lowered cap. A 429's or
    a 5xx's body is read at most 8 KB, never counted and never `too_large`. A `too_large`
    import records the cap it stopped under (`failed_cap`), and Try again is refused with 409
    `cap_unchanged` until the cap is higher.
  - Try again is refused with 409 `already_saved` ("This import was saving its dataset when it stopped.
    Look for it in My datasets before importing again.") once the import reached saving (`saved_path` or
    `dataset_ref` set), so it never makes a second Dataset. The retry's claim requires it in the same
    statement, so a save committed between the first check and the claim still wins.
  - At most one import of an entry per owner is queued or running (`active_key`, unique; null once it
    ends): a start that races another gets 409 `already_running`. Try again claims the paused import in
    the transaction that creates the new one (of two retries one starts, the other gets 409
    `not_retryable`), and a queue refusal restores the paused import and its documents in one commit.
  - A document is written by one conditional statement, only while its import runs and is not put away.
    A running import stops as `license_hidden`, discarding what it fetched, once the terms are Hide: it
    checks before every request (a repeat after a per-minute wait too), and its move to saving requires the
    terms not Hide in the same statement, so a Hide committed first means no Dataset. The sweep deletes
    every non-running import's documents while they are. Once `create_dataset`
    returns, nothing removes the CSV (any import).
  - Only `https://legaldatahunter.com` is called; a redirect to another host, or one with no
    `Location`, is refused. Each hop is prepared by a `requests.Session` and sent through its transport
    adapter, never `Session.send`, so a redirect's body is never read: only its `Location`.
  - The CSV's path is committed as `staging_path` before the file is written. A CSV never handed to the
    host (`saved_path` null) is deleted by the sweep (for the import it fails), `discard` and the purge
    (for imports whose documents go), and `erase`; one with `saved_path` set never is. Tests use
    recorded responses (`tests/fake_legal_data_hunter.py`), never the live API.
- Migration `dl_0010_legal_data_hunter`: tables `library_ldh_entries` and `library_ldh_documents` (with
  `nbytes`, nullable integer), and `library_imports.documents`, `skipped`, `missing`, `resume_after`,
  `saved_path` and `staging_path` (string 1000), `failed_cap` (big integer) and `active_key` (string 300, unique index
  `ix_library_imports_active_key`), all nullable.
- `legal_data_hunter.erase(session, owner_id)` for a host erasing a user, before it deletes the user's
  imports (so the CSVs they never handed over are found and deleted): it also deletes documents whose
  import is gone, and sets the user's entries' `created_by` to null. `imports.sweep` also deletes fetched documents no import will resume from.

### Changed

- `GET /featured` answers `{"featured": [...], "legalDataHunter": [...]}`; `GET /options` and `GET
  /admin/settings` carry `legalDataHunter`.
- `GET /options` leaves provider terms (`legal-data-hunter-terms`) and the "Provider terms" group out of
  the student's license filter; `GET /admin/settings` keeps them. `GET /search?license=legal-data-hunter-terms`
  asks Hugging Face nothing and answers no results, and `resolve_license` never gives a provider's terms.
- An admin's entry made from chosen documents stores their number as `query.limit`, so Refresh keeps its
  size; a Refresh that finds nothing is refused (400 `invalid`) and leaves the entry as it was.
- `run_import` claims an import with one conditional update from `queued`, so of two workers given the
  same ref only one runs it (every provider). A retry puts the old import away only when it made a new one.
- `dataset_library.__version__` is 0.16.0.

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
