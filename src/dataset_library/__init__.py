"""Search public legal datasets by license and task, and import a sample as the host's own Dataset."""
import logging
from dataclasses import dataclass
from typing import Callable, Protocol

from .licenses import License, resolve_license
from .limits import DEFAULT_MAX_DOWNLOAD_MB, HARD_MAX_ROWS

__version__ = "0.13.0"

__all__ = ["Cache", "License", "init_dataset_library", "resolve_license"]


class Cache(Protocol):
    """Where search results are kept for a while (Redis in BAIGLE).

    ``add(key, value, ttl)`` sets the key only if it is not set, for ``ttl`` seconds, and says whether it
    did; it is the lock that keeps one Community preview fetch running per key (#122). ``delete(key)``
    removes a key, so a lock is released as soon as its work ends."""

    def get(self, key: str): ...

    def set(self, key: str, value, ttl: int) -> None: ...

    def add(self, key: str, value, ttl: int) -> bool: ...

    def delete(self, key: str) -> None: ...


@dataclass
class _Host:
    db: object
    login_required: Callable
    current_owner: Callable[[], str]
    task_types: list
    cache: Cache
    is_admin: Callable[[], bool]
    audit: Callable[..., None]
    enqueue: Callable[[str], None] | None = None
    max_rows: Callable[[], int] = lambda: 1000
    hard_max_rows: int = 5000
    # Where the platform keeps copies of pinned datasets, and how a copy is queued for
    # the worker. Without both, no copy is made and a pin is imported only from Hugging Face.
    snapshot_dir: str | None = None
    enqueue_snapshot: Callable[[str], None] | None = None
    # The Community source's seams (community.Seams), or None: then it is not offered (#122).
    community: object | None = None
    # The most an import downloads, when no admin cap is set; the admin's cap never goes above it (#122).
    max_fetch_bytes: int = DEFAULT_MAX_DOWNLOAD_MB * 1_000_000

    def row_cap(self):
        """The cap a new dataset meets: the admin's, else the host's default."""
        from .limits import row_cap

        return row_cap(self.db.session) or self.max_rows()


def init_dataset_library(
    app,
    *,
    db,
    login_required,
    current_owner,
    task_types,
    cache,
    is_admin=None,
    audit=None,
    enqueue=None,
    max_rows=None,
    hard_max_rows=None,
    snapshot_dir=None,
    enqueue_snapshot=None,
    max_fetch_bytes=None,
    community_datasets=None,
    community_record=None,
    community_download=None,
    community_base_url=None,
    community_preview_seconds=None,
    community_preview_max_bytes=None,
    url_prefix="/api/library",
):
    """Mount the library API on a host Flask app.

    ``db`` is the host's Flask-SQLAlchemy handle; the tables come from
    ``dataset_library.models.LibraryBase.metadata`` or the package's
    migrations. ``login_required`` is the host's sign-in decorator, applied to
    every route; ``current_owner`` returns the signed-in user's id.
    ``task_types`` are the host's task type ids (legal-ml-core's TASK_TYPES
    values): the library offers no other. ``cache`` keeps search results
    (``get(key)``, ``set(key, value, ttl)``, ``add(key, value, ttl)`` and ``delete(key)``,
    with JSON-serialisable values). ``is_admin()`` says whether the signed-in user may change the library's
    settings (nobody, by default). ``audit(event, **fields)`` records each
    settings change; fields are counts and license ids, never personal data.
    ``enqueue(import_ref)`` hands a queued import to the host's worker, which
    calls ``dataset_library.imports.run_import``; without it there is no
    import. ``max_rows()`` is the most rows a dataset may have (legal-ml-core's
    LEGAL_ML_MAX_ROWS) when an admin set none, read once when an import is
    queued. An admin may set a cap from 1 to ``hard_max_rows`` (default
    ``limits.HARD_MAX_ROWS``); ``limits.row_cap(session)`` returns it, so the
    host's upload check can honour it too. ``max_fetch_bytes`` is the host's download
    limit (the worker's ``LIBRARY_MAX_FETCH_BYTES``): an import downloads at most the
    admin's cap, and never more than this (default ``limits.DEFAULT_MAX_DOWNLOAD_MB``).

    ``snapshot_dir`` is a directory the host keeps on disk (shared with the worker):
    copies of pinned datasets with an Allow license go there, so a pin still imports
    while Hugging Face is down. ``enqueue_snapshot(snapshot_ref)`` hands a copy to the
    worker, which calls ``dataset_library.snapshots.run_copy``. Both are needed; without
    them a pin has no copy. The worker also calls ``snapshots.sweep`` and
    ``snapshots.check_upstream`` from time to time.

    The Community source (datasets published openly to Zenodo) needs four seams, all or
    nothing, and the library imports none of the repository's client (ADR 0005):
    ``community_datasets()`` returns the open dataset Publications, as fair-publishing's
    ``open_datasets`` does, each with the Zenodo origin its record is on under ``origin``;
    ``community_record(record_id)`` returns the record's files, as ``Zenodo.record_files``
    does; ``community_download(record_id, key, max_bytes)`` returns one file's bytes, as
    ``Zenodo.download`` does; ``community_base_url()`` returns the Zenodo base URL the host
    reads from. Only a Publication on that origin is listed, previewed or imported. A host maps
    the repository's ``RecordNotDownloadable`` to ``providers.community.NotDownloadable``, its
    ``FileTooLarge`` to ``TooLarge``, and any other failure to ``ProviderError``. An admin
    still switches the source on. ``run_import`` takes the same seams.

    A preview fetches from Zenodo on a helper thread that the request waits for up to
    ``community_preview_seconds`` (default ``community.PREVIEW_SECONDS``), and reads a file of up to
    ``community_preview_max_bytes`` (default ``community.PREVIEW_MAX_BYTES``); a larger file is shown
    without its rows and stays importable within the import cap.
    """
    from .routes import library_bp

    community_seams = None
    given = (community_datasets, community_record, community_download, community_base_url)
    if all(seam is not None for seam in given):
        from .community import PREVIEW_MAX_BYTES, PREVIEW_SECONDS, Seams

        community_seams = Seams(
            datasets=community_datasets,
            record=community_record,
            download=community_download,
            base_url=community_base_url,
            preview_seconds=PREVIEW_SECONDS if community_preview_seconds is None else community_preview_seconds,
            preview_max_bytes=PREVIEW_MAX_BYTES if community_preview_max_bytes is None else community_preview_max_bytes,
        )
    elif any(seam is not None for seam in given):
        logging.getLogger("dataset_library").warning("library.community not offered: it needs all four seams")

    app.extensions["dataset_library"] = _Host(
        db=db,
        login_required=login_required,
        current_owner=current_owner,
        task_types=list(task_types),
        cache=cache,
        is_admin=is_admin or (lambda: False),
        audit=audit or (lambda event, **fields: None),
        enqueue=enqueue,
        max_rows=max_rows or (lambda: 1000),
        hard_max_rows=hard_max_rows or HARD_MAX_ROWS,
        snapshot_dir=snapshot_dir,
        enqueue_snapshot=enqueue_snapshot,
        community=community_seams,
        max_fetch_bytes=max_fetch_bytes or DEFAULT_MAX_DOWNLOAD_MB * 1_000_000,
    )
    app.register_blueprint(library_bp, url_prefix=url_prefix)
