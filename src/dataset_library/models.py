"""The library's own tables. Hosts create them from LibraryBase.metadata or
from the package's migrations."""
import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, BigInteger, Boolean, DateTime, Integer, String, UniqueConstraint, false
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _now():
    # Naive UTC, as SQLite and most hosts store it.
    return datetime.now(timezone.utc).replace(tzinfo=None)


class LibraryBase(DeclarativeBase):
    pass


class LibrarySettings(LibraryBase):
    """One row: what an admin set. Missing values take the package defaults."""

    __tablename__ = "library_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # {license id: "allow" | "warn" | "hide"}; a license not named keeps its default level.
    license_policy: Mapped[dict] = mapped_column(JSON, default=dict)
    # The most rows a dataset may have, and the most an import downloads (MB);
    # None takes the host's default.
    max_rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_download_mb: Mapped[int | None] = mapped_column(Integer, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)


class LibrarySource(LibraryBase):
    """Where search looks: a Hugging Face organisation, for now."""

    __tablename__ = "library_sources"
    __table_args__ = (UniqueConstraint("provider", "ref", name="uq_library_sources_ref"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    provider: Mapped[str] = mapped_column(String(40))
    kind: Mapped[str] = mapped_column(String(20))  # "org"
    ref: Mapped[str] = mapped_column(String(200))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    def to_dict(self):
        return {"id": self.id, "provider": self.provider, "kind": self.kind, "ref": self.ref, "enabled": self.enabled}


class LibraryPin(LibraryBase):
    """A dataset a course admin pinned as a starting point, with the card as it
    was pinned and whether it has labels (#119).

    Pins are course data, not personal data. `pinned_by` only says who pinned
    it; when a host erases a user's data it should keep this column or set it
    to null, not delete the pin with its owner.
    """

    __tablename__ = "library_pins"
    __table_args__ = (UniqueConstraint("provider", "external_id", name="uq_library_pins_dataset"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    provider: Mapped[str] = mapped_column(String(40))
    external_id: Mapped[str] = mapped_column(String(200))
    # The admin's display title, or null to use the card's.
    title: Mapped[str | None] = mapped_column(String(120), nullable=True)
    featured: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    labelled: Mapped[str] = mapped_column(String(10), default="unclear", server_default="unclear")
    # The search card as it was pinned; the license level is re-derived on read.
    card: Mapped[dict] = mapped_column(JSON)
    pinned_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    pinned_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class LibrarySnapshot(LibraryBase):
    """The platform's copy of a pinned dataset's sample-sized part, kept so the pin
    can still be imported while Hugging Face is down (#121). A copy is made only
    while the dataset's license is Allow, and has one row per pin. Once its pin is
    removed a copy still read by a running import stays as `removing` (pin_id
    null, not served) until those imports finish; there is no foreign key so the
    row can outlive its pin."""

    __tablename__ = "library_snapshots"
    __table_args__ = (UniqueConstraint("pin_id", name="uq_library_snapshots_pin"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    pin_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    provider: Mapped[str] = mapped_column(String(40))
    external_id: Mapped[str] = mapped_column(String(200))
    # The part of the dataset the copy holds: the config and split the preview chose.
    config: Mapped[str | None] = mapped_column(String(200), nullable=True)
    split: Mapped[str | None] = mapped_column(String(200), nullable=True)
    # Set when the copy is ready: the Hub revision it was read at, and what it holds.
    revision: Mapped[str | None] = mapped_column(String(80), nullable=True)
    license: Mapped[str | None] = mapped_column(String(80), nullable=True)
    columns: Mapped[list | None] = mapped_column(JSON, nullable=True)
    # The file's name inside the host's snapshot directory.
    file_path: Mapped[str | None] = mapped_column(String(200), nullable=True)
    row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="queued")
    error_code: Mapped[str | None] = mapped_column(String(40), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(500), nullable=True)
    # Set when the Hub no longer has the copy's revision; the copy is kept regardless.
    upstream_changed: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    upstream_checked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)


class LibraryImport(LibraryBase):
    """One import of a sample, from queued to completed, failed or cancelled. The
    database, not the queue, says where it is; a completed import is the
    dataset's provenance."""

    __tablename__ = "library_imports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id: Mapped[str] = mapped_column(String(255), index=True)
    provider: Mapped[str] = mapped_column(String(40))
    external_id: Mapped[str] = mapped_column(String(200))
    config: Mapped[str | None] = mapped_column(String(200), nullable=True)
    split: Mapped[str | None] = mapped_column(String(200), nullable=True)
    revision: Mapped[str | None] = mapped_column(String(80), nullable=True)
    url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    license: Mapped[str] = mapped_column(String(80))
    license_ack_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    text_column: Mapped[str] = mapped_column(String(200))
    label_column: Mapped[str | None] = mapped_column(String(200), nullable=True)
    task_type: Mapped[str] = mapped_column(String(40))
    sample_size: Mapped[int] = mapped_column(Integer)
    # Keep the label balance of the whole dataset (#115); needs a label column.
    balance: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    # "stratified" or "random", and why a asked-for balance was not kept.
    sample_method: Mapped[str | None] = mapped_column(String(20), nullable=True)
    sample_note: Mapped[str | None] = mapped_column(String(300), nullable=True)
    # The row cap when the import was queued; a later change does not touch it.
    max_rows: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(20), default="queued")
    error_code: Mapped[str | None] = mapped_column(String(40), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(500), nullable=True)
    dataset_ref: Mapped[str | None] = mapped_column(String(64), nullable=True)
    rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # What a Dataset cites (#120): the card's fields, pinned to the revision read
    # and when it was read. Null for imports made before citations existed.
    citation: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Where the sample was read: Hugging Face, or the platform's copy when Hugging
    # Face was down; then `snapshot_id` names the copy and `revision` is its (#121).
    source: Mapped[str] = mapped_column(String(20), default="huggingface", server_default="huggingface")
    snapshot_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    # A Community import (#122): the Publication it was started from, the Zenodo origin its record
    # is on, and the one file (by key) it reads. `revision` holds that file's checksum.
    publication_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    origin: Mapped[str | None] = mapped_column(String(200), nullable=True)
    file_key: Mapped[str | None] = mapped_column(String(500), nullable=True)
    # A Legal Data Hunter import (#126): `external_id` is the entry, `documents` the ids it fetches, frozen when it
    # was started (a Refresh of the entry never changes them), and what it skipped: documents with no text, and
    # documents Legal Data Hunter no longer has. `resume_after` is when its credits reset, as Retry-After said.
    documents: Mapped[list | None] = mapped_column(JSON, nullable=True)
    skipped: Mapped[int | None] = mapped_column(Integer, nullable=True)
    missing: Mapped[int | None] = mapped_column(Integer, nullable=True)
    resume_after: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # The CSV it hands `create_dataset`, saved with the move to saving, before the Dataset exists: the publish block
    # matches it while `dataset_ref` is not yet recorded. `failed_cap` is the download cap it stopped under as
    # `too_large`; Try again is refused until the cap is higher. `active_key` is `owner:entry` while it is queued or
    # running, null otherwise: a unique index allows one active import of an entry per owner. `staging_path` is the
    # CSV, committed before it is written: while `saved_path` is null the file was never handed to the host, so the
    # package deletes it when the import is swept, put away, purged or erased. Once claimed for that it is null again.
    saved_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    staging_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    failed_cap: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    active_key: Mapped[str | None] = mapped_column(String(300), nullable=True, unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)
    imported_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # When the worker running it last said it was alive; a stale one is swept (#114).
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # When the student put away a failed or cancelled import (or tried it again).
    dismissed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "status": self.status,
            "error": self.error_message,
            "code": self.error_code,
            "provider": self.provider,
            "externalId": self.external_id,
            "config": self.config,
            "split": self.split,
            "revision": self.revision,
            "url": self.url,
            "license": self.license,
            "licenseAckAt": self.license_ack_at.isoformat() + "Z" if self.license_ack_at else None,
            "name": self.name,
            "taskType": self.task_type,
            "textColumn": self.text_column,
            "labelColumn": self.label_column,
            "size": self.sample_size,
            "balance": self.balance,
            "sampleMethod": self.sample_method,
            "sampleNote": self.sample_note,
            "rows": self.rows,
            "source": self.source,
            "datasetRef": self.dataset_ref,
            # Legal Data Hunter (#126): how many documents it fetches, and how many it skipped.
            "documentCount": len(self.documents) if self.documents is not None else None,
            "skipped": self.skipped,
            "missing": self.missing,
            "resumeAfter": self.resume_after.isoformat() + "Z" if self.resume_after else None,
            "createdAt": self.created_at.isoformat() + "Z" if self.created_at else None,
            "importedAt": self.imported_at.isoformat() + "Z" if self.imported_at else None,
        }


class LibraryLegalDataHunterEntry(LibraryBase):
    """A course admin's Legal Data Hunter entry (#126): a search, and the documents it found, frozen.

    `query` is the search as the admin made it ({q, corpus, country, source, dateFrom, dateTo}); `documents`
    the ids it found, at most 100, each {country, source, sourceId, title}. A Refresh runs the query again and
    replaces them; an import copies them when it starts, so a Refresh never changes it. Entries are course data:
    `created_by` only says who made it, and a host erasing a user keeps it or sets it to null."""

    __tablename__ = "library_ldh_entries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title: Mapped[str] = mapped_column(String(120))
    description: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    task_type: Mapped[str] = mapped_column(String(40), default="classification", server_default="classification")
    query: Mapped[dict] = mapped_column(JSON)
    documents: Mapped[list] = mapped_column(JSON)
    created_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)
    refreshed_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class LibraryLegalDataHunterDocument(LibraryBase):
    """One document a Legal Data Hunter import has fetched, or skipped, so Try again resumes without fetching it
    again (#126, decision L12). `position` is its place in the import's frozen list; `row` is the dataset row (null
    when skipped). Rows are deleted once the import completes or is put away, and by the sweep after a week; a host
    erasing a user deletes the rows of that user's imports (`legal_data_hunter.erase`)."""

    __tablename__ = "library_ldh_documents"
    __table_args__ = (UniqueConstraint("import_id", "position", name="uq_library_ldh_documents_position"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    import_id: Mapped[str] = mapped_column(String(36), index=True)
    position: Mapped[int] = mapped_column(Integer)
    # "fetched", "no_text" or "missing".
    outcome: Mapped[str] = mapped_column(String(20))
    row: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # The response's body size in bytes, whatever the outcome: a retry counts what was already paid for.
    nbytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
