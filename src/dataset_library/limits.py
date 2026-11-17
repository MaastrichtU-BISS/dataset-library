"""The limits an admin sets (#118): the most rows a dataset may have and the
most an import downloads. Unset, each takes the host's default."""
from .models import LibrarySettings

# An admin may not go above these: training runs on the course's worker.
HARD_MAX_ROWS = 5000
DEFAULT_MAX_DOWNLOAD_MB = 200
HARD_MAX_DOWNLOAD_MB = 1000


def _settings(session):
    return session.get(LibrarySettings, 1)


def row_cap(session):
    """The admin's row cap, or None when they set none.

    A host passes this to its own row check (in BAIGLE, legal-ml-core's
    LEGAL_ML_MAX_ROWS), so uploads meet the same cap as imports."""
    settings = _settings(session)
    return settings.max_rows if settings else None


def download_cap_bytes(session):
    """The admin's download cap in bytes, or None when they set none."""
    settings = _settings(session)
    if settings is None or settings.max_download_mb is None:
        return None
    return settings.max_download_mb * 1_000_000


def check(limits, hard_max_rows):
    """Field errors for a `limits` body: {maxRows, maxDownloadMb}, each a whole
    number in range or null (back to the default)."""
    fields = {}
    if not isinstance(limits, dict):
        return {"limits": "Limits must be an object."}
    for key, hard_max, unit in (("maxRows", hard_max_rows, "rows"), ("maxDownloadMb", HARD_MAX_DOWNLOAD_MB, "MB")):
        value = limits.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= hard_max:
            fields[f"limits.{key}"] = f"Enter a whole number from 1 to {hard_max:,} {unit}."
    return fields
