"""Downloading one split's Parquet shards from Hugging Face, for an import (#113).

The viewer lists the shards it converted (``/parquet``); they are downloaded in
order, streamed, until the next would pass the byte cap. The listing answers
with ``X-Revision``, the commit of the dataset its conversion was built from
(#168), so the import records what its rows were read from, not the Hub's
current head. Every shard is then downloaded from the commit the Hub's
``refs/convert/parquet`` branch points at, so a conversion rebuilt between the
listing and the download cannot mix two builds' files, and one rebuilt while
the listing is read is not imported until it settles. Only huggingface.co and
datasets-server.huggingface.co are called. 429 and 5xx are retried three times
with backoff, then the provider is unavailable. An import queued at another
revision is refused, with the same refusal ``imports`` makes, before any shard
is downloaded (``expected_revision``, #174)."""
import os
import re
import time
from urllib.parse import unquote

import requests

from ..errors import ProviderUnavailable
from ..imports import Fetched, ImportFailed, refusal_for

LISTING = "https://datasets-server.huggingface.co/parquet"
HUB = "https://huggingface.co/api/datasets/"
# The branch the viewer publishes the converted Parquet on, and how a shard's
# URL names it: resolve/refs%2Fconvert%2Fparquet/...
CONVERT = "refs/convert/parquet"
ENCODED_CONVERT = "refs%2Fconvert%2Fparquet"
# The listing's header naming the dataset commit the conversion was built from (#168).
SOURCE_REVISION = "X-Revision"
TIMEOUT = 30
RETRIES = 3
# Worth asking again; 501 is the viewer saying it cannot read the dataset at all.
RETRY_STATUSES = {429, 500, 502, 503, 504}
CHUNK = 1 << 20
REBUILDING = "Hugging Face is rebuilding this dataset's converted files"
UNPINNABLE = "Hugging Face listed a file we cannot pin to one build of this dataset"
# A dataset revision as the Hub names one (providers/huggingface keeps the same).
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _get(url, sleep, **kwargs):
    return _ask(requests.get, url, sleep, **kwargs)


def _post(url, sleep, **kwargs):
    return _ask(requests.post, url, sleep, **kwargs)


def _ask(method, url, sleep, **kwargs):
    for attempt in range(RETRIES + 1):
        try:
            response = method(url, timeout=TIMEOUT, **kwargs)
        except requests.RequestException as exc:
            error = exc
        else:
            if response.status_code not in RETRY_STATUSES:
                return response
            error = f"answered {response.status_code}"
            response.close()
        if attempt < RETRIES:
            sleep(2**attempt)
    raise ProviderUnavailable(f"Hugging Face {error} after {RETRIES} retries")


def fetch_shards(external_id, config, split, max_bytes, dest_dir, *, expected_revision=None, sleep=time.sleep):
    listing = _get(LISTING, sleep, params={"dataset": external_id, "config": config})
    if listing.status_code != 200:
        raise ImportFailed(
            "not_importable",
            "Hugging Face cannot convert this dataset to files we can read, so it cannot be imported.",
        )
    revision = _source_revision(listing)
    shards = [s for s in listing.json().get("parquet_files") or [] if s.get("config") == config and s.get("split") == split]
    if not shards:
        raise ImportFailed("not_importable", "Hugging Face has no files for this part of the dataset.")

    chosen, total = [], 0
    for shard in shards:
        if total + shard["size"] > max_bytes:
            break
        chosen.append(shard)
        total += shard["size"]
    if not chosen:
        raise ImportFailed("too_large", "Too large to sample here: even one part of this dataset is over the download limit.")

    # The convert branch is read on both sides of a second listing: the same
    # commit both times, and the same source revision as the first listing,
    # means the listing describes the build the shards are pinned to. Anything
    # else is a rebuild under way, so the import waits rather than records a
    # revision its rows may not have come from.
    commit = _convert_commit(external_id, sleep)
    again = _get(LISTING, sleep, params={"dataset": external_id, "config": config})
    if again.status_code != 200 or _source_revision(again) != revision or _convert_commit(external_id, sleep) != commit:
        raise ProviderUnavailable(REBUILDING)
    # The listing can be cached while the branch has moved on: every chosen file
    # must be in the pinned build at the size the listing gives, or the listing
    # describes another build.
    _check_build(external_id, commit, chosen, sleep)
    head = _head(external_id, sleep) if revision else None
    # An import queued at another revision is refused before any shard is
    # downloaded (#174), with the refusal `imports` makes after a fetch.
    refusal = refusal_for(expected_revision, revision, head)
    if refusal is not None:
        raise refusal

    os.makedirs(dest_dir, exist_ok=True)
    paths = []
    for n, shard in enumerate(chosen):
        path = os.path.join(dest_dir, f"{n:05}.parquet")
        _download(_pinned(shard["url"], commit), path, shard["size"], sleep)
        paths.append(path)
    return Fetched(paths=paths, revision=revision, head=head)


def _source_revision(listing):
    """The dataset commit the listed conversion was built from, or None when the
    listing does not say (#168). Never the Hub's current head, which the rows
    may not have been read from."""
    revision = listing.headers.get(SOURCE_REVISION)
    return revision if isinstance(revision, str) and REVISION_PATTERN.match(revision) else None


def _convert_commit(external_id, sleep):
    """The commit the dataset's converted-Parquet branch points at, so every
    shard comes from one fixed build. The branch missing while the listing had
    its files means the conversion is being rebuilt: try again later."""
    refs = _get(f"{HUB}{external_id}/refs", sleep)
    if refs.status_code != 200:
        raise ProviderUnavailable(f"Hugging Face answered {refs.status_code} for the convert branch")
    try:
        commit = next(c["targetCommit"] for c in refs.json().get("converts") or [] if c.get("ref") == CONVERT)
    except (StopIteration, AttributeError, KeyError, TypeError, ValueError) as broken:
        raise ProviderUnavailable(REBUILDING) from broken
    if not (isinstance(commit, str) and REVISION_PATTERN.match(commit)):
        raise ProviderUnavailable(REBUILDING)
    return commit


def _check_build(external_id, commit, chosen, sleep):
    """Every chosen shard is in the build at `commit`, at the size the listing
    gave; otherwise the listing describes a different build (#168). One
    paths-info call answers for exactly these files, with no paging."""
    paths = [_in_build(shard["url"]) for shard in chosen]
    found = _post(f"{HUB}{external_id}/paths-info/{commit}", sleep, data={"paths": paths})
    if found.status_code != 200:
        raise ProviderUnavailable(REBUILDING)
    try:
        sizes = {entry["path"]: entry["size"] for entry in found.json() if entry.get("type") == "file"}
    except (AttributeError, KeyError, TypeError, ValueError) as broken:
        raise ProviderUnavailable(REBUILDING) from broken
    if any(sizes.get(path) != shard["size"] for path, shard in zip(paths, chosen)):
        raise ProviderUnavailable(REBUILDING)


def _head(external_id, sleep):
    """The dataset's current commit, only to tell a conversion that lags it from
    a dataset that changed after it was opened. Never recorded as read, and
    never a reason to fail: a Hub that cannot say gives None."""
    try:
        hub = _get(HUB + external_id, sleep)
    except ProviderUnavailable:
        return None
    if hub.status_code != 200:
        return None
    try:
        sha = hub.json().get("sha")
    except (AttributeError, ValueError):
        return None
    return sha if isinstance(sha, str) and REVISION_PATTERN.match(sha) else None


def _in_build(url):
    """A shard's path inside the converted build, from its listed URL."""
    marker = f"/resolve/{ENCODED_CONVERT}/"
    if marker not in url:
        raise ProviderUnavailable(UNPINNABLE)
    return unquote(url.split(marker, 1)[1])


def _pinned(url, commit):
    """The shard's URL at `commit`, not at the branch the conversion moves on.
    A URL that does not name the branch cannot be pinned, so it is refused."""
    marker = f"/resolve/{ENCODED_CONVERT}/"
    if marker not in url:
        raise ProviderUnavailable(UNPINNABLE)
    return url.replace(marker, f"/resolve/{commit}/", 1)


def _download(url, path, expected, sleep):
    response = _get(url, sleep, stream=True)
    if response.status_code != 200:
        raise ProviderUnavailable(f"Hugging Face answered {response.status_code} for a shard")
    written = 0
    with open(path, "wb") as f:
        for chunk in response.iter_content(CHUNK):
            written += len(chunk)
            # A shard larger than listed is not trusted past a small margin.
            if written > expected * 1.1 + CHUNK:
                raise ImportFailed("too_large", "Too large to sample here: a part of this dataset is bigger than Hugging Face said.")
            f.write(chunk)
