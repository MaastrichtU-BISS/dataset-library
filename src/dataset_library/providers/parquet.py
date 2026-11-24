"""Reading a sample of one split's Parquet shards from Hugging Face, for an import (#113).

The viewer lists the shards it converted (``/parquet``). Whole shards are not
downloaded (#178): a shard can be bigger than the byte cap while an import keeps
a few thousand rows. Up to ``MAX_SHARDS`` shards, spread across the split, have
their footers read with HTTP range requests (the last 8 bytes, then the footer);
then row groups are taken one shard at a time, each spread across its shard,
while the cap allows, and only their byte ranges are read. Footers and row
groups both count against the cap. Each shard's row groups are written as a
local Parquet file, so what an import or a copy reads is unchanged. A server
that answers a range with the whole file is closed before its body is read.
The only size refusal left, ``too_large``, is not even one row group fitting.

The listing answers with ``X-Revision``, the commit of the dataset its
conversion was built from (#168), so the import records what its rows were read
from, not the Hub's current head. Every range is read from the commit the Hub's
``refs/convert/parquet`` branch points at, so a conversion rebuilt between the
listing and the read cannot mix two builds' files, and one rebuilt while the
listing is read is not imported until it settles. Only huggingface.co and
datasets-server.huggingface.co are called. 429 and 5xx are retried three times
with backoff, then the provider is unavailable. An import queued at another
revision is refused, with the same refusal ``imports`` makes, before any byte
of a shard is read, footers included (``expected_revision``, #174)."""
import io
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
# Reading only what a sample needs (#178): a shard's footer is found from its
# last 8 bytes, a 4-byte length and the magic.
MAGIC = b"PAR1"
TRAILER = 8
# The most shards whose footers are read, spread across the split.
MAX_SHARDS = 16
# The one refusal for size left: not even one row group, with its shard's
# footer, fits under the download limit. Retrying cannot help; an admin can
# raise the limit.
TOO_LARGE = "Too large to sample here: even one block of rows in this dataset is over the download limit."
NO_RANGES = "Hugging Face sent a whole file when asked for part of it, so it was not downloaded"
WRONG_RANGE = "Hugging Face sent a different part of a file than the one asked for"
NOT_PARQUET = "Hugging Face listed a file that is not a Parquet file we can read"


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
    # A shard's size only says where its footer is: what is read is decided by
    # the footers, so a shard over the cap no longer stops an import (#178).
    chosen = _spread_shards(shards)

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
    # describes another build. The footers are found from these sizes.
    _check_build(external_id, commit, chosen, sleep)
    head = _head(external_id, sleep) if revision else None
    # An import queued at another revision is refused before any byte of a
    # shard is read, footers included (#174), with the refusal `imports` makes
    # after a fetch.
    refusal = refusal_for(expected_revision, revision, head)
    if refusal is not None:
        raise refusal

    budget = _Budget(max_bytes)
    urls = [_pinned(shard["url"], commit) for shard in chosen]
    footers = _read_footers(urls, [shard["size"] for shard in chosen], budget, sleep)
    plan = _plan(footers, budget)

    os.makedirs(dest_dir, exist_ok=True)
    paths = []
    for n, (i, groups) in enumerate(plan):
        path = os.path.join(dest_dir, f"{n:05}.parquet")
        _write_row_groups(urls[i], footers[i], groups, path, sleep)
        paths.append(path)
    return Fetched(paths=paths, revision=revision, head=head)


class _Budget:
    """The bytes an import may still read: footers and row groups both count."""

    def __init__(self, cap):
        self.cap, self.spent = cap, 0

    def fits(self, n):
        return self.spent + n <= self.cap

    def spend(self, n):
        self.spent += n


class _Footer:
    """A shard's parsed footer: its size, its metadata, and each row group's
    rows and byte range in the shard."""

    def __init__(self, size, metadata, data_end):
        self.size, self.metadata = size, metadata
        self.groups = [_group_range(metadata.row_group(i), data_end) for i in range(metadata.num_row_groups)]


def _spread_shards(shards):
    """At most MAX_SHARDS shards, spread evenly from first to last, so a sample
    is not only the head of the split."""
    k = min(len(shards), MAX_SHARDS)
    return [shards[i * len(shards) // k] for i in range(k)]


def _spread(n):
    """0..n-1 in an order that halves the gaps: 0, n/2, n/4, 3n/4, ... so the
    first row groups taken from a shard come from all over it."""
    order, seen, parts = [], set(), 1
    while len(order) < n:
        for j in range(parts):
            i = j * n // parts
            if i not in seen:
                seen.add(i)
                order.append(i)
        parts *= 2
    return order


def _read_footers(urls, sizes, budget, sleep):
    """Each shard's footer, read with two ranges: the last 8 bytes, which give
    its length, then the footer. The first may use the whole budget. Later ones
    stop at half of it, and never take the room the smallest row group read so
    far needs, so a later footer cannot make an import too large. The footers
    read, in order."""
    import pyarrow.parquet as pq

    footers = []
    for url, size in zip(urls, sizes):
        limit = budget.cap
        if footers:
            smallest = min((end - start for f in footers for rows, start, end in f.groups if rows), default=0)
            limit = min(budget.cap // 2, budget.cap - smallest)
        if size < len(MAGIC) + TRAILER:
            raise ProviderUnavailable(NOT_PARQUET)
        if budget.spent + TRAILER > limit:
            break
        trailer = _read_range(url, size - TRAILER, size, budget, sleep)
        length = int.from_bytes(trailer[:4], "little")
        if trailer[4:] != MAGIC or length > size - TRAILER - len(MAGIC):
            raise ProviderUnavailable(NOT_PARQUET)
        if budget.spent + length > limit:
            break
        data_end = size - TRAILER - length
        footer = _read_range(url, data_end, size - TRAILER, budget, sleep)
        try:
            metadata = pq.read_metadata(io.BytesIO(footer + trailer))
        except Exception as broken:  # noqa: BLE001 - whatever pyarrow raised, the footer cannot be read
            raise ProviderUnavailable(NOT_PARQUET) from broken
        footers.append(_Footer(size, metadata, data_end))
    if not footers:
        raise ImportFailed("too_large", TOO_LARGE)
    return footers


def _group_range(group, data_end):
    """The bytes [start, end) of one row group: from its first column chunk's
    first page to its last one's end. Outside the shard's data, the footer
    is not one we trust."""
    start, end = None, None
    for c in range(group.num_columns):
        column = group.column(c)
        first = column.data_page_offset
        if column.has_dictionary_page and column.dictionary_page_offset:
            first = min(first, column.dictionary_page_offset)
        last = first + column.total_compressed_size
        start = first if start is None else min(start, first)
        end = last if end is None else max(end, last)
    if start is None or start < len(MAGIC) or end > data_end:
        raise ProviderUnavailable(NOT_PARQUET)
    return group.num_rows, start, end


def _plan(footers, budget):
    """The row groups to read, by shard: one from each shard in turn, spread
    across it, while the budget allows. Returns [(footer index, [row group
    indexes])] for shards with any taken."""
    queues = [[g for g in _spread(len(f.groups)) if f.groups[g][0] > 0] for f in footers]
    if not any(queues):
        raise ImportFailed("not_importable", "Hugging Face has no rows for this part of the dataset.")
    taken = [[] for _ in footers]
    while any(queues):
        for i, queue in enumerate(queues):
            if not queue:
                continue
            g = queue.pop(0)
            _, start, end = footers[i].groups[g]
            if budget.fits(end - start):
                budget.spend(end - start)
                taken[i].append(g)
    if not any(taken):
        raise ImportFailed("too_large", TOO_LARGE)
    return [(i, sorted(groups)) for i, groups in enumerate(taken) if groups]


def _write_row_groups(url, footer, groups, path, sleep):
    """Read `groups` of one shard, adjacent ones in one range, into a spool
    file next to `path`, then write them to `path` as a Parquet file of their
    own. The spool holds only the ranges read."""
    import pyarrow.parquet as pq

    ranges = []
    for g in groups:
        _, start, end = footer.groups[g]
        if ranges and ranges[-1][1] == start:
            ranges[-1][1] = end
        else:
            ranges.append([start, end])
    spool_path = path + ".ranges"
    try:
        with open(spool_path, "w+b") as spool:
            segments = []
            for start, end in ranges:
                segments.append((start, end, spool.tell()))
                _stream_range(url, start, end, spool.write, sleep)
            source = _Ranges(spool, segments, footer.size)
            reader = pq.ParquetFile(source, metadata=footer.metadata, pre_buffer=False)
            writer = pq.ParquetWriter(path, reader.schema_arrow)
            try:
                for g in groups:
                    writer.write_table(reader.read_row_group(g))
            finally:
                writer.close()
    finally:
        try:
            os.remove(spool_path)
        except FileNotFoundError:
            pass


class _Ranges(io.RawIOBase):
    """A shard as pyarrow reads it, holding only the ranges read: a read
    outside them is an error, never a download."""

    def __init__(self, spool, segments, size):
        super().__init__()
        self._spool, self._segments, self._size, self._pos = spool, segments, size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence]
        self._pos = base + offset
        return self._pos

    def readinto(self, buffer):
        n = min(len(buffer), self._size - self._pos)
        if n <= 0:
            return 0
        for start, end, local in self._segments:
            if start <= self._pos and self._pos + n <= end:
                self._spool.seek(local + self._pos - start)
                data = self._spool.read(n)
                buffer[: len(data)] = data
                self._pos += len(data)
                return len(data)
        raise OSError(f"bytes {self._pos}-{self._pos + n} of the shard were not read")


def _read_range(url, start, end, budget, sleep):
    """Bytes [start, end) of a pinned shard, counted against the budget."""
    parts = []
    _stream_range(url, start, end, parts.append, sleep)
    budget.spend(end - start)
    return b"".join(parts)


def _stream_range(url, start, end, sink, sleep):
    """Bytes [start, end) of a pinned shard, passed to `sink` as they come. A
    server that answers with the whole file (200) is closed before its body is
    read; one that sends more or less than asked is refused, and never past
    what was asked."""
    want = end - start
    response = _get(url, sleep, stream=True, headers={"Range": f"bytes={start}-{end - 1}"})
    try:
        if response.status_code == 200:
            raise ProviderUnavailable(NO_RANGES)
        if response.status_code != 206:
            raise ProviderUnavailable(f"Hugging Face answered {response.status_code} for a part of a shard")
        named = response.headers.get("Content-Range")
        if named is not None and not named.startswith(f"bytes {start}-{end - 1}/"):
            raise ProviderUnavailable(WRONG_RANGE)
        got = 0
        for chunk in response.iter_content(CHUNK):
            if got + len(chunk) > want:
                raise ProviderUnavailable(WRONG_RANGE)
            sink(chunk)
            got += len(chunk)
        if got != want:
            raise ProviderUnavailable(WRONG_RANGE)
    finally:
        response.close()


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
