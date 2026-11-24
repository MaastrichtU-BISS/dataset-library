"""Reading a sample of one split's Parquet shards from Hugging Face, for an import (#113).

The viewer lists the shards it converted (``/parquet``). Whole shards are not
downloaded (#178): a shard can be bigger than the byte cap while an import keeps
a few thousand rows. Up to ``MAX_SHARDS`` shards, spread across the split, are
read from with HTTP range requests: first every shard's last 8 bytes, which give
its footer's length, then the footers, cheapest first, while each leaves room
for the cheapest row group found so far. Then row groups are taken one shard at
a time, each spread across its shard, while the cap allows and, for an import,
until there are enough rows for its sample (``max_rows``) from two shards; only
their byte ranges are read. Footers and row groups both count against the cap,
and no byte is read twice. Each shard's row groups are written as a local
Parquet file, so what an import or a copy reads is unchanged. A redirect is
followed by hand, its body never read. A server that answers a range with the
whole file is closed before its body is read, and one that sends another
range, or more or less than asked, is refused without being read past it or
past the cap. The only size refusal left, ``too_large``, is not even one row
group of any shard whose footer was read fitting. A file that is not Parquet we
can read raises ``NotParquet``, which an import refuses as ``not_importable``.

The listing answers with ``X-Revision``, the commit of the dataset its
conversion was built from (#168), so the import records what its rows were read
from, not the Hub's current head. Every range is read from the commit the Hub's
``refs/convert/parquet`` branch points at, so a conversion rebuilt between the
listing and the read cannot mix two builds' files, and one rebuilt while the
listing is read is not imported until it settles. Only huggingface.co,
datasets-server.huggingface.co and the https hosts the Hub redirects a range
to (its CDN) are called. 429 and 5xx are retried three times
with backoff, then the provider is unavailable. An import queued at another
revision is refused, with the same refusal ``imports`` makes, before any byte
of a shard is read, footers included (``expected_revision``, #174)."""
import io
import os
import re
import time
from urllib.parse import unquote, urljoin, urlsplit

import requests
import urllib3

from ..errors import NotParquet, ProviderUnavailable, RangeNotFetched
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
CUT_OFF = "Hugging Face stopped sending part of a file"
BAD_REDIRECT = "Hugging Face sent us somewhere else for part of a file that we do not read from"
# A range is asked for again where a redirect points, at most this many times,
# and only over https.
REDIRECTS = {301, 302, 303, 307, 308}
MAX_REDIRECTS = 5
# The rows an import reads for a sample of `max_rows` (see `row_target`).
ROW_HEADROOM = 3
MIN_EXTRA_ROWS = 1000
# PARQUET-816: files parquet-mr 1.2.8 and older wrote are read up to 100 bytes
# past each column chunk. Arrow's ApplicationVersion decides which, from the
# footer's created_by (cpp/src/parquet/metadata.cc, file_reader.cc).
PARQUET_816_FIXED = ("parquet-mr", (1, 2, 9))
PARQUET_816_PADDING = 100
SPACES = " \t\v\r\n\f"
DIGITS = "0123456789"


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


def fetch_shards(external_id, config, split, max_bytes, dest_dir, *, expected_revision=None, max_rows=None, sleep=time.sleep):
    """Read a sample of one split into local Parquet files, at most `max_bytes`
    in all. With `max_rows`, the rows an import keeps, the reading stops once
    there are ``row_target(max_rows)`` rows; without it (a copy), it fills the
    byte cap."""
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
    # describes another build. The footers are found from these sizes, and every
    # range answered must name this size as the file's.
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
    # Ranges are sent through one session's adapter, below its redirect handling (see `_open_range`).
    with requests.Session() as session:
        footers, skipped = _read_footers(session, urls, [shard["size"] for shard in chosen], budget, sleep)
        plan = _plan(footers, budget, None if max_rows is None else row_target(max_rows), skipped)

        os.makedirs(dest_dir, exist_ok=True)
        paths = []
        try:
            for n, (footer, groups) in enumerate(plan):
                path = os.path.join(dest_dir, f"{n:05}.parquet")
                _write_row_groups(session, footer, groups, path, budget, sleep)
                paths.append(path)
        except BaseException:
            # No file of a fetch that failed is left for anything to read.
            for path in paths:
                _remove(path)
            raise
    return Fetched(paths=paths, revision=revision, head=head)


def row_target(max_rows):
    """The rows read for an import that keeps `max_rows`: three times as many,
    and at least a thousand more. The random sampler draws from what was read,
    so a sample of nearly every row read would be little more than the first
    row groups; the stratified sampler needs every label's rows, and a label
    can be rare; and rows without text are dropped. ``ROW_HEADROOM`` and
    ``MIN_EXTRA_ROWS`` set that margin."""
    return max(ROW_HEADROOM * max_rows, max_rows + MIN_EXTRA_ROWS)


class _Budget:
    """The bytes an import may still read: footers and row groups both count."""

    def __init__(self, cap):
        self.cap, self.spent = cap, 0

    @property
    def left(self):
        return self.cap - self.spent

    def fits(self, n):
        return self.spent + n <= self.cap

    def spend(self, n):
        self.spent += n


class _Footer:
    """A shard's parsed footer: where it is read from, its size, its metadata,
    each row group's rows and the byte range pyarrow reads for it, and the
    bytes already held: the footer and trailer, [data_end, size)."""

    def __init__(self, url, size, metadata, data_end, tail):
        self.url, self.size, self.metadata = url, size, metadata
        self.data_end, self.tail = data_end, tail
        self.held = [(data_end, size)]
        pad = _padding(metadata.created_by)
        self.groups = []
        for i in range(metadata.num_row_groups):
            rows, start, end = _group_range(metadata.row_group(i), data_end)
            if rows:
                # pyarrow reads up to `pad` bytes past each column chunk, but
                # never past the end of the file (PARQUET-816); those in the
                # footer are already held.
                end += min(pad, size - end)
            self.groups.append((rows, start, end))

    def costs(self):
        """What reading each row group with rows alone would add."""
        return [_uncovered(self.held, start, end) for rows, start, end in self.groups if rows]


def _padding(created_by):
    """The bytes pyarrow reads past each column chunk of a file `created_by`
    wrote: parquet-mr 1.2.8 and older did not count the dictionary page's
    header in a chunk's size (PARQUET-816), so pyarrow reads up to 100 bytes
    more for files they wrote. Which files those are is Arrow's call, so its
    parse is followed exactly: a bare "parquet-mr", or one whose version
    cannot be parsed past some part, is version 0 in the parts not parsed."""
    application, version = _writer_version(created_by)
    if application == PARQUET_816_FIXED[0] and version < PARQUET_816_FIXED[1]:
        return PARQUET_816_PADDING
    return 0


def _writer_version(created_by):
    """(application, (major, minor, patch)) as Arrow's ApplicationVersion
    parses `created_by`: "NAME", "NAME version VERSION" or "NAME version
    VERSION (build BUILD)". pyarrow gives "" when the footer names no writer,
    which Arrow reads as "unknown"; neither is padded."""
    text = created_by or ""
    mark = text.find(" version ")
    application = (text if mark < 0 else text[:mark]).strip(SPACES)
    version = [0, 0, 0]
    if mark >= 0:
        start = mark + len(" version ")
        while start < len(text) and text[start] in SPACES:
            start += 1
        end = text.find(" (", start)
        if end < 0:
            end = len(text)
        while start < end - 1 and text[end - 1] in SPACES:
            end -= 1
        if start != end:
            _parse_version(text[start:end], version)
    return application, tuple(version)


def _parse_version(text, version):
    """MAJOR[.MINOR[.PATCH...]] into `version`, stopping where Arrow's parser
    gives up and leaving the parts after it 0."""
    pos = 0
    for part in (0, 1):
        end = _digits_end(text, pos)
        if end < len(text):
            if text[end] != "." or end == pos:
                return
            version[part], pos = _atoi(text[pos:end]), end + 1
        else:
            version[part], pos = _atoi(text[pos:]), len(text)
    end = _digits_end(text, pos)
    if end != pos:
        version[2] = _atoi(text[pos:end])


def _digits_end(text, pos):
    while pos < len(text) and text[pos] in DIGITS:
        pos += 1
    return pos


def _atoi(digits):
    """C's atoi as Arrow calls it: strtol clamped to a 64-bit long, then cut to a 32-bit int."""
    n = min(int(digits or "0"), 2**63 - 1) & 0xFFFFFFFF
    return n - 2**32 if n >= 2**31 else n


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


def _read_footers(session, urls, sizes, budget, sleep):
    """The shards' footers. Every shard's last 8 bytes are read first: they are
    cheap, and give its footer's length. Then the footers are read cheapest
    first, each only while it leaves room for the cheapest row group that fits
    of those read so far, or, before any is known, any room at all; a footer
    that does not is skipped, and so are the dearer ones after it. Returns the
    footers read, in the shards' order, and how many shards were skipped."""
    import pyarrow.parquet as pq

    trailers, skipped = [], 0
    for n, (url, size) in enumerate(zip(urls, sizes)):
        if size < len(MAGIC) + TRAILER:
            raise NotParquet(NOT_PARQUET)
        if not budget.fits(TRAILER):
            # Every trailer is 8 bytes: no later shard's fits either.
            skipped += len(urls) - n
            break
        trailer = _read_range(session, url, size - TRAILER, size, size, budget, sleep)
        length = int.from_bytes(trailer[:4], "little")
        if trailer[4:] != MAGIC or length == 0 or length > size - TRAILER - len(MAGIC):
            raise NotParquet(NOT_PARQUET)
        trailers.append((length, n, url, size, trailer))

    footers = {}
    trailers.sort(key=lambda t: t[:2])
    for k, (length, n, url, size, trailer) in enumerate(trailers):
        if length > budget.left - _reserve(footers.values(), budget):
            skipped += len(trailers) - k
            break
        data_end = size - TRAILER - length
        footer = _read_range(session, url, data_end, size - TRAILER, size, budget, sleep)
        try:
            metadata = pq.read_metadata(io.BytesIO(footer + trailer))
        except Exception as broken:  # noqa: BLE001 - whatever pyarrow raised, the footer cannot be read
            raise NotParquet(NOT_PARQUET) from broken
        footers[n] = _Footer(url, size, metadata, data_end, footer + trailer)
    return [footers[n] for n in sorted(footers)], skipped


def _reserve(footers, budget):
    """The room a further footer must leave. Before any row group that fits is
    known, any room at all. Once one shard has one, its cheapest; once two
    shards have, the two cheapest shards' cheapest together, so a footer read
    late never costs the sample its second shard."""
    cheapest = sorted(
        min(fitting) for fitting in ([c for c in f.costs() if c <= budget.left] for f in footers) if fitting
    )
    return sum(cheapest[:2]) if cheapest else 1


def _group_range(group, data_end):
    """The rows of one row group and its bytes [start, end): from its first
    column chunk's first page to its last one's end. An empty group is
    (0, 0, 0), as its offsets are not written. Outside the shard's data, or
    with a negative or empty chunk, the footer is not one we trust."""
    if group.num_rows == 0:
        return 0, 0, 0
    start, end = None, None
    for c in range(group.num_columns):
        column = group.column(c)
        first = column.data_page_offset
        if column.has_dictionary_page and column.dictionary_page_offset:
            first = min(first, column.dictionary_page_offset)
        if column.total_compressed_size < 0:
            raise NotParquet(NOT_PARQUET)
        last = first + column.total_compressed_size
        start = first if start is None else min(start, first)
        end = last if end is None else max(end, last)
    if start is None or group.num_rows < 0 or start < len(MAGIC) or end <= start or end > data_end:
        raise NotParquet(NOT_PARQUET)
    return group.num_rows, start, end


def _plan(footers, budget, target=None, skipped=0):
    """The row groups to read, by shard: one from each shard in turn, spread
    across it, while the budget allows and, with a `target`, until that many
    rows are taken. Until two shards have a row group, a shard does not take
    the room another shard's cheapest row group needs, so uneven costs still
    give a sample from two shards, and the target does not end the reading
    while a second shard's row group can fit; past the target only a shard with
    none is read from. Returns [(footer, [row group indexes])] for shards with
    any taken."""
    queues = [[g for g in _spread(len(f.groups)) if f.groups[g][0] > 0] for f in footers]
    if not any(queues):
        if skipped:
            raise ImportFailed("too_large", TOO_LARGE)
        raise ImportFailed("not_importable", "Hugging Face has no rows for this part of the dataset.")
    taken = [[] for _ in footers]
    rows = 0

    def cost(i, g):
        # What reading group g adds to shard i's bytes already held or taken:
        # padded ranges overlap their neighbours and the footer, and no byte
        # is read twice.
        _, start, end = footers[i].groups[g]
        return _uncovered(footers[i].held + [footers[i].groups[t][1:] for t in taken[i]], start, end)

    def reserve(i):
        # The room the cheapest fitting row group of another shard with none
        # needs, while taking from shard i would leave fewer than two shards.
        if any(taken[j] for j in range(len(footers)) if j != i):
            return 0
        others = [cost(j, g) for j, queue in enumerate(queues) if j != i for g in queue]
        others = [c for c in others if c <= budget.left]
        return min(others, default=0)

    def reached():
        # The target ends the reading once two shards have rows, or once no
        # shard without any has a row group that still fits.
        if target is None or rows < target:
            return False
        if sum(1 for t in taken if t) >= 2:
            return True
        return not any(cost(j, g) <= budget.left for j, queue in enumerate(queues) if not taken[j] for g in queue)

    reserving = True
    while not reached():
        progress = False
        for i, queue in enumerate(queues):
            if reached():
                break
            if taken[i] and target is not None and rows >= target:
                # Past the target, only a second shard is wanted.
                continue
            # A group that does not fit now never will: the budget only shrinks.
            while queue and cost(i, queue[0]) > budget.left:
                queue.pop(0)
            if not queue:
                continue
            pick = queue[0]
            if reserving:
                room = budget.left - reserve(i)
                if cost(i, pick) > room:
                    if taken[i]:
                        continue
                    pick = min((g for g in queue if cost(i, g) <= room), key=lambda g: cost(i, g), default=None)
                    if pick is None:
                        continue
            budget.spend(cost(i, pick))
            queue.remove(pick)
            taken[i].append(pick)
            rows += footers[i].groups[pick][0]
            progress = True
        if not progress:
            if not reserving:
                break
            # Nothing fits beside another shard's cheapest group: one shard it is.
            reserving = False
    if not any(taken):
        raise ImportFailed("too_large", TOO_LARGE)
    return [(footers[i], sorted(groups)) for i, groups in enumerate(taken) if groups]


def _uncovered(ranges, start, end):
    """The bytes of [start, end) outside every range in `ranges`."""
    left, pos = 0, start
    for a, b in sorted(ranges):
        if b <= pos:
            continue
        if a >= end:
            break
        if a > pos:
            left += a - pos
        pos = max(pos, b)
        if pos >= end:
            return left
    return left + max(0, end - pos)


def _merged(footer, groups):
    """The byte ranges covering `groups` of one shard, overlapping and adjacent
    ones joined, in order."""
    ranges = []
    for start, end in sorted(footer.groups[g][1:] for g in groups):
        if ranges and start <= ranges[-1][1]:
            ranges[-1][1] = max(ranges[-1][1], end)
        else:
            ranges.append([start, end])
    return ranges


def _write_row_groups(session, footer, groups, path, budget, sleep):
    """Read `groups` of one shard, overlapping and adjacent ones in one range,
    into a spool file next to `path`, then write them as a Parquet file of
    their own. Bytes in the footer, which padding can reach, are not read
    again: they are served from the footer already held. The file is written
    under a temporary name and moved to `path` only when whole; on failure
    neither it nor the spool is left. A shard pyarrow cannot decode is
    NOT_PARQUET; a spool or local file that cannot be read or written raises
    its own OSError, and a read of bytes never fetched raises RangeNotFetched."""
    import pyarrow.parquet as pq

    spool_path, partial = path + ".ranges", path + ".partial"
    try:
        with open(spool_path, "w+b") as spool:
            segments = []
            for start, end in _merged(footer, groups):
                end = min(end, footer.data_end)
                if start >= end:
                    continue
                segments.append((start, end, spool.tell()))
                # Everything planned is spent already: what is left is spare.
                _stream_range(session, footer.url, start, end, footer.size, spool.write, sleep, spare=budget.left)
            source = _Ranges(spool, segments, footer.size, held=[(footer.data_end, footer.tail)])
            reader = _decoded(source, lambda: pq.ParquetFile(source, metadata=footer.metadata, pre_buffer=False))
            writer = None
            try:
                for g in groups:
                    table = _decoded(source, lambda: reader.read_row_group(g))
                    if writer is None:
                        # A list's item keeps the name its writer gave it.
                        writer = pq.ParquetWriter(partial, reader.schema_arrow, use_compliant_nested_type=False)
                    writer.write_table(table)
            finally:
                if writer is not None:
                    writer.close()
        os.replace(partial, path)
    except BaseException:
        _remove(partial)
        raise
    finally:
        _remove(spool_path)


def _decoded(source, read):
    """What `read` returns from pyarrow decoding `source`. A failure of our own
    reading of `source` is raised as itself, whatever pyarrow made of it: a
    spool's OSError keeps its errno, and a read outside the fetched ranges is
    RangeNotFetched. Only what pyarrow raised decoding the bytes is
    NOT_PARQUET."""
    import pyarrow as pa

    try:
        return read()
    except Exception as broken:
        if source.failure is not None:
            raise source.failure from None
        if isinstance(broken, (pa.ArrowException, OSError, ValueError, EOFError)):
            raise NotParquet(NOT_PARQUET) from broken
        raise


def _remove(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


class _Ranges(io.RawIOBase):
    """A shard as pyarrow reads it, holding only the ranges read: `segments`,
    (start, end, offset) in the spool, and `held`, (start, bytes) in memory. A
    read outside them raises RangeNotFetched, never a download. The first
    failure of a read is kept as `failure`, so it is raised as itself however
    pyarrow passes it on."""

    def __init__(self, spool, segments, size, held=()):
        super().__init__()
        self._spool, self._segments, self._size, self._pos = spool, segments, size, 0
        self._held = list(held)
        self.failure = None

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
        out = memoryview(buffer).cast("B")
        done = 0
        while done < n:
            try:
                piece = self._piece(self._pos + done, n - done)
            except OSError as failed:
                self.failure = failed
                raise
            if not piece:
                self.failure = RangeNotFetched(f"bytes {self._pos + done}-{self._pos + n} of the shard were not fetched")
                raise self.failure
            out[done : done + len(piece)] = piece
            done += len(piece)
        self._pos += n
        return n

    def _piece(self, pos, n):
        """Up to `n` bytes from `pos`, from whichever range holds `pos`; b"" when none does."""
        for start, end, local in self._segments:
            if start <= pos < end:
                self._spool.seek(local + pos - start)
                return self._spool.read(min(n, end - pos))
        for start, data in self._held:
            if start <= pos < start + len(data):
                return data[pos - start : pos - start + n]
        return b""


def _read_range(session, url, start, end, size, budget, sleep):
    """Bytes [start, end) of a pinned shard of `size` bytes, counted against the budget."""
    parts = []
    _stream_range(session, url, start, end, size, parts.append, sleep, spare=budget.left - (end - start))
    budget.spend(end - start)
    return b"".join(parts)


def _stream_range(session, url, start, end, size, sink, sleep, spare=0):
    """Bytes [start, end) of a pinned shard of `size` bytes, passed to `sink` as
    they come. A server that answers with the whole file (200) is closed before
    its body is read. A 206 must name this range of a file of this size, and
    any length it gives must be the range's. With no length given, the body is
    read one byte past the range, to tell one that sends more, only when the
    cap has a byte to `spare`; otherwise exactly the range is read, which its
    Content-Range names. One that sends more or less is refused. A connection
    lost while reading is Hugging Face being unavailable; an error from `sink`
    (a disk that is full) is its own."""
    want = end - start
    reusable = False
    response = _open_range(session, url, {"Range": f"bytes={start}-{end - 1}", "Accept-Encoding": "identity"}, sleep)
    try:
        if response.status_code == 200:
            raise ProviderUnavailable(NO_RANGES)
        if response.status_code != 206:
            raise ProviderUnavailable(f"Hugging Face answered {response.status_code} for a part of a shard")
        headers = response.headers
        if headers.get("Content-Range") not in (f"bytes {start}-{end - 1}/{size}", f"bytes {start}-{end - 1}/*"):
            raise ProviderUnavailable(WRONG_RANGE)
        length = headers.get("Content-Length")
        if length is not None and length.strip() != str(want):
            raise ProviderUnavailable(WRONG_RANGE)
        if headers.get("Content-Encoding", "identity").strip().lower() != "identity":
            raise ProviderUnavailable(WRONG_RANGE)
        limit = want + 1 if length is None and spare > 0 else want
        got = 0
        while got < limit:
            try:
                chunk = response.raw.read(min(CHUNK, limit - got), decode_content=False)
            except (requests.RequestException, urllib3.exceptions.HTTPError, OSError) as lost:
                raise ProviderUnavailable(CUT_OFF) from lost
            if not chunk:
                break
            if got + len(chunk) > want:
                raise ProviderUnavailable(WRONG_RANGE)
            sink(chunk)
            got += len(chunk)
        if got != want:
            raise ProviderUnavailable(WRONG_RANGE)
        # A body read to its stated length leaves the connection reusable: hand it
        # back to the pool, so the next range does not open a new TLS connection.
        reusable = length is not None and hasattr(response.raw, "release_conn")
    except BaseException:
        reusable = False
        raise
    finally:
        if reusable:
            response.raw.release_conn()
        else:
            response.close()


def _open_range(session, url, headers, sleep):
    """The answer to a range request, its redirects followed here: each
    redirect is closed unread, and the range asked again where it points, with
    the same headers, at most MAX_REDIRECTS times and only over https. Each
    request goes straight to the session's adapter: a session's own `send`
    reads a redirect's body even with ``allow_redirects=False``, to make
    `Response.next`, and the Hub answers a pinned resolve/ URL with a redirect
    to its CDN."""
    if urlsplit(url).scheme != "https":
        raise ProviderUnavailable(BAD_REDIRECT)
    for _ in range(MAX_REDIRECTS + 1):
        response = _ask(_sender(session, headers), url, sleep)
        if response.status_code not in REDIRECTS:
            return response
        location = response.headers.get("Location")
        response.close()
        url = urljoin(url, location or "")
        if not location or urlsplit(url).scheme != "https":
            break
    raise ProviderUnavailable(BAD_REDIRECT)


def _sender(session, headers):
    """A GET of a URL with `headers`, sent by `session`'s adapter with the
    session's own headers, proxies and certificates, and no redirect handling."""

    def send(url, timeout):
        prepared = session.prepare_request(requests.Request("GET", url, headers=headers))
        settings = session.merge_environment_settings(prepared.url, {}, True, None, None)
        return session.get_adapter(prepared.url).send(
            prepared, stream=True, timeout=timeout,
            verify=settings["verify"], cert=settings["cert"], proxies=settings["proxies"],
        )

    return send


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
