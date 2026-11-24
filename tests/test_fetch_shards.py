"""Fetching a split's Parquet shards from Hugging Face (#113), from recorded listings."""
import io
import json
import pathlib
import random
from urllib.parse import unquote

import pytest
import requests
from requests.structures import CaseInsensitiveDict

from dataset_library.errors import NotParquet, ProviderUnavailable, RangeNotFetched
from dataset_library.imports import BEHIND, CHANGED, ImportFailed
from dataset_library.providers import parquet

LISTINGS = json.loads((pathlib.Path(__file__).parent / "fixtures" / "parquet" / "listings.json").read_text())
HUB = json.loads((pathlib.Path(__file__).parent / "fixtures" / "hub" / "datasets.json").read_text())
REFS = json.loads((pathlib.Path(__file__).parent / "fixtures" / "hub" / "refs.json").read_text())
MB = 1_000_000


class Body:
    """A response body read as `requests` gives it raw: every byte read is
    counted, so a test sees what was read, not only what was sent. `fail_after`
    loses the connection once that many bytes have been read."""

    def __init__(self, content, served=None, fail_after=None):
        self._content, self._served, self._fail_after = content, served, fail_after
        self.read_bytes = 0
        self.closed = False

    def close(self):
        self.closed = True

    def read(self, amt=None, decode_content=None):
        if self._fail_after is not None and self.read_bytes >= self._fail_after:
            raise requests.ConnectionError("connection reset")
        end = len(self._content) if amt is None else self.read_bytes + amt
        if self._fail_after is not None:
            end = min(end, self._fail_after)
        chunk = self._content[self.read_bytes : end]
        self.read_bytes += len(chunk)
        if self._served and chunk:
            self._served(len(chunk))
        return chunk


class Response:
    def __init__(self, status_code, body=None, content=b"", headers=None, served=None, fail_after=None):
        self.status_code = status_code
        self._body = body
        self.headers = CaseInsensitiveDict(headers or {})
        self.raw = Body(content, served, fail_after)

    @property
    def closed(self):
        return self.raw.closed

    def json(self):
        return self._body

    def close(self):
        self.raw.close()


class FakeAdapter(requests.adapters.BaseAdapter):
    """The transport under a real `requests.Session`: what it sends goes to
    the fake Hub, and its answer comes back as a real `requests.Response`, so
    the session's own redirect handling runs on it as it would."""

    def __init__(self, fake):
        super().__init__()
        self.fake = fake

    def send(self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None):
        answered = self.fake.serve(request.url, dict(request.headers))
        response = requests.Response()
        response.status_code, response.reason = answered.status_code, "fake"
        response.headers = CaseInsensitiveDict(answered.headers)
        response.raw, response.url, response.request, response.connection = answered.raw, request.url, request, self
        return response

    def close(self):
        pass


def parquet_shard(shard, groups, rows_per_group, text_bytes, labels=("a", "b"), nested=False, **options):
    """A real Parquet shard, written with pyarrow's defaults (dictionary,
    snappy, statistics) unless `options` say otherwise: `groups` row groups of
    `rows_per_group` rows, each text naming its shard and row ("s3-r17 ...") so
    a test can tell where a row came from. ClassLabel names are in the schema
    metadata. `nested` adds a list column whose item is named "custom" and a
    struct column."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = groups * rows_per_group
    # Random hex, so compression cannot make a shard much smaller than its text.
    texts = [f"s{shard}-r{r} " + random.Random(shard * 1_000_003 + r).randbytes(text_bytes // 2).hex() for r in range(rows)]
    features = {
        "text": {"dtype": "string", "_type": "Value"},
        "label": {"names": list(labels), "_type": "ClassLabel"},
    }
    columns = {"text": texts, "label": [r % len(labels) for r in range(rows)]}
    if nested:
        columns["tags"] = pa.array([[f"t{r}", f"u{r % 3}"] for r in range(rows)], pa.list_(pa.field("custom", pa.string())))
        columns["meta"] = pa.array([{"n": r, "w": f"w{r % 5}"} for r in range(rows)])
    table = pa.table(columns)
    table = table.replace_schema_metadata({"huggingface": json.dumps({"info": {"features": features}})})
    sink = io.BytesIO()
    pq.write_table(table, sink, row_group_size=rows_per_group, use_compliant_nested_type=False, **options)
    return sink.getvalue()


def default_shard(path):
    """What a recorded listing's shard serves: a small real Parquet file."""
    return parquet_shard(int(path.rsplit("/", 1)[1].split(".")[0]), groups=3, rows_per_group=4, text_bytes=16)


class FakeHuggingFace:
    """The viewer's /parquet, the Hub's datasets and refs, from recordings, and
    the shards themselves: real Parquet files, served by range as the Hub serves
    them. A recorded listing's shards are small files made here, and the listing
    and paths-info give their real sizes; `add_dataset` adds one made of the
    shards a test passes. Every byte of a shard sent is logged."""

    def __init__(self):
        self.downloads = []  # the shard URLs read from, in order, each once
        self.ranges = []  # (url, first byte, last byte) per range asked
        self.served = 0  # shard bytes sent
        self.failures = []  # statuses answered before the real one, per call
        self.refs_failures = []  # statuses the refs endpoint answers before the real one
        self.refs_bodies = {}  # dataset id: body answered instead of the recorded one
        self.hub_calls = []
        self.no_revision = False  # the listing answers without X-Revision
        self.listing_revisions = []  # X-Revision answered instead of the recorded one, per listing call
        self.refs_commits = []  # convert commit answered instead of the recorded one, per refs call
        self.build_sizes = {}  # path in the build: size answered instead of the listed one
        self.build_status = 200
        self.build_missing = set()  # paths the pinned build does not have
        self.head_failures = []  # statuses the Hub's dataset record answers before the real one
        self.ignore_ranges = False  # answer a range with 200 and the whole shard
        self.extra_bytes = 0  # sent past the asked range; or (first, last, size) -> bytes
        self.short_bytes = 0  # left off the end of the asked range
        self.send_length = True  # answer a range with its Content-Length
        self.content_range = None  # (url, first, last, size) -> Content-Range answered, or None for none
        self.fail_after = None  # bytes of a range body read before the connection is lost
        self.responses = []  # every range response, in order
        self.redirects = 0  # redirects answered before each range, each to the next hop
        self.redirect_body = b""  # the body each redirect carries
        self.redirect_location = None  # Location answered instead of the next hop's
        self.redirect_responses = []  # every redirect answered, in order
        self.asked_headers = []  # the headers each range was asked with
        self.listings = {key: recorded for key, recorded in LISTINGS.items()}
        self.files = {}  # path in the build: bytes, made on first use

    def add_dataset(self, dataset, config, split, shards, revision="1" * 40, commit="2" * 40):
        files = [
            {"dataset": dataset, "config": config, "split": split, "filename": f"{n:04}.parquet", "size": len(data),
             "url": f"https://huggingface.co/datasets/{dataset}/resolve/{parquet.ENCODED_CONVERT}/{config}/{split}/{n:04}.parquet"}
            for n, data in enumerate(shards)
        ]
        for n, data in enumerate(shards):
            self.files[f"{config}/{split}/{n:04}.parquet"] = data
        self.listings[f"{dataset}?config={config}"] = {"status": 200, "body": {"parquet_files": files}, "headers": {"X-Revision": revision}}
        self.refs_bodies[dataset] = {"tags": [], "branches": [], "converts": [{"name": "parquet", "ref": parquet.CONVERT, "targetCommit": commit}]}

    def file(self, path):
        if path not in self.files:
            self.files[path] = default_shard(path)
        return self.files[path]

    def listing(self, key):
        recorded = self.listings[key]
        files = recorded["body"].get("parquet_files")
        if not files:
            return recorded
        sized = [{**f, "size": len(self.file(parquet._in_build(f["url"])))} for f in files]
        return {**recorded, "body": {**recorded["body"], "parquet_files": sized}}

    def post(self, url, data=None, timeout=None, **kwargs):
        # paths-info at the pinned build, as the listings give its files.
        assert url.startswith(parquet.HUB) and "/paths-info/" in url
        self.hub_calls.append(url)
        dataset = url.removeprefix(parquet.HUB).split("/paths-info/", 1)[0]
        sizes = {
            parquet._in_build(s["url"]): s["size"]
            for key in self.listings if key.startswith(dataset + "?")
            for s in (self.listing(key)["body"].get("parquet_files") or [])
        }
        sizes.update(self.build_sizes)
        body = [{"type": "file", "path": p, "size": sizes[p]} for p in data["paths"] if p in sizes and p not in self.build_missing]
        return Response(self.build_status, body)

    def get(self, url, params=None, timeout=None, stream=False, headers=None, **kwargs):
        if self.failures:
            return Response(self.failures.pop(0), {"error": "busy"})
        if url == parquet.LISTING:
            recorded = self.listing(f"{params['dataset']}?config={params['config']}")
            answered = {} if self.no_revision else recorded.get("headers")
            if self.listing_revisions:
                answered = {"X-Revision": self.listing_revisions.pop(0)}
            return Response(recorded["status"], recorded["body"], headers=answered)
        if url.startswith(parquet.HUB):
            self.hub_calls.append(url)
            if url.endswith("/refs"):
                dataset = url.removeprefix(parquet.HUB).removesuffix("/refs")
                if self.refs_failures:
                    return Response(self.refs_failures.pop(0), {"error": "busy"})
                body = self.refs_bodies.get(dataset) or REFS.get(dataset, {"body": {"tags": [], "branches": [], "converts": []}})["body"]
                if self.refs_commits:
                    body = {**body, "converts": [{"name": "parquet", "ref": parquet.CONVERT, "targetCommit": self.refs_commits.pop(0)}]}
                return Response(200, body)
            if self.head_failures:
                return Response(self.head_failures.pop(0), {"error": "busy"})
            recorded = HUB.get(url.removeprefix(parquet.HUB), {"status": 404, "body": {"error": "Repository not found"}})
            return Response(recorded["status"], recorded["body"])
        raise AssertionError(f"a shard is read through a session's adapter, not requests.get: {url}")

    def install(self, monkeypatch):
        """Answer the listing and the Hub for `requests.get` and `.post`, and
        shards for a real `requests.Session` with this fake as its transport."""
        real_session = requests.Session

        def session():
            made = real_session()
            made.mount("https://", FakeAdapter(self))
            made.mount("http://", FakeAdapter(self))
            return made

        monkeypatch.setattr(parquet.requests, "get", self.get)
        monkeypatch.setattr(parquet.requests, "post", self.post)
        monkeypatch.setattr(parquet.requests, "Session", session)

    def serve(self, url, headers):
        """A shard request, as the Hub and its CDN answer it."""
        if self.failures:
            return Response(self.failures.pop(0), {"error": "busy"})
        url, _, query = url.partition("?")
        hop = int(query.removeprefix("hop=")) if query.startswith("hop=") else 0
        if hop < self.redirects and "Range" in headers:
            # The Hub answers a pinned resolve/ URL with a redirect to its CDN.
            location = self.redirect_location
            if location is None:
                location = f"https://cdn.example/resolve/{url.split('/resolve/', 1)[1]}?hop={hop + 1}"
            response = Response(302, content=self.redirect_body, headers={"Location": location, "Content-Length": str(len(self.redirect_body))})
            self.redirect_responses.append(response)
            self.asked_headers.append(dict(headers))
            return response
        if url not in self.downloads:
            self.downloads.append(url)
        data = self.file(url.split("/resolve/", 1)[1].split("/", 1)[1])
        asked = (headers or {}).get("Range")
        if self.ignore_ranges or asked is None:
            response = Response(200, content=data, served=self.count)
            self.responses.append(response)
            return response
        first, last = (int(n) for n in asked.removeprefix("bytes=").split("-"))
        self.ranges.append((url, first, last))
        self.asked_headers.append(dict(headers))
        extra = self.extra_bytes(first, last, len(data)) if callable(self.extra_bytes) else self.extra_bytes
        content = data[first : last + 1 + extra]
        content = content[: len(content) - self.short_bytes]
        answered = {"Content-Range": f"bytes {first}-{last}/{len(data)}"}
        if self.content_range is not None:
            answered = {} if self.content_range(url, first, last, len(data)) is None else {"Content-Range": self.content_range(url, first, last, len(data))}
        if self.send_length:
            answered["Content-Length"] = str(len(content))
        response = Response(206, content=content, headers=answered, served=self.count, fail_after=self.fail_after)
        self.responses.append(response)
        return response

    def count(self, n):
        self.served += n


@pytest.fixture
def hf(monkeypatch):
    fake = FakeHuggingFace()
    fake.install(monkeypatch)
    return fake


def fetch(tmp_path, dataset, config, split, max_bytes=200 * MB, expected_revision=None):
    return parquet.fetch_shards(
        dataset, config, split, max_bytes, str(tmp_path / "work"),
        expected_revision=expected_revision, sleep=lambda s: None,
    )


LEX_CONVERSION = "823338615adfc9049aadc2841a9b2d2bd531363c"
DWARF = "turhancan97/doppeldwarf-306"
DWARF_LISTED = "18d016c41417e9beb85fb83a94b4fbf866805cce"
DWARF_HEAD = "f5a645c5f00fe5acfe939d15e00c2faa3cb85375"


def test_the_chosen_splits_shards_are_read_from_the_converts_commit(hf, tmp_path):
    # The listing's URLs name the moving refs/convert/parquet branch; every range
    # is read from the commit the Hub says that branch points at, so the bytes
    # cannot change under a conversion rebuilt while the import reads.
    import pyarrow.parquet as pq

    fetched = fetch(tmp_path, "coastalcph/lex_glue", "ledgar", "train")

    pinned = f"https://huggingface.co/datasets/coastalcph/lex_glue/resolve/{LEX_CONVERSION}/ledgar/train/0000.parquet"
    assert hf.downloads == [pinned]
    assert {url for url, _, _ in hf.ranges} == {pinned}
    # The local file holds the shard's rows, read from its row groups.
    served = pq.read_table(io.BytesIO(hf.file("ledgar/train/0000.parquet")))
    assert [pq.read_table(p).to_pydict() for p in fetched.paths] == [served.to_pydict()]


def test_the_import_records_the_revision_the_rows_were_read_from(hf, tmp_path):
    # X-Revision names the commit the listed conversion was built from: for this
    # recording the same sha the Hub's record names.
    fetched = fetch(tmp_path, "coastalcph/lex_glue", "ledgar", "train")

    assert fetched.revision == "c23fdff1a6bf74e0e1a71cb86f1e781d37da888c"
    hub = "https://huggingface.co/api/datasets/coastalcph/lex_glue"
    assert hf.hub_calls == [f"{hub}/refs", f"{hub}/refs", f"{hub}/paths-info/{LEX_CONVERSION}", hub]
    # The head is asked only to explain a refusal; the recorded revision is the conversion's.
    assert fetched.head == "c23fdff1a6bf74e0e1a71cb86f1e781d37da888c"


def test_a_conversion_that_lags_the_queued_revision_records_its_own(hf, tmp_path):
    # turhancan97/doppeldwarf-306 was recorded while its conversion still sat at
    # 18d016c4: the Hub's head was already f5a645c5, and the convert branch tip
    # was ac78ba96 (checked live on 2026-11-23).
    fetched = fetch(tmp_path, DWARF, "classes", "reference")

    assert fetched.revision == DWARF_LISTED
    assert fetched.revision != DWARF_HEAD
    assert fetched.head == DWARF_HEAD
    assert hf.downloads == [f"https://huggingface.co/datasets/{DWARF}/resolve/ac78ba966b386ec617057c606445d435f36de67f/classes/reference/0000.parquet"]


def test_an_import_queued_at_the_datasets_head_downloads_no_shards(hf, tmp_path):
    # The conversion still sits at 18d016c4 while the Hub's head is f5a645c5, the
    # revision the import was queued at: opening the dataset again would not help
    # until the conversion catches up, and no shard is downloaded for the refusal (#174).
    with pytest.raises(ImportFailed) as refused:
        fetch(tmp_path, DWARF, "classes", "reference", expected_revision=DWARF_HEAD)

    assert refused.value.code == "conversion_behind"
    assert refused.value.message == BEHIND
    assert hf.downloads == []


def test_an_import_queued_at_another_revision_than_conversion_and_head_downloads_no_shards(hf, tmp_path):
    # The dataset moved on after the import was queued at a revision neither the
    # conversion nor the Hub's head is at: opening it again shows the new version (#174).
    with pytest.raises(ImportFailed) as refused:
        fetch(tmp_path, DWARF, "classes", "reference", expected_revision="a" * 40)

    assert refused.value.code == "changed"
    assert refused.value.message == CHANGED
    assert hf.downloads == []


def test_an_import_queued_at_the_conversion_s_revision_downloads(hf, tmp_path):
    fetched = fetch(tmp_path, DWARF, "classes", "reference", expected_revision=DWARF_LISTED)

    assert fetched.revision == DWARF_LISTED
    assert len(hf.downloads) == 1


def test_a_copy_with_no_expected_revision_downloads_even_when_the_conversion_lags(hf, tmp_path):
    # Copies pass no expected revision: a conversion at another revision still
    # downloads, as it did before the refusal existed (#174).
    fetched = fetch(tmp_path, DWARF, "classes", "reference")

    assert fetched.revision == DWARF_LISTED
    assert len(hf.downloads) == 1


def test_a_listing_that_names_no_revision_records_none(hf, tmp_path):
    # No X-Revision, nothing recorded: not the Hub's head, which the rows may not
    # have come from (#168). The recording does say one, so this is the fake
    # taking it away, as an answer without the header behaves.
    hf.no_revision = True

    fetched = fetch(tmp_path, "coastalcph/lex_glue", "ledgar", "train")

    assert hf.downloads == [f"https://huggingface.co/datasets/coastalcph/lex_glue/resolve/{LEX_CONVERSION}/ledgar/train/0000.parquet"]
    assert fetched.revision is None


def test_a_conversion_being_rebuilt_is_not_imported_yet(hf, tmp_path):
    # Agnuxo/P2PCLAW-Innovative-Benchmark was recorded with a listing and no
    # convert branch at all (checked live on 2026-11-23); a listing with files
    # whose branch has gone means the same rebuild is under way.
    hf.refs_bodies = {"coastalcph/lex_glue": REFS["Agnuxo/P2PCLAW-Innovative-Benchmark"]["body"]}

    with pytest.raises(ProviderUnavailable):
        fetch(tmp_path, "coastalcph/lex_glue", "ledgar", "train")

    assert hf.downloads == []


@pytest.mark.parametrize("moved", ["commit", "revision"])
def test_a_conversion_rebuilt_while_the_listing_is_read_is_not_imported_yet(hf, tmp_path, moved):
    # The listing named one build and the convert branch another: the shards
    # would be pinned to a build the recorded revision does not describe.
    if moved == "commit":
        hf.refs_commits = [LEX_CONVERSION, "0" * 40]
    else:
        hf.listing_revisions = ["c23fdff1a6bf74e0e1a71cb86f1e781d37da888c", "1" * 40]

    with pytest.raises(ProviderUnavailable) as waiting:
        fetch(tmp_path, "coastalcph/lex_glue", "ledgar", "train")

    assert str(waiting.value) == parquet.REBUILDING
    assert hf.downloads == []


def test_a_listing_describing_another_build_than_the_pinned_one_is_not_imported_yet(hf, tmp_path):
    # A cached listing while the branch has moved on: the pinned build's file
    # is not the size the listing gave.
    hf.build_sizes = {"ledgar/train/0000.parquet": 1}

    with pytest.raises(ProviderUnavailable) as waiting:
        fetch(tmp_path, "coastalcph/lex_glue", "ledgar", "train")

    assert str(waiting.value) == parquet.REBUILDING
    assert hf.downloads == []


def test_a_pinned_build_the_hub_cannot_list_is_not_imported_yet(hf, tmp_path):
    hf.build_status = 404

    with pytest.raises(ProviderUnavailable):
        fetch(tmp_path, "coastalcph/lex_glue", "ledgar", "train")

    assert hf.downloads == []


def test_a_shard_missing_from_the_pinned_build_is_not_imported_yet(hf, tmp_path):
    # paths-info leaves out a path the build does not have.
    hf.build_missing = {"ledgar/train/0000.parquet"}

    with pytest.raises(ProviderUnavailable) as waiting:
        fetch(tmp_path, "coastalcph/lex_glue", "ledgar", "train")

    assert str(waiting.value) == parquet.REBUILDING
    assert hf.downloads == []


def test_a_hub_that_cannot_name_its_head_does_not_fail_the_fetch(hf, tmp_path):
    # The head only explains a refusal; a busy Hub record gives None instead.
    hf.head_failures = [503, 503, 503, 503]

    fetched = fetch(tmp_path, "coastalcph/lex_glue", "ledgar", "train")

    assert fetched.paths and fetched.head is None
    assert fetched.revision == "c23fdff1a6bf74e0e1a71cb86f1e781d37da888c"


def test_a_listing_with_no_revision_does_not_ask_for_the_head(hf, tmp_path):
    hf.no_revision = True

    fetched = fetch(tmp_path, "coastalcph/lex_glue", "ledgar", "train")

    assert fetched.head is None
    assert "https://huggingface.co/api/datasets/coastalcph/lex_glue" not in hf.hub_calls


def test_a_convert_commit_that_is_not_a_revision_is_not_used(hf, tmp_path):
    hf.refs_commits = ["../main", "../main"]

    with pytest.raises(ProviderUnavailable) as waiting:
        fetch(tmp_path, "coastalcph/lex_glue", "ledgar", "train")

    assert str(waiting.value) == parquet.REBUILDING
    assert hf.downloads == []


def test_a_shard_url_that_does_not_name_the_convert_branch_cannot_be_pinned():
    with pytest.raises(ProviderUnavailable) as refused:
        parquet._pinned("https://huggingface.co/datasets/x/y/resolve/main/c/s/0000.parquet", "a" * 40)

    assert str(refused.value) == parquet.UNPINNABLE


def test_the_convert_branch_is_asked_again_through_5xx_and_429(hf, tmp_path):
    hf.refs_failures = [503, 429, 502]

    assert fetch(tmp_path, "coastalcph/lex_glue", "ledgar", "train").paths

    hf.refs_failures = [503, 503, 503, 503]
    with pytest.raises(ProviderUnavailable):
        fetch(tmp_path, "coastalcph/lex_glue", "ledgar", "train")


# --- Reading only what a sample needs (#178) ----------------------------------------
#
# FineFineWeb's conversion is 10 shards of 263-292 MB against a 200 MB default
# cap. These fakes keep its shape at a thousandth of the size: 10 shards of about
# 270 KB, each of 12 row groups of about 22 KB, against a 200 KB cap. Shards are
# over the cap; their row groups are well under it. Its texts are long, so
# pyarrow keeps no statistics for them and a footer is about 3 KB.
SCALED_CAP = 200_000
FINE = "m-a-p/FineFineWeb"
# The writers a shard is written with: pyarrow's defaults (dictionary, snappy,
# statistics), and zstd with a page index. Zstd halves the hex texts, so its
# row groups hold twice the rows to keep the same shape.
WRITERS = {
    "defaults": ({}, 5),
    "zstd-page-index": ({"compression": "zstd", "write_page_index": True}, 10),
}


def fine_shards(n=10, groups=12, rows_per_group=5, text_bytes=4400, **options):
    return [parquet_shard(s, groups, rows_per_group, text_bytes, **options) for s in range(n)]


def writer_shards(writer, **kwargs):
    options, rows_per_group = WRITERS[writer]
    return fine_shards(rows_per_group=rows_per_group, **options, **kwargs)


@pytest.fixture
def fine(hf):
    hf.add_dataset(FINE, "default", "train", fine_shards())
    return hf


def measured(data):
    """A shard's footer bytes (with its trailer) and each row group's bytes,
    read from its metadata here rather than by the code under test."""
    import pyarrow.parquet as pq

    metadata = pq.read_metadata(io.BytesIO(data))
    groups = []
    for g in range(metadata.num_row_groups):
        group = metadata.row_group(g)
        columns = [group.column(c) for c in range(group.num_columns)]
        start = min(min(c.data_page_offset, c.dictionary_page_offset or c.data_page_offset) for c in columns)
        end = max(min(c.data_page_offset, c.dictionary_page_offset or c.data_page_offset) + c.total_compressed_size for c in columns)
        groups.append(end - start)
    return int.from_bytes(data[-8:-4], "little") + 8, groups


def shards_read(paths):
    """Which shards the local files' rows came from, by the shard each text names."""
    import pyarrow.parquet as pq

    return {text.split("-", 1)[0] for p in paths for text in pq.read_table(p).column("text").to_pylist()}


def rows_in(paths):
    import pyarrow.parquet as pq

    return sum(pq.ParquetFile(p).metadata.num_rows for p in paths)


def assert_no_byte_read_twice(ranges):
    by_url = {}
    for url, first, last in ranges:
        by_url.setdefault(url, []).append((first, last))
    for spans in by_url.values():
        spans.sort()
        assert all(a[1] < b[0] for a, b in zip(spans, spans[1:])), spans


def test_a_dataset_whose_every_shard_is_over_the_cap_imports_a_sample(fine, tmp_path):
    sizes = [len(fine.file(f"default/train/{n:04}.parquet")) for n in range(10)]
    assert min(sizes) > SCALED_CAP

    fetched = fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP)

    footers = sum(measured(fine.file(f"default/train/{n:04}.parquet"))[0] for n in range(10))
    group = max(measured(fine.file("default/train/0000.parquet"))[1])
    assert rows_in(fetched.paths) >= (SCALED_CAP - footers) // group * 5 - 5
    assert fetched.revision == "1" * 40
    assert all(f"/resolve/{'2' * 40}/" in url for url in fine.downloads)


@pytest.mark.parametrize("writer", sorted(WRITERS))
@pytest.mark.parametrize("cap", [SCALED_CAP, 60_000, 30_000, 1_000_000, 5_000_000])
def test_an_import_reads_at_most_the_cap_footers_and_row_groups_included(hf, tmp_path, cap, writer):
    hf.add_dataset(FINE, "default", "train", writer_shards(writer))

    fetch(tmp_path, FINE, "default", "train", max_bytes=cap)

    assert 0 < hf.served <= cap
    # Every range asked was a footer or row groups: nothing was read twice.
    assert hf.served == sum(last + 1 - first for _, first, last in hf.ranges)
    assert_no_byte_read_twice(hf.ranges)


def test_a_cap_over_the_whole_split_reads_every_row(fine, tmp_path):
    fetched = fetch(tmp_path, FINE, "default", "train", max_bytes=10**9)

    assert rows_in(fetched.paths) == 10 * 12 * 5
    assert fine.served < sum(len(fine.file(f"default/train/{n:04}.parquet")) for n in range(10))


@pytest.mark.parametrize("writer", sorted(WRITERS))
def test_the_sample_draws_from_every_shard_the_cap_allows(hf, tmp_path, writer):
    # Ten footers leave room for some first row groups: one from each of as
    # many shards, in order, not all from the first.
    shards = writer_shards(writer)
    hf.add_dataset(FINE, "default", "train", shards)
    left = SCALED_CAP - sum(measured(s)[0] for s in shards)
    expected = set()
    for n, shard in enumerate(shards):
        cost = measured(shard)[1][0]
        if cost <= left:
            left -= cost
            expected.add(f"s{n}")
    assert 2 <= len(expected) < 10

    fetched = fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP)

    assert len(fetched.paths) == len(expected)
    assert shards_read(fetched.paths) == expected


@pytest.mark.parametrize("writer", sorted(WRITERS))
def test_a_shard_s_row_groups_are_spread_across_it(hf, tmp_path, writer):
    # Room for 21 row groups across ten shards: the first shard gives three,
    # from its start, its middle and its first quarter, not its first three.
    import pyarrow.parquet as pq

    shards = writer_shards(writer)
    hf.add_dataset(FINE, "default", "train", shards)
    costs = [c for s in shards for c in measured(s)[1]]
    cap = sum(measured(s)[0] for s in shards) + 21 * max(costs) + min(costs) // 2
    rows_per_group = WRITERS[writer][1]

    fetched = fetch(tmp_path, FINE, "default", "train", max_bytes=cap)

    first = pq.read_table(fetched.paths[0]).column("text").to_pylist()
    assert {int(t.split("-r", 1)[1].split(" ", 1)[0]) // rows_per_group for t in first} == {0, 6, 3}
    assert rows_in(fetched.paths) == 21 * rows_per_group


def test_shards_are_spread_across_a_long_split():
    shards = [{"n": n} for n in range(100)]

    picked = [s["n"] for s in parquet._spread_shards(shards)]

    assert len(picked) == parquet.MAX_SHARDS
    assert picked[0] == 0 and picked[-1] > 90


def test_the_spread_order_visits_every_index_once():
    assert parquet._spread(8) == [0, 4, 2, 6, 1, 3, 5, 7]
    for n in range(0, 40):
        assert sorted(parquet._spread(n)) == list(range(n))


def test_the_class_label_names_reach_the_local_files(fine, tmp_path):
    from dataset_library.imports import _label_names
    import pyarrow.parquet as pq

    fetched = fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP)

    assert _label_names(pq.ParquetFile(fetched.paths[0]), "label") == ["a", "b"]


def test_nested_columns_keep_their_field_names(hf, tmp_path):
    # A list whose item Hugging Face named "custom" keeps that name, and a
    # struct its fields, so the local file has the shard's own schema.
    import pyarrow.parquet as pq

    shard = parquet_shard(0, groups=4, rows_per_group=5, text_bytes=64, nested=True)
    hf.add_dataset(FINE, "default", "train", [shard])

    fetched = fetch(tmp_path, FINE, "default", "train", max_bytes=10**9)

    served = pq.read_table(io.BytesIO(shard))
    local = pq.read_table(fetched.paths[0])
    assert local.schema.field("tags").type.value_field.name == "custom"
    assert local.schema.equals(served.schema, check_metadata=True)
    assert local.to_pydict() == served.to_pydict()


def run_fine_import(app, tmp_path, fetch_shards, sample_size=200):
    from dataset_library import imports
    from dataset_library.models import LibraryImport
    from test_import import FakeCreateDataset

    create = FakeCreateDataset()
    with app.app_context():
        session = app.db.session
        record = LibraryImport(
            owner_id="alice", provider="huggingface", external_id=FINE,
            config="default", split="train", revision="1" * 40,
            text_column="text", label_column="label", task_type="classification",
            sample_size=sample_size, max_rows=1000, name="FineFineWeb", license="odc-by",
            url=f"https://huggingface.co/datasets/{FINE}", status="queued",
        )
        session.add(record)
        session.commit()
        ref = record.id
        imports.run_import(ref, session=session, fetch_shards=fetch_shards, create_dataset=create, data_dir=str(tmp_path / "data"), seed=7)
        return session.get(LibraryImport, ref).to_dict(), create


def test_an_import_samples_rows_from_several_shards(app, fine, tmp_path):
    # Through run_import with the real fetch, at the default sample method: the
    # random sample (#113) is not just the head of shard 0.
    def real_fetch(external_id, config, split, max_bytes, dest_dir, **kwargs):
        return parquet.fetch_shards(external_id, config, split, SCALED_CAP, dest_dir, sleep=lambda s: None, **kwargs)

    found, create = run_fine_import(app, tmp_path, real_fetch, sample_size=20)

    assert found["status"] == "completed", found
    assert found["rows"] == 20
    assert len({row["text"].split("-", 1)[0] for row in create.calls[0]["rows"]}) > 1
    assert fine.served <= SCALED_CAP


def test_a_row_group_over_the_cap_is_too_large_to_sample(hf, tmp_path):
    # Only the footers are read: no row group fits, and none is downloaded.
    hf.add_dataset(FINE, "default", "train", fine_shards(n=2, groups=2, rows_per_group=100))

    with pytest.raises(ImportFailed) as refused:
        fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP)

    assert refused.value.code == "too_large"
    assert refused.value.message == parquet.TOO_LARGE
    footers = sum(last + 1 - first for _, first, last in hf.ranges)
    assert hf.served == footers < 10_000


def test_a_footer_over_the_cap_is_too_large_to_sample(fine, tmp_path):
    # Every shard's trailer is read, in case a later footer is smaller; no footer fits.
    with pytest.raises(ImportFailed) as refused:
        fetch(tmp_path, FINE, "default", "train", max_bytes=1_000)

    assert refused.value.code == "too_large"
    assert refused.value.message == parquet.TOO_LARGE
    assert fine.served == 8 * 10 and len(fine.ranges) == 10
    assert all(last + 1 - first == 8 for _, first, last in fine.ranges)


def test_a_shard_whose_row_groups_are_all_over_the_cap_does_not_stop_the_next(hf, tmp_path):
    # Shard 0's row groups are each over the cap; shard 1's are small. Shard 0
    # must not stop shard 1's footer being read, nor make the import too large.
    big = parquet_shard(0, groups=2, rows_per_group=60, text_bytes=4400)
    small = parquet_shard(1, groups=6, rows_per_group=1, text_bytes=4400)
    hf.add_dataset(FINE, "default", "train", [big, small])
    cap = 100_000
    assert min(measured(big)[1]) > cap > len(small)

    fetched = fetch(tmp_path, FINE, "default", "train", max_bytes=cap)

    assert shards_read(fetched.paths) == {"s1"}
    assert rows_in(fetched.paths) == 6
    assert hf.served <= cap


def test_a_footer_that_does_not_fit_is_skipped_for_a_later_one(hf, tmp_path):
    # Shard 0's footer is over the cap, shard 1's is not: shard 1 is read.
    wide = parquet_shard(0, groups=40, rows_per_group=1, text_bytes=1000)
    narrow = parquet_shard(1, groups=1, rows_per_group=2, text_bytes=64)
    hf.add_dataset(FINE, "default", "train", [wide, narrow])
    cap = measured(narrow)[0] + measured(narrow)[1][0] + 16
    assert measured(wide)[0] > cap

    fetched = fetch(tmp_path, FINE, "default", "train", max_bytes=cap)

    assert shards_read(fetched.paths) == {"s1"}
    assert hf.served <= cap


# --- Empty row groups -------------------------------------------------------------


def shard_with_groups(shard, sizes, text_bytes=64):
    """A shard written group by group, `sizes` rows each; 0 writes an empty row group."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    sink = io.BytesIO()
    writer = None
    r = 0
    for size in sizes:
        texts = [f"s{shard}-r{r + i} " + "x" * text_bytes for i in range(size)]
        table = pa.table({"text": pa.array(texts, pa.string()), "label": pa.array([0] * size, pa.int64())})
        writer = writer or pq.ParquetWriter(sink, table.schema)
        writer.write_table(table)
        r += size
    writer.close()
    return sink.getvalue()


def test_an_empty_row_group_is_skipped(hf, tmp_path):
    import pyarrow.parquet as pq

    shard = shard_with_groups(0, [3, 0, 3])
    assert [pq.read_metadata(io.BytesIO(shard)).row_group(g).num_rows for g in range(3)] == [3, 0, 3]
    hf.add_dataset(FINE, "default", "train", [shard])

    fetched = fetch(tmp_path, FINE, "default", "train", max_bytes=10**9)

    assert rows_in(fetched.paths) == 6
    assert pq.ParquetFile(fetched.paths[0]).metadata.num_row_groups == 2


def test_an_empty_shard_before_a_good_one_is_skipped(hf, tmp_path):
    hf.add_dataset(FINE, "default", "train", [shard_with_groups(0, [0]), shard_with_groups(1, [2, 2])])

    fetched = fetch(tmp_path, FINE, "default", "train", max_bytes=10**9)

    assert shards_read(fetched.paths) == {"s1"}
    assert rows_in(fetched.paths) == 4


def test_a_split_of_only_empty_row_groups_has_no_rows(hf, tmp_path):
    hf.add_dataset(FINE, "default", "train", [shard_with_groups(0, [0]), shard_with_groups(1, [0, 0])])

    with pytest.raises(ImportFailed) as refused:
        fetch(tmp_path, FINE, "default", "train", max_bytes=10**9)

    assert refused.value.code == "not_importable"
    assert "no rows" in refused.value.message


# --- Reading no more than asked -----------------------------------------------------


def test_ranges_are_asked_without_compression(fine, tmp_path):
    fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP)

    assert fine.asked_headers and all(h["Accept-Encoding"] == "identity" for h in fine.asked_headers)


def test_a_server_that_ignores_range_is_closed_before_the_shard_is_read(fine, tmp_path):
    fine.ignore_ranges = True

    with pytest.raises(ProviderUnavailable) as refused:
        fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP)

    assert str(refused.value) == parquet.NO_RANGES
    assert fine.served == 0
    assert fine.responses and all(r.closed for r in fine.responses)


def test_a_server_that_says_it_sends_more_than_the_range_is_refused_unread(fine, tmp_path):
    fine.extra_bytes = 100

    with pytest.raises(ProviderUnavailable) as refused:
        fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP)

    assert str(refused.value) == parquet.WRONG_RANGE
    # The trailers end their files, so they came whole; the first footer's
    # Content-Length was 100 bytes too many, and its body was not read.
    assert len(fine.ranges) == 11
    assert fine.served == 8 * 10
    assert all(r.closed for r in fine.responses)


def test_a_server_that_sends_more_than_the_range_unannounced_is_read_one_byte_past_it(fine, tmp_path):
    fine.extra_bytes = 100_000
    fine.send_length = False

    with pytest.raises(ProviderUnavailable) as refused:
        fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP)

    assert str(refused.value) == parquet.WRONG_RANGE
    _, first, last = fine.ranges[-1]
    want = last + 1 - first
    assert fine.responses[-1].raw.read_bytes <= want + 1
    assert fine.served <= 8 * 10 + want + 1


def test_a_body_shorter_than_the_range_is_refused(fine, tmp_path):
    fine.short_bytes = 3
    fine.send_length = False

    with pytest.raises(ProviderUnavailable) as refused:
        fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP)

    assert str(refused.value) == parquet.WRONG_RANGE
    assert len(fine.ranges) == 1


def test_a_range_answer_without_content_range_is_refused(fine, tmp_path):
    fine.content_range = lambda url, first, last, size: None

    with pytest.raises(ProviderUnavailable) as refused:
        fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP)

    assert str(refused.value) == parquet.WRONG_RANGE
    assert fine.served == 0


def test_a_range_of_a_file_of_another_size_is_refused(fine, tmp_path):
    fine.content_range = lambda url, first, last, size: f"bytes {first}-{last}/{size + 1}"

    with pytest.raises(ProviderUnavailable) as refused:
        fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP)

    assert str(refused.value) == parquet.WRONG_RANGE
    assert fine.served == 0


def test_a_range_of_a_file_of_unknown_size_is_read(fine, tmp_path):
    fine.content_range = lambda url, first, last, size: f"bytes {first}-{last}/*"

    assert fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP).paths


def test_a_server_that_names_another_range_is_refused(fine, tmp_path):
    fine.content_range = lambda url, first, last, size: "bytes 0-7/1"

    with pytest.raises(ProviderUnavailable) as refused:
        fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP)

    assert str(refused.value) == parquet.WRONG_RANGE


def test_a_connection_lost_mid_range_is_hugging_face_unavailable(fine, tmp_path):
    fine.fail_after = 4

    with pytest.raises(ProviderUnavailable) as lost:
        fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP)

    assert str(lost.value) == parquet.CUT_OFF
    assert not isinstance(lost.value, NotParquet)
    assert all(r.closed for r in fine.responses)


# --- Files that are not Parquet we can read -----------------------------------------


def test_a_file_that_is_not_parquet_is_refused(hf, tmp_path):
    hf.add_dataset(FINE, "default", "train", [b"not a parquet file at all, just text"])

    with pytest.raises(NotParquet) as refused:
        fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP)

    assert str(refused.value) == parquet.NOT_PARQUET


def test_a_file_too_short_to_hold_a_footer_is_refused(hf, tmp_path):
    hf.add_dataset(FINE, "default", "train", [b"PAR1"])

    with pytest.raises(NotParquet) as refused:
        fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP)

    assert str(refused.value) == parquet.NOT_PARQUET
    assert hf.served == 0


def test_a_file_without_the_parquet_magic_is_refused_before_its_footer_is_read(hf, tmp_path):
    # The length before the magic is a real one: only the magic is wrong.
    shard = parquet_shard(0, groups=2, rows_per_group=2, text_bytes=64)
    hf.add_dataset(FINE, "default", "train", [shard[:-4] + b"PAR0"])

    with pytest.raises(NotParquet):
        fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP)

    assert hf.served == 8


def test_a_footer_of_length_zero_is_refused(hf, tmp_path):
    hf.add_dataset(FINE, "default", "train", [b"PAR1" + b"\0" * 32 + (0).to_bytes(4, "little") + b"PAR1"])

    with pytest.raises(NotParquet):
        fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP)

    # Only the trailer: no empty range is asked for a footer of no bytes.
    assert hf.served == 8 and len(hf.ranges) == 1


def corrupt_first_group(data):
    """`data` with its first row group's first text page header overwritten."""
    import pyarrow.parquet as pq

    column = pq.read_metadata(io.BytesIO(data)).row_group(0).column(0)
    first = min(column.data_page_offset, column.dictionary_page_offset or column.data_page_offset)
    return data[:first] + b"\xff" * 64 + data[first + 64 :]


def test_a_corrupt_page_is_not_parquet_and_leaves_no_file(hf, tmp_path):
    shards = fine_shards(n=3)
    shards[2] = corrupt_first_group(shards[2])
    hf.add_dataset(FINE, "default", "train", shards)

    with pytest.raises(NotParquet) as refused:
        fetch(tmp_path, FINE, "default", "train", max_bytes=10**9)

    assert str(refused.value) == parquet.NOT_PARQUET
    work = tmp_path / "work"
    assert sorted(p.name for p in work.iterdir()) == []


def test_a_local_file_that_cannot_be_written_raises_its_own_error(fine, tmp_path, monkeypatch):
    # A disk that is full is not Hugging Face being unavailable, nor a bad file.
    import errno
    import pyarrow.parquet as pq

    def full(self, table, *args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(pq.ParquetWriter, "write_table", full)

    with pytest.raises(OSError) as failed:
        fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP)

    assert failed.value.errno == errno.ENOSPC
    assert not isinstance(failed.value, ProviderUnavailable)
    assert list((tmp_path / "work").iterdir()) == []


class Column:
    def __init__(self, data_page_offset, total_compressed_size, dictionary_page_offset=None):
        self.data_page_offset = data_page_offset
        self.total_compressed_size = total_compressed_size
        self.dictionary_page_offset = dictionary_page_offset
        self.has_dictionary_page = dictionary_page_offset is not None


class Group:
    def __init__(self, num_rows, *columns):
        self.num_rows, self._columns = num_rows, columns
        self.num_columns = len(columns)

    def column(self, c):
        return self._columns[c]


@pytest.mark.parametrize("group", [
    Group(3, Column(0, 10)),  # starts in the magic
    Group(3, Column(4, 10), Column(14, 200)),  # ends past the data
    Group(3, Column(10, -5)),  # a negative size
    Group(3, Column(10, 50), Column(20, -5)),  # a negative size inside a group that looks whole
    Group(3, Column(10, 0)),  # rows in no bytes
    Group(3),  # rows in no columns
])
def test_a_row_group_outside_the_shard_s_data_is_not_parquet(group):
    with pytest.raises(NotParquet):
        parquet._group_range(group, data_end=100)


def test_a_row_group_s_range_runs_from_its_dictionary_to_its_last_page():
    group = Group(3, Column(20, 10, dictionary_page_offset=8), Column(30, 40))

    assert parquet._group_range(group, data_end=100) == (3, 8, 70)


def test_an_empty_row_group_has_no_range_whatever_its_offsets():
    assert parquet._group_range(Group(0, Column(0, 15, dictionary_page_offset=116)), data_end=100) == (0, 0, 0)


def with_created_by(data, created_by):
    """`data` with its footer's created_by replaced, as an older writer would
    have written it. pyarrow cannot write another writer's name, so the footer's
    Thrift string (a one-byte length, then the text) is rewritten here."""
    length = int.from_bytes(data[-8:-4], "little")
    start = len(data) - 8 - length
    footer = data[start:-8]
    old = b"parquet-cpp-arrow version "
    at = footer.index(old)
    size = footer[at - 1]
    new = created_by.encode()
    footer = footer[: at - 1] + bytes([len(new)]) + new + footer[at + size :]
    return data[:start] + footer + len(footer).to_bytes(4, "little") + b"PAR1"


def test_a_shard_from_an_old_parquet_mr_is_read_with_pyarrow_s_padding(hf, tmp_path):
    # PARQUET-816: pyarrow reads up to 100 bytes past each column chunk of a
    # file parquet-mr 1.2.8 wrote. Without those bytes, the read would ask for
    # bytes never fetched.
    import pyarrow.parquet as pq

    shard = with_created_by(parquet_shard(0, groups=6, rows_per_group=2, text_bytes=64), "parquet-mr version 1.2.8 (build 1)")
    assert pq.read_metadata(io.BytesIO(shard)).created_by == "parquet-mr version 1.2.8 (build 1)"
    hf.add_dataset(FINE, "default", "train", [shard])
    footer, groups = measured(shard)
    cap = footer + groups[0] + 100 + groups[3] + 100

    fetched = fetch(tmp_path, FINE, "default", "train", max_bytes=cap)

    assert rows_in(fetched.paths) == 4
    assert hf.served <= cap


def test_only_old_parquet_mr_writers_are_padded():
    assert parquet._padding("parquet-mr version 1.2.8 (build abc)") == 100
    assert parquet._padding("parquet-mr version 1.2.9 (build abc)") == 0
    assert parquet._padding("parquet-mr version 1.12.3 (build abc)") == 0
    assert parquet._padding("parquet-cpp-arrow version 1.0.0") == 0
    assert parquet._padding(None) == 0


# --- Planning ---------------------------------------------------------------------


class PlannedFooter:
    """A footer as `_plan` sees one: each row group's rows and byte range."""

    held = []

    def __init__(self, costs, rows=10):
        self.groups, at = [], 4
        for cost in costs:
            self.groups.append((rows, at, at + cost))
            at += cost


def test_uneven_costs_still_give_two_shards():
    # Shard 0's row group would take all the room shard 1's cheapest one needs.
    footers = [PlannedFooter([6_634]), PlannedFooter([1_834]), PlannedFooter([1_834])]
    budget = parquet._Budget(6_734)

    plan = parquet._plan(footers, budget)

    assert len(plan) >= 2
    assert budget.spent <= 6_734


def test_a_shard_that_alone_fits_is_still_read():
    footers = [PlannedFooter([5_000, 5_000]), PlannedFooter([9_000])]
    budget = parquet._Budget(6_000)

    plan = parquet._plan(footers, budget)

    assert [(footers.index(f), groups) for f, groups in plan] == [(0, [0])]


def test_a_shard_whose_first_group_is_dear_gives_its_cheapest_to_make_two():
    # Taking either shard's first group leaves no room for the other's: the
    # first shard gives its cheapest instead.
    footers = [PlannedFooter([5_000, 900]), PlannedFooter([5_000, 900])]
    budget = parquet._Budget(5_500)

    plan = parquet._plan(footers, budget)

    assert len(plan) == 2


def test_the_plan_stops_at_its_row_target():
    footers = [PlannedFooter([100] * 50) for _ in range(4)]
    budget = parquet._Budget(10**9)

    plan = parquet._plan(footers, budget, target=95)

    assert sum(len(groups) for _, groups in plan) == 10
    assert len(plan) == 4
    assert budget.spent == 1_000


def test_row_groups_whose_padded_ranges_overlap_are_budgeted_as_their_union():
    # Two neighbouring groups padded 100 bytes into each other: 216 bytes in all, not 236.
    footer = PlannedFooter([])
    footer.groups = [(10, 4, 120), (10, 100, 220)]
    budget = parquet._Budget(216)

    plan = parquet._plan([footer], budget)

    assert [groups for _, groups in plan] == [[0, 1]]
    assert budget.spent == 216


def test_overlapping_padded_ranges_are_budgeted_once():
    assert parquet._uncovered([(0, 10), (20, 30)], 5, 25) == 10
    assert parquet._uncovered([], 5, 25) == 20
    assert parquet._uncovered([(0, 100)], 5, 25) == 0


def test_the_row_target_leaves_headroom_for_sampling():
    assert parquet.row_target(5000) == 15_000
    assert parquet.row_target(100) == 1_100


def test_an_import_with_a_row_cap_reads_far_less_than_the_byte_cap(hf, tmp_path):
    # FineFineWeb's shape at a smaller scale: an import keeping 5000 rows reads
    # enough row groups for them and its headroom, not the whole cap.
    shards = fine_shards(n=10, groups=12, rows_per_group=200, text_bytes=64)
    hf.add_dataset(FINE, "default", "train", shards)
    cap = 50_000_000

    fetched = parquet.fetch_shards(FINE, "default", "train", cap, str(tmp_path / "work"), max_rows=5000, sleep=lambda s: None)

    rows = rows_in(fetched.paths)
    assert parquet.row_target(5000) <= rows < parquet.row_target(5000) + 10 * 200
    assert hf.served < cap // 10
    assert hf.served < sum(len(s) for s in shards)
    assert len(shards_read(fetched.paths)) == 10


def test_a_fetch_without_a_row_cap_fills_the_byte_cap(hf, tmp_path):
    # A copy passes no row cap: it reads as much as the cap allows.
    shards = fine_shards(n=4, groups=12, rows_per_group=200, text_bytes=64)
    hf.add_dataset(FINE, "default", "train", shards)

    fetched = fetch(tmp_path, FINE, "default", "train", max_bytes=10**9)

    assert rows_in(fetched.paths) == 4 * 12 * 200


def test_the_import_passes_its_sample_size_to_a_fetch_that_takes_it(app, fine, tmp_path):
    seen = {}

    def real_fetch(external_id, config, split, max_bytes, dest_dir, *, expected_revision=None, max_rows=None):
        seen["max_rows"] = max_rows
        return parquet.fetch_shards(
            external_id, config, split, SCALED_CAP, dest_dir,
            expected_revision=expected_revision, max_rows=max_rows, sleep=lambda s: None,
        )

    found, _ = run_fine_import(app, tmp_path, real_fetch, sample_size=20)

    assert found["status"] == "completed", found
    assert seen == {"max_rows": 20}


def test_a_host_fetch_without_the_row_cap_keyword_still_works(app, fine, tmp_path):
    def host_fetch(external_id, config, split, max_bytes, dest_dir, *, expected_revision=None):
        return parquet.fetch_shards(
            external_id, config, split, SCALED_CAP, dest_dir, expected_revision=expected_revision, sleep=lambda s: None,
        )

    found, _ = run_fine_import(app, tmp_path, host_fetch, sample_size=20)

    assert found["status"] == "completed", found
    assert found["rows"] == 20


def test_an_import_of_a_file_that_is_not_parquet_is_not_importable(app, hf, tmp_path, monkeypatch):
    # Asking again cannot help, so it is a refusal, not Hugging Face being busy,
    # and no copy stands in for it.
    from dataset_library import imports, snapshots

    monkeypatch.setattr(snapshots, "fallback_for", lambda *a, **k: pytest.fail("no copy for a file that is not Parquet"))
    hf.add_dataset(FINE, "default", "train", [b"not a parquet file at all, just text"])

    def real_fetch(external_id, config, split, max_bytes, dest_dir, **kwargs):
        return parquet.fetch_shards(external_id, config, split, SCALED_CAP, dest_dir, sleep=lambda s: None, **kwargs)

    found, _ = run_fine_import(app, tmp_path, real_fetch, sample_size=20)

    assert (found["status"], found["code"], found["error"]) == ("failed", "not_importable", parquet.NOT_PARQUET)


def test_an_import_whose_range_is_refused_still_falls_back_to_a_copy(app, fine, tmp_path, monkeypatch):
    # NO_RANGES stays Hugging Face being unavailable, so a copy may stand in (#121).
    from dataset_library import snapshots

    asked = []
    monkeypatch.setattr(snapshots, "fallback_for", lambda *a, **k: asked.append(True))
    fine.ignore_ranges = True

    def real_fetch(external_id, config, split, max_bytes, dest_dir, **kwargs):
        return parquet.fetch_shards(external_id, config, split, SCALED_CAP, dest_dir, sleep=lambda s: None, **kwargs)

    found, _ = run_fine_import(app, tmp_path, real_fetch, sample_size=20)

    assert asked == [True]
    assert (found["status"], found["code"]) == ("failed", "provider_busy")


def test_takes_tells_a_keyword_a_callable_accepts():
    from dataset_library.imports import _takes

    assert _takes(lambda a, *, max_rows=None: 0, "max_rows")
    assert _takes(lambda a, **kwargs: 0, "max_rows")
    assert _takes(lambda a, max_rows=None: 0, "max_rows")
    assert not _takes(lambda a, *, expected_revision=None: 0, "max_rows")
    assert not _takes(lambda a, *max_rows: 0, "max_rows")


def test_an_import_queued_at_another_revision_reads_no_footer(fine, tmp_path):
    # #174: the refusal comes before any byte of a shard, footers included.
    with pytest.raises(ImportFailed) as refused:
        fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP, expected_revision="a" * 40)

    assert refused.value.code == "changed"
    assert fine.served == 0 and fine.ranges == []


def test_the_build_check_names_the_shards_whose_footers_are_read(fine, tmp_path):
    fine.build_sizes = {"default/train/0005.parquet": 1}

    with pytest.raises(ProviderUnavailable) as waiting:
        fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP)

    assert str(waiting.value) == parquet.REBUILDING
    assert fine.served == 0


def test_a_dataset_the_viewer_cannot_convert_is_refused_plainly(hf, tmp_path):
    with pytest.raises(ImportFailed) as refused:
        fetch(tmp_path, "jfrenz/legalglue", "default", "train")

    assert refused.value.code == "not_importable"
    assert "Hugging Face cannot convert" in refused.value.message


def test_429_and_5xx_are_retried_three_times_then_hugging_face_is_busy(hf, tmp_path):
    hf.failures = [503, 429, 502]
    assert fetch(tmp_path, "coastalcph/lex_glue", "ledgar", "train").paths

    hf.failures = [503, 503, 503, 503]
    with pytest.raises(ProviderUnavailable):
        fetch(tmp_path, "coastalcph/lex_glue", "ledgar", "train")


def test_an_import_refused_for_its_revision_downloads_nothing_and_uses_no_copy(app, hf, tmp_path, monkeypatch):
    # Through run_import with the real fetch: the queued revision reaches
    # fetch_shards, which refuses before any download (#174), and the refusal is
    # not a ProviderError, so no copy stands in for it.
    from dataset_library import imports, snapshots
    from dataset_library.models import LibraryImport

    def no_copy(*args, **kwargs):
        raise AssertionError("a refusal must not fall back to a copy")

    monkeypatch.setattr(snapshots, "fallback_for", no_copy)

    def real_fetch(external_id, config, split, max_bytes, dest_dir, **kwargs):
        return parquet.fetch_shards(external_id, config, split, max_bytes, dest_dir, sleep=lambda s: None, **kwargs)

    with app.app_context():
        session = app.db.session
        record = LibraryImport(
            owner_id="alice", provider="huggingface", external_id=DWARF,
            config="classes", split="reference", revision=DWARF_HEAD,
            text_column="text", label_column="label", task_type="classification",
            sample_size=2, max_rows=1000, name="Doppeldwarf", license="cc-by-4.0",
            url=f"https://huggingface.co/datasets/{DWARF}", status="queued",
        )
        session.add(record)
        session.commit()
        ref = record.id

        imports.run_import(
            ref, session=session, fetch_shards=real_fetch,
            create_dataset=lambda *args, **kwargs: pytest.fail("no Dataset is made"),
            data_dir=str(tmp_path / "data"), seed=7,
        )
        found = session.get(LibraryImport, ref).to_dict()

    assert (found["status"], found["code"], found["error"]) == ("failed", "conversion_behind", imports.BEHIND)
    assert found["revision"] == DWARF_HEAD
    assert hf.downloads == []


# --- Council round 2 (#178) ---------------------------------------------------------


def plain_shard(shard, text_bytes, extra=0, pad=0, features=True):
    """A one-row shard written plainly (no compression, dictionary or
    statistics), so its size moves byte for byte with `extra` characters of
    text and its footer with `pad`, a schema metadata value; that is how the
    reviewers' exact sizes are made. `features` adds the `huggingface`
    metadata, which makes the footer heavy."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    text = f"s{shard}-r0 " + random.Random(shard * 1_000_003).randbytes(text_bytes // 2).hex() + "x" * extra
    table = pa.table({"text": pa.array([text], pa.string()), "label": pa.array([0], pa.int64())})
    metadata = {}
    if features:
        metadata["huggingface"] = json.dumps({"info": {"features": {
            "text": {"dtype": "string", "_type": "Value"}, "label": {"names": ["a", "b"], "_type": "ClassLabel"},
        }}})
    if pad:
        metadata["pad"] = "p" * pad
    table = table.replace_schema_metadata(metadata)
    sink = io.BytesIO()
    pq.write_table(table, sink, compression="none", use_dictionary=False, write_statistics=False)
    return sink.getvalue()


def footer_length(data):
    return int.from_bytes(data[-8:-4], "little")


def read_bytes(hf):
    """Every body byte read, from range answers and redirects alike."""
    return sum(r.raw.read_bytes for r in hf.responses + hf.redirect_responses)


# The row target, item 1.


def test_a_first_row_group_past_the_target_still_leaves_room_for_a_second_shard():
    # Row groups of 20,000 rows and a 5000-row import: one group passes the
    # 15,000-row target, and the sample would be one shard's.
    footers = [PlannedFooter([1_000] * 4, rows=20_000) for _ in range(10)]
    budget = parquet._Budget(10**9)

    plan = parquet._plan(footers, budget, target=parquet.row_target(5000))

    assert len(plan) == 2
    assert [len(groups) for _, groups in plan] == [1, 1]
    assert budget.spent == 2_000


def test_the_target_ends_the_reading_with_one_shard_when_no_other_fits():
    footers = [PlannedFooter([1_000] * 4, rows=20_000), PlannedFooter([9_000], rows=20_000)]
    budget = parquet._Budget(5_000)

    plan = parquet._plan(footers, budget, target=parquet.row_target(5000))

    assert [(footers.index(f), groups) for f, groups in plan] == [(0, [0])]


def test_an_import_whose_first_row_group_passes_the_target_reads_rows_from_two_shards(hf, tmp_path):
    hf.add_dataset(FINE, "default", "train", [shard_with_groups(0, [1200, 1200]), shard_with_groups(1, [1200, 1200])])
    cap = 10**7

    fetched = parquet.fetch_shards(FINE, "default", "train", cap, str(tmp_path / "work"), max_rows=100, sleep=lambda s: None)

    assert shards_read(fetched.paths) == {"s0", "s1"}
    assert rows_in(fetched.paths) == 2400
    assert hf.served == read_bytes(hf) <= cap
    assert_no_byte_read_twice(hf.ranges)


def texts_shard(texts_by_group):
    """A shard of the given texts, one row group per list, with ClassLabel names."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    features = {"text": {"dtype": "string", "_type": "Value"}, "label": {"names": ["a", "b"], "_type": "ClassLabel"}}
    sink = io.BytesIO()
    writer = None
    for texts in texts_by_group:
        table = pa.table({"text": pa.array(texts, pa.string()), "label": pa.array([r % 2 for r in range(len(texts))], pa.int64())})
        table = table.replace_schema_metadata({"huggingface": json.dumps({"info": {"features": features}})})
        writer = writer or pq.ParquetWriter(sink, table.schema)
        writer.write_table(table)
    writer.close()
    return sink.getvalue()


def test_an_import_whose_first_row_group_is_blank_keeps_rows_from_the_next_shard(app, hf, tmp_path):
    # Shard 0's first row group is 1,200 rows without text, past the 1,100-row
    # target of a 100-row import; shard 1 has 120 rows with text.
    blank = texts_shard([[""] * 1200, [""] * 1200])
    usable = texts_shard([[f"s1-r{r} a clause" for r in range(120)]])
    hf.add_dataset(FINE, "default", "train", [blank, usable])

    def real_fetch(external_id, config, split, max_bytes, dest_dir, **kwargs):
        return parquet.fetch_shards(external_id, config, split, 10**7, dest_dir, sleep=lambda s: None, **kwargs)

    found, create = run_fine_import(app, tmp_path, real_fetch, sample_size=100)

    assert found["status"] == "completed", found
    assert found["rows"] == 100
    assert {row["text"].split("-", 1)[0] for row in create.calls[0]["rows"]} == {"s1"}


# Every byte read counts, item 2.


def test_a_redirect_is_followed_without_reading_its_body(fine, tmp_path):
    # The Hub answers a pinned resolve/ URL with a 302 carrying a body, then its CDN answers 206.
    fine.redirects = 1
    fine.redirect_body = b"r" * 1_000

    fetched = fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP)

    assert fetched.paths
    asked = sum(last + 1 - first for _, first, last in fine.ranges)
    assert read_bytes(fine) == fine.served == asked <= SCALED_CAP
    assert len(fine.redirect_responses) == len(fine.ranges)
    assert all(r.closed and r.raw.read_bytes == 0 for r in fine.redirect_responses)
    # The CDN is asked for the same range, uncompressed.
    assert all(url.startswith("https://cdn.example/") for url, _, _ in fine.ranges)
    assert all(h["Range"].startswith("bytes=") and h["Accept-Encoding"] == "identity" for h in fine.asked_headers)


@pytest.mark.parametrize("redirects", [5, 6])
def test_at_most_five_redirects_are_followed(fine, tmp_path, redirects):
    fine.redirects = redirects
    fine.redirect_body = b"r" * 1_000

    if redirects == 5:
        assert fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP).paths
    else:
        with pytest.raises(ProviderUnavailable) as refused:
            fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP)
        assert str(refused.value) == parquet.BAD_REDIRECT
        assert fine.served == 0
    assert all(r.closed and r.raw.read_bytes == 0 for r in fine.redirect_responses)


@pytest.mark.parametrize("location", ["http://cdn.example/resolve/x", "ftp://cdn.example/x", ""])
def test_a_redirect_anywhere_but_https_is_refused_unread(fine, tmp_path, location):
    fine.redirects = 1
    fine.redirect_body = b"r" * 1_000
    fine.redirect_location = location

    with pytest.raises(ProviderUnavailable) as refused:
        fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP)

    assert str(refused.value) == parquet.BAD_REDIRECT
    assert read_bytes(fine) == 0
    assert all(r.closed for r in fine.redirect_responses)


def test_an_overlong_answer_with_no_length_is_not_read_past_the_cap(hf, tmp_path):
    # The reviewer's case: a 1,087-byte cap, exactly the shard's trailer,
    # footer and row group. Its row group is answered with more than asked and
    # no Content-Length; reading one byte past it would pass the cap, so
    # exactly the range is read, as its exact Content-Range names.
    shard = plain_shard(0, text_bytes=522, extra=8, features=False)
    assert len(shard) - 4 == 1_087
    hf.add_dataset(FINE, "default", "train", [shard])
    data_end = len(shard) - 8 - footer_length(shard)
    hf.extra_bytes = lambda first, last, size: 100 if last < data_end else 0
    hf.send_length = False

    fetched = fetch(tmp_path, FINE, "default", "train", max_bytes=1_087)

    assert rows_in(fetched.paths) == 1
    assert read_bytes(hf) <= 1_087


def test_an_overlong_answer_with_no_length_is_still_refused_while_the_cap_has_room(hf, tmp_path):
    shard = plain_shard(0, text_bytes=522, extra=8, features=False)
    hf.add_dataset(FINE, "default", "train", [shard])
    hf.extra_bytes = 100
    hf.send_length = False

    with pytest.raises(ProviderUnavailable) as refused:
        fetch(tmp_path, FINE, "default", "train", max_bytes=10_000)

    assert str(refused.value) == parquet.WRONG_RANGE
    assert read_bytes(hf) <= 10_000


# Footers, cheapest first, item 3.


def test_an_affordable_shard_listed_after_a_dear_one_is_imported(hf, tmp_path):
    # Shard 0's footer is the dearer and none of its row groups fits; reading
    # it first would leave no room for shard 1's row group.
    dear = parquet_shard(0, groups=8, rows_per_group=8, text_bytes=4400)
    cheap = plain_shard(1, text_bytes=16_000)
    hf.add_dataset(FINE, "default", "train", [dear, cheap])
    cap = 16 + footer_length(cheap) + measured(cheap)[1][0]
    assert footer_length(dear) > footer_length(cheap) and footer_length(dear) < cap
    assert min(measured(dear)[1]) > cap

    fetched = fetch(tmp_path, FINE, "default", "train", max_bytes=cap)

    assert shards_read(fetched.paths) == {"s1"}
    assert read_bytes(hf) <= cap
    assert_no_byte_read_twice(hf.ranges)


def test_a_cap_just_over_one_shard_s_whole_file_imports_it(hf, tmp_path):
    # The reviewer's case: a 10,799-byte cap, and shard 2's whole file is
    # 10,783 bytes. Shards 0 and 1 have dearer footers and row groups over the cap.
    dear = [plain_shard(n, text_bytes=12_000) for n in (0, 1)]
    affordable = plain_shard(2, text_bytes=10_000, extra=39, pad=47, features=False)
    assert len(affordable) == 10_783
    assert all(footer_length(d) > footer_length(affordable) and measured(d)[1][0] > 10_799 for d in dear)
    hf.add_dataset(FINE, "default", "train", dear + [affordable])

    fetched = fetch(tmp_path, FINE, "default", "train", max_bytes=10_799)

    assert shards_read(fetched.paths) == {"s2"}
    assert read_bytes(hf) == 24 + len(affordable) - 12 <= 10_799


def test_a_first_footer_that_would_crowd_out_a_later_shard_is_not_read(hf, tmp_path):
    # The reviewer's case: a 1,470-byte cap, an 896-byte first footer over a
    # row group that does not fit, and a later shard costing 1,454 bytes.
    first = plain_shard(0, text_bytes=3_000, pad=152, features=False)
    later = plain_shard(1, text_bytes=880, extra=17, features=False)
    assert footer_length(first) == 896 and measured(first)[1][0] > 1_470
    assert 8 + footer_length(later) + measured(later)[1][0] == 1_454
    hf.add_dataset(FINE, "default", "train", [first, later])

    fetched = fetch(tmp_path, FINE, "default", "train", max_bytes=1_470)

    assert shards_read(fetched.paths) == {"s1"}
    assert read_bytes(hf) == 8 + 1_454


def test_two_metadata_heavy_shards_are_both_read_without_a_row_target(hf, tmp_path):
    # The reviewer's case: two 1,268-byte shards, mostly footer, under a
    # 2,536-byte cap. A copy passes no row target, so this is what it reads.
    shards = [plain_shard(n, text_bytes=16, extra=38, pad=96) for n in (0, 1)]
    assert [len(s) for s in shards] == [1_268, 1_268]
    assert footer_length(shards[0]) > measured(shards[0])[1][0]
    hf.add_dataset(FINE, "default", "train", shards)

    fetched = fetch(tmp_path, FINE, "default", "train", max_bytes=2_536)

    assert shards_read(fetched.paths) == {"s0", "s1"}
    assert read_bytes(hf) <= 2_536
    assert_no_byte_read_twice(hf.ranges)


# Old writers' padding, item 4.


def test_an_old_writer_s_last_row_group_fits_a_cap_of_the_file_s_unique_bytes(hf, tmp_path):
    # The last row group's padding runs into the footer, already read: it is
    # neither read again nor counted again.
    shard = with_created_by(parquet_shard(0, groups=3, rows_per_group=2, text_bytes=64), "parquet-mr version 1.2.8 (build 1)")
    hf.add_dataset(FINE, "default", "train", [shard])
    cap = len(shard) - 4  # all but the leading magic

    fetched = fetch(tmp_path, FINE, "default", "train", max_bytes=cap)

    assert rows_in(fetched.paths) == 6
    assert read_bytes(hf) == hf.served == cap
    assert_no_byte_read_twice(hf.ranges)


def test_an_old_writer_s_shard_is_read_with_no_byte_fetched_twice(hf, tmp_path):
    shard = with_created_by(parquet_shard(0, groups=6, rows_per_group=2, text_bytes=64), "parquet-mr")
    hf.add_dataset(FINE, "default", "train", [shard])

    fetched = fetch(tmp_path, FINE, "default", "train", max_bytes=10**9)

    assert rows_in(fetched.paths) == 12
    assert_no_byte_read_twice(hf.ranges)


WRITERS_AS_ARROW_READS_THEM = [
    "parquet-mr",
    "parquet-mr version 1.2",
    "parquet-mr version 1",
    "parquet-mr version 1.2.8 (build abc)",
    "parquet-mr version 1.2.8-rc1",
    "parquet-mr version abc",
    "parquet-mr version 1.x.9",
    "  parquet-mr  version 1.2.8",
    "parquet-mr version 1.2.9 (build abc)",
    "parquet-mr version 1.2.10",
    "parquet-mr version 1.3",
    "parquet-mr-x version 1.2.8",
    "parquet-cpp version 1.0.0",
    "impala version 1.0",
]


@pytest.mark.parametrize("created_by", WRITERS_AS_ARROW_READS_THEM)
def test_the_padding_is_what_pyarrow_reads_past_a_column_chunk(created_by):
    # Not a copy of Arrow's rule: what this pyarrow really reads for a footer
    # naming this writer.
    import pyarrow.parquet as pq

    shard = with_created_by(parquet_shard(0, groups=3, rows_per_group=2, text_bytes=64), created_by)
    metadata = pq.read_metadata(io.BytesIO(shard))
    assert metadata.created_by == created_by
    reads = []

    class Recording(io.RawIOBase):
        def __init__(self):
            self.pos = 0

        def readable(self):
            return True

        def seekable(self):
            return True

        def tell(self):
            return self.pos

        def seek(self, offset, whence=io.SEEK_SET):
            self.pos = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: len(shard)}[whence] + offset
            return self.pos

        def readinto(self, buffer):
            n = min(len(buffer), len(shard) - self.pos)
            buffer[:n] = shard[self.pos : self.pos + n]
            reads.append(self.pos + n)
            self.pos += n
            return n

    pq.ParquetFile(Recording(), metadata=metadata, pre_buffer=False).read_row_group(0)

    _, _, end = parquet._group_range(metadata.row_group(0), len(shard))
    assert max(reads) - end == parquet._padding(created_by)


def test_only_old_parquet_mr_writers_are_padded_as_arrow_parses_them():
    assert parquet._padding("parquet-mr") == 100
    assert parquet._padding("parquet-mr version 1.2") == 100
    assert parquet._padding("parquet-mr version ") == 100
    assert parquet._padding("parquet-mr version 1.2.9") == 0
    assert parquet._padding("") == 0
    assert parquet._writer_version("parquet-mr version 1.x.9") == ("parquet-mr", (1, 0, 0))
    assert parquet._writer_version("parquet-mr version 1.2.8abc-rc1+x (build y)") == ("parquet-mr", (1, 2, 8))


# Our own failures are not "not Parquet", item 5.


class FailingSpool:
    """A spool file whose reads fail as a failing disk does."""

    def __init__(self, file):
        self._file = file

    def __getattr__(self, name):
        return getattr(self._file, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return self._file.__exit__(*exc)

    def read(self, n=-1):
        import errno

        raise OSError(errno.EIO, "Input/output error")


def test_a_spool_that_cannot_be_read_raises_its_own_error(fine, tmp_path, monkeypatch):
    import errno

    monkeypatch.setattr(parquet, "open", lambda path, mode: FailingSpool(open(path, mode)), raising=False)

    with pytest.raises(OSError) as failed:
        fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP)

    assert failed.value.errno == errno.EIO
    assert not isinstance(failed.value, (NotParquet, ProviderUnavailable))
    assert list((tmp_path / "work").iterdir()) == []


def fetch_one_byte_short(monkeypatch):
    # A plan bug: each range starts one byte after what pyarrow will read.
    real = parquet._merged
    monkeypatch.setattr(parquet, "_merged", lambda footer, groups: [[start + 1, end] for start, end in real(footer, groups)])


def test_a_read_of_bytes_never_fetched_is_our_own_error(fine, tmp_path, monkeypatch):
    fetch_one_byte_short(monkeypatch)

    with pytest.raises(RangeNotFetched):
        fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP)

    assert not issubclass(RangeNotFetched, (OSError, ProviderUnavailable))
    assert list((tmp_path / "work").iterdir()) == []


def test_an_import_that_reads_bytes_never_fetched_fails_as_unexpected_and_logs_it(app, fine, tmp_path, monkeypatch, caplog):
    fetch_one_byte_short(monkeypatch)

    def real_fetch(external_id, config, split, max_bytes, dest_dir, **kwargs):
        return parquet.fetch_shards(external_id, config, split, SCALED_CAP, dest_dir, sleep=lambda s: None, **kwargs)

    with caplog.at_level("ERROR", logger="dataset_library"):
        found, create = run_fine_import(app, tmp_path, real_fetch, sample_size=20)

    assert (found["status"], found["code"]) == ("failed", "unexpected")
    assert create.calls == []
    logged = [r for r in caplog.records if r.getMessage().startswith("library.import.error")]
    assert logged and logged[0].exc_info[0] is RangeNotFetched
