"""Fetching a split's Parquet shards from Hugging Face (#113), from recorded listings."""
import io
import json
import pathlib
from urllib.parse import unquote

import pytest

from dataset_library.errors import ProviderUnavailable
from dataset_library.imports import BEHIND, CHANGED, ImportFailed
from dataset_library.providers import parquet

LISTINGS = json.loads((pathlib.Path(__file__).parent / "fixtures" / "parquet" / "listings.json").read_text())
HUB = json.loads((pathlib.Path(__file__).parent / "fixtures" / "hub" / "datasets.json").read_text())
REFS = json.loads((pathlib.Path(__file__).parent / "fixtures" / "hub" / "refs.json").read_text())
MB = 1_000_000


class Response:
    def __init__(self, status_code, body=None, content=b"", headers=None, served=None):
        self.status_code = status_code
        self._body = body
        self._content = content
        self.headers = headers or {}
        self._served = served  # called with each chunk's length as it is sent
        self.closed = False

    def json(self):
        return self._body

    def iter_content(self, chunk_size):
        for i in range(0, len(self._content), chunk_size):
            chunk = self._content[i : i + chunk_size]
            if self._served:
                self._served(len(chunk))
            yield chunk

    def close(self):
        self.closed = True


def parquet_shard(shard, groups, rows_per_group, text_bytes, labels=("a", "b")):
    """A real Parquet shard as Hugging Face writes one, uncompressed so its size
    is predictable: `groups` row groups of `rows_per_group` rows, each text
    naming its shard and row ("s3-r17 ...") so a test can tell where a row came
    from. ClassLabel names are in the schema metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = groups * rows_per_group
    texts = [f"s{shard}-r{r} " + format(r * 7919 + shard, "x").rjust(8, "0") * (text_bytes // 8) for r in range(rows)]
    features = {
        "text": {"dtype": "string", "_type": "Value"},
        "label": {"names": list(labels), "_type": "ClassLabel"},
    }
    table = pa.table({"text": texts, "label": [r % len(labels) for r in range(rows)]})
    table = table.replace_schema_metadata({"huggingface": json.dumps({"info": {"features": features}})})
    sink = io.BytesIO()
    pq.write_table(table, sink, row_group_size=rows_per_group, compression="none", use_dictionary=False, write_statistics=False)
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
        self.extra_bytes = 0  # sent past the asked range
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
        assert stream, "shards are streamed"
        if url not in self.downloads:
            self.downloads.append(url)
        data = self.file(url.split("/resolve/", 1)[1].split("/", 1)[1])
        asked = (headers or {}).get("Range")
        if self.ignore_ranges or asked is None:
            return Response(200, content=data, served=self.count)
        first, last = (int(n) for n in asked.removeprefix("bytes=").split("-"))
        self.ranges.append((url, first, last))
        content = data[first : last + 1 + self.extra_bytes]
        return Response(206, content=content, headers={"Content-Range": f"bytes {first}-{last}/{len(data)}"}, served=self.count)

    def count(self, n):
        self.served += n


@pytest.fixture
def hf(monkeypatch):
    fake = FakeHuggingFace()
    monkeypatch.setattr(parquet.requests, "get", fake.get)
    monkeypatch.setattr(parquet.requests, "post", fake.post)
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
# over the cap; their row groups are well under it.
SCALED_CAP = 200_000
FINE = "m-a-p/FineFineWeb"


def fine_shards(n=10, groups=12, rows_per_group=40, text_bytes=544):
    return [parquet_shard(s, groups, rows_per_group, text_bytes) for s in range(n)]


@pytest.fixture
def fine(hf):
    hf.add_dataset(FINE, "default", "train", fine_shards())
    return hf


def shards_read(paths):
    """Which shards the local files' rows came from, by the shard each text names."""
    import pyarrow.parquet as pq

    return {text.split("-", 1)[0] for p in paths for text in pq.read_table(p).column("text").to_pylist()}


def test_a_dataset_whose_every_shard_is_over_the_cap_imports_a_sample(fine, tmp_path):
    import pyarrow.parquet as pq

    sizes = [len(fine.file(f"default/train/{n:04}.parquet")) for n in range(10)]
    assert min(sizes) > SCALED_CAP

    fetched = fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP)

    rows = sum(pq.ParquetFile(p).metadata.num_rows for p in fetched.paths)
    assert rows >= 200
    assert fetched.revision == "1" * 40
    assert all(f"/resolve/{'2' * 40}/" in url for url in fine.downloads)


@pytest.mark.parametrize("cap", [SCALED_CAP, 60_000, 30_000, 1_000_000, 5_000_000])
def test_an_import_reads_at_most_the_cap_footers_and_row_groups_included(fine, tmp_path, cap):
    fetch(tmp_path, FINE, "default", "train", max_bytes=cap)

    assert 0 < fine.served <= cap
    # Every range asked was a footer or row groups: nothing was read twice.
    assert fine.served == sum(last + 1 - first for _, first, last in fine.ranges)


def test_a_cap_over_the_whole_split_reads_every_row(fine, tmp_path):
    import pyarrow.parquet as pq

    fetched = fetch(tmp_path, FINE, "default", "train", max_bytes=10**9)

    assert sum(pq.ParquetFile(p).metadata.num_rows for p in fetched.paths) == 10 * 12 * 40
    assert fine.served < sum(len(fine.file(f"default/train/{n:04}.parquet")) for n in range(10))


def test_the_sample_draws_from_every_shard_the_cap_allows(fine, tmp_path):
    # Ten footers of about 2 KB leave room for seven row groups: one from each
    # of seven shards, not seven from the first.
    fetched = fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP)

    assert len(fetched.paths) == 7
    assert shards_read(fetched.paths) == {f"s{n}" for n in range(7)}


def test_a_shard_s_row_groups_are_spread_across_it(fine, tmp_path):
    # Room for 21 row groups across ten shards: the first shard gives three,
    # from its start, its middle and its first quarter, not its first three.
    import pyarrow.parquet as pq

    fetched = fetch(tmp_path, FINE, "default", "train", max_bytes=500_000)

    first = pq.read_table(fetched.paths[0]).column("text").to_pylist()
    assert {int(t.split("-r", 1)[1].split(" ", 1)[0]) // 40 for t in first} == {0, 6, 3}


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


def test_an_import_samples_rows_from_several_shards(app, fine, tmp_path):
    # Through run_import with the real fetch, at the default sample method: the
    # random sample (#113) is not just the head of shard 0.
    from dataset_library import imports
    from dataset_library.models import LibraryImport
    from test_import import FakeCreateDataset

    def real_fetch(external_id, config, split, max_bytes, dest_dir, **kwargs):
        return parquet.fetch_shards(external_id, config, split, SCALED_CAP, dest_dir, sleep=lambda s: None, **kwargs)

    create = FakeCreateDataset()
    with app.app_context():
        session = app.db.session
        record = LibraryImport(
            owner_id="alice", provider="huggingface", external_id=FINE,
            config="default", split="train", revision="1" * 40,
            text_column="text", label_column="label", task_type="classification",
            sample_size=200, max_rows=1000, name="FineFineWeb", license="odc-by",
            url=f"https://huggingface.co/datasets/{FINE}", status="queued",
        )
        session.add(record)
        session.commit()
        ref = record.id
        imports.run_import(ref, session=session, fetch_shards=real_fetch, create_dataset=create, data_dir=str(tmp_path / "data"), seed=7)
        found = session.get(LibraryImport, ref).to_dict()

    assert found["status"] == "completed", found
    assert found["rows"] == 200
    assert len({row["text"].split("-", 1)[0] for row in create.calls[0]["rows"]}) > 1
    assert fine.served <= SCALED_CAP


def test_a_row_group_over_the_cap_is_too_large_to_sample(hf, tmp_path):
    # Only the footers are read: no row group fits, and none is downloaded.
    hf.add_dataset(FINE, "default", "train", fine_shards(n=2, groups=2, rows_per_group=400))

    with pytest.raises(ImportFailed) as refused:
        fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP)

    assert refused.value.code == "too_large"
    assert refused.value.message == parquet.TOO_LARGE
    footers = sum(last + 1 - first for _, first, last in hf.ranges)
    assert hf.served == footers < 10_000


def test_a_footer_over_the_cap_is_too_large_to_sample(fine, tmp_path):
    with pytest.raises(ImportFailed) as refused:
        fetch(tmp_path, FINE, "default", "train", max_bytes=1_000)

    assert refused.value.code == "too_large"
    assert refused.value.message == parquet.TOO_LARGE
    # Only the 8 bytes that gave the first footer's length.
    assert fine.served == 8 and len(fine.ranges) == 1


def test_a_server_that_ignores_range_is_closed_before_the_shard_is_read(fine, tmp_path):
    fine.ignore_ranges = True

    with pytest.raises(ProviderUnavailable) as refused:
        fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP)

    assert str(refused.value) == parquet.NO_RANGES
    assert fine.served == 0


def test_a_server_that_sends_more_than_the_range_is_refused(fine, tmp_path):
    fine.extra_bytes = 100

    with pytest.raises(ProviderUnavailable) as refused:
        fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP)

    assert str(refused.value) == parquet.WRONG_RANGE
    # The trailer ends the file, so it came whole; the footer after it came
    # with 100 bytes too many, and nothing was read after that.
    assert len(fine.ranges) == 2


def test_a_server_that_names_another_range_is_refused(fine, tmp_path, monkeypatch):
    real = fine.get

    def wrong(url, **kwargs):
        response = real(url, **kwargs)
        if response.status_code == 206:
            response.headers["Content-Range"] = "bytes 0-7/1"
        return response

    monkeypatch.setattr(parquet.requests, "get", wrong)

    with pytest.raises(ProviderUnavailable) as refused:
        fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP)

    assert str(refused.value) == parquet.WRONG_RANGE


def test_a_file_that_is_not_parquet_is_refused(hf, tmp_path):
    hf.add_dataset(FINE, "default", "train", [b"not a parquet file at all, just text"])

    with pytest.raises(ProviderUnavailable) as refused:
        fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP)

    assert str(refused.value) == parquet.NOT_PARQUET


def test_a_file_too_short_to_hold_a_footer_is_refused(hf, tmp_path):
    hf.add_dataset(FINE, "default", "train", [b"PAR1"])

    with pytest.raises(ProviderUnavailable) as refused:
        fetch(tmp_path, FINE, "default", "train", max_bytes=SCALED_CAP)

    assert str(refused.value) == parquet.NOT_PARQUET
    assert hf.served == 0


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
