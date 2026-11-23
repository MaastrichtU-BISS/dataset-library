"""Fetching a split's Parquet shards from Hugging Face (#113), from recorded listings."""
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
    def __init__(self, status_code, body=None, content=b"", headers=None):
        self.status_code = status_code
        self._body = body
        self._content = content
        self.headers = headers or {}

    def json(self):
        return self._body

    def iter_content(self, chunk_size):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]

    def close(self):
        pass


class FakeHuggingFace:
    """The viewer's /parquet, the Hub's datasets and refs, from recordings; a
    shard's bytes are its URL, so a test can tell which shards were downloaded
    from where."""

    def __init__(self):
        self.downloads = []
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

    def post(self, url, data=None, timeout=None, **kwargs):
        # paths-info at the pinned build, as the listings recorded its files.
        assert url.startswith(parquet.HUB) and "/paths-info/" in url
        self.hub_calls.append(url)
        dataset = url.removeprefix(parquet.HUB).split("/paths-info/", 1)[0]
        sizes = {
            parquet._in_build(s["url"]): s["size"]
            for key, recorded in LISTINGS.items() if key.startswith(dataset + "?")
            for s in (recorded["body"].get("parquet_files") or [])
        }
        sizes.update(self.build_sizes)
        body = [{"type": "file", "path": p, "size": sizes[p]} for p in data["paths"] if p in sizes and p not in self.build_missing]
        return Response(self.build_status, body)

    def get(self, url, params=None, timeout=None, stream=False, **kwargs):
        if self.failures:
            return Response(self.failures.pop(0), {"error": "busy"})
        if url == parquet.LISTING:
            recorded = LISTINGS[f"{params['dataset']}?config={params['config']}"]
            headers = {} if self.no_revision else recorded.get("headers")
            if self.listing_revisions:
                headers = {"X-Revision": self.listing_revisions.pop(0)}
            return Response(recorded["status"], recorded["body"], headers=headers)
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
        self.downloads.append(url)
        return Response(200, content=url.encode())


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


def test_the_chosen_splits_shards_are_downloaded_from_the_converts_commit(hf, tmp_path):
    # The listing's URLs name the moving refs/convert/parquet branch; every shard
    # is downloaded from the commit the Hub says that branch points at, so the
    # bytes cannot change under a conversion rebuilt while the import reads.
    fetched = fetch(tmp_path, "coastalcph/lex_glue", "ledgar", "train")

    assert hf.downloads == [f"https://huggingface.co/datasets/coastalcph/lex_glue/resolve/{LEX_CONVERSION}/ledgar/train/0000.parquet"]
    assert [pathlib.Path(p).read_bytes() for p in fetched.paths] == [hf.downloads[0].encode()]


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


def test_shards_are_downloaded_until_the_byte_cap(hf, tmp_path):
    # Shards of 198 MB, 228 MB, 236 MB, …: a 450 MB cap takes the first two.
    fetched = fetch(tmp_path, "isaacus/open-australian-legal-corpus", "corpus", "corpus", max_bytes=450 * MB)

    assert len(fetched.paths) == 2
    assert [u.rsplit("/", 1)[1] for u in hf.downloads] == ["0000.parquet", "0001.parquet"]
    assert all("/resolve/ef6ae142f81f5c135978cde1fd0d7c9c4c765555/" in u for u in hf.downloads)


def test_a_first_shard_over_the_cap_is_too_large_to_sample(hf, tmp_path):
    with pytest.raises(ImportFailed) as refused:
        fetch(tmp_path, "isaacus/open-australian-legal-corpus", "corpus", "corpus", max_bytes=100 * MB)

    assert refused.value.code == "too_large"
    assert refused.value.message == "Too large to sample here: even one part of this dataset is over the download limit."
    assert hf.downloads == []


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
