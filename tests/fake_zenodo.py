"""The three Community seams, answered from recorded fixtures (tests/fixtures/zenodo).

The answers have the shape fair-publishing 0.2.0 returns from open_datasets, from
Zenodo.record_files and from Zenodo.download. The library never sees a repository
client, and no test reaches the network. A test changes what Zenodo says by editing
`records`, `listed`, `gone` or `busy`."""
import copy
import json
import pathlib

from dataset_library.errors import ProviderUnavailable
from dataset_library.providers.community import NotDownloadable, TooLarge

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "zenodo"
FILES = FIXTURES / "files"
LISTING = json.loads((FIXTURES / "open_datasets.json").read_text(encoding="utf-8"))
RECORDS = json.loads((FIXTURES / "records.json").read_text(encoding="utf-8"))


class FakeZenodo:
    def __init__(self):
        self.listed = copy.deepcopy(LISTING)  # what open_datasets lists
        self.records = copy.deepcopy(RECORDS)  # what record_files answers, by record id
        self.gone = set()  # record ids Zenodo no longer serves (404/410)
        self.busy = False  # every call fails as an unreachable repository
        self.contents = {}  # (record id, key) -> bytes that replace the recorded file
        self.calls = []  # (seam, arguments), in call order

    def datasets(self):
        self.calls.append(("datasets",))
        return copy.deepcopy(self.listed)

    def record(self, record_id):
        self.calls.append(("record", record_id))
        self._answer(record_id)
        return copy.deepcopy(self.records[record_id])

    def download(self, record_id, key, max_bytes):
        self.calls.append(("download", record_id, key, max_bytes))
        self._answer(record_id)
        record = self.records[record_id]
        if not record["downloadable"] or not any(entry["key"] == key for entry in record["files"]):
            raise NotDownloadable()
        if (record_id, key) in self.contents:
            data = self.contents[(record_id, key)]
        else:
            data = (FILES / record_id / key).read_bytes()
        if len(data) > max_bytes:
            raise TooLarge()
        return data

    def downloads(self):
        return [call for call in self.calls if call[0] == "download"]

    def _answer(self, record_id):
        if self.busy:
            raise ProviderUnavailable("Zenodo did not answer")
        if record_id in self.gone or record_id not in self.records:
            raise NotDownloadable()

    def publication(self, record_id, **changes):
        """A listing row for a record that exists on Zenodo but is not listed: the Publication, as
        open_datasets would list it if the record were open. Used to close a record after listing."""
        row = copy.deepcopy(LISTING[0])
        row.update(id=f"pub-{record_id}", record_id=record_id, doi=f"10.5281/zenodo.{record_id}", **changes)
        return row
