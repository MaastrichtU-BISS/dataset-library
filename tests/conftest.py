"""A minimal host: sign-in by an X-User header, an in-memory cache, and
Hugging Face faked at its HTTP boundary from recorded responses."""
import functools
import json
import pathlib
import re
import threading

import pytest
import requests
from flask import Flask, abort, g, request
from flask_sqlalchemy import SQLAlchemy

from dataset_library import init_dataset_library
from dataset_library.models import LibraryBase

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "huggingface"
VIEWER = json.loads((pathlib.Path(__file__).parent / "fixtures" / "viewer" / "responses.json").read_text())
HUB_DATASETS = json.loads((pathlib.Path(__file__).parent / "fixtures" / "hub" / "datasets.json").read_text())
README_FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "hub" / "readmes"
VIEWER_URL = "https://datasets-server.huggingface.co/"
DATASET_URL = "https://huggingface.co/api/datasets/"
README_PATTERN = re.compile(r"^https://huggingface\.co/datasets/([^/]+/[^/]+)/resolve/([0-9a-f]{40})/README\.md$")
TASK_TYPES = ["classification", "document_classification", "ner", "named_entity_recognition",
              "summarization", "topic_modeling", "clustering", "judgment_prediction"]


def login_required(view):
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        user = request.headers.get("X-User")
        if not user:
            abort(401)
        g.user = user
        return view(*args, **kwargs)

    return wrapped


class MemoryCache:
    def __init__(self):
        self.values = {}
        self.ttls = {}
        self.guard = threading.Lock()

    def get(self, key):
        with self.guard:
            return self.values.get(key)

    def set(self, key, value, ttl):
        with self.guard:
            self.values[key] = value
            self.ttls[key] = ttl

    def add(self, key, value, ttl):
        """Set the key only when it is not set: the lock a preview fetch takes (dataset_library.Cache)."""
        with self.guard:
            if key in self.values:
                return False
            self.values[key] = value
            self.ttls[key] = ttl
            return True

    def delete(self, key):
        with self.guard:
            self.values.pop(key, None)
            self.ttls.pop(key, None)

    def delete_if(self, key, value):
        """Remove the key only while it holds `value`: a lock is released only by the job that took it (dataset_library.Cache)."""
        with self.guard:
            if self.values.get(key) == value:
                del self.values[key]
                self.ttls.pop(key, None)

    def expire(self, prefix):
        """Forget fresh entries, as Redis would once their TTL ran out."""
        for key in [k for k in self.values if k.startswith(prefix)]:
            del self.values[key]


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        if isinstance(self._body, str):
            raise ValueError("not JSON")
        return self._body

    def iter_content(self, chunk_size=1):
        data = self.text.encode("utf-8")
        for start in range(0, len(data), chunk_size):
            yield data[start:start + chunk_size]

    def close(self):
        pass


class FakeHub:
    """Answers GET /api/datasets like the Hub: `search` matches the id, `author`
    the owner, every `filter` must be a tag. Records come from real responses."""

    def __init__(self):
        records = {}
        for path in sorted(FIXTURES.glob("*.json")):
            for record in json.loads(path.read_text()):
                records[record["id"]] = record
        self.records = list(records.values())
        self.calls = []
        self.fail = None  # an HTTP status, "malformed" or "timeout"
        self.viewer_fail = None  # an HTTP status for the dataset viewer alone: the Hub still answers
        self.gone_revisions = set()  # (dataset id, sha) pairs the Hub no longer has

    def get(self, url, params=None, timeout=None, **kwargs):
        params = dict(params or {})
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        if self.fail == "timeout":
            raise requests.Timeout("read timed out")
        if self.fail == "malformed":
            return FakeResponse(200, "<html>oops</html>")
        if self.fail:
            return FakeResponse(self.fail, {"error": "busy"})
        if url.startswith(VIEWER_URL):
            if self.viewer_fail:
                return FakeResponse(self.viewer_fail, {"error": "The viewer is down"})
            query = url.removeprefix(VIEWER_URL) + "?" + "&".join(f"{k}={params[k]}" for k in ("dataset", "config", "split") if k in params)
            recorded = VIEWER.get(query, {"status": 404, "body": {"error": "not recorded"}})
            return FakeResponse(recorded["status"], recorded["body"])
        if url.startswith(DATASET_URL) and "/revision/" in url:
            # GET /api/datasets/{id}/revision/{sha}: a revision exists while the Hub keeps it.
            dataset_id, _, revision = url.removeprefix(DATASET_URL).partition("/revision/")
            if dataset_id not in HUB_DATASETS:
                # As the live Hub does for a dataset that is gone or not public (checked 2026-11-19).
                return FakeResponse(401, {"error": "Invalid credentials in Authorization header"})
            record = HUB_DATASETS.get(dataset_id, {}).get("body") or {}
            if (dataset_id, revision) not in self.gone_revisions and record.get("sha") == revision:
                return FakeResponse(200, record)
            return FakeResponse(404, {"error": "Revision Not Found"})
        if url.startswith(DATASET_URL):
            recorded = HUB_DATASETS.get(url.removeprefix(DATASET_URL), {"status": 404, "body": {"error": "Repository not found"}})
            return FakeResponse(recorded["status"], recorded["body"])
        readme = README_PATTERN.match(url)
        if readme:
            # A README is served only at the revision the dataset's record names, as the Hub does.
            dataset_id, revision = readme.groups()
            record = HUB_DATASETS.get(dataset_id, {}).get("body") or {}
            path = README_FIXTURES / f"{dataset_id.replace('/', '__')}.md"
            if record.get("sha") == revision and path.exists():
                return FakeResponse(200, path.read_text(encoding="utf-8"))
            return FakeResponse(404, "Not found")
        assert url == "https://huggingface.co/api/datasets", url
        found = self.records
        if params.get("search"):
            found = [r for r in found if params["search"].lower() in r["id"].lower()]
        if params.get("author"):
            found = [r for r in found if r["id"].split("/")[0] == params["author"]]
        filters = params.get("filter") or []
        filters = [filters] if isinstance(filters, str) else filters
        found = [r for r in found if all(f in r["tags"] for f in filters)]
        return FakeResponse(200, found[: int(params.get("limit", 30))])


@pytest.fixture
def hub(monkeypatch):
    import dataset_library.providers.huggingface as hf

    fake = FakeHub()
    monkeypatch.setattr(hf.requests, "get", fake.get)
    return fake


@pytest.fixture
def cache():
    return MemoryCache()


@pytest.fixture
def audits():
    return []


@pytest.fixture
def make_app(cache, hub, audits):
    def make(**options):
        app = Flask(__name__)
        app.config.update({"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:", **options.pop("config", {})})
        db = SQLAlchemy()
        db.init_app(app)
        init_dataset_library(
            app,
            db=db,
            login_required=login_required,
            current_owner=lambda: g.user,
            task_types=TASK_TYPES,
            cache=cache,
            is_admin=lambda: request.headers.get("X-Admin") == "yes",
            audit=lambda event, **fields: audits.append({"event": event, **fields}),
            **options,
        )
        with app.app_context():
            LibraryBase.metadata.create_all(db.engine)
        app.db = db
        return app

    return make


@pytest.fixture
def app(make_app):
    return make_app()


@pytest.fixture
def client(app):
    return app.test_client()


ALICE = {"X-User": "alice"}
ADMIN = {"X-User": "ada", "X-Admin": "yes"}
