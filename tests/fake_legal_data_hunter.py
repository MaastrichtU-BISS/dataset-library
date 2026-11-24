"""Legal Data Hunter's REST API, answered from recorded fixtures (tests/fixtures/legal_data_hunter).

The answers have the shapes legaldatahunter.com/docs gives for `POST /v1/search`, `GET /v1/documents/...` and a
429 (checked 2026-11-24). No test reaches the network: the provider's `requests.request` is replaced by `FakeLegalDataHunter`.
A test changes what Legal Data Hunter says by setting `credits` (calls left before a 429), `keys` (the keys it
accepts), `redirects` (path -> Location), `gone` (document paths it answers 404 for) or `busy`."""
import copy
import json
import pathlib
from urllib.parse import parse_qsl, unquote, urlsplit

from dataset_library.providers import legal_data_hunter as provider

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "legal_data_hunter"
SEARCHES = json.loads((FIXTURES / "search.json").read_text(encoding="utf-8"))
DOCUMENTS = json.loads((FIXTURES / "documents.json").read_text(encoding="utf-8"))
ADMIN_KEY = "sk-admin-0000000000000000"
STUDENT_KEY = "sk-student-1111111111111111"


class Response:
    def __init__(self, status_code, body=None, headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}

    def json(self):
        if isinstance(self._body, str):
            raise ValueError("not JSON")
        return self._body


class FakeLegalDataHunter:
    def __init__(self):
        self.keys = {ADMIN_KEY, STUDENT_KEY}
        self.credits = None  # calls left before a 429; None is no limit
        self.retry_after = "3600"
        self.redirects = {}  # path -> Location header
        self.gone = set()  # "country/source/source_id" answered 404
        self.busy = False
        self.calls = []  # {method, url, path, params, json, key}

    def request(self, method, url, params=None, json=None, headers=None, timeout=None, allow_redirects=True, **kwargs):
        auth = (headers or {}).get("Authorization", "")
        parts = urlsplit(url)
        # A redirect's Location carries its query in the URL, not in `params`.
        params = {**dict(parse_qsl(parts.query)), **dict(params or {})}
        self.calls.append({
            "method": method, "url": url, "origin": f"{parts.scheme}://{parts.netloc}", "path": parts.path,
            "params": dict(params or {}), "json": copy.deepcopy(json), "key": auth.removeprefix("Bearer "),
            "allow_redirects": allow_redirects, "timeout": timeout,
        })
        assert allow_redirects is False, "the provider follows redirects itself, on Legal Data Hunter's origin only"
        if f"{parts.scheme}://{parts.netloc}" != provider.ORIGIN:
            return Response(200, {"hits": [], "text": "another host answered"})
        if parts.path in self.redirects:
            return Response(302, None, {"Location": self.redirects[parts.path]})
        if self.busy:
            return Response(500, {"detail": "Internal Server Error"})
        if auth.removeprefix("Bearer ") not in self.keys:
            return Response(401, {"detail": "Invalid API key"})
        if self.credits is not None:
            if self.credits <= 0:
                return Response(429, {
                    "type": "error",
                    "error": {
                        "type": "rate_limit_error", "message": "Daily request limit reached", "limit_type": "day",
                        "plan": "free", "upgrade_url": "https://legaldatahunter.com/pricing",
                        "period_resets_at": "2026-11-25T00:00:00Z",
                    },
                }, {"Retry-After": self.retry_after})
            self.credits -= 1
        if method == "POST" and parts.path == "/v1/search":
            found = copy.deepcopy(SEARCHES.get(json["namespace"], SEARCHES["doctrine"]))
            found["hits"] = found["hits"][: json.get("top_k", 10)]
            return Response(200, found)
        if method == "GET" and parts.path.startswith("/v1/documents/"):
            country, source = [unquote(part) for part in parts.path.removeprefix("/v1/documents/").split("/")]
            key = f"{country}/{source}/{(params or {}).get('source_id')}"
            if key in self.gone or key not in DOCUMENTS:
                return Response(404, {"detail": "Document not found in any table"})
            return Response(200, copy.deepcopy(DOCUMENTS[key]))
        return Response(404, {"detail": "Not Found"})

    def documents_fetched(self):
        """The documents asked for, as `country/source/source_id`, in call order."""
        found = []
        for call in self.calls:
            if call["method"] == "GET" and call["path"].startswith("/v1/documents/"):
                country, source = [unquote(p) for p in call["path"].removeprefix("/v1/documents/").split("/")]
                found.append(f"{country}/{source}/{call['params'].get('source_id')}")
        return found
