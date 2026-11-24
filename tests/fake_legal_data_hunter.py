"""Legal Data Hunter's REST API, answered from recorded fixtures (tests/fixtures/legal_data_hunter).

The answers have the shapes legaldatahunter.com/docs gives for `POST /v1/search`, `GET /v1/documents/...` and a
429 (checked 2026-11-24). No test reaches the network: `FakeLegalDataHunter` is a `requests` transport adapter, mounted
on every `requests.Session` a test makes, so the provider's requests go through a real Session to it. A test changes
what Legal Data Hunter says by setting `credits` (calls left before a 429), `keys` (the keys it accepts), `redirects`
(path -> Location), `gone` (document paths it answers 404 for), `rejected` (document paths -> 400 or 422) or `busy`.
The key is read from the prepared request's `Authorization` header, as it would go on the wire."""
import copy
import io
import json
import pathlib
from urllib.parse import parse_qsl, unquote, urlsplit

import requests
from requests.adapters import BaseAdapter
from requests.structures import CaseInsensitiveDict

from dataset_library.providers import legal_data_hunter as provider

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "legal_data_hunter"
SEARCHES = json.loads((FIXTURES / "search.json").read_text(encoding="utf-8"))
DOCUMENTS = json.loads((FIXTURES / "documents.json").read_text(encoding="utf-8"))
ADMIN_KEY = "sk-admin-0000000000000000"
STUDENT_KEY = "sk-student-1111111111111111"


class _Body(io.BytesIO):
    """A response's raw body, counting the bytes read from it."""

    def __init__(self, data):
        super().__init__(data)
        self.bytes_read = 0

    def read(self, size=-1):
        chunk = super().read(size)
        self.bytes_read += len(chunk)
        return chunk


class Response(requests.Response):
    """A real `requests.Response`, its body read as `requests` streams it; `bytes_read` counts what was read."""

    def __init__(self, status_code, body=None, headers=None):
        super().__init__()
        self.status_code = status_code
        self._body = body
        self.headers = CaseInsensitiveDict(headers or {})
        self.raw = _Body(self._bytes())
        self.closed = False

    def _bytes(self):
        body = self._body if isinstance(self._body, str) else json.dumps(self._body)
        return (body or "").encode("utf-8")

    @property
    def bytes_read(self):
        return self.raw.bytes_read

    def close(self):
        self.closed = True
        super().close()


class FakeLegalDataHunter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.keys = {ADMIN_KEY, STUDENT_KEY}
        self.credits = None  # calls left before a 429; None is no limit
        self.retry_after = "3600"
        self.redirects = {}  # path -> Location header
        self.gone = set()  # "country/source/source_id" answered 404
        self.rejected = {}  # "country/source/source_id" -> 400 or 422
        self.busy = False
        self.minute_limited = 0  # requests answered with a per-minute 429 before the next one goes through
        self.minute_retry_after = "7"  # the per-minute 429's Retry-After
        self.redirect_body = "<html><body>Moved</body></html>"  # what a redirect carries, which is never read
        self.padding = ""  # added to a 429's and a 500's error body, to make it longer
        self.calls = []  # {method, url, path, params, json, key}
        self.responses = []  # every Response given, in call order

    def send(self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None):
        """The transport: one prepared request, one response, never a redirect followed."""
        response = self._answer(request, stream=stream, timeout=timeout)
        response.url, response.request = request.url, request
        self.responses.append(response)
        return response

    def close(self):
        pass

    def _answer(self, request, *, stream, timeout):
        method, url = request.method, request.url
        auth = request.headers.get("Authorization", "")
        parts = urlsplit(url)
        params = dict(parse_qsl(parts.query))
        body = request.body
        sent = json.loads(body) if body else None
        self.calls.append({
            "method": method, "url": url, "origin": f"{parts.scheme}://{parts.netloc}", "path": parts.path,
            "params": params, "json": copy.deepcopy(sent), "key": auth.removeprefix("Bearer "),
            "timeout": timeout, "stream": stream,
        })
        if f"{parts.scheme}://{parts.netloc}" != provider.ORIGIN:
            return Response(200, {"hits": [], "text": "another host answered"})
        if parts.path in self.redirects:
            return Response(302, self.redirect_body, {"Location": self.redirects[parts.path]})
        if self.busy:
            return Response(500, {"detail": "Internal Server Error", **self._padded()})
        if auth.removeprefix("Bearer ") not in self.keys:
            return Response(401, {"detail": "Invalid API key"})
        if self.minute_limited > 0:
            self.minute_limited -= 1
            return Response(429, {
                "type": "error",
                "error": {"type": "rate_limit_error", "message": "Per-minute limit reached", "limit_type": "minute", "plan": "free"},
            }, {"Retry-After": self.minute_retry_after})
        if self.credits is not None:
            if self.credits <= 0:
                return Response(429, {
                    "type": "error",
                    "error": {
                        "type": "rate_limit_error", "message": "Daily request limit reached", "limit_type": "day",
                        "plan": "free", "upgrade_url": "https://legaldatahunter.com/pricing",
                        "period_resets_at": "2026-11-25T00:00:00Z",
                    },
                    **self._padded(),
                }, {"Retry-After": self.retry_after})
            self.credits -= 1
        if method == "POST" and parts.path == "/v1/search":
            found = copy.deepcopy(SEARCHES.get(sent["namespace"], SEARCHES["doctrine"]))
            found["hits"] = found["hits"][: sent.get("top_k", 10)]
            return Response(200, found)
        if method == "GET" and parts.path.startswith("/v1/documents/"):
            country, source = [unquote(part) for part in parts.path.removeprefix("/v1/documents/").split("/")]
            key = f"{country}/{source}/{params.get('source_id')}"
            if key in self.rejected:
                return Response(self.rejected[key], {"detail": "Invalid source_id"})
            if key in self.gone or key not in DOCUMENTS:
                return Response(404, {"detail": "Document not found in any table"})
            return Response(200, copy.deepcopy(DOCUMENTS[key]))
        return Response(404, {"detail": "Not Found"})

    def _padded(self):
        return {"padding": self.padding} if self.padding else {}

    def documents_fetched(self):
        """The documents asked for, as `country/source/source_id`, in call order."""
        found = []
        for call in self.calls:
            if call["method"] == "GET" and call["path"].startswith("/v1/documents/"):
                country, source = [unquote(p) for p in call["path"].removeprefix("/v1/documents/").split("/")]
                found.append(f"{country}/{source}/{call['params'].get('source_id')}")
        return found
