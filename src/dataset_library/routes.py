"""The HTTP API, under /api/library."""
from flask import Blueprint, current_app, jsonify, request

from . import admin as admin_module
from . import community as community_module
from . import imports as imports_module
from . import labels as labels_module
from . import legal_data_hunter as ldh_module
from . import pins as pins_module
from . import policy as policy_module
from . import preview as preview_module
from .errors import ProviderError, ProviderRateLimited, ProviderUnavailable
from . import search as search_module
from . import snapshots as snapshots_module
from .licenses import GROUPS, LICENSES
from .providers import community as community_provider
from .providers import legal_data_hunter as ldh_provider
from .providers.huggingface import DATASET_ID_PATTERN, PrivateDataset
from .tasks import hugging_face_filter

library_bp = Blueprint("dataset_library", __name__)


class ApiError(Exception):
    def __init__(self, status, message, code, **extra):
        super().__init__(message)
        self.status = status
        self.body = {"error": message, "code": code, **extra}


@library_bp.errorhandler(ApiError)
def _api_error(error):
    return jsonify(error.body), error.status


def _host():
    return current_app.extensions["dataset_library"]


@library_bp.before_request
def _sign_in():
    """The host's sign-in decorator around nothing: a refusal it returns or
    raises ends the request before any route runs."""
    return _host().login_required(lambda: None)()


@library_bp.route("/search", methods=["GET"])
def search():
    args = request.args
    task, license_id = args.get("task", "").strip(), args.get("license", "").strip()
    if task and task not in _host().task_types:
        raise ApiError(400, f"Unknown task type: {task}", "bad_filter")
    if license_id and license_id not in LICENSES:
        raise ApiError(400, f"Unknown license: {license_id}", "bad_filter")
    source = args.get("source", "").strip()
    sources = {s.ref for s in search_module.enabled_sources(_host().db)}
    if community_module.enabled(_host()):
        sources.add(community_provider.SOURCE)
    if source and source not in sources:
        raise ApiError(400, f"Not a library source: {source}", "bad_filter")
    return jsonify(
        search_module.search(
            _host(),
            query=args.get("q", "").strip(),
            task=task,
            license_id=license_id,
            source=source,
        )
    )


# What the host UI needs to label tasks; legal-ml-core's TASK_TYPES labels.
TASK_LABELS = {"classification": "Classification", "ner": "Named entities", "summarization": "Summarization"}


@library_bp.route("/options", methods=["GET"])
def options():
    """The filter choices, and the license table with this platform's levels. A provider's terms (#126) are not a
    license a dataset in search can have, so they are left out here; the admin's settings still list them."""
    host = _host()
    policy = policy_module.overrides(host.db)
    licenses = []
    for license in LICENSES.values():
        if license.terms:
            continue
        level = policy_module.level(license, policy)
        if level != "hide":
            licenses.append(license.at_level(level))
    terms_groups = {license.group for license in LICENSES.values() if license.terms}
    groups = [group for group in GROUPS if group not in terms_groups]
    return jsonify(
        {
            "tasks": [
                {"value": task, "label": TASK_LABELS.get(task, task)}
                for task in host.task_types
                if hugging_face_filter(task)
            ],
            "licenses": licenses,
            "groups": groups,
            "admin": bool(host.is_admin()),
            "importable": host.enqueue is not None,
            "maxRows": host.row_cap(),
            "sources": _source_choices(host),
            # Legal Data Hunter entries (#126): whether they are offered, whether this user has a key, and why not.
            "legalDataHunter": {
                "offered": ldh_module.offered(host),
                "hasKey": ldh_module.has_key(host, host.current_owner()),
                "needsKey": ldh_provider.NEEDS_KEY,
                "corpora": [{"value": value, "label": label} for value, label in ldh_provider.CORPORA.items()],
                "maxDocuments": ldh_provider.MAX_DOCUMENTS,
            },
        }
    )


def _source_choices(host):
    """The sources a search can be limited to: the enabled Hugging Face organisations, then Community when on."""
    choices = [{"value": s.ref, "label": s.ref, "provider": s.provider} for s in search_module.enabled_sources(host.db)]
    if community_module.enabled(host):
        choices.append({"value": community_provider.SOURCE, "label": "Community", "provider": community_provider.PROVIDER})
    return choices


MAX_LABEL_IDS = 30


def _hugging_face_ids(ids):
    """External ids from `huggingface:org/name` ids, refusing anything else."""
    external = []
    for library_id in ids:
        provider, _, external_id = library_id.partition(":")
        if provider != "huggingface" or not DATASET_ID_PATTERN.match(external_id):
            raise ApiError(400, f"Not a library dataset id: {library_id}", "bad_id")
        external.append(external_id)
    return external


@library_bp.route("/labels", methods=["GET"])
def labels():
    """Whether each dataset has labels: `yes`, `no` or `unclear`. The page
    asks for the cards it shows, so a search does not wait for this."""
    ids = [i for i in request.args.get("ids", "").split(",") if i]
    if len(ids) > MAX_LABEL_IDS:
        raise ApiError(400, f"At most {MAX_LABEL_IDS} ids at a time.", "too_many_ids")
    found = labels_module.for_ids(_host().cache, _hugging_face_ids(ids))
    return jsonify({"labels": {f"huggingface:{external_id}": verdict for external_id, verdict in found.items()}})


# The parameter is not called `*_id`: hosts may treat such URL parameters as
# their own database ids (BAIGLE refuses any that is not a UUID).
@library_bp.route("/datasets/huggingface/<path:dataset_ref>", methods=["GET"])
def dataset(dataset_ref):
    """One dataset's facts, configs and first rows (`?config=&split=`). While Hugging
    Face cannot be asked, the copy the platform keeps of this part stands in (#121)."""
    if not DATASET_ID_PATTERN.match(dataset_ref):
        raise ApiError(400, f"Not a Hugging Face dataset id: {dataset_ref}", "bad_id")
    config, split = request.args.get("config") or None, request.args.get("split") or None
    try:
        body = preview_module.preview(_host(), dataset_ref, config, split)
    except (preview_module.NotInLibrary, PrivateDataset):
        raise ApiError(404, "That dataset is not in the library.", "not_found")
    except preview_module.UnknownConfig:
        raise ApiError(400, "That dataset has no such configuration or split.", "bad_config")
    except ProviderError as down:
        view = _copy_view(dataset_ref, config, split)
        if view is not None:
            return view
        if isinstance(down, ProviderRateLimited):
            raise ApiError(503, "Hugging Face is limiting requests. Try again in a few minutes.", "provider_rate_limited")
        raise ApiError(503, "Hugging Face is unavailable right now. Try again in a few minutes.", "provider_unavailable")
    if body.get(preview_module.VIEWER_DOWN):
        # The viewer did not answer: the copy of this part, when there is one, shows the rows (#121).
        view = _copy_view(dataset_ref, config, split)
        if view is not None:
            return view
    return jsonify(body)


# Not `<record_id>`: a host may refuse `*_id` URL parameters that are not its own UUIDs (see the dataset route above).
@library_bp.route("/datasets/community/<record_ref>", methods=["GET"])
def community_dataset(record_ref):
    """One Community dataset (#122): its card, whether it can be imported and, if not, why, and the first
    rows of its table file. Zenodo is asked now, so a record closed since it was listed shows as closed."""
    if not community_provider.RECORD_ID.match(record_ref):
        raise ApiError(400, f"Not a Zenodo record id: {record_ref}", "bad_id")
    try:
        body = community_module.view(_host(), record_ref)
    except preview_module.NotInLibrary:
        raise ApiError(404, "That dataset is not in the library.", "not_found")
    except ProviderError:
        raise ApiError(503, "Zenodo is unavailable right now. Try again in a few minutes.", "provider_unavailable")
    return jsonify(body)


def _copy_view(dataset_ref, config, split):
    """The platform's copy of this part of the dataset as the page shows it, or None when
    there is no copy to show (#121). A copy whose license is now hidden is not in the library."""
    try:
        copy = snapshots_module.copy_for_offline(_host(), dataset_ref, config, split)
        return None if copy is None else jsonify(snapshots_module.offline_detail(_host(), copy))
    except preview_module.LicenseHidden:
        raise ApiError(404, "That dataset is not in the library.", "not_found")


@library_bp.route("/provenance/<dataset_ref>", methods=["GET"])
def provenance(dataset_ref):
    """Where a Dataset the signed-in user imported came from, under which
    license, and when the student confirmed it. Reads the database only, so it
    works without an import queue; 404 when this user has no such import."""
    record = imports_module.provenance(_host(), _host().current_owner(), dataset_ref)
    if record is None:
        raise ApiError(404, "No such import.", "not_found")
    return jsonify(record)


@library_bp.route("/featured", methods=["GET"])
def featured():
    """The course's featured pins, for the row any signed-in user sees, and its Legal Data Hunter entries (#126), each
    a featured card that says whether this user can import it."""
    host = _host()
    return jsonify({
        "featured": pins_module.featured(host),
        "legalDataHunter": ldh_module.cards(host, host.current_owner()),
    })


@library_bp.route("/legal-data-hunter/entries", methods=["GET"])
def ldh_cards():
    """The Legal Data Hunter entries as featured cards, and whether the signed-in user has a key (#126)."""
    host = _host()
    has_key = ldh_module.has_key(host, host.current_owner())
    return jsonify({
        "entries": ldh_module.cards(host, host.current_owner(), keyed=has_key),
        "offered": ldh_module.offered(host),
        "hasKey": has_key,
        "needsKey": None if has_key else ldh_provider.NEEDS_KEY,
    })


# Not `<entry_id>`: a host may refuse `*_id` URL parameters that are not its own UUIDs.
@library_bp.route("/datasets/legal_data_hunter/<entry_ref>", methods=["GET"])
def ldh_dataset(entry_ref):
    """One Legal Data Hunter entry's card, with the confirmation step 2 asks for (#126)."""
    host = _host()
    try:
        return jsonify(ldh_module.view(host, host.current_owner(), entry_ref))
    except ldh_module.NotFound:
        raise ApiError(404, ldh_provider.NOT_IN_LIBRARY, "not_found")


def _require_admin():
    if not _host().is_admin():
        raise ApiError(403, "Only the course's library admins can change these settings.", "not_admin")


@library_bp.route("/pins", methods=["GET"])
def list_pins():
    _require_admin()
    return jsonify({"pins": pins_module.list_all(_host())})


@library_bp.route("/pins", methods=["POST"])
def add_pin():
    """Pin a library dataset; the same provider and dataset id is one pin."""
    _require_admin()
    body = request.get_json(silent=True)
    if body is not None and not isinstance(body, dict):
        raise ApiError(400, "Send a JSON object.", "bad_request")
    submitted = (body or {}).get("dataset")
    if not isinstance(submitted, str):
        raise ApiError(400, "Pick a Hugging Face dataset id.", "bad_id")
    dataset_id = submitted.strip()
    if not DATASET_ID_PATTERN.match(dataset_id):
        raise ApiError(400, f"Not a Hugging Face dataset id: {dataset_id}", "bad_id")
    try:
        pin, created = pins_module.add(_host(), dataset_id)
    except preview_module.LicenseHidden:
        raise ApiError(403, "This dataset's license is not allowed on this platform, so it cannot be pinned.", "license_hidden")
    except preview_module.NotInLibrary:
        raise ApiError(404, "That dataset is not in the library.", "not_found")
    except PrivateDataset:
        raise ApiError(404, "That dataset is not in the library.", "not_found")
    except ProviderRateLimited:
        raise ApiError(503, "Hugging Face is limiting requests. Try again in a few minutes.", "provider_rate_limited")
    except ProviderUnavailable:
        raise ApiError(503, "Hugging Face could not be reached. Try again in a few minutes.", "provider_unavailable")
    return jsonify(pin), 201 if created else 200


@library_bp.route("/pins/<pin_ref>", methods=["PATCH"])
def patch_pin(pin_ref):
    """Feature or unfeature a pin, and set or clear its display title."""
    _require_admin()
    # A change needs a JSON object; `{}` is a valid no-op, null or broken JSON is not.
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ApiError(400, "Send a JSON object.", "bad_request")
    try:
        pin = pins_module.update(_host(), pin_ref, body)
    except pins_module.NotFound:
        raise ApiError(404, "No such pin.", "not_found")
    except pins_module.TooManyFeatured:
        raise ApiError(409, "At most 3 datasets can be featured. Unfeature one first.", "too_many_featured")
    except pins_module.BadFeatured:
        raise ApiError(400, "Say whether the pin is featured: true or false.", "bad_featured")
    except pins_module.BadTitle:
        raise ApiError(400, "Keep the title under 121 characters.", "bad_title")
    return jsonify(pin)


@library_bp.route("/pins/<pin_ref>", methods=["DELETE"])
def remove_pin(pin_ref):
    _require_admin()
    try:
        pins_module.remove(_host(), pin_ref)
    except pins_module.NotFound:
        raise ApiError(404, "No such pin.", "not_found")
    return "", 204


@library_bp.route("/pins/<pin_ref>/snapshot", methods=["POST"])
def take_snapshot(pin_ref):
    """Keep a copy of a pin that has none, or whose copy failed (#121). The worker makes it."""
    _require_admin()
    try:
        pin = pins_module.take_copy(_host(), pin_ref)
    except pins_module.NotFound:
        raise ApiError(404, "No such pin.", "not_found")
    except pins_module.NotOffered:
        raise ApiError(404, "This platform keeps no copies of datasets.", "not_offered")
    except pins_module.NotAllowed:
        raise ApiError(409, "Only a dataset whose license is Allow can be copied.", "not_allowed")
    except (preview_module.NotInLibrary, PrivateDataset):
        raise ApiError(404, "That dataset is not in the library.", "not_found")
    except ProviderRateLimited:
        raise ApiError(503, "Hugging Face is limiting requests. Try again in a few minutes.", "provider_rate_limited")
    except ProviderUnavailable:
        raise ApiError(503, "Hugging Face could not be reached. Try again in a few minutes.", "provider_unavailable")
    return jsonify(pin), 202


@library_bp.route("/admin/settings", methods=["GET"])
def get_settings():
    _require_admin()
    return jsonify({**admin_module.current(_host().db, _host()), "groups": GROUPS})


@library_bp.route("/admin/settings", methods=["PUT"])
def put_settings():
    _require_admin()
    host = _host()
    try:
        changes = admin_module.save(
            host.db,
            request.get_json(silent=True) or {},
            host.hard_max_rows,
            community_available=host.community is not None,
        )
    except admin_module.Invalid as invalid:
        host.db.session.rollback()
        raise ApiError(400, "Some settings need fixing.", "invalid", fields=invalid.fields)
    except (ProviderRateLimited, ProviderUnavailable):
        host.db.session.rollback()
        raise ApiError(503, "Hugging Face could not check the new sources. Try again in a few minutes.", "provider_unavailable")
    # A copy is kept only while its license is Allow: a changed level removes the copies it no longer allows (#121).
    if changes["licenses_changed"]:
        pins_module.reconcile_copies(host)
        # Legal Data Hunter's terms set to Hide: no import may resume, so what they fetched goes (#126).
        if ldh_module.TERMS.id in changes["licenses_changed"]:
            ldh_module.hidden(host.db.session)
    host.audit("library.settings_saved", **changes)
    return jsonify({**admin_module.current(host.db, host), "groups": GROUPS})


@library_bp.route("/admin/sources/check", methods=["POST"])
def check_source():
    """Whether a source can be added, before the admin saves."""
    _require_admin()
    ref = ((request.get_json(silent=True) or {}).get("ref") or "").strip()
    try:
        problem = admin_module.check_source(ref)
    except (ProviderRateLimited, ProviderUnavailable):
        raise ApiError(503, "Hugging Face could not be reached to check that. Try again in a few minutes.", "provider_unavailable")
    return jsonify({"ref": ref, "ok": True} if problem is None else {"ref": ref, "ok": False, "error": problem})


def _importing():
    if _host().enqueue is None:
        raise ApiError(404, "Importing is not offered here.", "not_found")


@library_bp.route("/imports", methods=["POST"])
def start_import():
    """Queue an import of a sample; the worker does the rest."""
    _importing()
    record, created = _starting(imports_module.start, request.get_json(silent=True) or {})
    # The same import already queued or running comes back as it is.
    return jsonify(record), 202 if created else 200


def _starting(action, *args):
    """Run an action that starts an import, turning its refusals into answers."""
    try:
        return action(_host(), _host().current_owner(), *args)
    except imports_module.Refused as refused:
        raise ApiError(refused.status, refused.message, refused.code)
    except ProviderRateLimited:
        raise ApiError(503, "Hugging Face is limiting requests. Try again in a few minutes.", "provider_rate_limited")
    except ProviderUnavailable:
        raise ApiError(503, "Hugging Face could not be reached. Try again in a few minutes.", "provider_unavailable")


@library_bp.route("/imports", methods=["GET"])
def list_imports():
    """The signed-in user's imports, newest first; with `?current=true`, only
    those still running and those failed in the last day and not put away."""
    _importing()
    owner = _host().current_owner()
    if request.args.get("current") == "true":
        return jsonify({"imports": imports_module.current(_host(), owner)})
    return jsonify({"imports": imports_module.for_owner(_host(), owner)})


# Not `<import_id>`: hosts may treat `*_id` URL parameters as their own database ids.
@library_bp.route("/imports/<import_ref>", methods=["GET"])
def follow_import(import_ref):
    _importing()
    record = imports_module.get(_host(), _host().current_owner(), import_ref)
    if record is None:
        raise ApiError(404, "No such import.", "not_found")
    return jsonify(record)


def _found(record):
    if record is None:
        raise ApiError(404, "No such import.", "not_found")
    return jsonify(record)


@library_bp.route("/imports/<import_ref>/cancel", methods=["POST"])
def cancel_import(import_ref):
    """Cancel an import before it saves the dataset (409 `too_late` after)."""
    _importing()
    return _found(_starting(imports_module.cancel, import_ref))


@library_bp.route("/imports/<import_ref>/retry", methods=["POST"])
def retry_import(import_ref):
    """Start a failed or cancelled import again with the same settings."""
    _importing()
    record = _starting(imports_module.retry, import_ref)
    if record is None:
        raise ApiError(404, "No such import.", "not_found")
    return jsonify(record), 202


@library_bp.route("/imports/<import_ref>/dismiss", methods=["POST"])
def dismiss_import(import_ref):
    """Put away a failed or cancelled import (409 `running` while it runs)."""
    _importing()
    return _found(_starting(imports_module.dismiss, import_ref))


# --- Legal Data Hunter entries, for a course admin (#126) -------------------------------------------------------


def _ldh_admin():
    _require_admin()
    if not ldh_module.offered(_host()):
        raise ApiError(404, ldh_provider.NOT_OFFERED, "not_offered")


def _ldh_call(action, *args):
    """Run a Legal Data Hunter action (an admin's or a user's search), turning its refusals into answers. A key is never in an answer."""
    host = _host()
    try:
        return action(host, *args)
    except ldh_module.NotFound:
        raise ApiError(404, "No such entry.", "not_found")
    except ldh_module.NotOffered:
        raise ApiError(404, ldh_provider.NOT_OFFERED, "not_offered")
    except ldh_module.NeedsKey:
        raise ApiError(409, ldh_provider.NEEDS_KEY, "needs_key")
    except ldh_module.Invalid as invalid:
        host.db.session.rollback()
        raise ApiError(400, "Some fields need fixing.", "invalid", fields=invalid.fields)
    except ldh_provider.InvalidQuery as invalid:
        raise ApiError(400, "Some fields need fixing.", "invalid", fields={f"query.{k}": v for k, v in invalid.fields.items()})
    except ldh_provider.BadKey:
        raise ApiError(409, ldh_provider.BAD_KEY, "bad_key")
    except ldh_provider.OutOfCredits as ran_out:
        if ran_out.per_minute:
            raise ApiError(429, ldh_provider.SLOW_DOWN, "slow_down", retryAfter=ran_out.retry_after)
        raise ApiError(429, ldh_provider.CREDITS_RAN_OUT, "out_of_credits", retryAfter=ran_out.retry_after)
    except ProviderError:
        raise ApiError(503, ldh_provider.BUSY, "provider_unavailable")


def _json_object():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ApiError(400, "Send a JSON object.", "bad_request")
    return body


@library_bp.route("/legal-data-hunter/search", methods=["POST"])
def ldh_search():
    """Search Legal Data Hunter with the signed-in user's own key (#15): {q, corpus, country, source, dateFrom,
    dateTo}; 50 hits are asked for, whatever `limit` says. A 429 carries `retryAfter` and a `Retry-After` header."""
    if not ldh_module.searchable(_host()):
        raise ApiError(404, ldh_provider.NOT_OFFERED, "not_offered")
    body = _json_object()
    try:
        return jsonify(_ldh_call(ldh_module.search, body))
    except ApiError as refused:
        retry = refused.body.get("retryAfter")
        if refused.status == 429 and retry is not None:
            return jsonify(refused.body), 429, {"Retry-After": str(retry)}
        raise


@library_bp.route("/admin/legal-data-hunter/preview", methods=["POST"])
def ldh_preview():
    """Search Legal Data Hunter with the admin's own key: {q, corpus, country, source, dateFrom, dateTo, limit}."""
    _ldh_admin()
    return jsonify(_ldh_call(ldh_module.preview, _json_object()))


@library_bp.route("/admin/legal-data-hunter/entries", methods=["GET"])
def ldh_list():
    _ldh_admin()
    return jsonify({"entries": ldh_module.list_all(_host()), "max": ldh_provider.MAX_DOCUMENTS})


@library_bp.route("/admin/legal-data-hunter/entries", methods=["POST"])
def ldh_create():
    """Save an entry: {title, description, taskType, query, documents}; up to 100 documents."""
    _ldh_admin()
    return jsonify(_ldh_call(ldh_module.create, _json_object())), 201


@library_bp.route("/admin/legal-data-hunter/entries/<entry_ref>", methods=["GET"])
def ldh_get(entry_ref):
    _ldh_admin()
    return jsonify(_ldh_call(ldh_module.get, entry_ref))


@library_bp.route("/admin/legal-data-hunter/entries/<entry_ref>", methods=["PATCH"])
def ldh_update(entry_ref):
    """Change an entry's title, description or task type."""
    _ldh_admin()
    return jsonify(_ldh_call(ldh_module.update, entry_ref, _json_object()))


@library_bp.route("/admin/legal-data-hunter/entries/<entry_ref>/refresh", methods=["POST"])
def ldh_refresh(entry_ref):
    """Run the entry's query again with the admin's key and replace its documents; past imports are untouched."""
    _ldh_admin()
    return jsonify(_ldh_call(ldh_module.refresh, entry_ref))


@library_bp.route("/admin/legal-data-hunter/entries/<entry_ref>", methods=["DELETE"])
def ldh_remove(entry_ref):
    _ldh_admin()
    _ldh_call(ldh_module.remove, entry_ref)
    return "", 204
