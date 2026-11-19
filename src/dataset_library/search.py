"""Search across sources, with results cached and the license policy applied.

The cache holds the provider's cards before the policy, so an admin's change
applies at once. A fresh copy lasts ten minutes; the last good copy lasts a
day and is served, marked stale, while Hugging Face is limiting requests."""
import hashlib
import json
import logging
import time

from . import community
from . import policy as policy_module
from .errors import ProviderRateLimited, ProviderUnavailable
from .licenses import LICENSES, UNKNOWN
from .models import LibraryPin, LibrarySource
from .providers import community as community_provider
from .providers import huggingface

log = logging.getLogger("dataset_library")

FRESH_SECONDS = 600
LAST_GOOD_SECONDS = 86_400


def enabled_sources(db):
    return (
        db.session.query(LibrarySource)
        .filter_by(provider="huggingface", enabled=True)
        .order_by(LibrarySource.ref)
        .all()
    )


def search(host, *, query, task, license_id, source):
    """`source` must be one of the enabled sources' refs, `community`, or empty for all.

    Community cards are found whatever the Hub is doing: a Hub outage leaves them in place (#122)."""
    policy = policy_module.overrides(host.db)
    if license_id and policy.get(license_id) == "hide":
        return _body("ok", [])
    results, status = [], "ok"
    if source != community_provider.SOURCE:
        results, status = _hugging_face(host, query, task, license_id, source, policy)
    if source in ("", community_provider.SOURCE):
        results = results + community.cards(host, query=query, task=task, license_id=license_id)
    return _body(status, results)


def _hugging_face(host, query, task, license_id, source, policy):
    """The Hub's cards with the policy applied, or the pinned datasets when the Hub cannot be asked, and the status."""
    authors = [s.ref for s in enabled_sources(host.db)]
    if source:
        authors = [source]
    # With Community on, no organisation and no query, the Hub would list its whole catalogue beside the Community
    # cards: nothing is asked for (#122). Without Community, the Hub's own answer stands, as it always has.
    if not authors and not (query or "").strip() and community.enabled(host):
        return [], "ok"
    # The Hub has no tag for "no license", so those are picked out here.
    hub_license = None if license_id == "unknown" else license_id
    started = time.monotonic()
    key = _key(query, task, hub_license, authors)
    cards, status, cache_hit = host.cache.get(f"library:search:{key}"), "ok", True
    if cards is None:
        cache_hit = False
        try:
            cards = huggingface.search(query, task, hub_license, authors, host.task_types)
        except ProviderRateLimited:
            cards, status = host.cache.get(f"library:last:{key}"), "stale"
            if cards is None:
                status = "unavailable"
        except ProviderUnavailable as exc:
            log.warning("library.search.provider_error provider=huggingface error=%s", exc)
            status = "unavailable"
        else:
            host.cache.set(f"library:search:{key}", cards, FRESH_SECONDS)
            host.cache.set(f"library:last:{key}", cards, LAST_GOOD_SECONDS)
    if status == "unavailable":
        # Hugging Face cannot be asked: the pinned datasets stand in, each with its copy (#121).
        from . import pins as pins_module  # lazy: pins imports this module

        results = pins_module.offline(host, query=query, task=task, license_id=license_id, source=source)
    else:
        results = _with_pin_refs(host.db, _apply_policy(cards, policy, license_id))
    log.info(
        "library.search provider=huggingface query=%r task=%s license=%s status=%s results=%d cache_hit=%s ms=%d",
        query, task, license_id, status, len(results), cache_hit, (time.monotonic() - started) * 1000,
    )
    return results, status


def _apply_policy(cards, policy, license_id):
    results = []
    for card in cards:
        license = LICENSES.get(card["license"], UNKNOWN)
        if license_id == "unknown" and not license.unknown:
            continue
        card_level = policy_module.level(license, policy)
        if card_level == "hide":
            continue
        results.append({**card, "license": license.at_level(card_level)})
    return results


def _with_pin_refs(db, results):
    """Every card carries the ref of its pin, when it has one; one query."""
    external_ids = [card["externalId"] for card in results]
    pins = {
        pin.external_id: pin.id
        for pin in db.session.query(LibraryPin).filter(
            LibraryPin.provider == "huggingface", LibraryPin.external_id.in_(external_ids)
        )
    } if external_ids else {}
    return [{**card, "pinRef": pins.get(card["externalId"])} for card in results]


def _key(*parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:32]


def _body(status, results):
    return {"status": status, "results": results, "count": len(results)}
