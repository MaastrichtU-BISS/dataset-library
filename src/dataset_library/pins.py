"""What a course admin pins: a starting point of library datasets for students.

A pin keeps the card as it was pinned and the labels verdict, so the badge
stays exact between visits to the Hub; the license level, though, is
re-derived on read, so an admin's later policy change applies at once. A
dataset must still be in the library to be pinned, and only what the library
still serves reaches the featured row (#119).
"""
from sqlalchemy.exc import IntegrityError

from . import labels as labels_module
from . import policy as policy_module
from . import preview as preview_module
from . import search as search_module
from .licenses import LICENSES, UNKNOWN
from .models import LibraryPin

PROVIDER = "huggingface"
MAX_FEATURED = 3
MAX_TITLE = 120

# The card fields a search result carries (`providers/huggingface._card`).
CARD_FIELDS = (
    "id", "provider", "externalId", "author", "title", "url", "tasks", "license",
    "size", "gated", "description", "updated", "revision", "downloads",
)


class NotFound(Exception):
    pass


class TooManyFeatured(Exception):
    pass


class BadFeatured(Exception):
    pass


class BadTitle(Exception):
    pass


def add(host, external_id):
    """Pin one library dataset, or return the pin that already exists.
    Refuses first what the preview refuses, so nothing outside the library
    is pinned (or named by a refusal)."""
    body = preview_module.preview(host, external_id)
    # The Hub's card may name the same dataset under another id (a rename or a
    # redirect); the pin, its labels and its audit all use that canonical id.
    canonical = body["externalId"]
    # A pin made before a rename keeps the old id: the canonical one wins,
    # the submitted one is the fallback.
    query = host.db.session.query(LibraryPin).filter_by(provider=PROVIDER)
    existing = query.filter_by(external_id=canonical).first() or query.filter_by(external_id=external_id).first()
    if existing is not None:
        return to_dict(existing, policy_module.overrides(host.db)), False
    labelled = labels_module.for_ids(host.cache, [canonical])[canonical]
    pin = LibraryPin(
        provider=PROVIDER,
        external_id=canonical,
        card={key: body[key] for key in CARD_FIELDS},
        labelled=labelled,
        pinned_by=host.current_owner(),
    )
    host.db.session.add(pin)
    try:
        host.db.session.commit()
    except IntegrityError:
        # Another request pinned it between our lookup and this insert.
        host.db.session.rollback()
        existing = host.db.session.query(LibraryPin).filter_by(provider=PROVIDER, external_id=canonical).first()
        if existing is not None:
            return to_dict(existing, policy_module.overrides(host.db)), False
        raise
    host.audit("library.pin_added", provider=PROVIDER, external_id=canonical)
    return to_dict(pin, policy_module.overrides(host.db)), True


def update(host, pin_ref, body):
    """Feature or unfeature a pin, and set or clear its display title."""
    # Featuring takes every pin's lock, always in id order and before any
    # other lock, so two requests featuring different pins cannot deadlock.
    locked = None
    if body.get("featured") is True:
        locked = host.db.session.query(LibraryPin).order_by(LibraryPin.id).populate_existing().with_for_update().all()
    # Lock and re-read the pin (already held when featuring), so a concurrent
    # change to it is seen before deciding what changes and what is audited.
    pin = (
        host.db.session.query(LibraryPin)
        .filter_by(id=pin_ref)
        .populate_existing()
        .with_for_update()
        .first()
    )
    if pin is None:
        raise NotFound()
    changes = {}
    audit = {"external_id": pin.external_id}
    if "featured" in body:
        featured = body["featured"]
        if not isinstance(featured, bool):
            raise BadFeatured()
        if featured != pin.featured:
            if featured:
                # Lock the pins so two requests cannot both see room for a fourth.
                count = sum(1 for other in locked if other.featured and other.id != pin.id)
                if count >= MAX_FEATURED:
                    raise TooManyFeatured()
            changes["featured"] = featured
            audit["featured"] = featured
    if "title" in body:
        title = body["title"]
        if isinstance(title, str):
            title = title.strip() or None
        elif title is not None:
            raise BadTitle()
        if title and len(title) > MAX_TITLE:
            raise BadTitle()
        if title != pin.title:
            changes["title"] = title
            # The title itself is free text; the audit only says what happened.
            audit["title_cleared" if title is None else "title_changed"] = True
    if not changes:
        return to_dict(pin, policy_module.overrides(host.db))
    for key, value in changes.items():
        setattr(pin, key, value)
    host.db.session.commit()
    host.audit("library.pin_updated", **audit)
    return to_dict(pin, policy_module.overrides(host.db))


def remove(host, pin_ref):
    pin = host.db.session.get(LibraryPin, pin_ref)
    if pin is None:
        raise NotFound()
    host.db.session.delete(pin)
    host.db.session.commit()
    host.audit("library.pin_removed", external_id=pin.external_id)


def list_all(host):
    """Every pin, oldest first, for the settings page's pins section."""
    policy = policy_module.overrides(host.db)
    pins = host.db.session.query(LibraryPin).order_by(LibraryPin.pinned_at, LibraryPin.id).all()
    return [to_dict(pin, policy) for pin in pins]


def featured(host):
    """The featured pins for the course's row: up to three, oldest pinned
    first, leaving out any the library's rules now refuse."""
    policy = policy_module.overrides(host.db)
    sources = {source.ref for source in search_module.enabled_sources(host.db)}
    found = []
    for pin in host.db.session.query(LibraryPin).filter_by(featured=True).order_by(LibraryPin.pinned_at, LibraryPin.id).all():
        if sources and pin.card.get("author") not in sources:
            continue
        if policy_module.level(_license_of(pin), policy) == "hide":
            continue
        found.append(to_dict(pin, policy))
        if len(found) == MAX_FEATURED:
            break
    return found


def to_dict(pin, policy):
    license = _license_of(pin)
    return {
        "ref": pin.id,
        "provider": pin.provider,
        "externalId": pin.external_id,
        "title": pin.title or pin.card.get("title"),
        "titleOverride": pin.title,
        "featured": pin.featured,
        "labelled": pin.labelled,
        "card": {**pin.card, "license": license.at_level(policy_module.level(license, policy))},
        "pinnedAt": pin.pinned_at.isoformat() + "Z" if pin.pinned_at else None,
    }


def _license_of(pin):
    return LICENSES.get((pin.card.get("license") or {}).get("id"), UNKNOWN)
