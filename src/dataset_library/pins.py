"""What a course admin pins: a starting point of library datasets for students.

A pin keeps the card as it was pinned and the labels verdict, so the badge
stays exact between visits to the Hub; the license level, though, is
re-derived on read, so an admin's later policy change applies at once. A
dataset must still be in the library to be pinned, and only what the library
still serves reaches the featured row (#119).
"""
from sqlalchemy import func

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
    existing = host.db.session.query(LibraryPin).filter_by(provider=PROVIDER, external_id=external_id).first()
    if existing is not None:
        return to_dict(existing, policy_module.overrides(host.db)), False
    labelled = labels_module.for_ids(host.cache, [external_id])[external_id]
    pin = LibraryPin(
        provider=PROVIDER,
        external_id=external_id,
        card={key: body[key] for key in CARD_FIELDS},
        labelled=labelled,
        pinned_by=host.current_owner(),
    )
    host.db.session.add(pin)
    host.db.session.commit()
    host.audit("library.pin_added", provider=PROVIDER, external_id=external_id)
    return to_dict(pin, policy_module.overrides(host.db)), True


def update(host, pin_ref, body):
    """Feature or unfeature a pin, and set or clear its display title."""
    pin = host.db.session.get(LibraryPin, pin_ref)
    if pin is None:
        raise NotFound()
    changes = {}
    if "featured" in body:
        featured = body["featured"]
        if not isinstance(featured, bool):
            raise BadFeatured()
        if featured and not pin.featured:
            count = (
                host.db.session.query(func.count())
                .select_from(LibraryPin)
                .filter(LibraryPin.featured.is_(True), LibraryPin.id != pin.id)
                .scalar()
            )
            if count >= MAX_FEATURED:
                raise TooManyFeatured()
        changes["featured"] = featured
    if "title" in body:
        title = body["title"]
        if isinstance(title, str):
            title = title.strip() or None
        elif title is not None:
            raise BadTitle()
        if title and len(title) > MAX_TITLE:
            raise BadTitle()
        changes["title"] = title
    if "featured" in body:
        pin.featured = body["featured"]
    if "title" in body:
        pin.title = changes["title"]
    host.db.session.commit()
    host.audit("library.pin_updated", external_id=pin.external_id, **changes)
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
