"""What a course admin pins: a starting point of library datasets for students.

A pin keeps the card as it was pinned and the labels verdict, so the badge
stays exact between visits to the Hub; the license level, though, is
re-derived on read, so an admin's later policy change applies at once. A
dataset must still be in the library to be pinned, and only what the library
still serves reaches the featured row (#119).

A pin whose license is Allow also gets a copy, so it can still be imported while
Hugging Face is down; `offline` lists the pins for a search that cannot reach the
Hub (#121).
"""
from sqlalchemy.exc import IntegrityError

from . import labels as labels_module
from . import policy as policy_module
from . import preview as preview_module
from . import search as search_module
from . import snapshots as snapshots_module
from .licenses import LICENSES, UNKNOWN
from .models import LibraryPin

PROVIDER = "huggingface"
MAX_FEATURED = 3
MAX_TITLE = 120

# The card fields a search result carries (`providers/huggingface._card`). The
# citation is kept too, so a featured card can still be cited (#120).
CARD_FIELDS = (
    "id", "provider", "externalId", "author", "title", "url", "tasks", "license",
    "size", "gated", "description", "updated", "revision", "downloads", "citation",
)


class NotFound(Exception):
    pass


class TooManyFeatured(Exception):
    pass


class BadFeatured(Exception):
    pass


class BadTitle(Exception):
    pass


class NotOffered(Exception):
    """This platform keeps no copies (the host set no snapshot directory or queue)."""


class NotAllowed(Exception):
    """The dataset's license is not Allow, so it gets no copy."""


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
        return to_dict(existing, policy_module.overrides(host.db), snapshots_module.for_pin(host.db.session, existing.id)), False
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
            return to_dict(existing, policy_module.overrides(host.db), snapshots_module.for_pin(host.db.session, existing.id)), False
        raise
    host.audit("library.pin_added", provider=PROVIDER, external_id=canonical)
    # The copy is made by the worker; this only queues it (#121).
    copy = snapshots_module.request_copy(host, pin, body)
    return to_dict(pin, policy_module.overrides(host.db), copy), True


def take_copy(host, pin_ref):
    """Keep a copy of a pin that has none, or whose copy failed, when its license is
    Allow. The preview is asked afresh: the copy takes the dataset's part and revision
    as they are now."""
    if host.snapshot_dir is None or host.enqueue_snapshot is None:
        raise NotOffered()
    session = host.db.session
    pin = session.get(LibraryPin, pin_ref)
    if pin is None:
        raise NotFound()
    if policy_module.level(_license_of(pin), policy_module.overrides(host.db)) != "allow":
        raise NotAllowed()
    body = preview_module.preview(host, pin.external_id)
    copy = snapshots_module.request_copy(host, pin, body)
    return to_dict(pin, policy_module.overrides(host.db), copy)


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
    copy = snapshots_module.for_pin(host.db.session, pin.id)
    if not changes:
        return to_dict(pin, policy_module.overrides(host.db), copy)
    for key, value in changes.items():
        setattr(pin, key, value)
    host.db.session.commit()
    host.audit("library.pin_updated", **audit)
    return to_dict(pin, policy_module.overrides(host.db), copy)


def remove(host, pin_ref):
    """Unpin: the pin goes, and so does its copy, unless a running import still reads
    the copy; then the copy goes when that import has finished (#121)."""
    session = host.db.session
    pin = session.get(LibraryPin, pin_ref)
    if pin is None:
        raise NotFound()
    external_id = pin.external_id
    copy = snapshots_module.for_pin(session, pin.id)
    session.delete(pin)
    if copy is None:
        session.commit()
    else:
        # One commit for the pin and its copy, so the copy is never served without its pin.
        snapshots_module.discard(session, host.snapshot_dir, [copy])
    host.audit("library.pin_removed", external_id=external_id)


def reconcile_copies(host):
    """Take down the copies whose license the admin's policy no longer allows. Returns
    how many went."""
    return snapshots_module.reconcile(host.db.session, host.snapshot_dir, policy_module.overrides(host.db))


def list_all(host):
    """Every pin, oldest first, for the settings page's pins section."""
    policy = policy_module.overrides(host.db)
    copies = snapshots_module.by_pin(host.db.session)
    pins = host.db.session.query(LibraryPin).order_by(LibraryPin.pinned_at, LibraryPin.id).all()
    return [to_dict(pin, policy, copies.get(pin.id)) for pin in pins]


def featured(host):
    """The featured pins for the course's row: up to three, oldest pinned
    first, leaving out any the library's rules now refuse."""
    policy = policy_module.overrides(host.db)
    sources = {source.ref for source in search_module.enabled_sources(host.db)}
    copies = snapshots_module.by_pin(host.db.session)
    found = []
    for pin in host.db.session.query(LibraryPin).filter_by(featured=True).order_by(LibraryPin.pinned_at, LibraryPin.id).all():
        if sources and pin.card.get("author") not in sources:
            continue
        if policy_module.level(_license_of(pin), policy) == "hide":
            continue
        found.append(to_dict(pin, policy, copies.get(pin.id)))
        if len(found) == MAX_FEATURED:
            break
    return found


def offline(host, *, query, task, license_id, source):
    """The pinned datasets a search returns while Hugging Face cannot be asked (#121):
    filtered as the search's filters say, leaving out what the library now refuses. Each
    card says whether its copy is ready, which is what lets it be imported meanwhile."""
    policy = policy_module.overrides(host.db)
    sources = {s.ref for s in search_module.enabled_sources(host.db)}
    copies = snapshots_module.by_pin(host.db.session)
    needle = query.lower()
    results = []
    for pin in host.db.session.query(LibraryPin).order_by(LibraryPin.pinned_at, LibraryPin.id).all():
        card, license = pin.card, _license_of(pin)
        if sources and card.get("author") not in sources:
            continue
        if source and card.get("author") != source:
            continue
        level = policy_module.level(license, policy)
        if level == "hide":
            continue
        if task and task not in (card.get("tasks") or []):
            continue
        if license_id == "unknown" and not license.unknown:
            continue
        if license_id and license_id != "unknown" and license.id != license_id:
            continue
        if needle and needle not in f"{pin.external_id} {pin.title or card.get('title') or ''}".lower():
            continue
        results.append({
            **card,
            "license": license.at_level(level),
            "pinRef": pin.id,
            "labelled": pin.labelled,
            "snapshot": snapshots_module.summary(copies.get(pin.id)),
        })
    return results


def to_dict(pin, policy, copy=None):
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
        # The copy this pin's imports read while Hugging Face is down; `upstreamChanged`
        # says the Hub no longer has the copy's revision (the copy is kept regardless).
        "snapshot": snapshots_module.summary(copy),
        "upstreamChanged": bool(copy is not None and copy.upstream_changed),
        "pinnedAt": pin.pinned_at.isoformat() + "Z" if pin.pinned_at else None,
    }


def _license_of(pin):
    return LICENSES.get((pin.card.get("license") or {}).get("id"), UNKNOWN)
