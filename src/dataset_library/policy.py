"""The license policy: each license's level, as an admin set it or by default."""
from sqlalchemy import select

from .models import LibrarySettings


def overrides(db):
    return for_session(db.session)


def for_session(session):
    """The admin's license levels as the database holds them now. A select rather than
    the session's identity map, so a check made mid-build sees a change made elsewhere."""
    policy = session.execute(select(LibrarySettings.license_policy).where(LibrarySettings.id == 1)).scalar_one_or_none()
    return dict(policy or {})


def level(license, policy):
    """The level a license is applied at. A provider's terms are never Allow, whatever was stored (#126)."""
    chosen = policy.get(license.id, license.level)
    return chosen if chosen in license.levels else license.level
