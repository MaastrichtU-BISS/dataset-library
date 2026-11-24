"""What a course admin controls: the sources search looks in, and each
license's level (design review DR14). Saved whole, validated whole: one bad
field and nothing changes."""
from . import community as community_module
from . import limits as limits_module
from .licenses import LEVELS, LICENSES
from .models import LibrarySettings, LibrarySource
from .policy import level, overrides
from .providers import community as community_provider
from .providers import huggingface


class Invalid(Exception):
    def __init__(self, fields):
        super().__init__("Invalid settings")
        self.fields = fields


def current(db, host):
    policy = overrides(db)
    settings = db.session.get(LibrarySettings, 1)
    sources = db.session.query(LibrarySource).filter_by(provider="huggingface").order_by(LibrarySource.ref).all()
    return {
        "sources": [s.to_dict() for s in sources],
        # Offered only where the host gave the Community seams (#122); the admin switches it on or off.
        "community": {"available": host.community is not None, "enabled": community_module.switched_on(db.session)},
        # Offered where the host gave the key seam (#126); its entries are managed under /admin/legal-data-hunter.
        "legalDataHunter": {"available": getattr(host, "legal_data_hunter_key", None) is not None},
        "licenses": [
            {**lic.at_level(level(lic, policy)), "default": lic.level} for lic in LICENSES.values()
        ],
        "limits": {
            "maxRows": settings.max_rows if settings else None,
            "maxRowsDefault": host.max_rows(),
            "maxRowsHardMax": host.hard_max_rows,
            "maxDownloadMb": settings.max_download_mb if settings else None,
            "maxDownloadMbDefault": limits_module.DEFAULT_MAX_DOWNLOAD_MB,
            "maxDownloadMbHardMax": limits_module.HARD_MAX_DOWNLOAD_MB,
        },
    }


def check_source(ref):
    """None when `ref` is a Hugging Face organisation with public datasets, else why not.
    Raises ProviderError when the Hub cannot answer."""
    if not huggingface.ORGANISATION_PATTERN.match(ref or ""):
        return "That is not an organisation name. Use the part before the slash in a dataset id, e.g. lexlms."
    if not huggingface.organisation_has_datasets(ref):
        return f"No public datasets from “{ref}” on Hugging Face. Check the organisation’s name."
    return None


def save(db, body, hard_max_rows=limits_module.HARD_MAX_ROWS, community_available=False):
    """Replace the sources, the license policy and (when given) the limits, and switch the Community
    source on or off when the body says so. Returns what changed, for the audit."""
    sources = body.get("sources")
    policy = body.get("license_policy")
    fields = {}
    if not isinstance(sources, list):
        fields["sources"] = "A list of sources is required."
        sources = []
    if not isinstance(policy, dict):
        fields["license_policy"] = "License levels are required."
        policy = {}
    for license_id, chosen in policy.items():
        if license_id not in LICENSES:
            fields[f"license_policy.{license_id}"] = "Not a license the library knows."
        elif chosen not in LEVELS:
            fields[f"license_policy.{license_id}"] = "Choose Allow, Warn or Hide."
        elif chosen not in LICENSES[license_id].levels:
            # A provider's terms differ for each user's plan, so they are never Allow (#126).
            fields[f"license_policy.{license_id}"] = "Choose Warn or Hide."

    limits = body.get("limits")
    if limits is not None:
        fields.update(limits_module.check(limits, hard_max_rows))

    community_body = body.get("community")
    community_on = None
    if community_body is not None:
        if not isinstance(community_body, dict) or not isinstance(community_body.get("enabled"), bool):
            fields["community"] = "Say whether the Community source is on or off."
        elif community_body["enabled"] and not community_available:
            fields["community"] = "The Community source is not set up on this platform."
        else:
            community_on = community_body["enabled"]

    existing = {s.ref: s for s in db.session.query(LibrarySource).filter_by(provider="huggingface")}
    seen = set()
    wanted = []
    for index, source in enumerate(sources):
        ref = (source or {}).get("ref", "").strip() if isinstance(source, dict) else ""
        # The search filter names the Community source as "community", so no organisation may take that name.
        if ref == community_provider.SOURCE:
            fields[f"sources.{index}"] = "That name belongs to the Community source, so it cannot be an organisation."
            continue
        if ref in seen:
            fields[f"sources.{index}"] = "This source is already listed."
            continue
        seen.add(ref)
        if ref not in existing:
            problem = check_source(ref)
            if problem:
                fields[f"sources.{index}"] = problem
                continue
        wanted.append((ref, bool(source.get("enabled", True))))
    if fields:
        raise Invalid(fields)

    before = {lic.id: level(lic, overrides(db)) for lic in LICENSES.values()}
    for ref, row in existing.items():
        if ref not in seen:
            db.session.delete(row)
    for ref, enabled in wanted:
        if ref in existing:
            existing[ref].enabled = enabled
        else:
            db.session.add(LibrarySource(provider="huggingface", kind="org", ref=ref, enabled=enabled))
    settings = db.session.get(LibrarySettings, 1)
    if settings is None:
        settings = LibrarySettings(id=1, license_policy={})
        db.session.add(settings)
    # Keep only levels that differ from the default, so a new default applies.
    settings.license_policy = {lid: lvl for lid, lvl in policy.items() if LICENSES[lid].level != lvl}
    limits_changed = []
    if limits is not None:
        for key, column in (("maxRows", "max_rows"), ("maxDownloadMb", "max_download_mb")):
            if getattr(settings, column) != limits.get(key):
                setattr(settings, column, limits.get(key))
                limits_changed.append(column)
    community_changed = False
    if community_on is not None:
        community_changed = community_module.switched_on(db.session) != community_on
        community_module.set_enabled(db.session, community_on)
    db.session.commit()
    after = {lic.id: level(lic, settings.license_policy) for lic in LICENSES.values()}
    changes = {
        "sources_added": len([ref for ref, _ in wanted if ref not in existing]),
        "sources_removed": len([ref for ref in existing if ref not in seen]),
        "licenses_changed": sorted(lid for lid in LICENSES if before[lid] != after[lid]),
        "limits_changed": sorted(limits_changed),
    }
    # Only a save that sets the Community switch says so, so other saves audit as they always have.
    if community_on is not None:
        changes["community_changed"] = community_changed
    return changes
