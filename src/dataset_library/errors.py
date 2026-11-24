"""What a provider raises. Routes turn each into a status, never a 500."""


class ProviderError(Exception):
    """The provider is down, timed out, or answered with something unreadable."""


class ProviderUnavailable(ProviderError):
    pass


class ProviderRateLimited(ProviderError):
    pass


class NotParquet(ProviderUnavailable):
    """A listed file is not a Parquet file that can be read. Asking again cannot
    help, so an import refuses it rather than calling the provider busy."""


class RangeNotFetched(Exception):
    """A read of a shard's bytes that were never fetched: a bug in what was
    planned, not the provider's fault nor a file that is not Parquet. Not a
    ProviderError and not an OSError, so it ends an import or a copy as
    unexpected, logged."""
