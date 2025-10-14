"""FURAX component separation utilities."""

from importlib import metadata


def __getattr__(name: str):
    """Expose package metadata attributes lazily."""
    if name == "__version__":
        try:
            return metadata.version("furax-cs")
        except metadata.PackageNotFoundError:
            return "unknown"
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
