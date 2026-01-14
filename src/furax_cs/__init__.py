"""FURAX component separation utilities."""

from importlib import metadata

from .kmeans_clusters import kmeans_clusters
from .multires_clusters import multires_clusters
from .noise import generate_noise_operator
from .optim import minimize

__all__ = [
    "kmeans_clusters",
    "multires_clusters",
    "generate_noise_operator",
    "minimize",
]


def __getattr__(name: str) -> str:
    """Expose package metadata attributes lazily."""
    if name == "__version__":
        try:
            return metadata.version("furax-cs")
        except metadata.PackageNotFoundError:
            return "unknown"
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
