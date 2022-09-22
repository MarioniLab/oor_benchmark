from importlib.metadata import version

from . import api, datasets, methods, metrics

__all__ = ["datasets", "metrics", "methods"]

__version__ = version("oor_benchmark")
