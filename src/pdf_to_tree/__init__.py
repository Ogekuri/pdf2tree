"""Pacchetto principale della CLI git-alias per uvx."""

__version__ = "0.0.1"

from .core import main  # noqa: F401

__all__ = ["__version__", "main"]
