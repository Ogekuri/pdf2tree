"""Compatibility package for the pdf2tree CLI."""

from pdf2tree.core import main

__version__ = "0.0.3"

from .core import main  # noqa: F401

__all__ = ["__version__", "main"]  
