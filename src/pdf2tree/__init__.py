"""Pacchetto di compatibilità per la CLI di pdf2tree."""

from pdf2tree.core import main

__version__ = "0.0.9"

from .core import main  # noqa: F401

__all__ = ["__version__", "main"]  
