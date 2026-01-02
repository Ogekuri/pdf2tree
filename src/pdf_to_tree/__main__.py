"""Consente l'esecuzione del tool come modulo."""
from .core import main
import sys


if __name__ == "__main__":
    sys.exit(main())
