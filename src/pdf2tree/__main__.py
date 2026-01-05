"""Consente l'esecuzione tramite `python -m pdf2tree`."""

import sys

from .core import main


if __name__ == "__main__":
    sys.exit(main())
