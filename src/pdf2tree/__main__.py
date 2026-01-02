"""Allow `python -m pdf2tree` execution."""

import sys

from .core import main


if __name__ == "__main__":
    sys.exit(main())
