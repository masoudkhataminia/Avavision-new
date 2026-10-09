"""Entry point of the packaged Windows app: opens the station window unless a command is given."""

import sys

from avavision.__main__ import main

if __name__ == "__main__":
    main(sys.argv[1:] or ["desktop"])
