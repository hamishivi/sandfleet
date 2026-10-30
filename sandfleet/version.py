"""Package version and minimum interpreter requirement."""

import sys

if sys.version_info < (3, 11):
    raise RuntimeError(
        "Sandfleet requires Python 3.11 or newer; "
        f"found Python {sys.version_info[0]}.{sys.version_info[1]}"
    )

VERSION = "0.16.1"
