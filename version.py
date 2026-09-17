"""Single source of truth for the project version.

Bumped by hand. Every command-line entry point accepts ``--version`` and
reports this string together with the Python and OS it is running on, which
is the first thing worth knowing in a bug report.
"""

from __future__ import annotations

import platform
import sys

__version__ = "1.0.0"


def version_string() -> str:
    """One line identifying the build, the interpreter and the platform."""
    return (f"2048-ai {__version__}  "
            f"(Python {platform.python_version()} on "
            f"{platform.system()} {platform.machine()}, "
            f"{sys.maxsize.bit_length() + 1}-bit)")
