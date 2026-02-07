"""TRExFitter config migration helpers (parser + import to analysis spec).

This module is intentionally dependency-light and does not require the Rust
extension to be built.
"""

from .parser import (  # noqa: F401
    TrexConfigBlock,
    TrexConfigDoc,
    TrexConfigEntry,
    TrexConfigParseError,
    TrexValue,
    parse_trex_config,
    parse_trex_config_file,
)

