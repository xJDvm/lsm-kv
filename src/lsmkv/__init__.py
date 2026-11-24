"""lsm-kv: an embedded, persistent key-value store built on an LSM tree."""

from lsmkv.db import DB
from lsmkv.errors import ClosedError, CorruptionError, LockedError, LSMError

__version__ = "0.1.0"

__all__ = ["DB", "ClosedError", "CorruptionError", "LSMError", "LockedError", "__version__"]
