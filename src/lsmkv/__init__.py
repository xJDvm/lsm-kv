"""lsm-kv: an embedded, persistent key-value store built on an LSM tree."""

from lsmkv.errors import ClosedError, CorruptionError, LSMError

__version__ = "0.1.0"

__all__ = ["ClosedError", "CorruptionError", "LSMError", "__version__"]
