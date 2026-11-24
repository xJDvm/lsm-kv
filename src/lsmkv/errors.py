"""Exception hierarchy for lsm-kv."""


class LSMError(Exception):
    """Base class for all errors raised by lsm-kv."""


class CorruptionError(LSMError):
    """Raised when on-disk data is malformed, truncated or fails a checksum."""


class LockedError(LSMError):
    """Raised when a database directory is already open elsewhere."""


class ClosedError(LSMError):
    """Raised when an operation is attempted on a closed database or file."""
