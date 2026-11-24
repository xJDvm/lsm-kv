"""Exception hierarchy for lsm-kv."""


class LSMError(Exception):
    """Base class for all errors raised by lsm-kv."""


class CorruptionError(LSMError):
    """Raised when on-disk data is malformed, truncated or fails a checksum."""


class ClosedError(LSMError):
    """Raised when an operation is attempted on a closed database or file."""
