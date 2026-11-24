"""Shared types, size limits and input validation."""

from typing import TypeAlias

Entry: TypeAlias = tuple[bytes, bytes | None]
"""A key and its value; a value of ``None`` marks a deletion (tombstone)."""

MAX_KEY_SIZE = 64 * 1024
"""Maximum key length in bytes (64 KiB)."""

MAX_VALUE_SIZE = 16 * 1024 * 1024
"""Maximum value length in bytes (16 MiB)."""


def validate_key(key: object) -> bytes:
    """Return ``key`` if it is a valid key.

    Only exact ``bytes`` are accepted; ``str``, ``bytearray`` and
    ``memoryview`` are rejected so that stored keys are immutable and
    hashable and callers make encoding decisions explicitly.

    Raises:
        TypeError: ``key`` is not ``bytes``.
        ValueError: ``key`` is empty or longer than ``MAX_KEY_SIZE``.
    """
    if not isinstance(key, bytes):
        raise TypeError(f"key must be bytes, not {type(key).__name__}")
    if not key:
        raise ValueError("key must not be empty")
    if len(key) > MAX_KEY_SIZE:
        raise ValueError(f"key is {len(key)} bytes; the maximum is {MAX_KEY_SIZE}")
    return key


def validate_value(value: object) -> bytes:
    """Return ``value`` if it is a valid value (empty values are allowed).

    Only exact ``bytes`` are accepted, as for :func:`validate_key`.

    Raises:
        TypeError: ``value`` is not ``bytes``.
        ValueError: ``value`` is longer than ``MAX_VALUE_SIZE``.
    """
    if not isinstance(value, bytes):
        raise TypeError(f"value must be bytes, not {type(value).__name__}")
    if len(value) > MAX_VALUE_SIZE:
        raise ValueError(f"value is {len(value)} bytes; the maximum is {MAX_VALUE_SIZE}")
    return value
