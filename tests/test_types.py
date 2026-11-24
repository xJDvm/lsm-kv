"""Tests for key/value validation and shared limits."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from lsmkv import ClosedError, CorruptionError, LSMError
from lsmkv.types import MAX_KEY_SIZE, MAX_VALUE_SIZE, validate_key, validate_value


def test_limits() -> None:
    assert MAX_KEY_SIZE == 65536
    assert MAX_VALUE_SIZE == 16 * 1024 * 1024


def test_error_hierarchy() -> None:
    assert issubclass(CorruptionError, LSMError)
    assert issubclass(ClosedError, LSMError)
    assert issubclass(LSMError, Exception)


def test_key_empty_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_key(b"")


def test_key_single_byte_ok() -> None:
    assert validate_key(b"\x00") == b"\x00"


def test_key_max_size_ok() -> None:
    key = b"k" * MAX_KEY_SIZE
    assert validate_key(key) is key


def test_key_over_max_rejected() -> None:
    with pytest.raises(ValueError, match="maximum"):
        validate_key(b"k" * (MAX_KEY_SIZE + 1))


@pytest.mark.parametrize("bad", ["key", bytearray(b"key"), memoryview(b"key"), 1, None, [b"k"]])
def test_key_wrong_type_rejected(bad: object) -> None:
    with pytest.raises(TypeError, match="bytes"):
        validate_key(bad)


def test_value_empty_ok() -> None:
    assert validate_value(b"") == b""


def test_value_max_size_ok() -> None:
    value = b"v" * MAX_VALUE_SIZE
    assert validate_value(value) is value


def test_value_over_max_rejected() -> None:
    with pytest.raises(ValueError, match="maximum"):
        validate_value(b"v" * (MAX_VALUE_SIZE + 1))


@pytest.mark.parametrize("bad", ["value", bytearray(b"v"), memoryview(b"v"), 0, None])
def test_value_wrong_type_rejected(bad: object) -> None:
    with pytest.raises(TypeError, match="bytes"):
        validate_value(bad)


@given(st.binary(min_size=1, max_size=256))
def test_key_roundtrip(key: bytes) -> None:
    assert validate_key(key) == key


@given(st.binary(max_size=256))
def test_value_roundtrip(value: bytes) -> None:
    assert validate_value(value) == value
