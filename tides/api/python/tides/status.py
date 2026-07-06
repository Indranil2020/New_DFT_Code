"""Status and Result objects for TIDES Python API.

Mirrors the C++ Status/Result pattern from core/common/status.hpp.
No exceptions cross the API boundary (coding standard 35-coding-standards.md).
No try/except control flow (linter rule ERR001).
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Generic, TypeVar, Optional

T = TypeVar("T")


class StatusCode(enum.IntEnum):
    """Typed status codes matching C++ enum class StatusCode."""
    OK = 0
    INVALID_ARGUMENT = 1
    OUT_OF_RANGE = 2
    IO_ERROR = 3
    CORRUPT_DATA = 4
    UNIMPLEMENTED = 5
    NUMERICAL_ERROR = 6
    CONVERGENCE_FAILED = 7


@dataclass(frozen=True)
class Status:
    """Immutable status object. Errors are values, not jumps."""
    code: StatusCode = StatusCode.OK
    message: str = ""

    @staticmethod
    def ok() -> "Status":
        return Status(code=StatusCode.OK, message="")

    @staticmethod
    def invalid_argument(msg: str) -> "Status":
        return Status(code=StatusCode.INVALID_ARGUMENT, message=msg)

    @staticmethod
    def out_of_range(msg: str) -> "Status":
        return Status(code=StatusCode.OUT_OF_RANGE, message=msg)

    @staticmethod
    def io_error(msg: str) -> "Status":
        return Status(code=StatusCode.IO_ERROR, message=msg)

    @staticmethod
    def corrupt_data(msg: str) -> "Status":
        return Status(code=StatusCode.CORRUPT_DATA, message=msg)

    @staticmethod
    def unimplemented(msg: str) -> "Status":
        return Status(code=StatusCode.UNIMPLEMENTED, message=msg)

    @staticmethod
    def numerical_error(msg: str) -> "Status":
        return Status(code=StatusCode.NUMERICAL_ERROR, message=msg)

    @staticmethod
    def convergence_failed(msg: str) -> "Status":
        return Status(code=StatusCode.CONVERGENCE_FAILED, message=msg)

    @property
    def is_ok(self) -> bool:
        return self.code == StatusCode.OK

    def __bool__(self) -> bool:
        return self.is_ok

    def __repr__(self) -> str:
        if self.is_ok:
            return "Status(OK)"
        return f"Status({self.code.name}: {self.message})"


@dataclass
class Result(Generic[T]):
    """Result wrapper: either a value (if Status.ok) or an error Status."""
    _status: Status = field(default_factory=lambda: Status(code=StatusCode.OK, message=""))
    _value: Optional[T] = None

    @staticmethod
    def ok(value: T) -> "Result[T]":
        return Result(_status=Status.ok(), _value=value)

    @staticmethod
    def err(status: Status) -> "Result[T]":
        assert not status.is_ok, "err() requires a non-OK status"
        return Result(_status=status, _value=None)

    @property
    def status(self) -> Status:
        return self._status

    @property
    def is_ok(self) -> bool:
        return self._status.is_ok

    @property
    def value(self) -> T:
        assert self.is_ok, f"Cannot access value of error result: {self._status}"
        assert self._value is not None
        return self._value

    def __repr__(self) -> str:
        if self.is_ok:
            return f"Result(ok={self._value!r})"
        return f"Result(err={self._status!r})"
