"""RFC 8785 JSON Canonicalization Scheme (JCS).

Two independent implementations must hash *the same bytes* for an evidence root to mean
anything. ``json.dumps(sort_keys=True)`` is not sufficient: it does not specify number
formatting, and Python's default escaping differs from the spec. This module pins the
serialization exactly.
"""

from __future__ import annotations

import json
import math
import re
from decimal import Decimal
from typing import Any

# JCS mandates the ECMAScript Number::toString algorithm. Python's repr() is also a
# shortest-round-trip representation, but chooses scientific notation too early:
# Python has ``1e-06`` where ECMAScript (and JCS) requires ``0.000001``. Normalize the
# shared representation around that boundary below.
_EXPONENT = re.compile(r"^(-?)(\d)(?:\.(\d+))?e([+-])(\d+)$")


def _format_number(value: float | int) -> str:
    """Serialize a number per RFC 8785 section 3.2.2.3."""

    if isinstance(value, bool):  # bool is a subclass of int; callers must not reach here
        raise TypeError("bool is not a JSON number")
    if isinstance(value, int):
        return str(value)
    if math.isnan(value) or math.isinf(value):
        raise ValueError("NaN and Infinity cannot be canonicalized")
    if value == 0:
        # Canonical form has no negative zero.
        return "0"
    if value == int(value) and abs(value) < 1e21:
        return str(int(value))

    text = repr(float(value))
    if 1e-6 <= abs(value) < 1e21 and "e" in text.lower():
        return format(Decimal(text), "f")
    match = _EXPONENT.match(text)
    if match:
        sign, lead, rest, exp_sign, exp_digits = match.groups()
        mantissa = f"{lead}.{rest}" if rest else lead
        return f"{sign}{mantissa}e{exp_sign}{int(exp_digits)}"
    return text


def _escape(text: str) -> str:
    """Escape a JSON string per RFC 8785 section 3.2.2.2."""

    out: list[str] = ['"']
    for char in text:
        code = ord(char)
        if char == '"':
            out.append('\\"')
        elif char == "\\":
            out.append("\\\\")
        elif char == "\b":
            out.append("\\b")
        elif char == "\f":
            out.append("\\f")
        elif char == "\n":
            out.append("\\n")
        elif char == "\r":
            out.append("\\r")
        elif char == "\t":
            out.append("\\t")
        elif code < 0x20:
            out.append(f"\\u{code:04x}")
        else:
            # Everything else is emitted literally; the output is UTF-8 encoded.
            out.append(char)
    out.append('"')
    return "".join(out)


def _serialize(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return _escape(value)
    if isinstance(value, int | float):
        return _format_number(value)
    if isinstance(value, list | tuple):
        return "[" + ",".join(_serialize(item) for item in value) + "]"
    if isinstance(value, dict):
        # JCS sorts object keys by their UTF-16 code-unit sequence.
        items = sorted(value.items(), key=lambda kv: _utf16_sort_key(kv[0]))
        return "{" + ",".join(f"{_escape(k)}:{_serialize(v)}" for k, v in items) + "}"
    raise TypeError(f"{type(value).__name__} is not JSON-serializable for canonicalization")


def _utf16_sort_key(text: str) -> tuple[int, ...]:
    """Key that orders strings by UTF-16 code units, as RFC 8785 requires.

    Python compares by code point, which differs from UTF-16 order for characters above
    the BMP (surrogates sort below U+E000). Encoding to UTF-16BE makes the order exact.
    """

    raw = text.encode("utf-16-be")
    return tuple(int.from_bytes(raw[i : i + 2], "big") for i in range(0, len(raw), 2))


def canonicalize(value: Any) -> bytes:
    """Return the RFC 8785 canonical UTF-8 encoding of a JSON-compatible value."""

    if not isinstance(value, str | int | float | bool | type(None) | list | tuple | dict):
        raise TypeError(f"cannot canonicalize {type(value).__name__}")
    return _serialize(value).encode("utf-8")


def loads_canonical(data: bytes) -> Any:
    """Parse canonical bytes back into Python values."""

    return json.loads(data.decode("utf-8"))


__all__ = ["canonicalize", "loads_canonical"]
