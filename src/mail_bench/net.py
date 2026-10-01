"""Length-prefixed JSON and NumPy-array messages for local policy servers."""

from __future__ import annotations

import json
import struct
from typing import Any


MAX_HEADER_BYTES = 16 * 1024 * 1024
MAX_ARRAY_BYTES = 512 * 1024 * 1024


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("NumPy is required for policy-server messages") from exc
    return np


def _encode(value: Any, arrays: list[Any]) -> Any:
    np = _numpy()
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        arrays.append(array)
        return {
            "__ndarray__": len(arrays) - 1,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "nbytes": int(array.nbytes),
        }
    if isinstance(value, dict):
        return {str(key): _encode(item, arrays) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item, arrays) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _decode(value: Any, buffers: list[bytes]) -> Any:
    np = _numpy()
    if isinstance(value, dict):
        if "__ndarray__" in value:
            array = np.frombuffer(
                buffers[int(value["__ndarray__"])], dtype=np.dtype(value["dtype"])
            )
            return array.reshape(value["shape"]).copy()
        return {key: _decode(item, buffers) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item, buffers) for item in value]
    return value


def _recv_exact(sock: Any, length: int) -> bytes:
    chunks = []
    remaining = length
    while remaining:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ConnectionError("socket closed in the middle of a message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(sock: Any, value: Any) -> None:
    """Send one message without pickle or Python-version-specific objects."""
    arrays: list[Any] = []
    header = json.dumps(_encode(value, arrays), separators=(",", ":")).encode()
    if len(header) > MAX_HEADER_BYTES:
        raise ValueError("policy message header exceeds the safety limit")
    if any(array.nbytes > MAX_ARRAY_BYTES for array in arrays):
        raise ValueError("policy message array exceeds the safety limit")
    sock.sendall(struct.pack("<I", len(header)))
    sock.sendall(header)
    for array in arrays:
        sock.sendall(array.tobytes())


def recv_message(sock: Any) -> Any:
    """Receive one message and reconstruct its NumPy arrays."""
    (header_length,) = struct.unpack("<I", _recv_exact(sock, 4))
    if header_length > MAX_HEADER_BYTES:
        raise ValueError("policy message header exceeds the safety limit")
    header = json.loads(_recv_exact(sock, header_length).decode())
    metadata: list[dict[str, Any]] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            if "__ndarray__" in value:
                metadata.append(value)
            else:
                for item in value.values():
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(header)
    metadata.sort(key=lambda item: int(item["__ndarray__"]))
    for expected, item in enumerate(metadata):
        if int(item["__ndarray__"]) != expected:
            raise ValueError("policy message contains invalid array indices")
        if int(item["nbytes"]) > MAX_ARRAY_BYTES:
            raise ValueError("policy message array exceeds the safety limit")
    buffers = [_recv_exact(sock, int(item["nbytes"])) for item in metadata]
    return _decode(header, buffers)
