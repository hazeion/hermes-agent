"""Private, structured telemetry for trusted local integrations.

The progress channel intentionally exposes only a small, versioned event
surface. It never writes tool arguments/results or raw model reasoning.
"""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional


_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,120}$")
_PROGRESS_TYPES = {"tool.started", "tool.completed", "reasoning.available"}
_MAX_PROGRESS_EVENTS = 200
_MAX_PROGRESS_BYTES = 256 * 1024
_MAX_USAGE_BYTES = 32 * 1024


def _positive_int(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def build_usage_report(result: dict, failure: Optional[str] = None) -> dict:
    """Return the bounded usage contract shared by quiet chat and oneshot."""
    context_tokens = _positive_int(result.get("last_prompt_tokens"))
    context_length = _positive_int(result.get("context_length"))
    if (
        context_tokens is None
        or context_length is None
        or context_tokens > context_length
    ):
        context_tokens = None
        context_length = None

    report = {
        "schema_version": 1,
        "estimated_cost_usd": result.get("estimated_cost_usd"),
        "cost_status": result.get("cost_status"),
        "cost_source": result.get("cost_source"),
        "input_tokens": result.get("input_tokens"),
        "output_tokens": result.get("output_tokens"),
        "cache_read_tokens": result.get("cache_read_tokens"),
        "cache_write_tokens": result.get("cache_write_tokens"),
        "reasoning_tokens": result.get("reasoning_tokens"),
        "total_tokens": result.get("total_tokens"),
        "context_tokens": context_tokens,
        "context_length": context_length,
        "api_calls": result.get("api_calls"),
        "model": result.get("model"),
        "provider": result.get("provider"),
        "session_id": result.get("session_id"),
        "completed": result.get("completed"),
        "failed": bool(result.get("failed")) or failure is not None,
        "service_tier": result.get("service_tier"),
    }
    if failure is not None:
        report["failure"] = str(failure)[:500]
    return report


def write_usage_file(
    path: Optional[str],
    result: dict,
    failure: Optional[str] = None,
    *,
    strict: bool = False,
) -> None:
    """Best-effort usage write; strict mode requires a server-owned file."""
    if not path:
        return
    try:
        out = Path(path).expanduser()
        if not strict and not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                out,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            os.close(descriptor)
        payload = (
            json.dumps(build_usage_report(result, failure=failure), indent=2) + "\n"
        ).encode("utf-8")
        if len(payload) > _MAX_USAGE_BYTES:
            return
        if strict:
            _write_server_owned_file(out, payload, append=False)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temp_name = tempfile.mkstemp(
                prefix=f".{out.name}.",
                suffix=".tmp",
                dir=out.parent,
            )
            try:
                view = memoryview(payload)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("short telemetry write")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            Path(temp_name).replace(out)
    except Exception:
        try:
            if "temp_name" in locals():
                Path(temp_name).unlink(missing_ok=True)
        except Exception:
            pass
        pass


def _write_server_owned_file(path: Path, payload: bytes, *, append: bool) -> None:
    """Write through a no-follow directory descriptor without creating paths."""
    if (
        os.open not in getattr(os, "supports_dir_fd", set())
        or not hasattr(os, "geteuid")
    ):
        raise OSError("strict telemetry writes are unavailable")
    parent_fd = os.open(
        path.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        flags = os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        if append:
            flags |= os.O_APPEND
        descriptor = os.open(path.name, flags, dir_fd=parent_fd)
        try:
            details = os.fstat(descriptor)
            if (
                not stat.S_ISREG(details.st_mode)
                or details.st_uid != os.geteuid()
                or details.st_mode & 0o077
            ):
                raise OSError("unsafe telemetry file")
            if not append:
                os.ftruncate(descriptor, 0)
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short telemetry write")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)


class ProgressWriter:
    """Thread-safe, best-effort JSONL progress writer."""

    def __init__(self, path: Optional[str], *, strict: bool = False):
        self.path = Path(path).expanduser() if path else None
        self.strict = strict
        self._sequence = 0
        self._bytes_written = 0
        self._disabled = False
        self._lock = threading.Lock()

    def callback(
        self,
        event_type: str,
        tool_name: Any = None,
        preview: Any = None,
        _display_args: Any = None,
        **metadata: Any,
    ) -> None:
        if self.path is None or event_type not in _PROGRESS_TYPES:
            return

        event: dict[str, Any] = {"schema_version": 1, "type": event_type}
        if event_type.startswith("tool."):
            name = str(tool_name or "").strip()
            if not _TOOL_NAME_RE.fullmatch(name):
                return
            event["tool"] = name
            if event_type == "tool.started":
                event["summary"] = f"Using {name}"
            else:
                event["summary"] = (
                    f"{name} reported an error"
                    if bool(metadata.get("is_error"))
                    else f"Finished {name}"
                )
                duration = metadata.get("duration")
                if isinstance(duration, (int, float)) and not isinstance(duration, bool):
                    event["duration_ms"] = max(0, min(round(float(duration) * 1000), 86_400_000))
        else:
            event["summary"] = "Reasoning about the next action"
        try:
            with self._lock:
                if self._disabled or self._sequence >= _MAX_PROGRESS_EVENTS:
                    self._disabled = True
                    return
                self._sequence += 1
                event["sequence"] = self._sequence
                payload = (
                    json.dumps(event, separators=(",", ":")) + "\n"
                ).encode("utf-8")
                if self._bytes_written + len(payload) > _MAX_PROGRESS_BYTES:
                    self._disabled = True
                    return
                if self.strict:
                    _write_server_owned_file(self.path, payload, append=True)
                else:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with self.path.open("ab") as handle:
                        handle.write(payload)
                        handle.flush()
                self._bytes_written += len(payload)
        except Exception:
            self._disabled = True
