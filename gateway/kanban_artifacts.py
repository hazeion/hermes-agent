"""Safe, bounded artifact reads for completed Kanban tasks.

This module is intentionally narrower than task attachments. It exposes only
files produced by a worker, validates the stored blob again on every manifest
or download read, and never returns a filesystem path in public metadata.
"""

from __future__ import annotations

import codecs
import hashlib
import os
import re
import stat
import tempfile
import warnings
from pathlib import Path
from typing import Any, BinaryIO

from PIL import Image, ImageOps, ImageSequence, UnidentifiedImageError

from gateway.kanban_api import KanbanApiError
from hermes_cli import kanban_db


ARTIFACT_API_VERSION = 1
MAX_ARTIFACT_BYTES = 100 * 1024 * 1024
MAX_ARTIFACTS = 10
MAX_TASK_ARTIFACT_BYTES = 250 * 1024 * 1024
MAX_ARTIFACT_INSPECTIONS = 10
MAX_IMAGE_PIXELS = 40_000_000
MAX_IMAGE_FRAMES = 100
MAX_ANIMATED_PIXELS = 80_000_000

_ARTIFACT_ID = re.compile(r"hart_[0-9a-f]{64}\Z")
_BOARD_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
_TASK_ID = re.compile(r"t_[a-f0-9]{8}\Z")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_SECRET_NAME = re.compile(
    r"(?i)(^|[._-])(\.env|id_rsa|id_ed25519|credentials?|secrets?|tokens?|auth)([._-]|$)"
)
_SECRET_EXTENSIONS = frozenset({".key", ".pem", ".p12", ".pfx", ".jks", ".keystore"})
_PRIVATE_KEY_MARKERS = (
    b"-----BEGIN PRIVATE KEY-----",
    b"-----BEGIN RSA PRIVATE KEY-----",
    b"-----BEGIN OPENSSH PRIVATE KEY-----",
    b"-----BEGIN EC PRIVATE KEY-----",
)
_TOKEN_PATTERN = re.compile(
    rb"(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}|"
    rb"github_pat_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{20,}|"
    rb"AKIA[0-9A-Z]{16}|[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\."
    rb"[A-Za-z0-9_-]{10,})"
)
_SECRET_ASSIGNMENT = re.compile(
    rb"(?i)(?:^|[\s,{])['\"]?"
    rb"[A-Za-z0-9_-]*(?:api[_-]?key|access[_-]?token|password|secret|"
    rb"credential|authorization|private[_-]?key)[A-Za-z0-9_-]*"
    rb"['\"]?\s*[:=]\s*['\"]?([A-Za-z0-9_./:@%?&+=-]{16,})"
)
_CREDENTIAL_URL = re.compile(
    rb"(?i)\b[a-z][a-z0-9+.-]*://[^/\s:@]+:[^@\s/]{4,}@[^\s'\"<>]+"
)
_BLOCKED_MAGIC = (
    b"MZ",
    b"\x7fELF",
    b"PK\x03\x04",
    b"PK\x05\x06",
    b"PK\x07\x08",
    b"\x1f\x8b",
    b"Rar!\x1a\x07",
    b"7z\xbc\xaf\x27\x1c",
    b"BZh",
    b"\xfd7zXZ\x00",
    b"%PDF-",
)
_IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
_TEXT_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".py": "text/x-python",
    ".pyi": "text/x-python",
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".cjs": "text/javascript",
    ".ts": "text/typescript",
    ".tsx": "text/typescript",
    ".jsx": "text/javascript",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".css": "text/css",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".toml": "application/toml",
    ".ini": "text/plain",
    ".cfg": "text/plain",
    ".conf": "text/plain",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".sql": "application/sql",
    ".c": "text/x-c",
    ".h": "text/x-c",
    ".cc": "text/x-c++",
    ".cpp": "text/x-c++",
    ".hpp": "text/x-c++",
    ".java": "text/x-java-source",
    ".go": "text/x-go",
    ".rs": "text/x-rust",
    ".rb": "text/x-ruby",
    ".php": "text/x-php",
    ".swift": "text/x-swift",
    ".kt": "text/x-kotlin",
    ".kts": "text/x-kotlin",
    ".cs": "text/x-csharp",
    ".vue": "text/plain",
    ".svelte": "text/plain",
    ".graphql": "text/plain",
    ".gql": "text/plain",
    ".log": "text/plain",
    ".diff": "text/x-diff",
    ".patch": "text/x-diff",
}
def _scope(board_value: Any, task_value: Any) -> tuple[str, str]:
    board = str(board_value or "").strip().lower()
    task_id = str(task_value or "").strip()
    if not _BOARD_ID.fullmatch(board):
        raise KanbanApiError(400, "invalid_board", "Board identifier is invalid")
    if board != kanban_db.DEFAULT_BOARD and not kanban_db.board_exists(board):
        raise KanbanApiError(404, "board_not_found", "Board was not found")
    if not _TASK_ID.fullmatch(task_id):
        raise KanbanApiError(400, "invalid_task", "Task identifier is invalid")
    return board, task_id


def _safe_name(value: str) -> tuple[str, str, str]:
    name = str(value or "").strip()
    if (
        not name
        or len(name) > 255
        or _CONTROL.search(name)
        or name in {".", ".."}
        or Path(name).name != name
        or "/" in name
        or "\\" in name
    ):
        raise KanbanApiError(409, "artifact_unsafe", "A generated artifact is unsafe")
    lower = name.lower()
    extension = Path(lower).suffix
    if extension in _SECRET_EXTENSIONS or _SECRET_NAME.search(lower):
        raise KanbanApiError(409, "artifact_unsafe", "A generated artifact is unsafe")
    if extension in _IMAGE_TYPES:
        return name, "image", _IMAGE_TYPES[extension]
    if extension in _TEXT_TYPES:
        return name, "text", _TEXT_TYPES[extension]
    raise KanbanApiError(409, "artifact_type_unsupported", "A generated artifact type is unsupported")


def _image_type(prefix: bytes) -> str | None:
    if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if prefix.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if prefix.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(prefix) >= 12 and prefix.startswith(b"RIFF") and prefix[8:12] == b"WEBP":
        return "image/webp"
    return None


def _image_container_ends_exactly(
    source: Path | BinaryIO,
    image_type: str,
    expected_size: int,
) -> bool:
    owned = isinstance(source, Path)
    handle = source.open("rb") if owned else source
    try:
        handle.seek(0)
        if image_type == "image/jpeg":
            if expected_size < 2:
                return False
            handle.seek(expected_size - 2)
            return handle.read(2) == b"\xff\xd9"
        if image_type == "image/gif":
            if expected_size < 1:
                return False
            handle.seek(expected_size - 1)
            return handle.read(1) == b"\x3b"
        if image_type == "image/webp":
            header = handle.read(12)
            return (
                len(header) == 12
                and header[:4] == b"RIFF"
                and header[8:12] == b"WEBP"
                and int.from_bytes(header[4:8], "little") + 8 == expected_size
            )
        if image_type != "image/png":
            return False
        if handle.read(8) != b"\x89PNG\r\n\x1a\n":
            return False
        offset = 8
        while offset + 12 <= expected_size:
            length_bytes = handle.read(4)
            chunk_type = handle.read(4)
            if len(length_bytes) != 4 or len(chunk_type) != 4:
                return False
            chunk_size = int.from_bytes(length_bytes, "big")
            offset += 8
            if chunk_size > expected_size - offset - 4:
                return False
            handle.seek(chunk_size + 4, os.SEEK_CUR)
            offset += chunk_size + 4
            if chunk_type == b"IEND":
                return chunk_size == 0 and offset == expected_size
        return False
    finally:
        if owned:
            handle.close()
        else:
            handle.seek(0)


def _validate_image_structure(
    source: Path | BinaryIO,
    image_type: str,
    expected_size: int,
) -> None:
    if not _image_container_ends_exactly(source, image_type, expected_size):
        raise KanbanApiError(
            409,
            "artifact_image_invalid",
            "A generated image is malformed or contains trailing content",
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            if not isinstance(source, Path):
                source.seek(0)
            with Image.open(source) as image:
                if Image.MIME.get(image.format or "") != image_type:
                    raise KanbanApiError(
                        409,
                        "artifact_content_mismatch",
                        "A generated image does not match its filename",
                    )
                image.verify()
            if not isinstance(source, Path):
                source.seek(0)
            with Image.open(source) as image:
                frame_count = int(getattr(image, "n_frames", 1) or 1)
                if frame_count > MAX_IMAGE_FRAMES:
                    raise KanbanApiError(
                        409,
                        "artifact_image_invalid",
                        "A generated image has too many frames",
                    )
                total_pixels = 0
                for frame in ImageSequence.Iterator(image):
                    width, height = frame.size
                    pixels = int(width) * int(height)
                    total_pixels += pixels
                    if (
                        width <= 0
                        or height <= 0
                        or pixels > MAX_IMAGE_PIXELS
                        or total_pixels > MAX_ANIMATED_PIXELS
                    ):
                        raise KanbanApiError(
                            409,
                            "artifact_image_invalid",
                            "A generated image exceeds the safe pixel limit",
                        )
                    frame.load()
    except KanbanApiError:
        raise
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
    ) as exc:
        raise KanbanApiError(
            409,
            "artifact_image_invalid",
            "A generated image is malformed or unsafe",
        ) from exc


def _canonical_image_snapshot(
    source: Path | BinaryIO,
    image_type: str,
) -> tuple[BinaryIO, int]:
    format_name = {
        "image/png": "PNG",
        "image/jpeg": "JPEG",
        "image/gif": "GIF",
        "image/webp": "WEBP",
    }[image_type]
    frames: list[Image.Image] = []
    durations: list[int] = []
    snapshot: BinaryIO | None = None
    loop = 0
    try:
        if not isinstance(source, Path):
            source.seek(0)
        with Image.open(source) as image:
            loop = max(0, min(int(image.info.get("loop", 0) or 0), 65535))
            for frame in ImageSequence.Iterator(image):
                frame.load()
                oriented = ImageOps.exif_transpose(frame)
                safe_mode = (
                    "RGBA"
                    if oriented.mode in {"RGBA", "LA"}
                    or "transparency" in oriented.info
                    else "RGB"
                )
                safe_frame = oriented.convert(safe_mode).copy()
                safe_frame.info.clear()
                frames.append(safe_frame)
                durations.append(
                    max(
                        0,
                        min(int(frame.info.get("duration", 0) or 0), 600_000),
                    )
                )
                if oriented is not frame:
                    oriented.close()
        if not frames or (format_name == "JPEG" and len(frames) != 1):
            raise KanbanApiError(
                409,
                "artifact_image_invalid",
                "A generated image has an unsupported frame structure",
            )
        first = frames[0].convert("RGB") if format_name == "JPEG" else frames[0]
        save_options: dict[str, Any] = {}
        if format_name == "PNG":
            save_options.update(compress_level=9, optimize=False)
        elif format_name == "JPEG":
            save_options.update(
                quality=95,
                subsampling=0,
                optimize=False,
                progressive=False,
            )
        elif format_name == "GIF":
            save_options.update(optimize=False)
        elif format_name == "WEBP":
            save_options.update(lossless=True, quality=100, method=6)
        if len(frames) > 1:
            save_options.update(
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=loop,
            )
        snapshot = tempfile.TemporaryFile(mode="w+b")
        first.save(snapshot, format=format_name, **save_options)
        size = snapshot.tell()
        if size <= 0 or size > MAX_ARTIFACT_BYTES:
            raise KanbanApiError(
                409,
                "artifact_size_invalid",
                "A canonical generated image exceeds the artifact limit",
            )
        snapshot.seek(0)
        return snapshot, size
    except KanbanApiError:
        if snapshot is not None:
            snapshot.close()
        raise
    except (OSError, ValueError, KeyError) as exc:
        if snapshot is not None:
            snapshot.close()
        raise KanbanApiError(
            409,
            "artifact_image_invalid",
            "A generated image could not be canonicalized safely",
        ) from exc
    finally:
        for frame in frames:
            frame.close()


def _artifact_id(board: str, task_id: str, attachment_id: int, digest: str) -> str:
    material = f"{board}\0{task_id}\0{attachment_id}\0{digest}".encode("utf-8")
    return "hart_" + hashlib.sha256(material).hexdigest()


def _safe_stored_path(board: str, task_id: str, attachment: kanban_db.Attachment) -> Path:
    source = Path(attachment.stored_path)
    try:
        details = source.lstat()
        root = kanban_db.attachments_root(board=board).resolve(strict=True)
        task_root = kanban_db.task_attachments_dir(task_id, board=board).resolve(strict=True)
        resolved = source.resolve(strict=True)
    except OSError as exc:
        raise KanbanApiError(410, "artifact_unavailable", "A generated artifact is unavailable") from exc
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
        or task_root.parent != root
        or resolved.parent != task_root
    ):
        raise KanbanApiError(409, "artifact_unsafe", "A generated artifact is unsafe")
    return resolved


def _inspect(
    board: str,
    task_id: str,
    attachment: kanban_db.Attachment,
) -> tuple[dict[str, Any], Path, BinaryIO | None]:
    name, kind, mime_type = _safe_name(attachment.filename)
    path = _safe_stored_path(board, task_id, attachment)
    try:
        details = path.lstat()
    except OSError as exc:
        raise KanbanApiError(410, "artifact_unavailable", "A generated artifact is unavailable") from exc
    size = int(details.st_size)
    if (
        size <= 0
        or size != int(attachment.size)
        or size > MAX_ARTIFACT_BYTES
    ):
        raise KanbanApiError(409, "artifact_size_invalid", "A generated artifact has an invalid size")

    digest = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8-sig")("strict") if kind == "text" else None
    raw_image_snapshot = (
        tempfile.TemporaryFile(mode="w+b") if kind == "image" else None
    )
    prefix = b""
    carry = b""
    try:
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or opened.st_size != size
                or (opened.st_dev, opened.st_ino)
                != (details.st_dev, details.st_ino)
            ):
                raise KanbanApiError(409, "artifact_changed", "A generated artifact changed during validation")
            with os.fdopen(descriptor, "rb") as handle:
                descriptor = -1
                remaining = size
                while remaining:
                    chunk = handle.read(min(64 * 1024, remaining))
                    if not chunk:
                        raise KanbanApiError(
                            409,
                            "artifact_changed",
                            "A generated artifact changed during validation",
                        )
                    remaining -= len(chunk)
                    if not prefix:
                        prefix = chunk[:4096]
                    digest.update(chunk)
                    if raw_image_snapshot is not None:
                        raw_image_snapshot.write(chunk)
                    if decoder is not None:
                        decoder.decode(chunk, final=False)
                    scanned = carry + chunk
                    if (
                        any(marker in scanned for marker in _PRIVATE_KEY_MARKERS)
                        or _TOKEN_PATTERN.search(scanned)
                        or _SECRET_ASSIGNMENT.search(scanned)
                        or _CREDENTIAL_URL.search(scanned)
                    ):
                        raise KanbanApiError(
                            409,
                            "artifact_secret_detected",
                            "A generated artifact appears to contain credentials",
                        )
                    carry = scanned[-4096:]
                if handle.read(1):
                    raise KanbanApiError(
                        409,
                        "artifact_changed",
                        "A generated artifact changed during validation",
                    )
                if decoder is not None:
                    decoder.decode(b"", final=True)
                final_details = os.fstat(handle.fileno())
                if (
                    final_details.st_dev,
                    final_details.st_ino,
                    final_details.st_size,
                ) != (opened.st_dev, opened.st_ino, opened.st_size):
                    raise KanbanApiError(
                        409,
                        "artifact_changed",
                        "A generated artifact changed during validation",
                    )
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    except KanbanApiError:
        if raw_image_snapshot is not None:
            raw_image_snapshot.close()
        raise
    except UnicodeDecodeError as exc:
        if raw_image_snapshot is not None:
            raw_image_snapshot.close()
        raise KanbanApiError(409, "artifact_text_invalid", "A generated text artifact is not UTF-8") from exc
    except OSError as exc:
        if raw_image_snapshot is not None:
            raw_image_snapshot.close()
        raise KanbanApiError(410, "artifact_unavailable", "A generated artifact is unavailable") from exc

    stripped = prefix.lstrip(b"\xef\xbb\xbf\x00\t\r\n ").lower()
    if any(prefix.startswith(value) for value in _BLOCKED_MAGIC):
        if raw_image_snapshot is not None:
            raw_image_snapshot.close()
        raise KanbanApiError(409, "artifact_type_unsupported", "A generated artifact type is unsupported")
    if stripped.startswith(b"<svg") or (stripped.startswith(b"<?xml") and b"<svg" in stripped):
        if raw_image_snapshot is not None:
            raw_image_snapshot.close()
        raise KanbanApiError(409, "artifact_type_unsupported", "SVG artifacts are unsupported")
    if kind == "image" and _image_type(prefix) != mime_type:
        if raw_image_snapshot is not None:
            raw_image_snapshot.close()
        raise KanbanApiError(409, "artifact_content_mismatch", "A generated image does not match its filename")
    canonical: BinaryIO | None = None
    if kind == "image":
        assert raw_image_snapshot is not None
        try:
            raw_image_snapshot.seek(0)
            _validate_image_structure(raw_image_snapshot, mime_type, size)
            raw_image_snapshot.seek(0)
            canonical, size = _canonical_image_snapshot(
                raw_image_snapshot,
                mime_type,
            )
        finally:
            raw_image_snapshot.close()
        digest = hashlib.sha256()
        while chunk := canonical.read(64 * 1024):
            digest.update(chunk)
        canonical.seek(0)

    hex_digest = digest.hexdigest()
    return (
        {
            "id": _artifact_id(board, task_id, int(attachment.id), hex_digest),
            "object": "hermes.kanban.artifact",
            "name": name,
            "kind": kind,
            "mime_type": mime_type,
            "byte_size": size,
            "digest": "sha256:" + hex_digest,
            "created_at": int(attachment.created_at),
        },
        path,
        canonical,
    )


def _collect_artifacts(
    board: str,
    task_id: str,
) -> tuple[list[tuple[dict[str, Any], Path, BinaryIO | None]], int, int]:
    with kanban_db.connect_closing(board=board) as connection:
        task = kanban_db.get_task(connection, task_id)
        if task is None:
            raise KanbanApiError(404, "task_not_found", "Task was not found")
        if task.created_by != "api_server":
            return [], 0, 0
        if task.status not in {"done", "archived"}:
            raise KanbanApiError(409, "task_not_complete", "Task artifacts are available only after completion")
        candidates, candidate_count = kanban_db.list_completion_attachments(
            connection,
            task_id,
            limit=MAX_ARTIFACT_INSPECTIONS,
        )

    accepted: list[tuple[dict[str, Any], Path, BinaryIO | None]] = []
    rejected = max(0, candidate_count - len(candidates))
    total = 0
    for attachment in candidates:
        if len(accepted) >= MAX_ARTIFACTS:
            rejected += 1
            continue
        try:
            metadata, path, canonical = _inspect(board, task_id, attachment)
        except KanbanApiError:
            rejected += 1
            continue
        if total + metadata["byte_size"] > MAX_TASK_ARTIFACT_BYTES:
            if canonical is not None:
                canonical.close()
            rejected += 1
            continue
        accepted.append((metadata, path, canonical))
        total += metadata["byte_size"]
    return accepted, rejected, total


def list_artifacts(board: Any, task_id: Any) -> dict[str, Any]:
    board, task_id = _scope(board, task_id)
    accepted, rejected, total = _collect_artifacts(board, task_id)
    try:
        return {
            "object": "hermes.kanban.artifact_list",
            "version": ARTIFACT_API_VERSION,
            "complete": True,
            "task_id": task_id,
            "data": [metadata for metadata, _path, _canonical in accepted],
            "rejected_count": rejected,
            "total_bytes": total,
        }
    finally:
        for _metadata, _path, canonical in accepted:
            if canonical is not None:
                canonical.close()


def open_artifact(
    board: Any,
    task_id: Any,
    artifact_id: str,
) -> tuple[dict[str, Any], BinaryIO]:
    """Return verified metadata and a descriptor-owned private snapshot."""
    board, task_id = _scope(board, task_id)
    if not isinstance(artifact_id, str) or _ARTIFACT_ID.fullmatch(artifact_id) is None:
        raise KanbanApiError(404, "artifact_not_found", "Artifact was not found")
    accepted, _rejected, _total = _collect_artifacts(board, task_id)
    matched = next(
        (
            (metadata, path, canonical)
            for metadata, path, canonical in accepted
            if metadata["id"] == artifact_id
        ),
        None,
    )
    if matched is None:
        for _metadata, _path, canonical in accepted:
            if canonical is not None:
                canonical.close()
        raise KanbanApiError(404, "artifact_not_found", "Artifact was not found")
    expected, matched_path, canonical = matched
    for metadata, _path, other in accepted:
        if metadata["id"] != artifact_id and other is not None:
            other.close()
    if canonical is not None:
        canonical.seek(0)
        return expected, canonical

    snapshot: BinaryIO | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(matched_path, flags)
        snapshot = tempfile.TemporaryFile(mode="w+b")
        digest = hashlib.sha256()
        copied = 0
        try:
            details = os.fstat(descriptor)
            if (
                not stat.S_ISREG(details.st_mode)
                or details.st_nlink != 1
                or details.st_size != expected["byte_size"]
            ):
                raise KanbanApiError(409, "artifact_changed", "A generated artifact changed before download")
            with os.fdopen(descriptor, "rb") as handle:
                descriptor = -1
                while chunk := handle.read(64 * 1024):
                    copied += len(chunk)
                    if copied > expected["byte_size"]:
                        raise KanbanApiError(
                            409,
                            "artifact_changed",
                            "A generated artifact changed before download",
                        )
                    digest.update(chunk)
                    snapshot.write(chunk)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    except OSError as exc:
        if snapshot is not None:
            snapshot.close()
        raise KanbanApiError(410, "artifact_unavailable", "A generated artifact is unavailable") from exc
    actual_digest = "sha256:" + digest.hexdigest()
    if copied != expected["byte_size"] or actual_digest != expected["digest"]:
        if snapshot is not None:
            snapshot.close()
        raise KanbanApiError(409, "artifact_changed", "A generated artifact changed before download")
    assert snapshot is not None
    snapshot.seek(0)
    return expected, snapshot


def read_artifact(
    board: Any,
    task_id: Any,
    artifact_id: str,
) -> tuple[dict[str, Any], bytes]:
    """Compatibility helper for bounded in-process callers and tests."""
    metadata, snapshot = open_artifact(board, task_id, artifact_id)
    try:
        return metadata, snapshot.read(MAX_ARTIFACT_BYTES + 1)
    finally:
        snapshot.close()
