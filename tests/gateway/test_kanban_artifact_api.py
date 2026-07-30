"""Security and HTTP contract tests for generated Kanban artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import base64

import pytest

from gateway.config import PlatformConfig
from gateway.kanban_artifacts import list_artifacts, read_artifact
from gateway import kanban_artifacts
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli import kanban_db


class _Request:
    def __init__(
        self,
        *,
        authorization: str = "",
        board: str = "default",
        task_id: str = "",
        artifact_id: str = "",
    ):
        self.headers = {"Authorization": authorization} if authorization else {}
        self.query = {"board": board}
        self.match_info = {"task_id": task_id, "artifact_id": artifact_id}
        self.transport = None
        self.remote = "127.0.0.1"
        self.method = "GET"
        self.path_qs = f"/v1/kanban/tasks/{task_id}/artifacts"


def _completed_task(
    *,
    name: str = "report.md",
    content: bytes = b"# Safe report\n",
    uploaded_by: str = "kanban_complete",
) -> tuple[str, kanban_db.Attachment]:
    with kanban_db.connect_closing(board="default") as connection:
        task_id = kanban_db.create_task(
            connection,
            title="Create a deliverable",
            created_by="api_server",
            initial_status="running",
        )
        metadata = None
        if uploaded_by == "kanban_complete":
            workspace = kanban_db.resolve_workspace(
                kanban_db.get_task(connection, task_id),
                board="default",
            )
            kanban_db.set_workspace_path(connection, task_id, workspace)
            generated = workspace / name
            generated.parent.mkdir(parents=True, exist_ok=True)
            generated.write_bytes(content)
            metadata = {"artifacts": [str(generated.resolve())]}
        else:
            kanban_db.store_attachment_bytes(
                connection,
                task_id,
                name,
                content,
                content_type="application/octet-stream",
                uploaded_by=uploaded_by,
                board="default",
                max_bytes=100 * 1024 * 1024,
            )
        assert kanban_db.complete_task(
            connection,
            task_id,
            summary="Deliverable completed",
            metadata=metadata,
        )
        attachments = kanban_db.list_attachments(connection, task_id)
        if uploaded_by == "kanban_complete":
            return task_id, attachments[0]
        return task_id, attachments[0]


def test_manifest_and_download_expose_only_bounded_metadata_and_exact_bytes():
    task_id, _attachment = _completed_task()
    manifest = list_artifacts("default", task_id)

    assert manifest["object"] == "hermes.kanban.artifact_list"
    assert manifest["version"] == 1
    assert manifest["rejected_count"] == 0
    assert manifest["total_bytes"] == len(b"# Safe report\n")
    assert len(manifest["data"]) == 1
    artifact = manifest["data"][0]
    assert set(artifact) == {
        "id",
        "object",
        "name",
        "kind",
        "mime_type",
        "byte_size",
        "digest",
        "created_at",
    }
    assert artifact["id"].startswith("hart_")
    assert artifact["name"] == "report.md"
    assert artifact["kind"] == "text"
    assert "/" not in artifact["name"]
    assert "stored_path" not in json.dumps(manifest)

    downloaded, content = read_artifact("default", task_id, artifact["id"])
    assert downloaded == artifact
    assert content == b"# Safe report\n"


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("secrets.txt", b"API_KEY=sk-abcdefghijklmnopqrstuvwxyz\n"),
        (
            "aws.txt",
            b"AWS_SECRET_ACCESS_KEY=abcdefghijklmnopqrstuvwx1234567890ABCD",
        ),
        (
            "database.txt",
            b"DATABASE_URL=postgres://user:verysecretpassword@example.test/db",
        ),
        ("page.html", b"<html><body>unsafe active content</body></html>"),
        ("archive.zip", b"PK\x03\x04payload"),
        ("settings.json", b'{"password":"abcdefghijklmnopQRSTUVWX"}'),
        ("wrong.png", b"not a png"),
        ("header-only.png", b"\x89PNG\r\n\x1a\nnot-an-image"),
        (
            "polyglot.png",
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC"
            )
            + b"PK\x03\x04hidden-archive",
        ),
        (
            "leaky.png",
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC"
            )
            + b"sk-abcdefghijklmnopqrstuvwxyz",
        ),
    ],
)
def test_unsupported_secret_or_mismatched_artifacts_are_not_published(name, content):
    task_id, _attachment = _completed_task(name=name, content=content)
    manifest = list_artifacts("default", task_id)
    assert manifest["data"] == []
    assert manifest["rejected_count"] == 1
    assert manifest["total_bytes"] == 0


def test_user_uploaded_task_input_is_not_mistaken_for_generated_output():
    task_id, _attachment = _completed_task(uploaded_by="dashboard")
    manifest = list_artifacts("default", task_id)
    assert manifest["data"] == []
    assert manifest["rejected_count"] == 0


def test_structurally_valid_png_is_published():
    content = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC"
    )
    task_id, _attachment = _completed_task(
        name="diagram.png",
        content=content,
    )
    manifest = list_artifacts("default", task_id)
    assert len(manifest["data"]) == 1
    assert manifest["data"][0]["mime_type"] == "image/png"


def test_generic_agent_attachment_is_not_a_completion_artifact():
    task_id, _attachment = _completed_task(uploaded_by="agent")
    manifest = list_artifacts("default", task_id)
    assert manifest["data"] == []
    assert manifest["rejected_count"] == 0


def test_symlinked_stored_blob_fails_closed(tmp_path):
    task_id, attachment = _completed_task()
    stored = Path(attachment.stored_path)
    replacement = tmp_path / "replacement.md"
    replacement.write_text("changed", encoding="utf-8")
    stored.unlink()
    try:
        stored.symlink_to(replacement)
    except OSError:
        pytest.skip("symlink creation unavailable")
    manifest = list_artifacts("default", task_id)
    assert manifest["data"] == []
    assert manifest["rejected_count"] == 1


def test_hardlinked_stored_blob_fails_closed(tmp_path):
    task_id, attachment = _completed_task()
    stored = Path(attachment.stored_path)
    outside = tmp_path / "outside.md"
    outside.write_bytes(stored.read_bytes())
    stored.unlink()
    os.link(outside, stored)

    manifest = list_artifacts("default", task_id)

    assert manifest["data"] == []
    assert manifest["rejected_count"] == 1


def test_api_created_task_does_not_promote_a_path_mentioned_only_in_prose(tmp_path):
    with kanban_db.connect_closing(board="default") as connection:
        task_id = kanban_db.create_task(
            connection,
            title="Do not trust prose",
            created_by="api_server",
            initial_status="running",
        )
        workspace = kanban_db.resolve_workspace(
            kanban_db.get_task(connection, task_id),
            board="default",
        )
        kanban_db.set_workspace_path(connection, task_id, workspace)
        candidate = workspace / "mentioned.md"
        candidate.write_text("not explicitly exported", encoding="utf-8")
        assert kanban_db.complete_task(
            connection,
            task_id,
            summary=f"Saved the result to {candidate}",
        )
        assert kanban_db.list_attachments(connection, task_id) == []


@pytest.mark.parametrize("explicit", [True, False])
def test_local_task_completion_files_are_never_exposed_by_remote_api(explicit):
    with kanban_db.connect_closing(board="default") as connection:
        task_id = kanban_db.create_task(
            connection,
            title="Local-only deliverable",
            initial_status="running",
        )
        workspace = kanban_db.resolve_workspace(
            kanban_db.get_task(connection, task_id),
            board="default",
        )
        kanban_db.set_workspace_path(connection, task_id, workspace)
        generated = workspace / "local-report.md"
        generated.write_text("local result", encoding="utf-8")
        assert kanban_db.complete_task(
            connection,
            task_id,
            summary=(
                "Local result complete"
                if explicit
                else f"Saved the result to {generated}"
            ),
            metadata=(
                {"artifacts": [str(generated.resolve())]}
                if explicit
                else None
            ),
        )

    manifest = list_artifacts("default", task_id)
    assert manifest["data"] == []
    assert manifest["rejected_count"] == 0
    assert manifest["total_bytes"] == 0


def test_manifest_enforces_count_and_combined_bounds(monkeypatch):
    monkeypatch.setattr("gateway.kanban_artifacts.MAX_ARTIFACTS", 2)
    monkeypatch.setattr("gateway.kanban_artifacts.MAX_TASK_ARTIFACT_BYTES", 8)
    with kanban_db.connect_closing(board="default") as connection:
        task_id = kanban_db.create_task(
            connection,
            title="Bound output",
            created_by="api_server",
            initial_status="running",
        )
        workspace = kanban_db.resolve_workspace(
            kanban_db.get_task(connection, task_id),
            board="default",
        )
        kanban_db.set_workspace_path(connection, task_id, workspace)
        generated = []
        for index in range(3):
            path = workspace / f"file-{index}.txt"
            path.write_bytes(b"four")
            generated.append(str(path.resolve()))
        assert kanban_db.complete_task(
            connection,
            task_id,
            summary="Done",
            metadata={"artifacts": generated},
        )
    manifest = list_artifacts("default", task_id)
    assert len(manifest["data"]) == 2
    assert manifest["total_bytes"] == 8
    assert manifest["rejected_count"] == 1


def test_manifest_inspection_work_is_bounded(monkeypatch):
    task_id, _attachment = _completed_task()
    with kanban_db.connect_closing(board="default") as connection:
        for index in range(8):
            kanban_db.store_attachment_bytes(
                connection,
                task_id,
                f"extra-{index}.txt",
                b"safe",
                uploaded_by="kanban_complete",
                board="default",
            )
    inspected = 0
    real_inspect = kanban_artifacts._inspect

    def counting_inspect(*args, **kwargs):
        nonlocal inspected
        inspected += 1
        return real_inspect(*args, **kwargs)

    monkeypatch.setattr(kanban_artifacts, "MAX_ARTIFACT_INSPECTIONS", 3)
    monkeypatch.setattr(kanban_artifacts, "_inspect", counting_inspect)

    manifest = list_artifacts("default", task_id)

    assert inspected == 3
    assert manifest["rejected_count"] == 6


@pytest.mark.asyncio
async def test_http_download_requires_bearer_auth_and_sets_safe_headers():
    task_id, _attachment = _completed_task()
    artifact = list_artifacts("default", task_id)["data"][0]
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-secret"}))

    denied = await adapter._handle_kanban_artifact(
        _Request(task_id=task_id, artifact_id=artifact["id"])
    )
    assert denied.status == 401

    allowed = await adapter._handle_kanban_artifact(
        _Request(
            authorization="Bearer sk-secret",
            task_id=task_id,
            artifact_id=artifact["id"],
        )
    )
    assert allowed.status == 200
    assert allowed.body._value.read() == b"# Safe report\n"
    assert allowed.headers["Cache-Control"] == "private, no-store"
    assert allowed.headers["X-Content-Type-Options"] == "nosniff"
    assert allowed.headers["X-Hermes-Artifact-Id"] == artifact["id"]
    assert allowed.headers["X-Hermes-Artifact-Digest"] == artifact["digest"]
    assert allowed.headers["Content-Disposition"].startswith("attachment;")
