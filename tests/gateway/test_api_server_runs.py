"""Tests for /v1/runs endpoints: start, status, events, and stop.

Covers:
- POST /v1/runs — start a run (202)
- GET /v1/runs/{run_id} — poll run status
- GET /v1/runs/{run_id}/events — SSE event stream
- POST /v1/runs/{run_id}/stop — interrupt a running agent
- Auth, error handling, and cleanup
"""

import asyncio
from collections import deque
import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    RUN_INLINE_IMAGE_MAX_COUNT,
    _RUN_EVENT_JOURNAL_MAX_BYTES,
    _RUN_EVENT_RETENTION,
    _agent_run_usage,
    _approval_event_choices,
    _safe_approval_preview,
    _run_session_continuation_revision,
    cors_middleware,
    security_headers_middleware,
)
from hermes_state import SessionDB
from tools import approval as approval_mod
from tools import clarify_gateway as clarify_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("smart_denied", "allow_permanent", "expected"),
    [
        (False, True, ["once", "session", "always", "deny"]),
        (False, False, ["once", "session", "deny"]),
        (True, True, ["once", "deny"]),
        (True, False, ["once", "deny"]),
    ],
)
def test_approval_event_choices_follow_backend_capabilities(
    smart_denied, allow_permanent, expected
):
    assert _approval_event_choices(
        smart_denied=smart_denied,
        allow_permanent=allow_permanent,
    ) == expected


def test_safe_approval_preview_uses_only_server_owned_labels():
    private_path = "/Users/alice/private-project"
    credential = "sk-super-secret"
    preview = _safe_approval_preview({
        "command": f"rm -rf {private_path} --token {credential}",
        "description": f"Delete {private_path} with {credential}",
        "pattern_key": "recursive delete",
        "pattern_keys": [
            "recursive delete",
            f"plugin_rule:untrusted:{private_path}:{credential}",
        ],
    })

    assert preview == {
        "version": 1,
        "category": "terminal_command",
        "title": "Terminal command approval",
        "summary": "Hermes stopped a terminal command that matched a protected action.",
        "risk_labels": ["recursive delete"],
    }
    encoded = json.dumps(preview)
    assert private_path not in encoded
    assert credential not in encoded


def test_safe_approval_preview_falls_back_without_echoing_unknown_input():
    attacker_text = "unknown policy /private/customer-data token-123"
    preview = _safe_approval_preview({
        "command": attacker_text,
        "description": attacker_text,
        "pattern_key": attacker_text,
        "pattern_keys": [attacker_text],
    })

    assert preview == {
        "version": 1,
        "category": "protected_action",
        "title": "Protected action approval",
        "summary": "Hermes stopped an action that requires explicit approval.",
        "risk_labels": [],
    }
    assert attacker_text not in json.dumps(preview)


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    """Create an adapter with optional API key."""
    extra = {}
    if api_key:
        extra["key"] = api_key
    config = PlatformConfig(enabled=True, extra=extra)
    adapter = APIServerAdapter(config)
    return adapter


def _create_runs_app(adapter: APIServerAdapter) -> web.Application:
    """Create an aiohttp app with /v1/runs routes registered."""
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    app.router.add_post("/v1/runs/{run_id}/approval", adapter._handle_run_approval)
    app.router.add_post(
        "/v1/runs/{run_id}/clarification",
        adapter._handle_run_clarification,
    )
    app.router.add_post("/v1/runs/{run_id}/stop", adapter._handle_stop_run)
    return app


def _make_slow_agent(**kwargs):
    """Create a mock agent that blocks in run_conversation until interrupted.

    Returns (mock_agent, agent_ready_event, interrupt_event) where
    agent_ready_event is set once run_conversation starts, and
    interrupt_event is set when interrupt() is called.
    """
    ready = threading.Event()
    interrupted = threading.Event()

    mock_agent = MagicMock()

    def _do_interrupt(message=None):
        interrupted.set()

    mock_agent.interrupt = MagicMock(side_effect=_do_interrupt)

    def _slow_run(user_message=None, conversation_history=None, task_id=None):
        ready.set()
        # Block until interrupt() is called
        interrupted.wait(timeout=10)
        return {"final_response": "interrupted"}

    mock_agent.run_conversation.side_effect = _slow_run
    mock_agent.session_prompt_tokens = 0
    mock_agent.session_completion_tokens = 0
    mock_agent.session_total_tokens = 0

    return mock_agent, ready, interrupted


def _continuation_descriptor(db: SessionDB, session_id: str) -> dict:
    snapshot = db.get_continuation_snapshot(session_id)
    assert snapshot is not None
    resolved_id, history, message_ids = snapshot
    return {
        "version": 1,
        "session_id": resolved_id,
        "revision": _run_session_continuation_revision(
            resolved_id,
            history,
            message_ids,
        ),
    }


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture
def auth_adapter():
    return _make_adapter(api_key="sk-secret")


# ---------------------------------------------------------------------------
# POST /v1/runs — start a run
# ---------------------------------------------------------------------------


class TestStartRun:
    @pytest.mark.asyncio
    async def test_start_returns_202(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 10
                mock_agent.session_completion_tokens = 5
                mock_agent.session_total_tokens = 15
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                assert data["status"] == "started"
                assert data["run_id"].startswith("run_")

                status_resp = await cli.get(f"/v1/runs/{data['run_id']}")
                assert status_resp.status == 200
                status = await status_resp.json()
                assert status["run_id"] == data["run_id"]
                assert status["status"] in {"queued", "running", "completed"}
                assert status["object"] == "hermes.run"


class TestExactSessionContinuation:
    @pytest.fixture(autouse=True)
    def _configured_key_for_continuation_logic(self, adapter):
        """Exercise continuation state logic with its required key configured.

        The HTTP bearer gate itself has dedicated coverage below; these tests
        focus on descriptor binding, races, and lifecycle behavior.
        """
        adapter._api_key = "sk-secret"
        adapter._check_auth = lambda _request: None

    @pytest.mark.asyncio
    async def test_verified_continuation_loads_history_into_stoppable_run(
        self,
        adapter,
        tmp_path,
    ):
        db = SessionDB(tmp_path / "continuation.db")
        adapter._session_db = db
        try:
            session_id = db.create_session("continued-session", "api_server")
            db.append_message(session_id, "user", "earlier question")
            db.append_message(session_id, "assistant", "earlier answer")
            descriptor = _continuation_descriptor(db, session_id)

            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "continued"}
            mock_agent.session_prompt_tokens = 3
            mock_agent.session_completion_tokens = 2
            mock_agent.session_total_tokens = 5

            app = _create_runs_app(adapter)
            with patch.object(adapter, "_create_agent", return_value=mock_agent):
                async with TestClient(TestServer(app)) as cli:
                    resp = await cli.post(
                        "/v1/runs",
                        json={"input": "next step", "continuation": descriptor},
                    )
                    assert resp.status == 202, await resp.text()
                    run_id = (await resp.json())["run_id"]

                    for _ in range(40):
                        status_resp = await cli.get(f"/v1/runs/{run_id}")
                        status = await status_resp.json()
                        if status["status"] == "completed":
                            break
                        await asyncio.sleep(0.025)

            assert status["status"] == "completed"
            assert status["session_id"] == session_id
            assert status["continuation_version"] == 1
            assert status["output"] == "continued"
            call = mock_agent.run_conversation.call_args.kwargs
            assert call["task_id"] == session_id
            assert call["user_message"] == "next step"
            assert [
                (message["role"], message["content"])
                for message in call["conversation_history"]
            ] == [
                ("user", "earlier question"),
                ("assistant", "earlier answer"),
            ]
            assert adapter._active_continuation_sessions == {}
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_continuation_submission_refuses_an_unkeyed_api_server(
        self, tmp_path
    ):
        adapter = _make_adapter()
        db = SessionDB(tmp_path / "unkeyed-continuation.db")
        adapter._session_db = db
        try:
            session_id = db.create_session("private-session", "api_server")
            db.append_message(session_id, "user", "private history")
            descriptor = _continuation_descriptor(db, session_id)
            app = _create_runs_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                response = await cli.post(
                    "/v1/runs",
                    json={"input": "continue", "continuation": descriptor},
                )
                payload = await response.json()

            assert response.status == 403
            assert payload["error"]["code"] == "session_continuation_auth_required"
            assert "private" not in str(payload)
            assert adapter._run_statuses == {}
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_changed_revision_fails_before_run_allocation(self, adapter, tmp_path):
        db = SessionDB(tmp_path / "changed.db")
        adapter._session_db = db
        try:
            session_id = db.create_session("changed-session", "api_server")
            db.append_message(session_id, "user", "original")
            descriptor = _continuation_descriptor(db, session_id)
            db.append_message(session_id, "assistant", "new state")

            app = _create_runs_app(adapter)
            with patch.object(adapter, "_create_agent") as create_agent:
                async with TestClient(TestServer(app)) as cli:
                    resp = await cli.post(
                        "/v1/runs",
                        json={"input": "continue", "continuation": descriptor},
                    )
                    payload = await resp.json()

            assert resp.status == 409
            assert payload["error"]["code"] == "session_continuation_changed"
            assert "original" not in str(payload)
            assert "new state" not in str(payload)
            create_agent.assert_not_called()
            assert adapter._run_streams == {}
            assert adapter._run_statuses == {}
            assert adapter._active_continuation_sessions == {}
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_cross_session_revision_fails_closed(self, adapter, tmp_path):
        db = SessionDB(tmp_path / "cross-session.db")
        adapter._session_db = db
        try:
            first_id = db.create_session("first-session", "api_server")
            second_id = db.create_session("second-session", "api_server")
            db.append_message(first_id, "user", "first history")
            db.append_message(second_id, "user", "second history")
            descriptor = _continuation_descriptor(db, first_id)
            descriptor["session_id"] = second_id

            app = _create_runs_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "continue", "continuation": descriptor},
                )
                payload = await resp.json()

            assert resp.status == 409
            assert payload["error"]["code"] == "session_continuation_changed"
            assert adapter._run_statuses == {}
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_compression_rotation_makes_tip_descriptor_stale(
        self,
        adapter,
        tmp_path,
    ):
        db = SessionDB(tmp_path / "compressed.db")
        adapter._session_db = db
        try:
            tip_id = db.create_session("old-tip", "api_server")
            db.append_message(tip_id, "user", "old tip history")
            descriptor = _continuation_descriptor(db, tip_id)
            db.end_session(tip_id, "compression")
            new_tip = db.create_session(
                "new-tip",
                "api_server",
                parent_session_id=tip_id,
            )
            db.append_message(new_tip, "user", "new tip history")

            app = _create_runs_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "continue", "continuation": descriptor},
                )
                payload = await resp.json()

            assert resp.status == 409
            assert payload["error"]["code"] == "session_continuation_changed"
            assert adapter._run_statuses == {}
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_continuation_rejects_ambiguous_or_malformed_input(
        self,
        adapter,
        tmp_path,
    ):
        db = SessionDB(tmp_path / "invalid.db")
        adapter._session_db = db
        try:
            session_id = db.create_session("invalid-session", "api_server")
            db.append_message(session_id, "user", "history")
            descriptor = _continuation_descriptor(db, session_id)
            cases = [
                ({"input": "continue", "continuation": "invalid"}, 400),
                (
                    {
                        "input": "continue",
                        "continuation": {**descriptor, "extra": True},
                    },
                    400,
                ),
                (
                    {
                        "input": "continue",
                        "session_id": session_id,
                        "continuation": descriptor,
                    },
                    400,
                ),
                (
                    {
                        "input": "continue",
                        "conversation_history": [],
                        "continuation": descriptor,
                    },
                    400,
                ),
                (
                    {
                        "input": [
                            {"role": "user", "content": "old"},
                            {"role": "user", "content": "new"},
                        ],
                        "continuation": descriptor,
                    },
                    400,
                ),
                (
                    {
                        "input": "continue",
                        "continuation": {**descriptor, "version": True},
                    },
                    400,
                ),
                (
                    {
                        "input": "continue",
                        "continuation": {**descriptor, "revision": "sessionrev_bad"},
                    },
                    400,
                ),
            ]

            app = _create_runs_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                for body, expected_status in cases:
                    resp = await cli.post("/v1/runs", json=body)
                    assert resp.status == expected_status, await resp.text()

            assert adapter._run_statuses == {}
            assert adapter._run_streams == {}
            assert adapter._active_continuation_sessions == {}
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_only_one_continuation_run_can_own_a_session_and_stop_releases_it(
        self,
        adapter,
        tmp_path,
    ):
        db = SessionDB(tmp_path / "active.db")
        adapter._session_db = db
        try:
            session_id = db.create_session("active-session", "api_server")
            db.append_message(session_id, "user", "history")
            descriptor = _continuation_descriptor(db, session_id)
            slow_agent, ready, interrupted = _make_slow_agent()

            app = _create_runs_app(adapter)
            with patch.object(adapter, "_create_agent", return_value=slow_agent):
                async with TestClient(TestServer(app)) as cli:
                    first = await cli.post(
                        "/v1/runs",
                        json={"input": "first", "continuation": descriptor},
                    )
                    assert first.status == 202, await first.text()
                    first_id = (await first.json())["run_id"]
                    assert await asyncio.to_thread(ready.wait, 2)

                    second = await cli.post(
                        "/v1/runs",
                        json={"input": "second", "continuation": descriptor},
                    )
                    second_payload = await second.json()
                    assert second.status == 409
                    assert (
                        second_payload["error"]["code"]
                        == "session_continuation_active"
                    )

                    stopped = await cli.post(f"/v1/runs/{first_id}/stop")
                    assert stopped.status == 200
                    assert interrupted.wait(timeout=2)
                    for _ in range(40):
                        if not adapter._active_continuation_sessions:
                            break
                        await asyncio.sleep(0.025)

            assert adapter._active_continuation_sessions == {}
            assert len(adapter._run_statuses) == 1
            assert adapter._run_statuses[first_id]["status"] == "cancelled"
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_verification_reservation_closes_concurrent_snapshot_race(
        self,
        adapter,
        tmp_path,
    ):
        db = SessionDB(tmp_path / "verification-race.db")
        try:
            session_id = db.create_session("verification-race", "api_server")
            db.append_message(session_id, "user", "history")
            descriptor = _continuation_descriptor(db, session_id)
            snapshot_started = threading.Event()
            release_snapshot = threading.Event()

            class BlockingSnapshotDB:
                calls = 0

                def get_continuation_snapshot(self, requested_id):
                    self.calls += 1
                    snapshot_started.set()
                    release_snapshot.wait(timeout=5)
                    return db.get_continuation_snapshot(requested_id)

            blocking_db = BlockingSnapshotDB()
            adapter._session_db = blocking_db
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "done"}
            mock_agent.session_prompt_tokens = 0
            mock_agent.session_completion_tokens = 0
            mock_agent.session_total_tokens = 0

            app = _create_runs_app(adapter)
            with patch.object(adapter, "_create_agent", return_value=mock_agent):
                async with TestClient(TestServer(app)) as cli:
                    first_task = asyncio.create_task(
                        cli.post(
                            "/v1/runs",
                            json={"input": "first", "continuation": descriptor},
                        )
                    )
                    assert await asyncio.to_thread(snapshot_started.wait, 2)

                    second = await cli.post(
                        "/v1/runs",
                        json={"input": "second", "continuation": descriptor},
                    )
                    second_payload = await second.json()
                    assert second.status == 409
                    assert (
                        second_payload["error"]["code"]
                        == "session_continuation_active"
                    )
                    assert blocking_db.calls == 1

                    release_snapshot.set()
                    first = await first_task
                    assert first.status == 202, await first.text()
                    for _ in range(40):
                        if not adapter._active_continuation_sessions:
                            break
                        await asyncio.sleep(0.025)

            assert adapter._active_continuation_sessions == {}
            assert len(adapter._run_statuses) == 1
        finally:
            if "release_snapshot" in locals():
                release_snapshot.set()
            db.close()


class TestStartRunValidation:

    @pytest.mark.asyncio
    async def test_start_invalid_json_returns_400(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs",
                data="not json",
                headers={"Content-Type": "application/json"},
            )
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_start_missing_input_returns_400(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json={"model": "test"})
            assert resp.status == 400
            data = await resp.json()
            assert "input" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_start_empty_input_returns_400(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json={"input": ""})
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_start_invalid_history_does_not_allocate_run(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs",
                json={"input": "hello", "conversation_history": {"role": "user"}},
            )
        assert resp.status == 400
        assert adapter._run_streams == {}
        assert adapter._run_statuses == {}

    @pytest.mark.asyncio
    async def test_start_requires_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json={"input": "hello"})
        assert resp.status == 401

    @pytest.mark.asyncio
    async def test_start_with_valid_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "ok"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                assert resp.status == 202

    @pytest.mark.asyncio
    async def test_start_accepts_bounded_inline_images_through_runs_lifecycle(self, adapter):
        app = _create_runs_app(adapter)
        image_input = [
            {"type": "input_text", "text": "Describe this image."},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
        ]
        expected_message = [
            {"type": "text", "text": "Describe this image."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "A tiny image."}
                mock_agent.session_prompt_tokens = 1
                mock_agent.session_completion_tokens = 1
                mock_agent.session_total_tokens = 2
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": image_input})
                assert resp.status == 202, await resp.text()
                run_id = (await resp.json())["run_id"]
                for _ in range(20):
                    await asyncio.sleep(0)
                    if adapter._run_statuses[run_id]["status"] == "completed":
                        break

        _, kwargs = mock_agent.run_conversation.call_args
        assert kwargs["user_message"] == expected_message
        assert adapter._run_statuses[run_id]["status"] == "completed"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("image_url", "expected_code"),
        [
            ("https://private.example/image.png", "run_image_data_url_required"),
            ("data:image/svg+xml;base64,PHN2Zy8+", "run_image_data_url_required"),
            ("data:image/png;base64,not base64", "run_image_data_url_required"),
        ],
    )
    async def test_start_rejects_nonprivate_or_invalid_inline_image_inputs(
        self, adapter, image_url, expected_code
    ):
        app = _create_runs_app(adapter)
        payload = {"input": [{"type": "input_image", "image_url": image_url}]}
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json=payload)
            body = await resp.json()
        assert resp.status == 400
        assert body["error"]["code"] == expected_code
        assert adapter._run_streams == {}
        assert adapter._run_statuses == {}

    @pytest.mark.asyncio
    async def test_start_rejects_too_many_inline_images_before_allocating_run(self, adapter):
        app = _create_runs_app(adapter)
        payload = {
            "input": [
                {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}
                for _ in range(RUN_INLINE_IMAGE_MAX_COUNT + 1)
            ]
        }
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json=payload)
            body = await resp.json()
        assert resp.status == 400
        assert body["error"]["code"] == "run_image_limit"
        assert adapter._run_streams == {}
        assert adapter._run_statuses == {}

    @pytest.mark.asyncio
    async def test_start_rejects_unbounded_image_detail_before_allocating_run(self, adapter):
        app = _create_runs_app(adapter)
        payload = {
            "input": [{
                "type": "input_image",
                "image_url": {
                    "url": "data:image/png;base64,AAAA",
                    "detail": "operator supplied unrestricted detail text",
                },
            }]
        }
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json=payload)
            body = await resp.json()
        assert resp.status == 400
        assert body["error"]["code"] == "run_image_detail_unsupported"
        assert adapter._run_streams == {}
        assert adapter._run_statuses == {}


# ---------------------------------------------------------------------------
# GET /v1/runs/{run_id} — poll run status
# ---------------------------------------------------------------------------


class TestRunStatus:
    @pytest.mark.asyncio
    async def test_status_completed_run_includes_output_and_usage(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 4
                mock_agent.session_completion_tokens = 2
                mock_agent.session_total_tokens = 6
                mock_agent.context_compressor.latest_provider_prompt_tokens = 2_048
                mock_agent.context_compressor.context_length = 128_000
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(20):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    assert status_resp.status == 200
                    status = await status_resp.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

                assert status["status"] == "completed"
                assert status["output"] == "done"
                assert status["usage"]["total_tokens"] == 6
                assert status["usage"]["context_tokens"] == 2_048
                assert status["usage"]["context_length"] == 128_000
                assert status["last_event"] == "run.completed"

    def test_run_usage_omits_impossible_context_values(self):
        mock_agent = MagicMock()
        mock_agent.session_prompt_tokens = 4
        mock_agent.session_completion_tokens = 2
        mock_agent.session_total_tokens = 6
        mock_agent.context_compressor.latest_provider_prompt_tokens = 128_001
        mock_agent.context_compressor.context_length = 128_000

        usage = _agent_run_usage(mock_agent)
        assert usage == {
            "input_tokens": 4,
            "output_tokens": 2,
            "total_tokens": 6,
        }
        mock_agent.context_compressor.latest_provider_prompt_tokens = 0
        assert _agent_run_usage(mock_agent) == usage
        mock_agent.context_compressor.latest_provider_prompt_tokens = 2_048
        mock_agent.context_compressor.context_length = 128_000
        mock_agent.provider = "moa"
        assert _agent_run_usage(mock_agent) == usage

    @pytest.mark.asyncio
    async def test_structured_failure_status_retains_context_usage(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {
                    "failed": True,
                    "error": "provider rejected the request",
                }
                mock_agent.session_prompt_tokens = 4
                mock_agent.session_completion_tokens = 2
                mock_agent.session_total_tokens = 6
                mock_agent.context_compressor.latest_provider_prompt_tokens = 2_048
                mock_agent.context_compressor.context_length = 128_000
                mock_create.return_value = mock_agent

                started = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await started.json())["run_id"]
                for _ in range(20):
                    status = await (await cli.get(f"/v1/runs/{run_id}")).json()
                    if status["status"] == "failed":
                        break
                    await asyncio.sleep(0.05)

        assert status["status"] == "failed"
        assert status["usage"]["context_tokens"] == 2_048
        assert status["usage"]["context_length"] == 128_000

    @pytest.mark.asyncio
    async def test_post_result_cancellation_retains_context_usage(self, adapter):
        app = _create_runs_app(adapter)
        worker_started = threading.Event()
        release_worker = threading.Event()

        def finish_after_stop(**_kwargs):
            worker_started.set()
            release_worker.wait(timeout=3)
            return {"final_response": "finished while stop was pending"}

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.side_effect = finish_after_stop
                mock_agent.session_prompt_tokens = 4
                mock_agent.session_completion_tokens = 2
                mock_agent.session_total_tokens = 6
                mock_agent.context_compressor.latest_provider_prompt_tokens = 2_048
                mock_agent.context_compressor.context_length = 128_000
                mock_create.return_value = mock_agent

                started = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await started.json())["run_id"]
                assert await asyncio.to_thread(worker_started.wait, 2)
                adapter._stopping_run_ids.add(run_id)
                release_worker.set()
                for _ in range(20):
                    status = await (await cli.get(f"/v1/runs/{run_id}")).json()
                    if status["status"] == "cancelled":
                        break
                    await asyncio.sleep(0.05)

        assert status["status"] == "cancelled"
        assert status["usage"]["context_tokens"] == 2_048
        assert status["usage"]["context_length"] == 128_000

    @pytest.mark.asyncio
    async def test_status_reflects_explicit_session_id(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", "session_id": "space-session"},
                )
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(20):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

                mock_agent.run_conversation.assert_called_once()
                assert mock_agent.run_conversation.call_args.kwargs["task_id"] == "space-session"
                assert status["session_id"] == "space-session"

    @pytest.mark.asyncio
    async def test_status_not_found_returns_404(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/run_nonexistent")
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_status_requires_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/run_any")
        assert resp.status == 401


# ---------------------------------------------------------------------------
# GET /v1/runs/{run_id}/events — SSE event stream
# ---------------------------------------------------------------------------


class TestRunEvents:
    @pytest.mark.asyncio
    async def test_events_stream_returns_completed(self, adapter):
        """Events stream should receive run.completed when agent finishes."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "Hello!"}
                mock_agent.session_prompt_tokens = 10
                mock_agent.session_completion_tokens = 5
                mock_agent.session_total_tokens = 15
                mock_agent.context_compressor.latest_provider_prompt_tokens = 4_096
                mock_agent.context_compressor.context_length = 200_000
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                # Subscribe to events
                events_resp = await cli.get(f"/v1/runs/{run_id}/events")
                assert events_resp.status == 200
                body = await events_resp.text()

                # Should contain run.completed
                assert "run.completed" in body
                assert "Hello!" in body
                events = [
                    json.loads(line.removeprefix("data: "))
                    for line in body.splitlines()
                    if line.startswith("data: ")
                ]
                completed = next(
                    event for event in events if event["event"] == "run.completed"
                )
                assert completed["usage"]["context_tokens"] == 4_096
                assert completed["usage"]["context_length"] == 200_000

    @pytest.mark.asyncio
    async def test_approval_event_includes_request_id_and_safe_preview(self, adapter):
        app = _create_runs_app(adapter)
        private_path = "/Users/alice/private-project"
        credential = "sk-event-secret"
        finish_run = threading.Event()

        def _run_with_approval_event(**_kwargs):
            session_key = approval_mod.get_current_session_key()
            with approval_mod._lock:
                notify = approval_mod._gateway_notify_cbs[session_key]
            entry = approval_mod._ApprovalEntry({
                "command": f"rm -rf {private_path} --token {credential}",
                "description": f"Delete {private_path} using {credential}",
                "pattern_key": "recursive delete",
                "pattern_keys": ["recursive delete"],
            })
            with approval_mod._lock:
                approval_mod._gateway_queues.setdefault(session_key, []).append(entry)
            notify(entry.data)
            finish_run.wait(timeout=5)
            with approval_mod._lock:
                approval_mod._gateway_queues.pop(session_key, None)
            return {"final_response": "stopped for review"}

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.side_effect = _run_with_approval_event
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                response = await cli.post("/v1/runs", json={"input": "do it"})
                run_id = (await response.json())["run_id"]
                for _ in range(100):
                    if adapter._run_statuses[run_id]["status"] == "waiting_for_approval":
                        break
                    await asyncio.sleep(0.01)
                assert adapter._run_statuses[run_id]["status"] == "waiting_for_approval"
                pending_action = adapter._run_statuses[run_id]["pending_action"]
                assert pending_action == {
                    "version": 1,
                    "kind": "approval",
                    "request_id": pending_action["request_id"],
                    "preview": {
                        "version": 1,
                        "category": "terminal_command",
                        "title": "Terminal command approval",
                        "summary": "Hermes stopped a terminal command that matched a protected action.",
                        "risk_labels": ["recursive delete"],
                    },
                    "choices": ["once", "session", "always", "deny"],
                }
                await asyncio.sleep(0)
                finish_run.set()
                events_response = await cli.get(f"/v1/runs/{run_id}/events")
                body = await events_response.text()

        events = [
            json.loads(line.removeprefix("data: "))
            for line in body.splitlines()
            if line.startswith("data: ")
        ]
        approval_event = next(event for event in events if event["event"] == "approval.request")
        assert approval_event["request_id"]
        assert approval_event["preview"] == {
            "version": 1,
            "category": "terminal_command",
            "title": "Terminal command approval",
            "summary": "Hermes stopped a terminal command that matched a protected action.",
            "risk_labels": ["recursive delete"],
        }
        assert private_path not in json.dumps(approval_event["preview"])
        assert credential not in json.dumps(approval_event["preview"])

    @pytest.mark.asyncio
    async def test_reconnect_replays_only_events_after_last_event_id(self, adapter):
        run_id = "run_replay_cursor"
        adapter._run_streams[run_id] = asyncio.Queue(maxsize=_RUN_EVENT_RETENTION)
        adapter._run_streams_created[run_id] = time.time()
        adapter._run_event_logs[run_id] = deque(maxlen=_RUN_EVENT_RETENTION)
        adapter._run_event_sequences[run_id] = 0
        adapter._run_subscriber_queues[run_id] = set()
        adapter._set_run_status(run_id, "running")
        adapter._publish_run_event(run_id, {
            "event": "tool.started", "run_id": run_id, "tool": "web_search",
        })
        adapter._publish_run_event(run_id, {
            "event": "tool.completed", "run_id": run_id, "tool": "web_search",
        })
        adapter._publish_run_event(run_id, {
            "event": "run.completed", "run_id": run_id, "output": "done",
        })
        adapter._set_run_status(run_id, "completed", output="done")

        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.get(
                f"/v1/runs/{run_id}/events",
                headers={"Last-Event-ID": "1"},
            )
            body = await response.text()

        assert response.status == 200
        assert "id: 1" not in body
        assert "id: 2" in body
        assert "id: 3" in body
        assert body.index("id: 2") < body.index("id: 3")

    @pytest.mark.asyncio
    async def test_reconnect_rejects_invalid_ahead_or_expired_cursor(self, adapter):
        run_id = "run_replay_invalid_cursor"
        adapter._run_streams[run_id] = asyncio.Queue(maxsize=_RUN_EVENT_RETENTION)
        adapter._run_streams_created[run_id] = time.time()
        adapter._run_event_logs[run_id] = deque([
            {
                "event": "tool.completed",
                "run_id": run_id,
                "tool": "web_search",
                "sequence": 2,
                "event_id": 2,
            },
            {
                "event": "run.completed",
                "run_id": run_id,
                "output": "done",
                "sequence": 3,
                "event_id": 3,
            },
        ], maxlen=_RUN_EVENT_RETENTION)
        adapter._run_event_sequences[run_id] = 3
        adapter._run_subscriber_queues[run_id] = set()
        adapter._set_run_status(run_id, "completed", output="done")

        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            invalid = await cli.get(
                f"/v1/runs/{run_id}/events",
                headers={"Last-Event-ID": "not-a-cursor"},
            )
            ahead = await cli.get(
                f"/v1/runs/{run_id}/events",
                headers={"Last-Event-ID": "4"},
            )
            expired = await cli.get(
                f"/v1/runs/{run_id}/events",
                headers={"Last-Event-ID": "0"},
            )
            invalid_payload = await invalid.json()
            ahead_payload = await ahead.json()
            expired_payload = await expired.json()

        assert invalid.status == 400
        assert invalid_payload["error"]["code"] == "run_event_cursor_invalid"
        assert ahead.status == 409
        assert ahead_payload["error"]["code"] == "run_event_cursor_invalid"
        assert expired.status == 409
        assert expired_payload["error"]["code"] == "run_event_cursor_expired"

    def test_terminal_close_preserves_the_terminal_event_before_sentinel(self, adapter):
        run_id = "run_terminal_queue_order"
        adapter._run_streams[run_id] = asyncio.Queue(maxsize=2)
        adapter._run_event_logs[run_id] = deque(maxlen=_RUN_EVENT_RETENTION)
        adapter._run_event_sequences[run_id] = 0
        subscriber = asyncio.Queue(maxsize=2)
        adapter._run_subscriber_queues[run_id] = {subscriber}

        adapter._publish_run_event(run_id, {
            "event": "run.completed",
            "run_id": run_id,
            "output": "done",
        })
        adapter._close_run_event_stream(run_id)

        assert subscriber.get_nowait()["event"] == "run.completed"
        assert subscriber.get_nowait() is None

    def test_saturated_terminal_close_preserves_every_queued_sequence(self, adapter):
        run_id = "run_terminal_saturated_queue"
        adapter._run_streams[run_id] = asyncio.Queue(maxsize=2)
        adapter._run_event_logs[run_id] = deque(maxlen=_RUN_EVENT_RETENTION)
        adapter._run_event_sequences[run_id] = 0
        subscriber = asyncio.Queue(maxsize=2)
        adapter._run_subscriber_queues[run_id] = {subscriber}

        adapter._publish_run_event(run_id, {
            "event": "tool.started",
            "run_id": run_id,
            "tool": "web_search",
        })
        adapter._publish_run_event(run_id, {
            "event": "run.completed",
            "run_id": run_id,
            "output": "done",
        })
        adapter._close_run_event_stream(run_id)

        first = subscriber.get_nowait()
        terminal = subscriber.get_nowait()
        assert [first["sequence"], terminal["sequence"]] == [1, 2]
        assert terminal["event"] == "run.completed"
        assert subscriber.empty()

    def test_replay_journal_normalizes_private_and_oversized_event_content(self, adapter):
        run_id = "run_public_event_normalization"
        adapter._run_streams[run_id] = asyncio.Queue(maxsize=_RUN_EVENT_RETENTION)
        adapter._run_event_logs[run_id] = deque(maxlen=_RUN_EVENT_RETENTION)
        adapter._run_event_sequences[run_id] = 0
        adapter._run_subscriber_queues[run_id] = set()
        private_path = "/Users/alice/private-project/credentials.txt"
        secret = "sk-proj-" + ("x" * 48)
        raw_command = f"rm -rf {private_path} --token {secret}"

        adapter._publish_run_event(run_id, {
            "event": "tool.started",
            "run_id": run_id,
            "tool": "terminal",
            "preview": raw_command,
        })
        adapter._publish_run_event(run_id, {
            "event": "reasoning.available",
            "run_id": run_id,
            "text": f"Inspect {private_path} with {secret}",
        })
        for _ in range(400):
            adapter._publish_run_event(run_id, {
                "event": "message.delta",
                "run_id": run_id,
                "delta": ("safe output " * 1_300) + private_path + secret,
            })
        adapter._publish_run_event(run_id, {
            "event": "run.completed",
            "run_id": run_id,
            "output": raw_command + (" result" * 10_000),
        })

        serialized = json.dumps(list(adapter._run_event_logs[run_id]))
        assert private_path not in serialized
        assert secret not in serialized
        assert raw_command not in serialized
        assert '"preview"' not in serialized
        assert '"text"' not in serialized
        assert adapter._run_event_log_bytes[run_id] <= _RUN_EVENT_JOURNAL_MAX_BYTES
        assert len(serialized.encode("utf-8")) <= _RUN_EVENT_JOURNAL_MAX_BYTES + 4096

    def test_streaming_redaction_covers_secrets_split_across_adjacent_deltas(self, adapter):
        run_id = "run_split_secret_delta"
        adapter._run_streams[run_id] = asyncio.Queue(maxsize=_RUN_EVENT_RETENTION)
        adapter._run_event_logs[run_id] = deque(maxlen=_RUN_EVENT_RETENTION)
        adapter._run_event_sequences[run_id] = 0
        adapter._run_subscriber_queues[run_id] = set()
        provider_secret = "sk-proj-" + ("p" * 48)
        bearer_token = "b" * 400
        bearer_secret = "Bearer " + bearer_token

        for delta in (
            "Provider key: sk-proj-",
            "p" * 48,
            " ",
            "Authorization: Bear",
            "er ",
            bearer_token,
            " ",
            "Path: /Users/",
            "alice/private.txt",
            " ",
        ):
            adapter._publish_run_event(run_id, {
                "event": "message.delta",
                "run_id": run_id,
                "delta": delta,
            })
        adapter._publish_run_event(run_id, {
            "event": "run.completed",
            "run_id": run_id,
            "output": "done",
        })

        reconstructed = "".join(
            event.get("delta", "")
            for event in adapter._run_event_logs[run_id]
            if event["event"] == "message.delta"
        )
        assert provider_secret not in reconstructed
        assert bearer_secret not in reconstructed
        assert bearer_token not in reconstructed
        assert "/Users/alice/private.txt" not in reconstructed

    def test_buffered_delta_is_sequenced_before_following_tool_event(self, adapter):
        run_id = "run_delta_tool_order"
        adapter._run_streams[run_id] = asyncio.Queue(maxsize=_RUN_EVENT_RETENTION)
        adapter._run_event_logs[run_id] = deque(maxlen=_RUN_EVENT_RETENTION)
        adapter._run_event_sequences[run_id] = 0
        adapter._run_subscriber_queues[run_id] = set()

        adapter._publish_run_event(run_id, {
            "event": "message.delta",
            "run_id": run_id,
            "delta": "I will check that now. ",
        })
        adapter._publish_run_event(run_id, {
            "event": "tool.started",
            "run_id": run_id,
            "tool": "web_search",
        })
        adapter._publish_run_event(run_id, {
            "event": "run.completed",
            "run_id": run_id,
            "output": "done",
        })

        assert [
            event["event"]
            for event in adapter._run_event_logs[run_id]
        ] == ["message.delta", "tool.started", "run.completed"]

    def test_incomplete_bearer_marker_survives_tool_boundary_without_token_leak(self, adapter):
        run_id = "run_bearer_tool_boundary"
        adapter._run_streams[run_id] = asyncio.Queue(maxsize=_RUN_EVENT_RETENTION)
        adapter._run_event_logs[run_id] = deque(maxlen=_RUN_EVENT_RETENTION)
        adapter._run_event_sequences[run_id] = 0
        adapter._run_subscriber_queues[run_id] = set()
        token = "q" * 400

        for event in (
            {
                "event": "message.delta",
                "run_id": run_id,
                "delta": "Authorization: Bearer ",
            },
            {
                "event": "tool.started",
                "run_id": run_id,
                "tool": "web_search",
            },
            {
                "event": "message.delta",
                "run_id": run_id,
                "delta": token,
            },
            {
                "event": "message.delta",
                "run_id": run_id,
                "delta": " done ",
            },
            {
                "event": "run.completed",
                "run_id": run_id,
                "output": "done",
            },
        ):
            adapter._publish_run_event(run_id, event)

        reconstructed = "".join(
            event.get("delta", "")
            for event in adapter._run_event_logs[run_id]
        )
        assert token not in reconstructed
        assert [
            event["event"]
            for event in adapter._run_event_logs[run_id]
        ][1] == "tool.started"

    @pytest.mark.parametrize(
        "boundary_event",
        [
            {"event": "tool.started", "tool": "web_search"},
            {"event": "reasoning.available"},
            {
                "event": "approval.request",
                "request_id": "approval-boundary",
                "preview": {
                    "version": 1,
                    "category": "terminal_command",
                    "risk_labels": ["protected action"],
                },
                "choices": ["once", "deny"],
            },
            {
                "event": "clarify.request",
                "request_id": "clarify_0123456789abcdef0123456789abcdef",
                "prompt": {
                    "version": 1,
                    "type": "text",
                    "question": "Continue?",
                },
            },
            {"event": "run.completed", "output": "boundary"},
        ],
        ids=["tool", "reasoning", "approval", "clarification", "terminal"],
    )
    def test_every_credential_marker_split_survives_lifecycle_boundaries(
        self,
        adapter,
        boundary_event,
    ):
        token = "boundary-secret-" + ("q" * 300)
        marker_cases = (
            ("Authorization: Bearer ", "\nvisible"),
            ('Authorization: "Bearer ', '"\nvisible'),
            ("Authorization='Bearer ", "'\nvisible"),
        )

        for marker_index, (marker, suffix) in enumerate(marker_cases):
            for split in range(1, len(marker)):
                run_id = (
                    f"run_marker_boundary_{boundary_event['event']}_"
                    f"{marker_index}_{split}"
                )
                adapter._run_streams[run_id] = asyncio.Queue(
                    maxsize=_RUN_EVENT_RETENTION
                )
                adapter._run_event_logs[run_id] = deque(
                    maxlen=_RUN_EVENT_RETENTION
                )
                adapter._run_event_sequences[run_id] = 0
                subscriber = asyncio.Queue(maxsize=_RUN_EVENT_RETENTION)
                adapter._run_subscriber_queues[run_id] = {subscriber}

                adapter._publish_run_event(run_id, {
                    "event": "message.delta",
                    "run_id": run_id,
                    "delta": marker[:split],
                })
                adapter._publish_run_event(
                    run_id,
                    {**boundary_event, "run_id": run_id},
                )
                for delta in (marker[split:], token, suffix):
                    adapter._publish_run_event(run_id, {
                        "event": "message.delta",
                        "run_id": run_id,
                        "delta": delta,
                    })
                adapter._publish_run_event(run_id, {
                    "event": "run.completed",
                    "run_id": run_id,
                    "output": "done",
                })

                serialized = json.dumps(list(adapter._run_event_logs[run_id]))
                reconstructed = "".join(
                    event.get("delta", "")
                    for event in adapter._run_event_logs[run_id]
                )
                delivered = []
                while not subscriber.empty():
                    delivered.append(subscriber.get_nowait())
                assert token not in serialized
                assert token not in reconstructed
                assert token not in json.dumps(delivered)

    @pytest.mark.parametrize(
        ("assignment", "suffix"),
        [
            (": ", "\nvisible"),
            (': "', '"\nvisible'),
            ("='", "'\nvisible"),
        ],
        ids=["plain", "double-quoted", "single-quoted"],
    )
    def test_lifecycle_probe_ignores_long_assignment_padding(
        self,
        adapter,
        assignment,
        suffix,
    ):
        run_id = f"run_long_credential_padding_{len(assignment)}"
        adapter._run_streams[run_id] = asyncio.Queue(
            maxsize=_RUN_EVENT_RETENTION
        )
        adapter._run_event_logs[run_id] = deque(
            maxlen=_RUN_EVENT_RETENTION
        )
        adapter._run_event_sequences[run_id] = 0
        subscriber = asyncio.Queue(maxsize=_RUN_EVENT_RETENTION)
        adapter._run_subscriber_queues[run_id] = {subscriber}
        token = "long-padding-secret-" + ("x" * 300)

        for event in (
            {
                "event": "message.delta",
                "run_id": run_id,
                "delta": "Authorization",
            },
            {
                "event": "reasoning.available",
                "run_id": run_id,
            },
            {
                "event": "message.delta",
                "run_id": run_id,
                "delta": (" " * 140) + assignment + token + suffix,
            },
            {
                "event": "run.completed",
                "run_id": run_id,
                "output": "done",
            },
        ):
            adapter._publish_run_event(run_id, event)

        journal = json.dumps(list(adapter._run_event_logs[run_id]))
        delivered = []
        while not subscriber.empty():
            delivered.append(subscriber.get_nowait())
        assert token not in journal
        assert token not in json.dumps(delivered)

    @pytest.mark.parametrize("padding", [257, 400, 1024])
    @pytest.mark.parametrize(
        ("assignment", "suffix"),
        [
            (": ", "\nvisible"),
            (': "', '"\nvisible'),
            ("='", "'\nvisible"),
        ],
        ids=["plain", "double-quoted", "single-quoted"],
    )
    def test_routine_buffer_emission_preserves_credential_probe(
        self,
        adapter,
        padding,
        assignment,
        suffix,
    ):
        run_id = f"run_buffer_probe_{padding}_{len(assignment)}"
        adapter._run_streams[run_id] = asyncio.Queue(
            maxsize=_RUN_EVENT_RETENTION
        )
        adapter._run_event_logs[run_id] = deque(
            maxlen=_RUN_EVENT_RETENTION
        )
        adapter._run_event_sequences[run_id] = 0
        subscriber = asyncio.Queue(maxsize=_RUN_EVENT_RETENTION)
        adapter._run_subscriber_queues[run_id] = {subscriber}
        token = "buffer-boundary-secret-" + ("y" * 400)

        for delta in (
            "Authorization" + (" " * padding),
            assignment + token + suffix,
        ):
            adapter._publish_run_event(run_id, {
                "event": "message.delta",
                "run_id": run_id,
                "delta": delta,
            })
        adapter._publish_run_event(run_id, {
            "event": "run.completed",
            "run_id": run_id,
            "output": "done",
        })

        journal = json.dumps(list(adapter._run_event_logs[run_id]))
        delivered = []
        while not subscriber.empty():
            delivered.append(subscriber.get_nowait())
        assert token not in journal
        assert token not in json.dumps(delivered)

    def test_repeated_lifecycle_boundaries_advance_probe_once_per_delta(
        self,
        adapter,
    ):
        run_id = "run_repeated_probe_boundaries"
        adapter._run_streams[run_id] = asyncio.Queue(
            maxsize=_RUN_EVENT_RETENTION
        )
        adapter._run_event_logs[run_id] = deque(
            maxlen=_RUN_EVENT_RETENTION
        )
        adapter._run_event_sequences[run_id] = 0
        subscriber = asyncio.Queue(maxsize=_RUN_EVENT_RETENTION)
        adapter._run_subscriber_queues[run_id] = {subscriber}
        token = "repeated-boundary-secret-" + ("r" * 300)

        for event in (
            {
                "event": "message.delta",
                "run_id": run_id,
                "delta": "Au",
            },
            {
                "event": "tool.started",
                "run_id": run_id,
                "tool": "web_search",
            },
            {
                "event": "message.delta",
                "run_id": run_id,
                "delta": "t",
            },
            {
                "event": "reasoning.available",
                "run_id": run_id,
            },
            {
                "event": "message.delta",
                "run_id": run_id,
                "delta": f'horization: "{token}"\nvisible',
            },
            {
                "event": "run.completed",
                "run_id": run_id,
                "output": "done",
            },
        ):
            adapter._publish_run_event(run_id, event)

        journal = json.dumps(list(adapter._run_event_logs[run_id]))
        delivered = []
        while not subscriber.empty():
            delivered.append(subscriber.get_nowait())
        assert token not in journal
        assert token not in json.dumps(delivered)

    def test_mismatched_probe_still_captures_new_marker_suffix(self, adapter):
        run_id = "run_replaced_probe_suffix"
        adapter._run_streams[run_id] = asyncio.Queue(
            maxsize=_RUN_EVENT_RETENTION
        )
        adapter._run_event_logs[run_id] = deque(
            maxlen=_RUN_EVENT_RETENTION
        )
        adapter._run_event_sequences[run_id] = 0
        adapter._run_subscriber_queues[run_id] = set()
        token = "replacement-probe-secret-" + ("s" * 300)

        for event in (
            {
                "event": "message.delta",
                "run_id": run_id,
                "delta": "A",
            },
            {
                "event": "tool.started",
                "run_id": run_id,
                "tool": "web_search",
            },
            {
                "event": "message.delta",
                "run_id": run_id,
                "delta": "x Authorization",
            },
            {
                "event": "reasoning.available",
                "run_id": run_id,
            },
            {
                "event": "message.delta",
                "run_id": run_id,
                "delta": f': "{token}"\nvisible',
            },
            {
                "event": "run.completed",
                "run_id": run_id,
                "output": "done",
            },
        ):
            adapter._publish_run_event(run_id, event)

        assert token not in json.dumps(list(adapter._run_event_logs[run_id]))

    @pytest.mark.parametrize("line_break", ["\n", "\r\n"])
    def test_line_break_terminates_same_line_credential_probe(
        self,
        adapter,
        line_break,
    ):
        run_id = f"run_line_break_probe_{len(line_break)}"
        adapter._run_streams[run_id] = asyncio.Queue(
            maxsize=_RUN_EVENT_RETENTION
        )
        adapter._run_event_logs[run_id] = deque(
            maxlen=_RUN_EVENT_RETENTION
        )
        adapter._run_event_sequences[run_id] = 0
        adapter._run_subscriber_queues[run_id] = set()

        for event in (
            {
                "event": "message.delta",
                "run_id": run_id,
                "delta": "Authorization:",
            },
            {
                "event": "tool.started",
                "run_id": run_id,
                "tool": "web_search",
            },
            {
                "event": "message.delta",
                "run_id": run_id,
                "delta": line_break + "ordinary next-line output" + line_break,
            },
            {
                "event": "run.completed",
                "run_id": run_id,
                "output": "done",
            },
        ):
            adapter._publish_run_event(run_id, event)

        reconstructed = "".join(
            event.get("delta", "")
            for event in adapter._run_event_logs[run_id]
        )
        assert "ordinary next-line output" in reconstructed

    def test_oversized_single_delta_never_enters_retained_buffer(self, adapter):
        run_id = "run_oversized_single_delta"
        adapter._run_streams[run_id] = asyncio.Queue(maxsize=_RUN_EVENT_RETENTION)
        adapter._run_event_logs[run_id] = deque(maxlen=_RUN_EVENT_RETENTION)
        adapter._run_event_sequences[run_id] = 0
        adapter._run_subscriber_queues[run_id] = set()
        oversized = "opaque" * 500_000

        adapter._publish_run_event(run_id, {
            "event": "message.delta",
            "run_id": run_id,
            "delta": oversized,
        })

        assert oversized not in adapter._run_delta_buffers.get(run_id, "")
        assert len(adapter._run_delta_buffers.get(run_id, "")) < 100

    @pytest.mark.parametrize(
        "template",
        [
            "Authorization: Bearer {token}",
            'Authorization: "Bearer {token}"',
            "Authorization='Bearer {token}'",
            'Bearer "{token}"',
        ],
    )
    def test_terminal_status_redacts_opaque_bearer_value(self, adapter, template):
        token = "z" * 400
        status = adapter._set_run_status(
            "run_terminal_bearer_redaction",
            "completed",
            output=template.format(token=token),
        )
        assert token not in status["output"]
        assert "credential removed" in status["output"]

    def test_runtime_event_rejects_embedded_secret_token(self, adapter):
        run_id = "run_embedded_runtime_secret"
        adapter._run_streams[run_id] = asyncio.Queue(maxsize=_RUN_EVENT_RETENTION)
        adapter._run_event_logs[run_id] = deque(maxlen=_RUN_EVENT_RETENTION)
        adapter._run_event_sequences[run_id] = 0
        adapter._run_subscriber_queues[run_id] = set()

        published = adapter._publish_run_event(run_id, {
            "event": "runtime.updated",
            "run_id": run_id,
            "provider": "openrouter",
            "model": "openai/sk-proj-" + ("x" * 40),
        })

        assert published is None
        assert list(adapter._run_event_logs[run_id]) == []

    @pytest.mark.asyncio
    async def test_two_subscribers_receive_the_same_live_events(self, adapter):
        run_id = "run_two_subscribers"
        adapter._run_streams[run_id] = asyncio.Queue(maxsize=_RUN_EVENT_RETENTION)
        adapter._run_streams_created[run_id] = time.time()
        adapter._run_event_logs[run_id] = deque(maxlen=_RUN_EVENT_RETENTION)
        adapter._run_event_sequences[run_id] = 0
        adapter._run_subscriber_queues[run_id] = set()
        adapter._set_run_status(run_id, "running")

        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            first = await cli.get(f"/v1/runs/{run_id}/events")
            second = await cli.get(f"/v1/runs/{run_id}/events")
            assert len(adapter._run_subscriber_queues[run_id]) == 2
            adapter._publish_run_event(run_id, {
                "event": "tool.started", "run_id": run_id, "tool": "terminal",
            })
            adapter._publish_run_event(run_id, {
                "event": "run.completed", "run_id": run_id, "output": "done",
            })
            adapter._set_run_status(run_id, "completed", output="done")
            first_body, second_body = await asyncio.gather(
                first.text(), second.text()
            )

        for body in (first_body, second_body):
            assert "tool.started" in body
            assert "run.completed" in body
            assert "id: 1" in body
            assert "id: 2" in body



    @pytest.mark.asyncio
    async def test_approval_response_without_pending_returns_409(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                data = await resp.json()
                run_id = data["run_id"]

                approval_resp = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once"},
                )
                assert approval_resp.status == 409
                approval_data = await approval_resp.json()
                assert approval_data["error"]["code"] in {
                    "approval_not_active",
                    "approval_not_pending",
                }

    @pytest.mark.asyncio
    async def test_approval_string_false_does_not_resolve_all(self, adapter):
        """Quoted false must not fan out approval resolution across the queue."""
        app = _create_runs_app(adapter)
        run_id = "run_bool_parse"
        adapter._run_statuses[run_id] = {"run_id": run_id, "status": "running"}
        adapter._run_approval_sessions[run_id] = "session-123"

        async with TestClient(TestServer(app)) as cli:
            with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
                approval_resp = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once", "all": "false"},
                )

        assert approval_resp.status == 200
        mock_resolve.assert_called_once_with(
            "session-123",
            "once",
            resolve_all=False,
        )

    @pytest.mark.asyncio
    async def test_approval_request_id_resolves_only_the_matching_entry(self, adapter):
        app = _create_runs_app(adapter)
        run_id = "run_exact_approval"
        adapter._run_statuses[run_id] = {"run_id": run_id, "status": "waiting_for_approval"}
        adapter._run_approval_sessions[run_id] = run_id
        adapter._run_streams[run_id] = asyncio.Queue()
        first = approval_mod._ApprovalEntry({"request_id": "req-first"})
        second = approval_mod._ApprovalEntry({"request_id": "req-second"})
        with approval_mod._lock:
            approval_mod._gateway_queues[run_id] = [first, second]

        try:
            async with TestClient(TestServer(app)) as cli:
                response = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "deny", "request_id": "req-second"},
                )
                data = await response.json()

                assert response.status == 200
                assert data["request_id"] == "req-second"
                assert second.result == "deny"
                assert second.event.is_set()
                assert first.result is None
                assert not first.event.is_set()
                assert adapter._run_statuses[run_id]["status"] == "waiting_for_approval"
                responded_event = await adapter._run_streams[run_id].get()
                assert responded_event["event"] == "approval.responded"
                assert responded_event["request_id"] == "req-second"

                stale = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once", "request_id": "req-second"},
                )
                stale_data = await stale.json()
                assert stale.status == 409
                assert stale_data["error"]["code"] == "approval_request_not_pending"
                assert first.result is None
                assert not first.event.is_set()
        finally:
            with approval_mod._lock:
                approval_mod._gateway_queues.pop(run_id, None)
            adapter._run_streams.pop(run_id, None)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("fanout_field", ["all", "resolve_all"])
    async def test_exact_approval_rejects_resolve_all(self, adapter, fanout_field):
        app = _create_runs_app(adapter)
        run_id = "run_exact_no_fanout"
        adapter._run_statuses[run_id] = {"run_id": run_id, "status": "waiting_for_approval"}
        adapter._run_approval_sessions[run_id] = run_id
        pending = approval_mod._ApprovalEntry({"request_id": "req-only"})
        with approval_mod._lock:
            approval_mod._gateway_queues[run_id] = [pending]

        try:
            async with TestClient(TestServer(app)) as cli:
                response = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={
                        "choice": "once",
                        "request_id": "req-only",
                        fanout_field: True,
                    },
                )
                data = await response.json()

            assert response.status == 400
            assert data["error"]["code"] == "invalid_approval_scope"
            assert pending.result is None
            assert not pending.event.is_set()
        finally:
            with approval_mod._lock:
                approval_mod._gateway_queues.pop(run_id, None)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("request_id", ["", 123, "bad/request", "x" * 129])
    async def test_exact_approval_rejects_invalid_request_id(self, adapter, request_id):
        app = _create_runs_app(adapter)
        run_id = "run_invalid_exact_id"
        adapter._run_statuses[run_id] = {"run_id": run_id, "status": "waiting_for_approval"}
        adapter._run_approval_sessions[run_id] = run_id

        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                f"/v1/runs/{run_id}/approval",
                json={"choice": "deny", "request_id": request_id},
            )
            data = await response.json()

        assert response.status == 400
        assert data["error"]["code"] == "invalid_approval_request_id"

    @pytest.mark.asyncio
    async def test_clarification_event_waits_for_exact_choice_response(self, adapter):
        app = _create_runs_app(adapter)
        callback_result = {}

        def make_agent(**kwargs):
            mock_agent = MagicMock()

            def run_conversation(**_run_kwargs):
                callback_result["answer"] = kwargs["clarify_callback"](
                    "Pick a color OPENAI_API_KEY=sk-secret-12345678901234567890",
                    ["Red", "Blue"],
                )
                return {"final_response": callback_result["answer"]}

            mock_agent.run_conversation.side_effect = run_conversation
            mock_agent.session_prompt_tokens = 0
            mock_agent.session_completion_tokens = 0
            mock_agent.session_total_tokens = 0
            return mock_agent

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", side_effect=make_agent) as create:
                started = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await started.json())["run_id"]
                event = await asyncio.wait_for(adapter._run_streams[run_id].get(), timeout=3)

                assert event["event"] == "clarify.request"
                assert event["run_id"] == run_id
                assert event["request_id"].startswith("clarify_")
                assert event["prompt"] == {
                    "version": 1,
                    "type": "choice",
                    "question": "Pick a color OPENAI_API_KEY=***",
                    "choices": [
                        {"id": "choice-1", "label": "Red"},
                        {"id": "choice-2", "label": "Blue"},
                    ],
                }
                assert create.call_args.kwargs["enable_clarify"] is True

                response = await cli.post(
                    f"/v1/runs/{run_id}/clarification",
                    json={
                        "request_id": event["request_id"],
                        "response": {"type": "choice", "choice_id": "choice-2"},
                    },
                )
                assert response.status == 200
                payload = await response.json()
                assert payload["request_id"] == event["request_id"]
                assert payload["choice_id"] == "choice-2"

                for _ in range(40):
                    status = adapter._run_statuses[run_id]
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

                assert callback_result["answer"] == "Blue"
                assert status["status"] == "completed"
                assert run_id not in adapter._run_clarify_sessions

    @pytest.mark.asyncio
    async def test_clarification_is_bound_to_run_and_single_use(self, adapter):
        app = _create_runs_app(adapter)
        first_run = "run_first"
        second_run = "run_second"
        request_id = "clarify_" + "a" * 32
        for run_id in (first_run, second_run):
            adapter._run_statuses[run_id] = {"run_id": run_id, "status": "running"}
            adapter._run_clarify_sessions[run_id] = run_id
        clarify_mod.register(request_id, second_run, "Question?", ["One", "Two"])

        async with TestClient(TestServer(app)) as cli:
            cross_run = await cli.post(
                f"/v1/runs/{first_run}/clarification",
                json={
                    "request_id": request_id,
                    "response": {"type": "choice", "choice_id": "choice-1"},
                },
            )
            assert cross_run.status == 409

            exact = await cli.post(
                f"/v1/runs/{second_run}/clarification",
                json={
                    "request_id": request_id,
                    "response": {"type": "choice", "choice_id": "choice-2"},
                },
            )
            assert exact.status == 200
            assert clarify_mod.wait_for_response(request_id, timeout=0.1) == "Two"

            duplicate = await cli.post(
                f"/v1/runs/{second_run}/clarification",
                json={
                    "request_id": request_id,
                    "response": {"type": "choice", "choice_id": "choice-2"},
                },
            )
            assert duplicate.status == 409

    @pytest.mark.asyncio
    async def test_clarification_rejects_malformed_and_oversized_text(self, adapter):
        app = _create_runs_app(adapter)
        run_id = "run_validation"
        request_id = "clarify_" + "b" * 32
        adapter._run_statuses[run_id] = {"run_id": run_id, "status": "running"}
        adapter._run_clarify_sessions[run_id] = run_id
        clarify_mod.register(request_id, run_id, "Question?", None)

        async with TestClient(TestServer(app)) as cli:
            malformed = await cli.post(
                f"/v1/runs/{run_id}/clarification",
                json={"request_id": "../not-safe", "response": {"type": "text", "text": "ok"}},
            )
            assert malformed.status == 400

            too_long = await cli.post(
                f"/v1/runs/{run_id}/clarification",
                json={
                    "request_id": request_id,
                    "response": {"type": "text", "text": "x" * 2001},
                },
            )
            assert too_long.status == 400
            assert clarify_mod.get_pending_by_id(request_id, session_key=run_id) is not None

            accepted = await cli.post(
                f"/v1/runs/{run_id}/clarification",
                json={
                    "request_id": request_id,
                    "response": {"type": "text", "text": "A bounded answer"},
                },
            )
            assert accepted.status == 200
            assert clarify_mod.wait_for_response(request_id, timeout=0.1) == "A bounded answer"

    @pytest.mark.asyncio
    async def test_clarification_requires_api_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        run_id = "run_auth"
        request_id = "clarify_" + "c" * 32
        auth_adapter._run_statuses[run_id] = {"run_id": run_id, "status": "running"}
        auth_adapter._run_clarify_sessions[run_id] = run_id
        clarify_mod.register(request_id, run_id, "Question?", None)

        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                f"/v1/runs/{run_id}/clarification",
                json={
                    "request_id": request_id,
                    "response": {"type": "text", "text": "answer"},
                },
            )
            assert response.status == 401
            assert clarify_mod.get_pending_by_id(request_id, session_key=run_id) is not None

        clarify_mod.clear_session(run_id)

    @pytest.mark.asyncio
    async def test_approval_resolve_all_is_scoped_to_target_run(self, auth_adapter):
        """Same client session_id must not let one run approve another run's queue."""
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_create_agent") as mock_create:
                victim_agent, victim_ready, victim_interrupted = _make_slow_agent()
                attacker_agent, attacker_ready, attacker_interrupted = _make_slow_agent()
                mock_create.side_effect = [victim_agent, attacker_agent]

                victim_resp = await cli.post(
                    "/v1/runs",
                    json={"input": "victim", "session_id": "shared-project"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                attacker_resp = await cli.post(
                    "/v1/runs",
                    json={"input": "attacker", "session_id": "shared-project"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                assert victim_resp.status == 202
                assert attacker_resp.status == 202
                victim_run = (await victim_resp.json())["run_id"]
                attacker_run = (await attacker_resp.json())["run_id"]

                victim_ready.wait(timeout=3.0)
                attacker_ready.wait(timeout=3.0)
                assert auth_adapter._run_approval_sessions[victim_run] == victim_run
                assert auth_adapter._run_approval_sessions[attacker_run] == attacker_run
                assert auth_adapter._run_approval_sessions[victim_run] != auth_adapter._run_approval_sessions[attacker_run]

                victim_entry = approval_mod._ApprovalEntry({
                    "command": "bash -c victim-danger",
                    "description": "victim approval",
                    "pattern_keys": ["shell-c"],
                })
                attacker_entry = approval_mod._ApprovalEntry({
                    "command": "bash -c attacker-danger",
                    "description": "attacker approval",
                    "pattern_keys": ["shell-c"],
                })
                with approval_mod._lock:
                    approval_mod._gateway_queues[victim_run] = [victim_entry]
                    approval_mod._gateway_queues[attacker_run] = [attacker_entry]

                approval_resp = await cli.post(
                    f"/v1/runs/{attacker_run}/approval",
                    json={"choice": "always", "resolve_all": True},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                approval_data = await approval_resp.json()

                assert approval_resp.status == 200
                assert approval_data["resolved"] == 1
                assert attacker_entry.result == "always"
                assert attacker_entry.event.is_set()
                assert victim_entry.result is None
                assert not victim_entry.event.is_set()
                legacy_event = next(
                    event
                    for event in auth_adapter._run_event_logs[attacker_run]
                    if event["event"] == "approval.responded"
                )
                assert "request_id" not in legacy_event
                assert legacy_event["resolved"] == 1
                with approval_mod._lock:
                    assert approval_mod._gateway_queues[victim_run] == [victim_entry]
                    assert victim_run in approval_mod._gateway_queues
                    assert attacker_run not in approval_mod._gateway_queues

                # Clean up the synthetic pending victim approval and unblock the
                # slow test agents so their background run tasks can finish.
                with approval_mod._lock:
                    approval_mod._gateway_queues.pop(victim_run, None)
                victim_interrupted.set()
                attacker_interrupted.set()

    @pytest.mark.asyncio
    async def test_approval_response_keeps_next_registered_action_status_coherent(self, adapter):
        run_id = "run_back_to_back_approval"
        first_id = "approval_first"
        next_id = "approval_next"
        first_action = {
            "version": 1,
            "kind": "approval",
            "request_id": first_id,
            "preview": {
                "version": 1,
                "category": "protected_action",
                "title": "Protected action approval",
                "summary": "Hermes stopped an action that requires explicit approval.",
                "risk_labels": [],
            },
            "choices": ["once", "deny"],
        }
        next_action = {
            **first_action,
            "request_id": next_id,
        }
        adapter._run_streams[run_id] = asyncio.Queue(maxsize=_RUN_EVENT_RETENTION)
        adapter._run_event_logs[run_id] = deque(maxlen=_RUN_EVENT_RETENTION)
        adapter._run_event_sequences[run_id] = 0
        adapter._run_subscriber_queues[run_id] = set()
        adapter._run_approval_sessions[run_id] = run_id
        adapter._run_pending_approvals[run_id] = {first_id: first_action}
        adapter._set_run_status(
            run_id,
            "waiting_for_approval",
            pending_action=first_action,
        )

        def resolve_and_register_next(*_args, **_kwargs):
            adapter._run_pending_approvals[run_id][next_id] = next_action
            return 1

        app = _create_runs_app(adapter)
        with patch.object(
            approval_mod,
            "resolve_gateway_approval",
            side_effect=resolve_and_register_next,
        ):
            async with TestClient(TestServer(app)) as cli:
                response = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"request_id": first_id, "choice": "once"},
                )

        assert response.status == 200
        status = adapter._run_statuses[run_id]
        assert status["status"] == "waiting_for_approval"
        assert status["pending_action"]["request_id"] == next_id
        assert first_id not in adapter._run_pending_approvals[run_id]

    def test_resolved_approval_is_not_mirrored_by_delayed_loop_callback(self, adapter):
        run_id = "run_delayed_resolved_approval"
        request_id = "approval_already_resolved"
        adapter._run_streams[run_id] = asyncio.Queue(maxsize=_RUN_EVENT_RETENTION)
        adapter._run_event_logs[run_id] = deque(maxlen=_RUN_EVENT_RETENTION)
        adapter._run_event_sequences[run_id] = 0
        adapter._run_subscriber_queues[run_id] = set()
        adapter._set_run_status(run_id, "running")

        published = adapter._register_run_approval_event(
            run_id,
            run_id,
            {
                "request_id": request_id,
                "pattern_keys": ["shell-c"],
            },
        )

        assert published is None
        assert adapter._run_statuses[run_id]["status"] == "running"
        assert adapter._run_pending_approvals.get(run_id) in (None, {})
        assert list(adapter._run_event_logs[run_id]) == []


    @pytest.mark.asyncio
    async def test_events_not_found_returns_404(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/run_nonexistent/events")
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_events_requires_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/run_any/events")
        assert resp.status == 401


# ---------------------------------------------------------------------------
# Run lifecycle TTL sweeping
# ---------------------------------------------------------------------------


class TestRunLifecycleSweep:
    def test_sweep_keeps_transport_with_active_subscriber(self, adapter):
        run_id = "run_subscribed"
        queue = asyncio.Queue()
        adapter._run_streams[run_id] = queue
        adapter._run_streams_created[run_id] = 0
        adapter._run_stream_subscribers.add(run_id)

        adapter._sweep_orphaned_runs_once(time.time())

        assert adapter._run_streams[run_id] is queue
        assert run_id in adapter._run_streams_created

    @pytest.mark.asyncio
    async def test_live_run_keeps_bounded_replay_transport_past_idle_ttl(self, adapter):
        """A waiting live run keeps its bounded journal and control state."""
        app = _create_runs_app(adapter)
        adapter._max_concurrent_runs = 1

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                start_resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert start_resp.status == 202
                run_id = (await start_resp.json())["run_id"]
                assert agent_ready.wait(timeout=3.0)

                task = adapter._active_run_tasks[run_id]
                assert isinstance(task, asyncio.Task)
                assert not task.done()

                pending = approval_mod._ApprovalEntry({
                    "command": "bash -c long-running",
                    "description": "approval after stream TTL",
                    "pattern_keys": ["shell-c"],
                })
                with approval_mod._lock:
                    approval_mod._gateway_queues[run_id] = [pending]

                adapter._run_streams_created[run_id] -= adapter._RUN_STREAM_TTL + 1
                # Exercise one real sweeper iteration without waiting 60 seconds.
                with patch(
                    "gateway.platforms.api_server.asyncio.sleep",
                    side_effect=[None, asyncio.CancelledError()],
                ):
                    with pytest.raises(asyncio.CancelledError):
                        await adapter._sweep_orphaned_runs()

                assert adapter._active_run_tasks[run_id] is task
                assert adapter._active_run_agents[run_id] is mock_agent
                assert run_id in adapter._run_streams
                assert run_id in adapter._run_streams_created
                assert run_id in adapter._run_event_logs
                assert adapter._run_approval_sessions[run_id] == run_id

                limited = adapter._concurrency_limited_response()
                assert limited is not None
                assert limited.status == 429

                approval_resp = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once"},
                )
                assert approval_resp.status == 200
                assert pending.event.is_set()
                assert pending.result == "once"

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                mock_agent.interrupt.assert_called_once_with("Stop requested via API")

    @pytest.mark.asyncio
    async def test_live_replay_transport_remains_bounded_after_idle_ttl(self, adapter):
        """Long-lived disconnected runs retain only the fixed replay window."""
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                start_resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await start_resp.json())["run_id"]
                assert agent_ready.wait(timeout=3.0)
                replay_queue = adapter._run_streams[run_id]
                stream_delta = mock_create.call_args.kwargs["stream_delta_callback"]

                adapter._run_streams_created[run_id] -= adapter._RUN_STREAM_TTL + 1
                adapter._sweep_orphaned_runs_once(time.time())
                assert run_id in adapter._run_streams
                for index in range(_RUN_EVENT_RETENTION + 25):
                    stream_delta(f"bounded-{index}")
                await asyncio.sleep(0)
                assert replay_queue.qsize() <= _RUN_EVENT_RETENTION
                assert 0 < len(adapter._run_event_logs[run_id]) <= _RUN_EVENT_RETENTION
                assert adapter._run_event_log_bytes[run_id] <= _RUN_EVENT_JOURNAL_MAX_BYTES
                mock_agent.interrupt("finish test")
                for _ in range(40):
                    if run_id not in adapter._active_run_tasks:
                        break
                    await asyncio.sleep(0.05)

                assert run_id in adapter._run_streams

    @pytest.mark.asyncio
    async def test_expired_orphan_run_state_is_reaped(self, adapter):
        run_id = "run_expired_orphan"
        adapter._run_streams[run_id] = asyncio.Queue()
        adapter._run_streams_created[run_id] = 0
        adapter._run_approval_sessions[run_id] = run_id

        pending = approval_mod._ApprovalEntry({
            "command": "bash -c orphaned",
            "description": "orphaned approval",
            "pattern_keys": ["shell-c"],
        })
        with approval_mod._lock:
            approval_mod._gateway_queues[run_id] = [pending]

        with patch(
            "gateway.platforms.api_server.asyncio.sleep",
            side_effect=[None, asyncio.CancelledError()],
        ):
            with pytest.raises(asyncio.CancelledError):
                await adapter._sweep_orphaned_runs()

        assert run_id not in adapter._run_streams
        assert run_id not in adapter._run_streams_created
        assert run_id not in adapter._run_approval_sessions
        assert pending.event.is_set()
        with approval_mod._lock:
            assert run_id not in approval_mod._gateway_queues


# ---------------------------------------------------------------------------
# POST /v1/runs/{run_id}/stop — interrupt a running agent
# ---------------------------------------------------------------------------


class TestStopRun:
    @pytest.mark.asyncio
    async def test_stop_releases_pending_clarification(self, adapter):
        app = _create_runs_app(adapter)

        def make_agent(**kwargs):
            mock_agent = MagicMock()
            mock_agent.run_conversation.side_effect = lambda **_run_kwargs: {
                "final_response": kwargs["clarify_callback"](
                    "Q" * 2500,
                    ["X" * 600, "Y" * 600, "Z" * 600, "W" * 600, "extra"],
                )
            }
            mock_agent.session_prompt_tokens = 0
            mock_agent.session_completion_tokens = 0
            mock_agent.session_total_tokens = 0
            return mock_agent

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", side_effect=make_agent):
                started = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await started.json())["run_id"]
                event = await asyncio.wait_for(
                    adapter._run_streams[run_id].get(), timeout=3
                )
                assert event["event"] == "clarify.request"
                assert len(event["prompt"]["question"]) == 2000
                assert len(event["prompt"]["choices"]) == 4
                assert all(
                    len(choice["label"]) == 500
                    for choice in event["prompt"]["choices"]
                )

                stopped = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stopped.status == 200
                assert clarify_mod.get_pending_by_id(
                    event["request_id"], session_key=run_id
                ) is None

                for _ in range(40):
                    if run_id not in adapter._active_run_tasks:
                        break
                    await asyncio.sleep(0.05)

                assert adapter._run_statuses[run_id]["status"] == "cancelled"
                assert run_id not in adapter._run_clarify_sessions

    @pytest.mark.asyncio
    async def test_stop_before_agent_creation_prevents_run_start(self, adapter):
        """A stop accepted while queued must prevent agent construction."""
        app = _create_runs_app(adapter)
        original_create_task = asyncio.create_task
        task_started = asyncio.Event()
        allow_task = asyncio.Event()

        def _delayed_create_task(coro):
            async def _delayed():
                task_started.set()
                await allow_task.wait()
                return await coro

            return original_create_task(_delayed())

        with patch("gateway.platforms.api_server.asyncio.create_task", side_effect=_delayed_create_task), \
             patch.object(adapter, "_create_agent") as mock_create:
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await resp.json())["run_id"]
                await task_started.wait()

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                allow_task.set()

                for _ in range(20):
                    if run_id not in adapter._active_run_tasks:
                        break
                    await asyncio.sleep(0.05)

                mock_create.assert_not_called()
                assert adapter._run_statuses[run_id]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_stop_keeps_uncooperative_executor_tracked_until_exit(self, adapter):
        """Cancelling an asyncio wrapper must not hide its live executor thread."""
        app = _create_runs_app(adapter)
        run_can_finish = threading.Event()
        run_finished = threading.Event()

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                started = threading.Event()

                def _run_conversation(*_args, **_kwargs):
                    started.set()
                    run_can_finish.wait(timeout=5)
                    run_finished.set()
                    return {"final_response": "late result"}

                mock_agent.run_conversation.side_effect = _run_conversation
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await resp.json())["run_id"]
                assert started.wait(timeout=3)

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                await asyncio.sleep(0.1)

                assert not run_finished.is_set()
                assert run_id in adapter._active_run_agents
                assert run_id in adapter._active_run_tasks
                assert adapter._run_statuses[run_id]["status"] == "stopping"

                run_can_finish.set()
                for _ in range(40):
                    if run_id not in adapter._active_run_tasks:
                        break
                    await asyncio.sleep(0.05)

                assert run_id not in adapter._active_run_agents
                assert run_id not in adapter._active_run_tasks
                assert adapter._run_statuses[run_id]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_stop_running_agent(self, adapter):
        """Stop should interrupt the agent and cancel the task."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                # Wait for agent to start running in the thread
                agent_ready.wait(timeout=3.0)
                await asyncio.sleep(0.1)

                # Verify agent ref is stored
                assert run_id in adapter._active_run_agents

                # Stop the run
                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                stop_data = await stop_resp.json()
                assert stop_data["run_id"] == run_id
                assert stop_data["status"] == "stopping"

                # Agent interrupt should have been called
                mock_agent.interrupt.assert_called_once_with("Stop requested via API")

                status_resp = await cli.get(f"/v1/runs/{run_id}")
                assert status_resp.status == 200
                status_data = await status_resp.json()
                assert status_data["status"] in {"stopping", "cancelled"}

                # Refs should be cleaned up
                await asyncio.sleep(0.5)
                assert run_id not in adapter._active_run_agents
                assert run_id not in adapter._active_run_tasks

    @pytest.mark.asyncio
    async def test_stop_nonexistent_run_returns_404(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_nonexistent/stop")
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_stop_requires_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_any/stop")
        assert resp.status == 401

    @pytest.mark.asyncio
    async def test_stop_already_completed_run_returns_404(self, adapter):
        """Stopping a run that already finished should return 404 (refs cleaned up)."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                # Start and wait for completion
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                await asyncio.sleep(0.3)

                # Run should be done, refs cleaned up
                assert run_id not in adapter._active_run_agents

                # Stop should return 404
                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 404

    @pytest.mark.asyncio
    async def test_stop_interrupt_exception_does_not_crash(self, adapter):
        """If agent.interrupt() raises, stop should still succeed."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, interrupted = _make_slow_agent()

                # Override the interrupt side_effect to raise. Still trip
                # ``interrupted`` so the slow_run thread unblocks at teardown
                # — without this the agent thread blocks the full 10s
                # timeout and the test teardown waits the same amount.
                def _raising_interrupt(message=None):
                    interrupted.set()
                    raise RuntimeError("interrupt failed")

                mock_agent.interrupt = MagicMock(side_effect=_raising_interrupt)
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                agent_ready.wait(timeout=3.0)
                await asyncio.sleep(0.1)

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                stop_data = await stop_resp.json()
                assert stop_data["status"] == "stopping"

    @pytest.mark.asyncio
    async def test_stop_sends_sentinel_to_events_stream(self, adapter):
        """After stop, the events stream should close."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                agent_ready.wait(timeout=3.0)
                await asyncio.sleep(0.1)

                # Subscribe to events in background
                events_task = asyncio.ensure_future(
                    cli.get(f"/v1/runs/{run_id}/events")
                )

                await asyncio.sleep(0.1)

                # Stop the run
                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200

                # Events stream should close
                events_resp = await asyncio.wait_for(events_task, timeout=5.0)
                assert events_resp.status == 200
                body = await events_resp.text()
                # Stream should have received run.failed and closed
                assert "run.failed" in body or "stream closed" in body
