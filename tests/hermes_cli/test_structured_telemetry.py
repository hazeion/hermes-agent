"""Tests for the private structured progress channel."""

import json
from types import SimpleNamespace

from agent.conversation_loop import (
    _provider_reasoning_present,
    _reset_turn_context_measurement,
)
import hermes_cli.structured_telemetry as telemetry
from hermes_cli.structured_telemetry import ProgressWriter


def _server_owned(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(mode=0o600)
    return path


def _events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_progress_writer_exposes_exact_tool_name_without_args_or_result(tmp_path):
    path = _server_owned(tmp_path / "progress.jsonl")
    writer = ProgressWriter(str(path), strict=True)

    writer.callback(
        "tool.started",
        "browser.search",
        "secret path /private/tmp/example",
        {"token": "never-write-me"},
    )
    writer.callback(
        "tool.completed",
        "browser.search",
        None,
        None,
        duration=1.234,
        result="private result",
    )

    events = _events(path)
    assert events == [
        {
            "schema_version": 1,
            "type": "tool.started",
            "tool": "browser.search",
            "summary": "Using browser.search",
            "sequence": 1,
        },
        {
            "schema_version": 1,
            "type": "tool.completed",
            "tool": "browser.search",
            "summary": "Finished browser.search",
            "duration_ms": 1234,
            "sequence": 2,
        },
    ]
    serialized = path.read_text()
    assert "never-write-me" not in serialized
    assert "private result" not in serialized
    assert "/private/tmp/example" not in serialized


def test_genuine_reasoning_event_is_fixed_and_never_exposes_text(tmp_path):
    path = _server_owned(tmp_path / "progress.jsonl")
    writer = ProgressWriter(str(path), strict=True)
    writer.callback(
        "reasoning.available",
        "_thinking",
        "I will inspect /Users/person/private and token=secret",
        None,
    )

    assert _events(path) == [{
        "schema_version": 1,
        "type": "reasoning.available",
        "summary": "Reasoning about the next action",
        "sequence": 1,
    }]
    assert "/Users/person/private" not in path.read_text()
    assert "token=secret" not in path.read_text()


def test_assistant_text_alone_is_not_a_reasoning_signal():
    assert not _provider_reasoning_present(
        SimpleNamespace(content="I updated the implementation", reasoning=None)
    )
    assert _provider_reasoning_present(
        SimpleNamespace(content="Final answer", reasoning="provider metadata")
    )


def test_new_turn_clears_prior_provider_context_before_any_response():
    compressor = SimpleNamespace(latest_provider_prompt_tokens=24000)
    _reset_turn_context_measurement(SimpleNamespace(context_compressor=compressor))
    assert compressor.latest_provider_prompt_tokens == 0


def test_unknown_events_and_invalid_tool_names_are_dropped(tmp_path):
    path = _server_owned(tmp_path / "progress.jsonl")
    writer = ProgressWriter(str(path), strict=True)
    writer.callback("tool.output_risk", "terminal", "ignored", None)
    writer.callback("tool.started", "../../bad tool", "ignored", None)
    assert path.read_text() == ""


def test_write_failures_never_escape():
    writer = ProgressWriter(
        "/proc/definitely/not/writable/progress.jsonl",
        strict=True,
    )
    writer.callback("tool.started", "terminal", None, None)


def test_writer_refuses_file_and_parent_symlinks(tmp_path):
    target = _server_owned(tmp_path / "target.jsonl")
    target.write_text("keep\n")
    alias = tmp_path / "progress.jsonl"
    alias.symlink_to(target)
    ProgressWriter(str(alias), strict=True).callback(
        "tool.started",
        "terminal",
        None,
        None,
    )
    assert target.read_text() == "keep\n"

    real_parent = tmp_path / "real"
    real_parent.mkdir()
    nested_target = _server_owned(real_parent / "progress.jsonl")
    parent_alias = tmp_path / "alias"
    parent_alias.symlink_to(real_parent, target_is_directory=True)
    ProgressWriter(
        str(parent_alias / "progress.jsonl"),
        strict=True,
    ).callback(
        "tool.started",
        "terminal",
        None,
        None,
    )
    assert nested_target.read_text() == ""


def test_writer_stops_at_event_bound(tmp_path):
    path = _server_owned(tmp_path / "progress.jsonl")
    writer = ProgressWriter(str(path), strict=True)
    for _index in range(250):
        writer.callback("tool.started", "terminal", None, None)
    assert len(_events(path)) == 200


def test_strict_channel_fails_closed_when_secure_dir_fd_is_unavailable(
    tmp_path,
    monkeypatch,
):
    path = _server_owned(tmp_path / "progress.jsonl")
    monkeypatch.setattr(telemetry.os, "supports_dir_fd", set())
    ProgressWriter(str(path), strict=True).callback(
        "tool.started",
        "terminal",
        None,
        None,
    )
    assert path.read_text() == ""
