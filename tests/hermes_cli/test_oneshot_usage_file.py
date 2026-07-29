"""Tests for hermes -z --usage-file (per-run JSON usage report)."""

import json
import stat

from hermes_cli.oneshot import _write_usage_file


def _server_owned(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(mode=0o600)
    return path


def _result(**overrides):
    base = {
        "estimated_cost_usd": 0.1234,
        "cost_status": "estimated",
        "cost_source": "pricing-table",
        "input_tokens": 1000,
        "output_tokens": 200,
        "cache_read_tokens": 800,
        "cache_write_tokens": 0,
        "reasoning_tokens": 50,
        "total_tokens": 1250,
        "last_prompt_tokens": 24000,
        "context_length": 128000,
        "api_calls": 3,
        "model": "openai/gpt-5.5",
        "provider": "openrouter",
        "session_id": "abc123",
        "completed": True,
        "failed": False,
    }
    base.update(overrides)
    return base


class TestWriteUsageFile:
    def test_writes_report_with_cost_and_tokens(self, tmp_path):
        path = _server_owned(tmp_path / "usage.json")
        _write_usage_file(str(path), _result())
        report = json.loads(path.read_text())
        assert report["estimated_cost_usd"] == 0.1234
        assert report["input_tokens"] == 1000
        assert report["output_tokens"] == 200
        assert report["model"] == "openai/gpt-5.5"
        assert report["api_calls"] == 3
        assert report["context_tokens"] == 24000
        assert report["context_length"] == 128000
        assert report["failed"] is False
        assert "failure" not in report

    def test_none_path_is_noop(self, tmp_path):
        # Must not raise and must not create a report file.
        _write_usage_file(None, _result())
        assert not (tmp_path / "usage.json").exists()

    def test_failure_marks_failed_and_records_message(self, tmp_path):
        path = _server_owned(tmp_path / "usage.json")
        _write_usage_file(str(path), {}, failure="boom")
        report = json.loads(path.read_text())
        assert report["failed"] is True
        assert report["failure"] == "boom"
        # Missing result fields serialize as null, not KeyError.
        assert report["estimated_cost_usd"] is None

    def test_explicit_usage_path_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "usage.json"
        _write_usage_file(str(path), _result())
        assert json.loads(path.read_text())["total_tokens"] == 1250

    def test_explicit_usage_path_remains_owner_only(self, tmp_path):
        path = _server_owned(tmp_path / "usage.json")
        _write_usage_file(str(path), _result())
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_legacy_fixed_temp_symlink_is_not_followed(self, tmp_path):
        path = tmp_path / "usage.json"
        target = tmp_path / "outside.json"
        target.write_text("keep\n")
        fixed_temp = tmp_path / ".usage.json.tmp"
        fixed_temp.symlink_to(target)
        _write_usage_file(str(path), _result())
        assert target.read_text() == "keep\n"
        assert json.loads(path.read_text())["total_tokens"] == 1250

    def test_unwritable_path_never_raises(self):
        # Root-owned path — the write must be swallowed, not raised.
        _write_usage_file("/proc/definitely/not/writable/usage.json", _result())

    def test_symlink_destination_is_refused(self, tmp_path):
        target = _server_owned(tmp_path / "target.json")
        target.write_text("keep\n")
        alias = tmp_path / "usage.json"
        alias.symlink_to(target)
        _write_usage_file(str(alias), _result())
        assert target.read_text() == "keep\n"

    def test_result_failed_flag_carries_through(self, tmp_path):
        path = _server_owned(tmp_path / "usage.json")
        _write_usage_file(str(path), _result(failed=True))
        assert json.loads(path.read_text())["failed"] is True

    def test_context_is_unavailable_when_not_positive_exact_integers(self, tmp_path):
        path = _server_owned(tmp_path / "usage.json")
        _write_usage_file(
            str(path),
            _result(last_prompt_tokens=0, context_length="128000"),
        )
        report = json.loads(path.read_text())
        assert report["context_tokens"] is None
        assert report["context_length"] is None

    def test_inconsistent_context_pair_is_unavailable(self, tmp_path):
        path = _server_owned(tmp_path / "usage.json")
        _write_usage_file(
            str(path),
            _result(last_prompt_tokens=200000, context_length=128000),
        )
        report = json.loads(path.read_text())
        assert report["context_tokens"] is None
        assert report["context_length"] is None
