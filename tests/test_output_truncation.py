"""Tests for output truncation and async improvements."""
import asyncio
import time
from pathlib import Path

import pytest

from codex_as_mcp import server
from tests.conftest import DummyContext, FakeProcess


@pytest.mark.asyncio
async def test_output_truncation_when_exceeds_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that output is truncated when it exceeds MAX_OUTPUT_LENGTH."""
    monkeypatch.setenv(server.LOG_ROOT_ENV_VAR, str(tmp_path))
    monkeypatch.setattr(server, "_resolve_codex_executable", lambda: "codex")

    # Create a long output that exceeds MAX_OUTPUT_LENGTH (4000 chars)
    long_output = "x" * 5000  # 5000 characters

    # Track which output file is created and write to it
    output_file_path = None

    async def fake_create_subprocess_exec(*args: object, **kwargs: object) -> FakeProcess:
        nonlocal output_file_path
        # Extract output path from command args
        cmd_args = list(args)
        for i, arg in enumerate(cmd_args):
            if arg == "--output-last-message" and i + 1 < len(cmd_args):
                output_file_path = Path(cmd_args[i + 1])
                output_file_path.write_text(long_output)
                break
        return FakeProcess([b"stdout line\n"], [])

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    ctx = DummyContext()
    result = await server.spawn_agent(ctx, "test truncation")

    # Result should be truncated
    assert len(result) < len(long_output) + 200  # Allow some overhead for log note
    assert "[Output truncated:" in result
    assert "5000 chars total" in result
    assert "showing first 4000" in result
    assert "Log file:" in result

    # Verify log file contains truncation event
    log_path = Path(result.split("Log file: ", 1)[1].strip())
    log_content = log_path.read_text()
    assert "output_truncated" in log_content


@pytest.mark.asyncio
async def test_output_not_truncated_when_below_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that output is NOT truncated when below MAX_OUTPUT_LENGTH."""
    monkeypatch.setenv(server.LOG_ROOT_ENV_VAR, str(tmp_path))
    monkeypatch.setattr(server, "_resolve_codex_executable", lambda: "codex")

    # Create output below limit
    short_output = "This is a short output message."

    async def fake_create_subprocess_exec(*args: object, **kwargs: object) -> FakeProcess:
        cmd_args = list(args)
        for i, arg in enumerate(cmd_args):
            if arg == "--output-last-message" and i + 1 < len(cmd_args):
                Path(cmd_args[i + 1]).write_text(short_output)
                break
        return FakeProcess([b"stdout line\n"], [])

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    ctx = DummyContext()
    result = await server.spawn_agent(ctx, "test no truncation")

    # Result should contain full output
    assert short_output in result
    assert "[Output truncated:" not in result
    assert "Log file:" in result


@pytest.mark.asyncio
async def test_fallback_output_truncation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that fallback output (recent_lines) is also truncated."""
    monkeypatch.setenv(server.LOG_ROOT_ENV_VAR, str(tmp_path))
    monkeypatch.setattr(server, "_resolve_codex_executable", lambda: "codex")

    # Create very long stdout that will be used as fallback
    long_lines = [f"Line {i} with some padding text\n".encode() for i in range(300)]

    async def fake_create_subprocess_exec(*args: object, **kwargs: object) -> FakeProcess:
        # Write empty output file to force fallback to recent_lines
        cmd_args = list(args)
        for i, arg in enumerate(cmd_args):
            if arg == "--output-last-message" and i + 1 < len(cmd_args):
                Path(cmd_args[i + 1]).write_text("")  # Empty output file
                break
        return FakeProcess(long_lines, [])

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    ctx = DummyContext()
    result = await server.spawn_agent(ctx, "test fallback truncation")

    # Should use fallback and truncate
    assert "Warning: Codex agent completed but output file was empty." in result
    assert "Recent output:" in result
    assert "[Truncated]" in result or len(result) < 6000  # Either explicit truncation or implicit


@pytest.mark.asyncio
async def test_spawn_agent_async_has_delay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that spawn_agent_async has a delay before returning."""
    monkeypatch.setenv(server.LOG_ROOT_ENV_VAR, str(tmp_path))
    monkeypatch.setattr(server, "_resolve_codex_executable", lambda: "codex")

    # Mock _run_codex_agent to avoid actual execution
    async def fake_run_codex_agent(*args, **kwargs):  # type: ignore
        await asyncio.sleep(10)  # Simulate long task
        return "Agent completed"

    monkeypatch.setattr(server, "_run_codex_agent", fake_run_codex_agent)

    ctx = DummyContext()
    start_time = time.time()
    result = await server.spawn_agent_async(ctx, "test delay")
    elapsed = time.time() - start_time

    # Should have ~1.5s delay before returning
    assert elapsed >= 1.4  # Allow some variance
    assert elapsed < 3.0  # But not too long
    assert result["status"] == "running"
    assert "note" in result
    assert "DO NOT poll immediately" in result["note"]
    assert "wait at least 30 seconds" in result["note"]


@pytest.mark.asyncio
async def test_get_agent_status_warns_on_early_poll(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that get_agent_status warns when polling too early."""
    monkeypatch.setenv(server.LOG_ROOT_ENV_VAR, str(tmp_path))
    monkeypatch.setattr(server, "_resolve_codex_executable", lambda: "codex")

    # Mock _run_codex_agent to simulate long task
    async def fake_run_codex_agent(*args, **kwargs):  # type: ignore
        await asyncio.sleep(20)
        return "Agent completed"

    monkeypatch.setattr(server, "_run_codex_agent", fake_run_codex_agent)

    ctx = DummyContext()

    # Start async task
    start_result = await server.spawn_agent_async(ctx, "test early poll")
    task_id = start_result["task_id"]

    # Poll immediately (within 15 seconds)
    await asyncio.sleep(0.5)  # Small delay to ensure task is registered
    status = await server.get_agent_status(task_id)

    # Should have warning
    assert status["status"] == "running"
    assert "warning" in status
    assert "Most Codex tasks take 30-120 seconds" in status["warning"]
    assert "Avoid frequent polling" in status["warning"]


@pytest.mark.asyncio
async def test_get_agent_status_no_warning_after_15_seconds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that get_agent_status doesn't warn after 15 seconds."""
    monkeypatch.setenv(server.LOG_ROOT_ENV_VAR, str(tmp_path))
    monkeypatch.setattr(server, "_resolve_codex_executable", lambda: "codex")

    # Create a task that was started >15 seconds ago
    from datetime import datetime, timedelta, timezone

    task_id = "test_task_old"
    old_timestamp = (datetime.now(timezone.utc) - timedelta(seconds=20)).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    # Manually register an old running task
    server.running_tasks[task_id] = {
        "task": None,
        "status": "running",
        "output": None,
        "error": None,
        "log_file": str(tmp_path / "fake.log"),
        "started_at": old_timestamp,
        "completed_at": None,
        "elapsed_seconds": None,
    }

    status = await server.get_agent_status(task_id)

    # Should NOT have warning
    assert "warning" not in status
    assert status["status"] == "running"
    assert status["elapsed_seconds"] >= 19  # Should show ~20 seconds

    # Cleanup
    del server.running_tasks[task_id]


@pytest.mark.asyncio
async def test_get_agent_status_no_warning_when_completed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that get_agent_status doesn't warn for completed tasks."""
    from datetime import datetime, timezone

    task_id = "test_task_completed"
    timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    # Register a completed task
    server.running_tasks[task_id] = {
        "task": None,
        "status": "completed",
        "output": "Task done",
        "error": None,
        "log_file": str(tmp_path / "fake.log"),
        "started_at": timestamp,
        "completed_at": timestamp,
        "elapsed_seconds": 5.2,
    }

    status = await server.get_agent_status(task_id)

    # Should NOT have warning even though it completed quickly
    assert "warning" not in status
    assert status["status"] == "completed"
    assert status["output"] == "Task done"

    # Cleanup
    del server.running_tasks[task_id]
