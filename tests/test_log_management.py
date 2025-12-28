"""Tests for log management tools (list_agent_logs, cleanup_old_logs)."""
import time
from pathlib import Path

import pytest

from codex_as_mcp import server


@pytest.mark.asyncio
async def test_list_agent_logs_empty_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test list_agent_logs with no log files."""
    monkeypatch.setenv(server.LOG_ROOT_ENV_VAR, str(tmp_path))

    result = await server.list_agent_logs(max_count=20)

    assert result["total"] == 0
    assert result["showing"] == 0
    assert result["log_dir"] == str(tmp_path)
    assert result["logs"] == []


@pytest.mark.asyncio
async def test_list_agent_logs_with_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test list_agent_logs returns log files sorted by modification time."""
    monkeypatch.setenv(server.LOG_ROOT_ENV_VAR, str(tmp_path))

    # Create some fake log files with different timestamps
    log1 = tmp_path / "codex_agent_20251228_100000_123_abc.log"
    log2 = tmp_path / "codex_agent_20251228_110000_456_def.log"
    log3 = tmp_path / "codex_agent_20251228_120000_789_ghi.log"

    log1.write_text("log 1 content" * 100)
    time.sleep(0.01)
    log2.write_text("log 2 content" * 200)
    time.sleep(0.01)
    log3.write_text("log 3 content" * 50)

    result = await server.list_agent_logs(max_count=20)

    assert result["total"] == 3
    assert result["showing"] == 3
    assert result["log_dir"] == str(tmp_path)
    assert len(result["logs"]) == 3

    # Should be sorted by modification time (most recent first)
    # log3 was created last, so it should be first
    logs = result["logs"]
    assert Path(logs[0]["path"]).name == log3.name
    assert Path(logs[1]["path"]).name == log2.name
    assert Path(logs[2]["path"]).name == log1.name

    # Check log info structure
    for log_info in logs:
        assert "path" in log_info
        assert "size_bytes" in log_info
        assert "size_human" in log_info
        assert "modified" in log_info
        assert log_info["size_bytes"] > 0
        assert "KB" in log_info["size_human"] or "MB" in log_info["size_human"]


@pytest.mark.asyncio
async def test_list_agent_logs_respects_max_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that list_agent_logs respects max_count parameter."""
    monkeypatch.setenv(server.LOG_ROOT_ENV_VAR, str(tmp_path))

    # Create 5 log files
    for i in range(5):
        log_file = tmp_path / f"codex_agent_2025122{i}_100000_123_abc{i}.log"
        log_file.write_text(f"log {i}")
        time.sleep(0.01)

    result = await server.list_agent_logs(max_count=3)

    assert result["total"] == 5
    assert result["showing"] == 3  # Should only show 3
    assert len(result["logs"]) == 3


@pytest.mark.asyncio
async def test_list_agent_logs_ignores_non_log_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that list_agent_logs only lists .log files."""
    monkeypatch.setenv(server.LOG_ROOT_ENV_VAR, str(tmp_path))

    # Create log files and other files
    (tmp_path / "codex_agent_20251228_100000_123_abc.log").write_text("log")
    (tmp_path / "codex_agent_20251228_100000_123_abc.log.1").write_text("rotated")  # Should be ignored
    (tmp_path / "other_file.txt").write_text("not a log")
    (tmp_path / "README.md").write_text("docs")

    result = await server.list_agent_logs(max_count=20)

    # Should only count the main .log file, not rotated (.log.1) or other files
    assert result["total"] == 1
    assert result["showing"] == 1
    assert len(result["logs"]) == 1


@pytest.mark.asyncio
async def test_cleanup_old_logs_dry_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test cleanup_old_logs in dry_run mode (default)."""
    monkeypatch.setenv(server.LOG_ROOT_ENV_VAR, str(tmp_path))

    # Create old and new log files
    old_log = tmp_path / "codex_agent_20200101_100000_123_old.log"
    new_log = tmp_path / "codex_agent_20251228_100000_123_new.log"

    old_log.write_text("old log content" * 100)
    new_log.write_text("new log content" * 100)

    # Set old_log's modification time to 10 days ago
    old_mtime = time.time() - (10 * 24 * 60 * 60)
    old_log.touch()
    import os
    os.utime(old_log, (old_mtime, old_mtime))

    result = await server.cleanup_old_logs(days=7, dry_run=True)

    # Should report what would be deleted
    assert result["dry_run"] is True
    assert result["deleted_count"] == 1
    assert result["freed_bytes"] > 0
    assert str(old_log) in result["files"]
    assert str(new_log) not in result["files"]

    # Files should still exist (dry_run)
    assert old_log.exists()
    assert new_log.exists()


@pytest.mark.asyncio
async def test_cleanup_old_logs_actual_deletion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test cleanup_old_logs actually deletes files when dry_run=False."""
    monkeypatch.setenv(server.LOG_ROOT_ENV_VAR, str(tmp_path))

    # Create old and new log files
    old_log = tmp_path / "codex_agent_20200101_100000_123_old.log"
    new_log = tmp_path / "codex_agent_20251228_100000_123_new.log"

    old_log.write_text("old log content" * 100)
    new_log.write_text("new log content" * 100)

    # Set old_log's modification time to 10 days ago
    old_mtime = time.time() - (10 * 24 * 60 * 60)
    import os
    os.utime(old_log, (old_mtime, old_mtime))

    result = await server.cleanup_old_logs(days=7, dry_run=False)

    # Should actually delete
    assert result["dry_run"] is False
    assert result["deleted_count"] == 1
    assert result["freed_bytes"] > 0

    # Old file should be deleted, new file should remain
    assert not old_log.exists()
    assert new_log.exists()


@pytest.mark.asyncio
async def test_cleanup_old_logs_includes_rotated_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that cleanup includes rotated log files (.log.1, .log.2, etc)."""
    monkeypatch.setenv(server.LOG_ROOT_ENV_VAR, str(tmp_path))

    # Create old log and rotated files
    old_log = tmp_path / "codex_agent_20200101_100000_123_old.log"
    old_log_rotated1 = tmp_path / "codex_agent_20200101_100000_123_old.log.1"
    old_log_rotated2 = tmp_path / "codex_agent_20200101_100000_123_old.log.2"

    for log_file in [old_log, old_log_rotated1, old_log_rotated2]:
        log_file.write_text("old content")

    # Set modification time to 10 days ago
    old_mtime = time.time() - (10 * 24 * 60 * 60)
    import os
    for log_file in [old_log, old_log_rotated1, old_log_rotated2]:
        os.utime(log_file, (old_mtime, old_mtime))

    result = await server.cleanup_old_logs(days=7, dry_run=False)

    # Should delete all 3 files
    assert result["deleted_count"] == 3
    assert not old_log.exists()
    assert not old_log_rotated1.exists()
    assert not old_log_rotated2.exists()


@pytest.mark.asyncio
async def test_cleanup_old_logs_validates_days_parameter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that cleanup_old_logs validates days parameter."""
    monkeypatch.setenv(server.LOG_ROOT_ENV_VAR, str(tmp_path))

    # Should raise error for invalid days
    with pytest.raises(server.ToolExecutionError) as exc_info:
        await server.cleanup_old_logs(days=0, dry_run=True)

    assert exc_info.value.payload.code == "invalid_days"
    assert exc_info.value.payload.kind == server.ErrorCategory.VALIDATION


@pytest.mark.asyncio
async def test_cleanup_old_logs_handles_no_old_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test cleanup_old_logs when there are no old files to delete."""
    monkeypatch.setenv(server.LOG_ROOT_ENV_VAR, str(tmp_path))

    # Create only new log files
    new_log = tmp_path / "codex_agent_20251228_100000_123_new.log"
    new_log.write_text("new log content")

    result = await server.cleanup_old_logs(days=7, dry_run=True)

    assert result["deleted_count"] == 0
    assert result["freed_bytes"] == 0
    assert result["files"] == []
    assert new_log.exists()


@pytest.mark.asyncio
async def test_cleanup_old_logs_human_readable_sizes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that cleanup_old_logs provides human-readable size info."""
    monkeypatch.setenv(server.LOG_ROOT_ENV_VAR, str(tmp_path))

    # Create old log with known size
    old_log = tmp_path / "codex_agent_20200101_100000_123_old.log"
    old_log.write_text("x" * 2000)  # 2000 bytes

    old_mtime = time.time() - (10 * 24 * 60 * 60)
    import os
    os.utime(old_log, (old_mtime, old_mtime))

    result = await server.cleanup_old_logs(days=7, dry_run=True)

    assert "freed_human" in result
    # 2000 bytes should show as KB
    assert "KB" in result["freed_human"]
    assert result["freed_bytes"] >= 2000
