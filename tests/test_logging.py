import asyncio
from pathlib import Path

import pytest

from codex_as_mcp import server


class DummyContext:
    def __init__(self) -> None:
        self.progress: list[tuple[int | None, int | None, str]] = []

    async def report_progress(self, index: int | None, total: int | None, message: str) -> None:
        self.progress.append((index, total, message))


class FakeProcess:
    def __init__(self, stdout_chunks: list[bytes], stderr_chunks: list[bytes], returncode: int = 0) -> None:
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        for chunk in stdout_chunks:
            self.stdout.feed_data(chunk)
        self.stdout.feed_eof()
        for chunk in stderr_chunks:
            self.stderr.feed_data(chunk)
        self.stderr.feed_eof()
        self._returncode = returncode

    async def wait(self) -> int:
        return self._returncode

    def terminate(self) -> None:  # pragma: no cover - used only on timeouts
        self._returncode = -1

    def kill(self) -> None:  # pragma: no cover - used only on timeouts
        self._returncode = -9


@pytest.mark.asyncio
async def test_spawn_agent_writes_persistent_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(server.LOG_ROOT_ENV_VAR, str(tmp_path))
    monkeypatch.setattr(server, "LOG_ROTATE_BYTES", 50)
    monkeypatch.setattr(server, "_resolve_codex_executable", lambda: "codex")

    stdout_chunks = [b"x" * 60 + b"\n", b"second line\n"]
    stderr_chunks = [b"warning\n"]

    async def fake_create_subprocess_exec(*_: object, **__: object) -> FakeProcess:
        return FakeProcess(stdout_chunks, stderr_chunks)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    ctx = DummyContext()
    result = await server.spawn_agent(ctx, "log something", agent_index=2, agent_count=3)

    assert "Log file: " in result
    log_path = Path(result.split("Log file: ", 1)[1].splitlines()[0])
    assert log_path.exists()
    assert log_path.parent == tmp_path

    rotated = log_path.with_name(log_path.name + ".1")
    assert rotated.exists()
    assert "[stdout]" in rotated.read_text(encoding="utf-8")
    assert "[stderr] warning" in log_path.read_text(encoding="utf-8")

    assert not list(tmp_path.glob("*.last_message.md"))
