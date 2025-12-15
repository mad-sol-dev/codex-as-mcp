import asyncio
import json
from pathlib import Path

import pytest

from codex_as_mcp import server
from tests.conftest import DummyContext, FakeProcess


@pytest.mark.asyncio
async def test_spawn_agent_writes_persistent_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(server.LOG_ROOT_ENV_VAR, str(tmp_path))
    monkeypatch.setattr(server, "LOG_ROTATE_BYTES", 1000)
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

    log_files = [log_path] + [
        log_path.with_name(log_path.name + f".{i}") for i in range(1, server.LOG_ROTATE_COUNT + 1)
    ]
    entries = []
    for lf in log_files:
        if lf.exists():
            for line in lf.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entries.append(json.loads(line))

    assert any(entry.get("event") == "stdout" for entry in entries)
    assert any(entry.get("event") == "stderr" for entry in entries)
    assert all("level" in entry for entry in entries)

    assert not list(tmp_path.glob("*.last_message.md"))


@pytest.mark.asyncio
async def test_spawn_agents_parallel_respects_max_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    concurrency_state = {"active": 0, "max": 0}
    lock = asyncio.Lock()

    async def fake_spawn_agent(ctx: server.Context, prompt: str, **_: object) -> str:  # type: ignore[override]
        async with lock:
            concurrency_state["active"] += 1
            concurrency_state["max"] = max(concurrency_state["max"], concurrency_state["active"])

        await asyncio.sleep(0.05)

        async with lock:
            concurrency_state["active"] -= 1

        return f"ok:{prompt}"

    monkeypatch.setattr(server, "spawn_agent", fake_spawn_agent)

    ctx = DummyContext()
    agent_specs = [{"prompt": f"task {i}"} for i in range(5)]

    results = await server.spawn_agents_parallel(ctx, agent_specs, max_parallel=2)

    assert concurrency_state["max"] <= 2
    assert sorted(result["index"] for result in results) == [str(i) for i in range(len(agent_specs))]
    assert all(result.get("output", "").startswith("ok:") for result in results)
