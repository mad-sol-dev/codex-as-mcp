import asyncio
from pathlib import Path

import pytest

from codex_as_mcp import server
from tests.conftest import DummyContext, FakeProcess


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prompt", "expected_code"),
    [
        (None, "invalid_prompt_type"),
        ("   ", "empty_prompt"),
    ],
)
async def test_spawn_agent_validates_prompt(prompt: str | None, expected_code: str) -> None:
    ctx = DummyContext()
    with pytest.raises(server.ToolExecutionError) as err:
        await server.spawn_agent(ctx, prompt)  # type: ignore[arg-type]

    payload = err.value.payload.to_dict()
    assert payload["code"] == expected_code
    assert payload["kind"] == server.ErrorCategory.VALIDATION.value
    assert "prompt" in payload["message"]


@pytest.mark.asyncio
async def test_spawn_agent_applies_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(server.LOG_ROOT_ENV_VAR, str(tmp_path))
    monkeypatch.setenv("CODEX_REASONING_EFFORT", "low")
    monkeypatch.setenv("CODEX_MODEL", "env-model")
    monkeypatch.setattr(server, "_resolve_codex_executable", lambda: "/usr/bin/codex")

    captured: dict[str, list[str]] = {}

    async def fake_create_subprocess_exec(*cmd: object, **_: object) -> FakeProcess:
        captured["cmd"] = [str(part) for part in cmd]
        return FakeProcess([b"done\n"], [])

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    ctx = DummyContext()
    prompt = 'Do "things"'
    result = await server.spawn_agent(ctx, prompt, reasoning_effort="xhigh", model="param-model")

    assert "Log file: " in result
    cmd = captured["cmd"]
    assert "-m" in cmd and cmd[cmd.index("-m") + 1] == "param-model"
    assert "-c" in cmd and cmd[cmd.index("-c") + 1] == "model_reasoning_effort=xhigh"
    assert cmd[-1] == prompt


@pytest.mark.asyncio
async def test_spawn_agent_preserves_prompt_special_chars(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(server.LOG_ROOT_ENV_VAR, str(tmp_path))
    monkeypatch.setattr(server, "_resolve_codex_executable", lambda: "/usr/bin/codex")

    captured: dict[str, list[str]] = {}
    prompt = 'Line1 "quote"\nLine2 with $dollar and `backtick` \\slash'

    async def fake_create_subprocess_exec(*cmd: object, **_: object) -> FakeProcess:
        captured["cmd"] = [str(part) for part in cmd]
        return FakeProcess([b"done\n"], [])

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    ctx = DummyContext()
    await server.spawn_agent(ctx, prompt)

    cmd = captured["cmd"]
    assert cmd[-1] == prompt


@pytest.mark.asyncio
async def test_spawn_agents_parallel_mixed_results(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_spawn_agent(ctx: server.Context, prompt: str, **_: object) -> str:  # type: ignore[override]
        if "fail" in prompt:
            raise server.ToolExecutionError(
                code="boom",
                kind=server.ErrorCategory.RUNTIME,
                message="boom",
                log_file="/tmp/fail.log",
            )
        return f"ok:{prompt}\nLog file: /tmp/success.log"

    monkeypatch.setattr(server, "spawn_agent", fake_spawn_agent)

    ctx = DummyContext()
    agents = [{"prompt": "do 1"}, {"prompt": "fail 2"}, {"prompt": "do 3"}]

    results = await server.spawn_agents_parallel(ctx, agents, max_parallel=2)

    assert sorted(result["index"] for result in results) == ["0", "1", "2"]
    assert any("error" in result for result in results)
    assert any("output" in result for result in results)
    assert sum(1 for r in results if "error" in r) == 1
    assert all("log_file" in r for r in results)
    error_payload = next(r for r in results if "error" in r)["error"]
    assert error_payload["kind"] == server.ErrorCategory.RUNTIME.value
    assert error_payload["code"] == "boom"


@pytest.mark.asyncio
async def test_spawn_agent_handles_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(server.LOG_ROOT_ENV_VAR, str(tmp_path))
    monkeypatch.setenv("CODEX_TIMEOUT_SECONDS", "0.01")
    monkeypatch.setattr(server, "_resolve_codex_executable", lambda: "codex")

    async def fake_create_subprocess_exec(*_: object, **__: object) -> FakeProcess:
        return FakeProcess(wait_delay=0.2)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    ctx = DummyContext()
    with pytest.raises(server.ToolExecutionError) as err:
        await server.spawn_agent(ctx, "long running task")

    payload = err.value.payload.to_dict()
    assert payload["code"] == "agent_timeout"
    assert payload["kind"] == server.ErrorCategory.TIMEOUT.value
    assert payload["details"]["timeout_seconds"] == pytest.approx(0.01, rel=0.2)

    log_path = Path(payload["log_file"])
    assert log_path.exists()


@pytest.mark.asyncio
async def test_spawn_agent_handles_missing_codex(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(server.LOG_ROOT_ENV_VAR, str(tmp_path))

    def _raise_missing() -> str:
        raise FileNotFoundError("Codex CLI missing")

    monkeypatch.setattr(server, "_resolve_codex_executable", _raise_missing)

    ctx = DummyContext()
    with pytest.raises(server.ToolExecutionError) as err:
        await server.spawn_agent(ctx, "task")

    payload = err.value.payload.to_dict()
    assert payload["code"] == "codex_not_found"
    assert payload["kind"] == server.ErrorCategory.RUNTIME.value
    assert not any(tmp_path.iterdir())
