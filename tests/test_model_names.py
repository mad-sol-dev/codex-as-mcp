"""Tests to ensure correct Codex CLI model names are used."""
import asyncio
from pathlib import Path

import pytest

from codex_as_mcp import server
from tests.conftest import DummyContext, FakeProcess


SUPPORTED_MODELS = [
    "gpt-5.2-codex",
    "gpt-5.1-codex-max",
    "gpt-5.1-codex-mini",
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5.1-codex",
    "gpt-5-codex",
    "gpt-5-codex-mini",
    "gpt-5",
]

UNSUPPORTED_MODELS = [
    "o3-mini",
    "o1-preview",
    "o1-mini",
    "codex-1",
    "codex-mini-latest",
]


@pytest.mark.asyncio
async def test_spawn_agent_accepts_valid_model_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that spawn_agent accepts valid Codex CLI model names."""
    monkeypatch.setenv(server.LOG_ROOT_ENV_VAR, str(tmp_path))
    monkeypatch.setattr(server, "_resolve_codex_executable", lambda: "codex")

    for model in SUPPORTED_MODELS:
        # Track which model was passed to codex CLI
        captured_model = None

        async def fake_create_subprocess_exec(*args: object, **kwargs: object) -> FakeProcess:
            nonlocal captured_model
            cmd_args = list(args)
            # Find -m flag and capture the model name, and write output file
            for i, arg in enumerate(cmd_args):
                if arg == "-m" and i + 1 < len(cmd_args):
                    captured_model = cmd_args[i + 1]
                elif arg == "--output-last-message" and i + 1 < len(cmd_args):
                    Path(cmd_args[i + 1]).write_text(f"Output from {model}")
            return FakeProcess([b"stdout\n"], [])

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

        ctx = DummyContext()
        result = await server.spawn_agent(ctx, f"test {model}", model=model)

        # Verify the model was passed correctly to codex CLI
        assert captured_model == model, f"Model {model} should be passed to codex CLI"
        assert f"Output from {model}" in result


@pytest.mark.asyncio
async def test_documentation_does_not_reference_unsupported_models(tmp_path: Path) -> None:
    """Test that documentation files don't reference unsupported model names."""
    repo_root = Path(__file__).parent.parent

    # Files to check
    files_to_check = [
        repo_root / "README.md",
        repo_root / "CLAUDE.md",
        repo_root / "src" / "codex_as_mcp" / "server.py",
    ]

    for file_path in files_to_check:
        if not file_path.exists():
            continue

        content = file_path.read_text()

        # Check for unsupported models (excluding notes/warnings about them)
        for model in UNSUPPORTED_MODELS:
            # Count occurrences
            occurrences = content.count(f'"{model}"') + content.count(f"'{model}'") + content.count(f"`{model}`")

            # Check if it's in a warning/note context
            warning_patterns = [
                f"NOT supported",
                f"not supported",
                f"do not use",
                f"Models like `{model}`",
                f"**NOT**",
            ]
            is_warning = any(pattern in content for pattern in warning_patterns)

            if occurrences > 0 and not is_warning:
                pytest.fail(
                    f"File {file_path.name} references unsupported model '{model}' "
                    f"({occurrences} times) without warning context. "
                    f"Use supported models: {', '.join(SUPPORTED_MODELS[:3])}"
                )


@pytest.mark.asyncio
async def test_recommended_models_in_examples() -> None:
    """Test that code examples use recommended model names."""
    repo_root = Path(__file__).parent.parent
    readme = repo_root / "README.md"

    if not readme.exists():
        pytest.skip("README.md not found")

    content = readme.read_text()

    # Check that examples section exists and uses correct models
    assert "gpt-5.2-codex" in content, "README should mention gpt-5.2-codex"
    assert "gpt-5.1-codex-max" in content, "README should mention gpt-5.1-codex-max"
    assert "gpt-5.1-codex-mini" in content, "README should mention gpt-5.1-codex-mini"

    # Verify examples section exists
    assert "## Examples" in content, "README should have an Examples section"


def test_supported_models_list_is_documented() -> None:
    """Test that all supported models are documented."""
    repo_root = Path(__file__).parent.parent
    claude_md = repo_root / "CLAUDE.md"

    if not claude_md.exists():
        pytest.skip("CLAUDE.md not found")

    content = claude_md.read_text()

    # Check for model documentation
    assert "gpt-5.2-codex" in content
    assert "gpt-5.1-codex-max" in content
    assert "gpt-5.1-codex-mini" in content

    # Check for warning about unsupported models
    assert "NOT supported" in content or "not supported" in content
