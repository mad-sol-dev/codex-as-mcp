"""
Minimal MCP server (v2) exposing a single tool: `spawn_agent`.

Tool: spawn_agent(prompt: str) -> str
- Runs the Codex CLI agent and returns its final response as the tool result.

Command executed:
    codex e --cd {os.getcwd()} --skip-git-repo-check --full-auto \
        --output-last-message {temp_output} "{prompt}"

Notes:
- No Authorization headers or extra auth flows are used.
- Uses a generous default timeout to allow long-running agent sessions.
- Designed to be run via: `uv run python -m codex_as_mcp`
"""

import asyncio
import contextlib
import os
import shutil
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Deque

from mcp.server.fastmcp import FastMCP, Context


# Default timeout (seconds) for the spawned agent run.
# Chosen to be long to accommodate non-trivial editing tasks.
DEFAULT_TIMEOUT_SECONDS: int = 8 * 60 * 60  # 8 hours

# Log management defaults
LOG_ROOT_ENV_VAR = "CODEX_AS_MCP_LOG_DIR"
DEFAULT_LOG_ROOT = Path.home() / ".cache" / "codex-as-mcp" / "logs"
LOG_ROTATE_BYTES = 512_000


mcp = FastMCP("codex-subagent")


def _resolve_codex_executable() -> str:
    """Resolve the `codex` executable path or raise a clear error.

    Returns:
        str: Absolute path to the `codex` executable.

    Raises:
        FileNotFoundError: If the executable cannot be found in PATH.
    """
    codex = shutil.which("codex")
    if not codex:
        raise FileNotFoundError(
            "Codex CLI not found in PATH. Please install it (e.g. `npm i -g @openai/codex`) "
            "and ensure your shell PATH includes the npm global bin."
        )
    return codex


def _prepare_log_root() -> Path:
    """Return the directory for storing Codex logs, creating it if needed."""

    override = os.environ.get(LOG_ROOT_ENV_VAR)
    root = Path(override).expanduser() if override else DEFAULT_LOG_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def _allocate_log_paths(agent_index: int | None) -> tuple[Path, Path]:
    """Allocate deterministic paths for log and last message files."""

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    suffix_parts = [str(os.getpid())]
    if agent_index is not None:
        suffix_parts.append(str(agent_index))
    suffix_parts.append(uuid.uuid4().hex[:8])
    base_name = f"codex_agent_{timestamp}_{'_'.join(suffix_parts)}"
    log_root = _prepare_log_root()
    log_file = log_root / f"{base_name}.log"
    output_path = log_root / f"{base_name}.last_message.md"
    return log_file, output_path


@mcp.tool()
async def spawn_agent(
    ctx: Context,
    prompt: str,
    timeout_seconds: float | int | None = DEFAULT_TIMEOUT_SECONDS,
    agent_index: int | None = None,
    agent_count: int | None = None,
) -> str:
    """Spawn a Codex agent to work inside the current working directory.

    The server resolves the working directory via ``os.getcwd()`` so it inherits
    whatever environment the MCP process currently has.

    Args:
        prompt: All instructions/context the agent needs for the task.
        timeout_seconds: Max seconds to allow the Codex CLI to run. Defaults to
            ``DEFAULT_TIMEOUT_SECONDS``.

    Returns:
        The agent's final response (clean output from Codex CLI) with the log
        file path appended, or a clear timeout/error message.
    """
    # Basic validation to avoid confusing UI errors
    if not isinstance(prompt, str):
        return "Error: 'prompt' must be a string."
    if not prompt.strip():
        return "Error: 'prompt' is required and cannot be empty."

    if timeout_seconds is None:
        timeout_seconds = DEFAULT_TIMEOUT_SECONDS
    try:
        timeout_seconds = float(timeout_seconds)
        if timeout_seconds <= 0:
            raise ValueError
    except (ValueError, TypeError):
        return "Error: 'timeout_seconds' must be a positive number."

    try:
        codex_exec = _resolve_codex_executable()
    except FileNotFoundError as e:
        return f"Error: {e}"

    work_directory = os.getcwd()

    try:
        log_file, output_path = _allocate_log_paths(agent_index)
        output_path.touch()
        log_file.touch()
    except Exception as e:
        return f"Error: Unable to prepare log directory: {e}"

    max_log_bytes = LOG_ROTATE_BYTES  # Rotate to keep logs lightweight on stdio
    log_lock = asyncio.Lock()
    log_suffix = ".1"
    recent_lines: Deque[str] = deque(maxlen=50)
    log_note = f"Log file: {log_file}"

    agent_label = f"agent {agent_index}" if agent_index is not None else "agent"

    # Quote the prompt so Codex CLI receives it wrapped in "..."
    quoted_prompt = '"' + prompt.replace('"', '\\"') + '"'

    cmd = [
        codex_exec,
        "e",
        "--cd",
        work_directory,
        "--skip-git-repo-check",
        "--full-auto",
        "--output-last-message",
        str(output_path),
        quoted_prompt,
    ]

    try:
        # Initial progress ping
        try:
            await ctx.report_progress(
                agent_index if agent_index is not None else 0,
                agent_count,
                f"[{agent_label}] Launching Codex agent...",
            )
        except Exception:
            pass

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as e:
            return f"Error: Failed to launch Codex agent: {e}"

        stdout_lines = 0
        stderr_lines = 0
        bytes_seen = 0
        last_non_empty_line = ""

        async def _write_log(prefix: str, text: str) -> None:
            nonlocal bytes_seen
            async with log_lock:
                if log_file.exists() and log_file.stat().st_size > max_log_bytes:
                    rotated = log_file.with_name(log_file.name + log_suffix)
                    try:
                        if rotated.exists():
                            rotated.unlink()
                        log_file.rename(rotated)
                    except Exception:
                        pass
                entry = f"[{prefix}] {text}"
                with log_file.open("a", encoding="utf-8") as handle:
                    handle.write(entry)
                bytes_seen += len(entry.encode("utf-8", errors="replace"))

        async def _consume_stream(stream: asyncio.StreamReader | None, prefix: str) -> None:
            nonlocal stdout_lines, stderr_lines, last_non_empty_line
            if not stream:
                return
            while True:
                try:
                    line = await stream.readline()
                except Exception:
                    break
                if not line:
                    break
                decoded = line.decode(errors="replace")
                if prefix == "stdout":
                    stdout_lines += 1
                else:
                    stderr_lines += 1
                if decoded.strip():
                    last_non_empty_line = f"{prefix}: {decoded.strip()}"
                recent_lines.append(f"{prefix}: {decoded.rstrip()}".rstrip())
                await _write_log(prefix, decoded)

        stop_progress = asyncio.Event()

        async def _progress_reporter() -> None:
            start = time.monotonic()
            while not stop_progress.is_set():
                await asyncio.sleep(2)
                elapsed = int(time.monotonic() - start)
                message_parts = [
                    f"[{agent_label}] {elapsed}s elapsed",
                    f"stdout lines={stdout_lines}",
                    f"stderr lines={stderr_lines}",
                    f"bytes={bytes_seen}",
                ]
                if last_non_empty_line:
                    message_parts.append(f"last: {last_non_empty_line}")
                try:
                    await ctx.report_progress(
                        agent_index if agent_index is not None else 1,
                        agent_count,
                        "; ".join(message_parts),
                    )
                except Exception:
                    pass

        stdout_task = asyncio.create_task(_consume_stream(proc.stdout, "stdout"))
        stderr_task = asyncio.create_task(_consume_stream(proc.stderr, "stderr"))
        progress_task = asyncio.create_task(_progress_reporter())

        async def _wait_with_heartbeats() -> int:
            last_ping = time.monotonic()
            while True:
                try:
                    return await asyncio.wait_for(proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    now = time.monotonic()
                    if now - last_ping >= 2.0:
                        last_ping = now
                        try:
                            await ctx.report_progress(
                                agent_index if agent_index is not None else 1,
                                agent_count,
                                f"[{agent_label}] Codex agent running...",
                            )
                        except Exception:
                            pass

        try:
            returncode = await asyncio.wait_for(
                _wait_with_heartbeats(), timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            try:
                proc.terminate()
            except Exception:
                pass

            try:
                await asyncio.wait_for(proc.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass

            for task in (stdout_task, stderr_task, progress_task):
                if task:
                    task.cancel()
                    try:
                        await task
                    except Exception:
                        pass

            return "\n".join(
                [
                    f"Error: Codex agent timed out after {timeout_seconds} seconds.",
                    log_note,
                ]
            )

        stop_progress.set()
        for task in (stdout_task, stderr_task, progress_task):
            if task:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        output = output_path.read_text(encoding="utf-8").strip()

        if returncode != 0:
            details = [
                "Error: Codex agent exited with a non-zero status.",
                f"Command: {' '.join(cmd)}",
                f"Exit Code: {returncode}",
            ]
            if recent_lines:
                details.append("Recent output:\n" + "\n".join(recent_lines))
            if output:
                details.append(f"Captured Output: {output}")
            details.append(log_note)
            return "\n".join(details)

        return "\n".join([output, log_note]) if output else log_note
    finally:
        with contextlib.suppress(Exception):
            if output_path.exists():
                output_path.unlink()


@mcp.tool()
async def spawn_agents_parallel(
    ctx: Context,
    agents: list[dict[str, str]],
    timeout_seconds: float | int | None = DEFAULT_TIMEOUT_SECONDS,
) -> list[dict[str, str]]:
    """Spawn multiple Codex agents in parallel.

    Each spawned agent reuses the server's current working directory
    (``os.getcwd()``).

    Args:
        agents: List of agent specs, each with a 'prompt' entry and optional
            'timeout_seconds' override.
                Example: [
                    {"prompt": "Create math.md"},
                    {"prompt": "Create story.md", "timeout_seconds": 600}
                ]
        timeout_seconds: Default max seconds for each agent unless overridden
            per spec. Defaults to ``DEFAULT_TIMEOUT_SECONDS``.

    Returns:
        List of results with 'index', 'output', and optional 'error' and
        'log_file' fields.
    """
    if not isinstance(agents, list):
        return [{"index": "0", "error": "Error: 'agents' must be a list of agent specs."}]

    if not agents:
        return [{"index": "0", "error": "Error: 'agents' list cannot be empty."}]

    async def run_one(index: int, spec: dict) -> dict:
        """Run a single agent and return result with index."""
        try:
            # Validate spec
            if not isinstance(spec, dict):
                return {
                    "index": str(index),
                    "error": f"Agent {index}: spec must be a dictionary with a 'prompt' field."
                }

            prompt = spec.get("prompt", "")

            # Report progress for this agent
            try:
                await ctx.report_progress(
                    index,
                    len(agents),
                    f"[agent {index}] Starting agent {index + 1}/{len(agents)}...",
                )
            except Exception:
                pass

            agent_timeout = spec.get("timeout_seconds", timeout_seconds)

            # Run the agent
            output = await spawn_agent(
                ctx,
                prompt,
                agent_timeout,
                agent_index=index,
                agent_count=len(agents),
            )

            log_file_line = ""
            for line in output.splitlines():
                if line.startswith("Log file: "):
                    log_file_line = line.replace("Log file: ", "", 1).strip()
                    break

            # Check if output contains an error
            if output.startswith("Error:"):
                result: dict[str, str] = {"index": str(index), "error": output}
            else:
                result = {"index": str(index), "output": output}

            if log_file_line:
                result["log_file"] = log_file_line

            return result

        except Exception as e:
            return {"index": str(index), "error": f"Agent {index}: {str(e)}"}

    # Run all agents concurrently
    tasks = [run_one(i, agent) for i, agent in enumerate(agents)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle any exceptions that weren't caught
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            final_results.append({"index": str(i), "error": f"Unexpected error: {str(result)}"})
        else:
            final_results.append(result)

    return final_results


def main() -> None:
    """Entry point for the MCP server v2."""
    mcp.run()


if __name__ == "__main__":
    main()
