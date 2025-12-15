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
import json
import os
import shutil
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Optional

from mcp.server.fastmcp import FastMCP, Context


# Default timeout (seconds) for the spawned agent run.
# Chosen to be long to accommodate non-trivial editing tasks.
DEFAULT_TIMEOUT_SECONDS: int = 8 * 60 * 60  # 8 hours

# Log management defaults
LOG_ROOT_ENV_VAR = "CODEX_AS_MCP_LOG_DIR"
DEFAULT_LOG_ROOT = Path.home() / ".cache" / "codex-as-mcp" / "logs"
LOG_ROTATE_BYTES = 1_000_000
LOG_ROTATE_COUNT = 5


mcp = FastMCP("codex-subagent")


class ErrorCategory(str, Enum):
    VALIDATION = "validation_error"
    RUNTIME = "runtime_error"
    TIMEOUT = "timeout_error"
    IO = "io_error"


@dataclass(slots=True)
class ToolErrorPayload:
    code: str
    kind: ErrorCategory
    message: str
    log_file: str | None = None
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "kind": self.kind.value,
            "message": self.message,
        }
        if self.log_file:
            payload["log_file"] = self.log_file
        if self.details:
            payload["details"] = self._normalize(self.details)
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def _normalize(cls, value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Exception):
            return str(value)
        if isinstance(value, dict):
            return {key: cls._normalize(val) for key, val in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._normalize(val) for val in value]
        return value


class ToolExecutionError(Exception):
    """Structured tool error surfaced to MCP clients."""

    def __init__(
        self,
        *,
        code: str,
        kind: ErrorCategory,
        message: str,
        log_file: Path | str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        log_path = str(log_file) if log_file else None
        self.payload = ToolErrorPayload(
            code=code,
            kind=kind,
            message=message,
            log_file=log_path,
            details=details,
        )
        super().__init__(self.payload.to_json())
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


def _allocate_log_paths(agent_index: Optional[int]) -> tuple[Path, Path]:
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


def _rotate_logs(log_file: Path, max_bytes: int, max_files: int) -> None:
    """Rotate the main log file when it exceeds ``max_bytes``."""

    if not log_file.exists() or log_file.stat().st_size <= max_bytes:
        return

    oldest = log_file.with_name(log_file.name + f".{max_files}")
    if oldest.exists():
        oldest.unlink()

    for idx in range(max_files - 1, 0, -1):
        src = log_file.with_name(log_file.name + (f".{idx}" if idx > 0 else ""))
        dst = log_file.with_name(log_file.name + f".{idx + 1}")
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)

    rotated_first = log_file.with_name(log_file.name + ".1")
    if rotated_first.exists():
        rotated_first.unlink()
    log_file.rename(rotated_first)
    log_file.touch()


def _format_log_line(
    level: str,
    event: str,
    message: str,
    context: Optional[dict] = None,
) -> str:
    payload = {
        "ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
        "level": level,
        "event": event,
        "message": message,
    }
    if context:
        payload["context"] = context
    return json.dumps(payload, ensure_ascii=False) + "\n"


@mcp.tool()
async def spawn_agent(
    ctx: Context,
    prompt: str,
    reasoning_effort: Optional[str] = None,
    model: Optional[str] = None,
    agent_index: Optional[int] = None,
    agent_count: Optional[int] = None,
) -> str:
    """Spawn a Codex agent to work inside the current working directory.

    The server resolves the working directory via ``os.getcwd()`` so it inherits
    whatever environment the MCP process currently has.

    Args:
        prompt: All instructions/context the agent needs for the task.
        reasoning_effort: Reasoning level for the task. Defaults to env var
            CODEX_REASONING_EFFORT or Codex's default.
            - "low": Fast, simple tasks (file operations, basic edits)
            - "medium": Balanced approach (default, most tasks)
            - "high": Complex analysis (architecture design, debugging)
            - "xhigh": Very complex (reverse engineering, security analysis)
        model: Override the Codex model. Defaults to user's config (~/.codex/config.toml).
            Examples: "o3-mini", "o1-preview", "gpt-5.1-codex-max"

    Returns:
        The agent's final response (clean output from Codex CLI) with the log
        file path appended, or a clear timeout/error message.
    """
    log_file: Path | None = None
    output_path: Path | None = None

    # Basic validation to avoid confusing UI errors
    if not isinstance(prompt, str):
        raise ToolExecutionError(
            code="invalid_prompt_type",
            kind=ErrorCategory.VALIDATION,
            message="'prompt' must be a string.",
        )
    if not prompt.strip():
        raise ToolExecutionError(
            code="empty_prompt",
            kind=ErrorCategory.VALIDATION,
            message="'prompt' is required and cannot be empty.",
        )

    # Get timeout from environment variable or use default
    timeout_seconds = DEFAULT_TIMEOUT_SECONDS
    timeout_env = os.environ.get("CODEX_TIMEOUT_SECONDS")
    if timeout_env:
        try:
            timeout_seconds = float(timeout_env)
            if timeout_seconds <= 0:
                raise ValueError
        except (ValueError, TypeError):
            pass  # Silently fall back to default on invalid value

    try:
        codex_exec = _resolve_codex_executable()
    except FileNotFoundError as e:
        raise ToolExecutionError(
            code="codex_not_found",
            kind=ErrorCategory.RUNTIME,
            message=str(e),
        )

    work_directory = os.getcwd()

    try:
        log_file, output_path = _allocate_log_paths(agent_index)
        output_path.touch()
        log_file.touch()
    except Exception as e:
        raise ToolExecutionError(
            code="log_init_failed",
            kind=ErrorCategory.IO,
            message="Unable to prepare log directory",
            details={"error": str(e)},
        )

    max_log_bytes = LOG_ROTATE_BYTES  # Rotate to keep logs lightweight on stdio
    log_lock = asyncio.Lock()
    recent_lines: Deque[str] = deque(maxlen=50)
    log_note = f"Log file: {log_file}"

    agent_label = f"agent {agent_index}" if agent_index is not None else "agent"
    agent_id = str(agent_index) if agent_index is not None else "single"
    prompt_preview = " ".join(prompt.strip().split())[:160]
    base_context = {
        "agent_id": agent_id,
        "agent_index": agent_index,
        "agent_count": agent_count,
        "workdir": work_directory,
        "prompt_preview": prompt_preview,
        "pid": os.getpid(),
    }

    cmd = [
        codex_exec,
        "e",
        "--cd",
        work_directory,
        "--skip-git-repo-check",
        "--full-auto",
        "--output-last-message",
        str(output_path),
    ]

    # Add model override if specified (parameter takes precedence over env var)
    model_to_use = model or os.environ.get("CODEX_MODEL")
    if model_to_use:
        cmd.extend(["-m", model_to_use])

    # Add reasoning_effort config (parameter > env var > codex default)
    reasoning_to_use = reasoning_effort or os.environ.get("CODEX_REASONING_EFFORT")
    if reasoning_to_use:
        cmd.extend(["-c", f"model_reasoning_effort={reasoning_to_use}"])

    # Pass the raw prompt; subprocess_exec handles argument separation safely.
    cmd.append(prompt)

    stdout_lines = 0
    stderr_lines = 0
    bytes_seen = 0
    last_non_empty_line = ""

    async def _write_log(level: str, event: str, message: str, context: Optional[dict] = None) -> None:
        nonlocal bytes_seen
        async with log_lock:
            try:
                _rotate_logs(log_file, max_log_bytes, LOG_ROTATE_COUNT)
            except Exception:
                pass
            entry = _format_log_line(level, event, message, context)
            with log_file.open("a", encoding="utf-8") as handle:
                handle.write(entry)
            bytes_seen += len(entry.encode("utf-8", errors="replace"))

    start_time = time.monotonic()

    try:
        await _write_log(
            "info",
            "agent_start",
            "Launching Codex agent",
            {**base_context, "timeout_seconds": timeout_seconds},
        )
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
            await _write_log(
                "info",
                "process_start",
                "Codex agent subprocess started",
                {**base_context, "command": cmd},
            )
        except Exception as e:
            await _write_log(
                "error",
                "process_start_failed",
                f"Failed to launch Codex agent: {e}",
                {**base_context, "command": cmd},
            )
            raise ToolExecutionError(
                code="process_start_failed",
                kind=ErrorCategory.RUNTIME,
                message="Failed to launch Codex agent",
                log_file=log_file,
                details={"error": str(e), "command": " ".join(map(str, cmd))},
            )

        async def _consume_stream(stream: Optional[asyncio.StreamReader], prefix: str) -> None:
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
                clean_line = decoded.rstrip("\n")
                if prefix == "stdout":
                    stdout_lines += 1
                else:
                    stderr_lines += 1
                if clean_line.strip():
                    last_non_empty_line = f"{prefix}: {clean_line.strip()}"
                recent_lines.append(f"{prefix}: {clean_line}".rstrip())
                await _write_log(
                    "info",
                    prefix,
                    f"{prefix}: {clean_line}",
                    {**base_context, "stream": prefix},
                )

        stop_progress = asyncio.Event()

        async def _progress_reporter() -> None:
            while not stop_progress.is_set():
                await asyncio.sleep(2)
                elapsed = time.monotonic() - start_time
                message_parts = [
                    f"[{agent_label}] {int(elapsed)}s elapsed",
                    f"stdout lines={stdout_lines}",
                    f"stderr lines={stderr_lines}",
                    f"bytes={bytes_seen}",
                ]
                if last_non_empty_line:
                    message_parts.append(f"last: {last_non_empty_line}")
                try:
                    await _write_log(
                        "info",
                        "progress",
                        "; ".join(message_parts),
                        {
                            **base_context,
                            "elapsed_seconds": round(elapsed, 2),
                            "stdout_lines": stdout_lines,
                            "stderr_lines": stderr_lines,
                            "bytes_written": bytes_seen,
                            "last_non_empty_line": last_non_empty_line or None,
                        },
                    )
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
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        pass

            await _write_log(
                "error",
                "timeout",
                f"Codex agent timed out after {timeout_seconds} seconds",
                {**base_context, "timeout_seconds": timeout_seconds, "elapsed_seconds": round(time.monotonic() - start_time, 2)},
            )
            raise ToolExecutionError(
                code="agent_timeout",
                kind=ErrorCategory.TIMEOUT,
                message=f"Codex agent timed out after {timeout_seconds} seconds.",
                log_file=log_file,
                details={
                    "timeout_seconds": timeout_seconds,
                    "elapsed_seconds": round(time.monotonic() - start_time, 2),
                },
            )

        stop_progress.set()
        for task in (stdout_task, stderr_task, progress_task):
            if task:
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        # Log completion
        await _write_log(
            "info",
            "agent_complete",
            f"Codex agent process completed with exit code: {returncode}",
            {**base_context, "exit_code": returncode, "elapsed_seconds": round(time.monotonic() - start_time, 2)},
        )

        # Robustly read output file with retries (Codex might still be flushing)
        output = ""
        output_error: ToolExecutionError | None = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if output_path.exists():
                    file_size = output_path.stat().st_size
                    await _write_log(
                        "info",
                        "output_check",
                        f"Output file exists, size: {file_size} bytes (attempt {attempt + 1}/{max_retries})",
                        {**base_context, "attempt": attempt + 1, "file_size": file_size},
                    )

                    # Wait a moment for file system flush
                    if attempt > 0:
                        await asyncio.sleep(0.5)

                    output = output_path.read_text(encoding="utf-8", errors="replace").strip()
                    if output:
                        break  # Success
                    elif attempt < max_retries - 1:
                        await _write_log(
                            "warning",
                            "output_empty_retry",
                            f"Output file empty on attempt {attempt + 1}, retrying...",
                            {**base_context, "attempt": attempt + 1},
                        )
                    else:
                        await _write_log(
                            "warning",
                            "output_missing",
                            "Output file is empty after all retries",
                            {**base_context, "attempt": attempt + 1},
                        )
                else:
                    await _write_log(
                        "error",
                        "output_missing",
                        f"Output file does not exist: {output_path}",
                        {**base_context, "attempt": attempt + 1},
                    )
                    output_error = ToolExecutionError(
                        code="output_missing",
                        kind=ErrorCategory.IO,
                        message=f"Output file does not exist: {output_path}",
                        log_file=log_file,
                        details={"attempt": attempt + 1},
                    )
                    break
            except Exception as e:
                await _write_log(
                    "error",
                    "output_read_error",
                    f"Failed to read output file (attempt {attempt + 1}): {e}",
                    {**base_context, "attempt": attempt + 1},
                )
                output_error = ToolExecutionError(
                    code="output_read_failed",
                    kind=ErrorCategory.IO,
                    message="Failed to read output file",
                    log_file=log_file,
                    details={"attempt": attempt + 1, "error": str(e)},
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)
                break

        if returncode != 0:
            await _write_log(
                "error",
                "agent_failed",
                "Codex agent exited with non-zero status",
                {
                    **base_context,
                    "exit_code": returncode,
                    "recent_lines": list(recent_lines)[-5:],
                },
            )
            raise ToolExecutionError(
                code="agent_non_zero_exit",
                kind=ErrorCategory.RUNTIME,
                message="Codex agent exited with a non-zero status.",
                log_file=log_file,
                details={
                    "command": cmd,
                    "exit_code": returncode,
                    "recent_output": list(recent_lines),
                    "captured_output": output or None,
                },
            )

        # Return output if available, otherwise include recent lines as fallback
        if output_error and returncode == 0:
            raise output_error
        if output:
            return "\n".join([output, log_note])
        elif recent_lines:
            # Fallback: if output file is empty but we have stdout/stderr
            await _write_log(
                "warning",
                "output_fallback",
                "Output file empty, using recent lines as fallback",
                {**base_context, "recent_line_count": len(recent_lines)},
            )
            return "\n".join([
                "Warning: Codex agent completed but output file was empty.",
                "Recent output:",
                "\n".join(list(recent_lines)[-20:]),  # Last 20 lines
                log_note
            ])
        else:
            return f"Warning: Codex agent completed but produced no output.\n{log_note}"
    finally:
        # Ensure cleanup happens even on exceptions
        cleanup_retries = 3
        for cleanup_attempt in range(cleanup_retries):
            try:
                if output_path.exists():
                    output_path.unlink()
                    break
            except Exception as cleanup_error:
                if cleanup_attempt < cleanup_retries - 1:
                    await asyncio.sleep(0.2)
                # Silently fail on last attempt - file will be cleaned up later


@mcp.tool()
async def spawn_agents_parallel(
    ctx: Context,
    agents: list[dict[str, str]],
    max_parallel: int | None = None,
) -> list[dict[str, Any]]:
    """Spawn multiple Codex agents in parallel.

    Each spawned agent reuses the server's current working directory
    (``os.getcwd()``).

    Args:
        agents: List of agent specs, each with a 'prompt' entry and optional
            'reasoning_effort' and 'model' overrides.
                Example: [
                    {"prompt": "Create math.md"},
                    {"prompt": "Analyze binary", "reasoning_effort": "high"},
                    {"prompt": "Debug issue", "model": "o3-mini"}
                ]
        max_parallel: Optional limit on how many agents run concurrently.
            If ``None`` (default), all agents run at once.

    Returns:
        List of results with 'index', 'output', and optional 'error' and
        'log_file' fields.
    """
    if not isinstance(agents, list):
        raise ToolExecutionError(
            code="invalid_agents_type",
            kind=ErrorCategory.VALIDATION,
            message="'agents' must be a list of agent specs.",
        )

    if not agents:
        raise ToolExecutionError(
            code="agents_empty",
            kind=ErrorCategory.VALIDATION,
            message="'agents' list cannot be empty.",
        )

    if max_parallel is not None and (not isinstance(max_parallel, int) or max_parallel <= 0):
        raise ToolExecutionError(
            code="invalid_max_parallel",
            kind=ErrorCategory.VALIDATION,
            message="'max_parallel' must be a positive integer or None.",
        )

    semaphore = asyncio.Semaphore(max_parallel) if max_parallel else None

    async def run_one(index: int, spec: dict) -> dict[str, Any]:
        """Run a single agent and return result with index."""
        try:
            # Validate spec
            if not isinstance(spec, dict):
                return {
                    "index": str(index),
                    "error": ToolErrorPayload(
                        code="invalid_agent_spec",
                        kind=ErrorCategory.VALIDATION,
                        message=f"Agent {index}: spec must be a dictionary with a 'prompt' field.",
                    ).to_dict(),
                }

            prompt = spec.get("prompt", "")
            reasoning_effort = spec.get("reasoning_effort")
            model = spec.get("model")

            # Report progress for this agent
            try:
                await ctx.report_progress(
                    index,
                    len(agents),
                    f"[agent {index}] Starting agent {index + 1}/{len(agents)}...",
                )
            except Exception:
                pass

            def _extract_log_file(value: str) -> str:
                for line in value.splitlines():
                    if line.startswith("Log file: "):
                        return line.replace("Log file: ", "", 1).strip()
                return ""

            # Run the agent
            output = await spawn_agent(
                ctx,
                prompt,
                reasoning_effort=reasoning_effort,
                model=model,
                agent_index=index,
                agent_count=len(agents),
            )

            log_file_line = _extract_log_file(output)
            result: dict[str, Any] = {"index": str(index), "output": output}
            if log_file_line:
                result["log_file"] = log_file_line

            return result

        except ToolExecutionError as exc:
            payload = exc.payload.to_dict()
            result: dict[str, Any] = {"index": str(index), "error": payload}
            if payload.get("log_file"):
                result["log_file"] = payload["log_file"]
            return result
        except Exception as e:
            fallback_payload = ToolErrorPayload(
                code="unexpected_agent_error",
                kind=ErrorCategory.RUNTIME,
                message=f"Agent {index} failed unexpectedly.",
                details={"error": str(e)},
            ).to_dict()
            return {"index": str(index), "error": fallback_payload}

    async def run_with_limit(index: int, spec: dict) -> dict:
        if semaphore is None:
            return await run_one(index, spec)
        async with semaphore:
            return await run_one(index, spec)

    # Run all agents concurrently, honoring the semaphore limit when provided
    tasks = [run_with_limit(i, agent) for i, agent in enumerate(agents)]
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
