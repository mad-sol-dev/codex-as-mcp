"""
Minimal MCP server (v2) exposing a single tool: `spawn_agent`.

Tool: spawn_agent(prompt: str) -> str
- Runs the Codex CLI agent and returns its final response as the tool result.

Command executed:
    codex e --cd {CODEX_AGENT_CWD or os.getcwd()} --skip-git-repo-check --full-auto \
        --output-last-message {temp_output} "{prompt}"

Notes:
- No Authorization headers or extra auth flows are used.
- Uses a generous default timeout to allow long-running agent sessions.
- The working directory passed to Codex can be overridden via
  CODEX_AGENT_CWD when the MCP server itself must run elsewhere.
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

# Output size limit to prevent Claude Desktop sync issues
# Claude Desktop has response size limits that cause "message_store_sync_loss"
# when exceeded, leading to conversation deletion
MAX_OUTPUT_LENGTH = 4000  # characters


mcp = FastMCP("codex-subagent")

# Global task tracking for async agent spawning
running_tasks: dict[str, dict[str, Any]] = {}


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
    human_readable: bool = True,
) -> str:
    """Format a log line. By default, use human-readable text format.

    Args:
        level: Log level (info, warning, error)
        event: Event type (agent_start, stdout, stderr, progress, etc.)
        message: The message to log
        context: Optional context dict (only used for JSON format)
        human_readable: If True, use simple text format. If False, use JSON.

    Returns:
        Formatted log line with newline
    """
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    if human_readable:
        # Simple text format: [timestamp] LEVEL event: message
        level_str = level.upper().ljust(7)
        return f"[{timestamp}] {level_str} {event}: {message}\n"
    else:
        # JSON format for structured logging (used for important events)
        payload = {
            "ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "level": level,
            "event": event,
            "message": message,
        }
        if context:
            payload["context"] = context
        return json.dumps(payload, ensure_ascii=False) + "\n"


async def _run_codex_agent(
    ctx: Optional[Context],
    prompt: str,
    reasoning_effort: Optional[str] = None,
    model: Optional[str] = None,
    agent_index: Optional[int] = None,
    agent_count: Optional[int] = None,
) -> str:
    """Internal function that runs a Codex agent.

    This is the core implementation used by both spawn_agent (blocking)
    and spawn_agent_async (non-blocking).

    Args:
        ctx: MCP context for progress reporting (optional, can be None for background tasks)
        prompt: All instructions/context the agent needs
        reasoning_effort: "low", "medium", "high", or "xhigh"
        model: Model override (e.g., "gpt-5.2-codex", "gpt-5.1-codex-mini")
        agent_index: Index of this agent (for parallel execution)
        agent_count: Total number of agents (for parallel execution)

    Returns:
        The agent's final response with log file path appended
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

    # Allow overriding the working directory for spawned agents
    working_dir = os.environ.get("CODEX_AGENT_CWD", os.getcwd())

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
        "workdir": working_dir,
        "prompt_preview": prompt_preview,
        "pid": os.getpid(),
    }

    cmd = [
        codex_exec,
        "e",
        "--cd",
        working_dir,
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

    async def _write_log(
        level: str,
        event: str,
        message: str,
        context: Optional[dict] = None,
        as_json: bool = False,
    ) -> None:
        """Write a log entry. Use as_json=True for important events, False for streaming output."""
        nonlocal bytes_seen
        async with log_lock:
            try:
                _rotate_logs(log_file, max_log_bytes, LOG_ROTATE_COUNT)
            except Exception:
                pass
            entry = _format_log_line(level, event, message, context, human_readable=not as_json)
            with log_file.open("a", encoding="utf-8") as handle:
                handle.write(entry)
            bytes_seen += len(entry.encode("utf-8", errors="replace"))

    start_time = time.monotonic()

    try:
        # Log start as JSON (important event)
        await _write_log(
            "info",
            "agent_start",
            f"Launching Codex agent (timeout: {timeout_seconds}s)",
            {**base_context, "timeout_seconds": timeout_seconds},
            as_json=True,
        )
        # Initial progress ping
        if ctx:
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
            # Log process start as JSON (important event)
            await _write_log(
                "info",
                "process_start",
                f"Codex agent subprocess started (PID: {proc.pid})",
                {**base_context, "command": cmd, "pid": proc.pid},
                as_json=True,
            )
        except Exception as e:
            # Log error as JSON
            await _write_log(
                "error",
                "process_start_failed",
                f"Failed to launch Codex agent: {e}",
                {**base_context, "command": cmd},
                as_json=True,
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
                    last_non_empty_line = clean_line.strip()
                recent_lines.append(f"{prefix}: {clean_line}".rstrip())
                # Log stdout/stderr as simple text (human-readable)
                await _write_log(
                    "info",
                    prefix,
                    clean_line,  # Just the line, no prefix duplication
                    as_json=False,  # Use text format for streaming output
                )

        stop_progress = asyncio.Event()
        last_logged_line = ""
        last_log_time = start_time
        warned_about_desktop_timeout = False

        async def _progress_reporter() -> None:
            """Report progress every 2s via MCP, but only log when something changes or every 10s."""
            nonlocal last_logged_line, last_log_time, warned_about_desktop_timeout
            while not stop_progress.is_set():
                await asyncio.sleep(2)
                elapsed = time.monotonic() - start_time
                message_parts = [
                    f"[{agent_label}] {int(elapsed)}s elapsed",
                    f"stdout={stdout_lines}",
                    f"stderr={stderr_lines}",
                ]
                if last_non_empty_line:
                    message_parts.append(f"last: {last_non_empty_line[:80]}")

                # Always send MCP progress (for timeout reset)
                if ctx:
                    try:
                        await ctx.report_progress(
                            agent_index if agent_index is not None else 1,
                            agent_count,
                            "; ".join(message_parts),
                        )
                    except Exception:
                        pass

                # Warn about Claude Desktop timeout at 50 seconds
                if not warned_about_desktop_timeout and elapsed >= 50:
                    warned_about_desktop_timeout = True
                    try:
                        await _write_log(
                            "warning",
                            "timeout_warning",
                            "Task running >50s: Claude Desktop will timeout at ~60s. Use Claude Code CLI for long tasks. See CLAUDE.md",
                            {
                                **base_context,
                                "elapsed_seconds": round(elapsed, 2),
                                "desktop_timeout_at": 60,
                            },
                            as_json=True,  # Important warning
                        )
                    except Exception:
                        pass

                # Only write to log if:
                # 1. last_non_empty_line changed, OR
                # 2. 10 seconds passed since last log
                should_log = (
                    last_non_empty_line != last_logged_line or
                    (elapsed - (last_log_time - start_time)) >= 10
                )

                if should_log:
                    try:
                        await _write_log(
                            "info",
                            "progress",
                            "; ".join(message_parts),
                            as_json=False,  # Use text format
                        )
                        last_logged_line = last_non_empty_line
                        last_log_time = time.monotonic()
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
                        if ctx:
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

            # Log timeout as JSON (important event)
            await _write_log(
                "error",
                "timeout",
                f"Codex agent timed out after {timeout_seconds} seconds (elapsed: {round(time.monotonic() - start_time, 2)}s)",
                {**base_context, "timeout_seconds": timeout_seconds, "elapsed_seconds": round(time.monotonic() - start_time, 2)},
                as_json=True,
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

        # Log completion as JSON (important event)
        elapsed_final = round(time.monotonic() - start_time, 2)
        await _write_log(
            "info",
            "agent_complete",
            f"Codex agent completed with exit code {returncode} (elapsed: {elapsed_final}s)",
            {**base_context, "exit_code": returncode, "elapsed_seconds": elapsed_final},
            as_json=True,
        )

        # Robustly read output file with retries (Codex might still be flushing)
        output = ""
        output_error: ToolExecutionError | None = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if output_path.exists():
                    file_size = output_path.stat().st_size
                    # Only log on first attempt or if retrying
                    if attempt == 0 or attempt == max_retries - 1:
                        await _write_log(
                            "info",
                            "output_check",
                            f"Output file size: {file_size} bytes (attempt {attempt + 1}/{max_retries})",
                            as_json=False,
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
                            as_json=True,  # Important warning
                        )
                    else:
                        await _write_log(
                            "warning",
                            "output_missing",
                            "Output file is empty after all retries",
                            {**base_context, "attempt": attempt + 1},
                            as_json=True,  # Important warning
                        )
                else:
                    await _write_log(
                        "error",
                        "output_missing",
                        f"Output file does not exist: {output_path}",
                        {**base_context, "attempt": attempt + 1},
                        as_json=True,  # Important error
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
                    {**base_context, "attempt": attempt + 1, "error": str(e)},
                    as_json=True,  # Important error
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
                f"Codex agent exited with non-zero status (exit code: {returncode})",
                {
                    **base_context,
                    "exit_code": returncode,
                    "recent_lines": list(recent_lines)[-5:],
                },
                as_json=True,  # Important error
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
        # IMPORTANT: Truncate output to prevent Claude Desktop sync issues
        if output_error and returncode == 0:
            raise output_error
        if output:
            # Truncate if output exceeds Claude Desktop's sync limit
            if len(output) > MAX_OUTPUT_LENGTH:
                truncated = output[:MAX_OUTPUT_LENGTH]
                truncation_note = f"\n\n[Output truncated: {len(output)} chars total, showing first {MAX_OUTPUT_LENGTH}. See full output in log file]"
                await _write_log(
                    "info",
                    "output_truncated",
                    f"Output truncated from {len(output)} to {MAX_OUTPUT_LENGTH} chars to prevent Claude Desktop sync issues",
                    {**base_context, "original_length": len(output), "truncated_length": MAX_OUTPUT_LENGTH},
                    as_json=True,
                )
                return "\n".join([truncated, truncation_note, log_note])
            return "\n".join([output, log_note])
        elif recent_lines:
            # Fallback: if output file is empty but we have stdout/stderr
            await _write_log(
                "warning",
                "output_fallback",
                "Output file empty, using recent lines as fallback",
                {**base_context, "recent_line_count": len(recent_lines)},
                as_json=True,  # Important warning
            )
            recent_output = "\n".join(list(recent_lines)[-20:])  # Last 20 lines
            # Truncate fallback output too
            if len(recent_output) > MAX_OUTPUT_LENGTH:
                recent_output = recent_output[:MAX_OUTPUT_LENGTH] + "\n[Truncated]"
            return "\n".join([
                "Warning: Codex agent completed but output file was empty.",
                "Recent output:",
                recent_output,
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
async def spawn_agent(
    ctx: Context,
    prompt: str,
    reasoning_effort: Optional[str] = None,
    model: Optional[str] = None,
    agent_index: Optional[int] = None,
    agent_count: Optional[int] = None,
) -> str:
    """Spawn a Codex agent to work inside the configured working directory.

    This is a BLOCKING tool - it waits for the agent to complete before returning.

    ⚠️  IMPORTANT - Claude Desktop Compatibility:
    - Claude Desktop has a FIXED 60-second timeout that CANNOT be configured
    - Progress updates do NOT reset this timeout (known limitation)
    - For tasks longer than 50 seconds, use `spawn_agent_async` instead
    - Recommended: Use Claude Code CLI for long-running tasks

    For non-blocking execution with Claude Desktop, use `spawn_agent_async`.

    The server resolves the working directory via ``CODEX_AGENT_CWD`` if set,
    otherwise ``os.getcwd()`` so it inherits whatever environment the MCP
    process currently has. Use the environment variable when the server runs
    from one location but you want agents to work in another workspace.

    Args:
        prompt: All instructions/context the agent needs for the task.
        reasoning_effort: Reasoning level for the task. Defaults to env var
            CODEX_REASONING_EFFORT or Codex's default.
            - "low": Fast, simple tasks (file operations, basic edits)
            - "medium": Balanced approach (default, most tasks)
            - "high": Complex analysis (architecture design, debugging)
            - "xhigh": Very complex (reverse engineering, security analysis)
        model: Override the Codex model. Defaults to user's config (~/.codex/config.toml).
            Examples: "gpt-5.2-codex", "gpt-5.1-codex-max", "gpt-5.1-codex-mini"

    Returns:
        The agent's final response (clean output from Codex CLI) with the log
        file path appended, or a clear timeout/error message.
    """
    return await _run_codex_agent(ctx, prompt, reasoning_effort, model, agent_index, agent_count)


@mcp.tool()
async def spawn_agent_async(
    ctx: Context,
    prompt: str,
    reasoning_effort: Optional[str] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """Spawn a Codex agent asynchronously (non-blocking, returns immediately).

    ✅ RECOMMENDED for Claude Desktop - avoids the 60-second timeout limitation.

    This tool starts a Codex agent in the background and returns immediately with
    a task ID. Use `get_agent_status(task_id)` to check progress and retrieve results.

    Workflow:
    1. Call spawn_agent_async() → returns task_id and log_file
    2. Optionally read log_file while running (using Read tool)
    3. Call get_agent_status(task_id) to check if completed
    4. When status is "completed", retrieve the output

    The server resolves the working directory via ``CODEX_AGENT_CWD`` if set,
    otherwise ``os.getcwd()``.

    Args:
        prompt: All instructions/context the agent needs for the task.
        reasoning_effort: Reasoning level ("low", "medium", "high", "xhigh")
        model: Override the Codex model (e.g., "gpt-5.2-codex", "gpt-5.1-codex-mini")

    Returns:
        dict with:
        - task_id: Unique identifier for this task
        - status: "running"
        - log_file: Path to log file (can be read while task is running)
        - started_at: ISO timestamp when task started
    """
    # Basic validation
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

    # Generate unique task ID
    task_id = f"codex_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    # Allocate log file paths
    log_file, output_path = _allocate_log_paths(None)

    # Enhance prompt to prevent agent from waiting for user input
    enhanced_prompt = f"""{prompt}

IMPORTANT: After completing the task, provide a final summary and terminate immediately.
Do NOT ask for further instructions or wait for user input."""

    started_at = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
    start_time = time.time()

    # Create background task
    async def _background_runner():
        try:
            output = await _run_codex_agent(
                None,  # No context for background tasks
                enhanced_prompt,
                reasoning_effort,
                model,
                agent_index=None,
                agent_count=None,
            )
            async with asyncio.Lock():
                running_tasks[task_id]["status"] = "completed"
                running_tasks[task_id]["output"] = output
                running_tasks[task_id]["completed_at"] = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
                running_tasks[task_id]["elapsed_seconds"] = round(time.time() - start_time, 2)
        except ToolExecutionError as e:
            async with asyncio.Lock():
                running_tasks[task_id]["status"] = "failed"
                running_tasks[task_id]["error"] = e.payload.to_dict()
                running_tasks[task_id]["completed_at"] = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
                running_tasks[task_id]["elapsed_seconds"] = round(time.time() - start_time, 2)
        except Exception as e:
            async with asyncio.Lock():
                running_tasks[task_id]["status"] = "failed"
                running_tasks[task_id]["error"] = {
                    "code": "unexpected_error",
                    "kind": "runtime_error",
                    "message": f"Unexpected error: {str(e)}",
                }
                running_tasks[task_id]["completed_at"] = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
                running_tasks[task_id]["elapsed_seconds"] = round(time.time() - start_time, 2)

    # Start the background task
    task = asyncio.create_task(_background_runner())

    # Register task in global tracking
    running_tasks[task_id] = {
        "task": task,
        "status": "running",
        "output": None,
        "error": None,
        "log_file": str(log_file),
        "started_at": started_at,
        "completed_at": None,
        "elapsed_seconds": None,
    }

    # Small delay before returning to allow task to initialize
    # This prevents potential issues with immediate Claude Desktop sync
    await asyncio.sleep(1.5)

    # Return immediately
    return {
        "task_id": task_id,
        "status": "running",
        "log_file": str(log_file),
        "started_at": started_at,
        "note": "Task is running in background. DO NOT poll immediately - wait at least 30 seconds before checking status. Most tasks take 30-120 seconds.",
    }


@mcp.tool()
async def get_agent_status(task_id: str) -> dict[str, Any]:
    """Get the status of an asynchronously spawned agent.

    Use this tool to check if an agent started with `spawn_agent_async` has
    completed, and to retrieve its results.

    Args:
        task_id: The task ID returned by spawn_agent_async

    Returns:
        dict with:
        - task_id: The task identifier
        - status: "running", "completed", or "failed"
        - log_file: Path to log file
        - started_at: ISO timestamp
        - completed_at: ISO timestamp (only if completed/failed)
        - elapsed_seconds: Runtime in seconds (only if completed/failed)
        - output: Agent's final response (only if status="completed")
        - error: Error details (only if status="failed")
        - warning: Warning message if polling too frequently
    """
    if task_id not in running_tasks:
        raise ToolExecutionError(
            code="task_not_found",
            kind=ErrorCategory.VALIDATION,
            message=f"Task ID '{task_id}' not found. Use list_agent_tasks to see available tasks.",
        )

    info = running_tasks[task_id]
    result: dict[str, Any] = {
        "task_id": task_id,
        "status": info["status"],
        "log_file": info["log_file"],
        "started_at": info["started_at"],
    }

    # Calculate elapsed time
    started_ts = datetime.fromisoformat(info["started_at"].replace("Z", "+00:00"))
    elapsed = (datetime.now(started_ts.tzinfo) - started_ts).total_seconds()

    # Warning if polling too early (task still running and started < 15 seconds ago)
    if info["status"] == "running" and elapsed < 15:
        result["warning"] = (
            f"Task started only {elapsed:.1f}s ago. Most Codex tasks take 30-120 seconds. "
            "Avoid frequent polling - wait at least 30 seconds between checks. "
            "You can read the log file to monitor progress instead."
        )

    result["elapsed_seconds"] = round(elapsed, 1)

    if info["completed_at"]:
        result["completed_at"] = info["completed_at"]
        result["elapsed_seconds"] = info["elapsed_seconds"]

    if info["status"] == "completed" and info["output"]:
        result["output"] = info["output"]

    if info["status"] == "failed" and info["error"]:
        result["error"] = info["error"]

    return result


@mcp.tool()
async def list_agent_tasks() -> dict[str, Any]:
    """List all agent tasks (running and completed).

    Returns:
        dict with:
        - total: Total number of tasks
        - running: Number of running tasks
        - completed: Number of completed tasks
        - failed: Number of failed tasks
        - tasks: List of task summaries with task_id, status, started_at
    """
    tasks_summary = []
    counts = {"running": 0, "completed": 0, "failed": 0}

    for task_id, info in running_tasks.items():
        status = info["status"]
        counts[status] = counts.get(status, 0) + 1

        task_info: dict[str, Any] = {
            "task_id": task_id,
            "status": status,
            "started_at": info["started_at"],
            "log_file": info["log_file"],
        }

        if info["completed_at"]:
            task_info["completed_at"] = info["completed_at"]
            task_info["elapsed_seconds"] = info["elapsed_seconds"]

        tasks_summary.append(task_info)

    return {
        "total": len(running_tasks),
        "running": counts["running"],
        "completed": counts["completed"],
        "failed": counts["failed"],
        "tasks": tasks_summary,
    }


@mcp.tool()
async def spawn_agents_parallel(
    ctx: Context,
    agents: list[dict[str, str]],
    max_parallel: int | None = None,
) -> list[dict[str, Any]]:
    """Spawn multiple Codex agents in parallel.

    Each spawned agent reuses the working directory resolved for the server:
    ``CODEX_AGENT_CWD`` if set, otherwise the current working directory
    (``os.getcwd()``). This allows the server to run from one place while
    agents edit a different workspace.

    Args:
        agents: List of agent specs, each with a 'prompt' entry and optional
            'reasoning_effort' and 'model' overrides.
                Example: [
                    {"prompt": "Create math.md"},
                    {"prompt": "Analyze binary", "reasoning_effort": "high"},
                    {"prompt": "Debug issue", "model": "gpt-5.1-codex-mini"}
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


@mcp.tool()
async def list_agent_logs(max_count: int = 20) -> dict[str, Any]:
    """List recent agent log files.

    Args:
        max_count: Maximum number of log files to return (default: 20)

    Returns:
        dict with:
        - total: Total number of log files found
        - logs: List of log file info (path, size, modified time)
        - log_dir: Path to log directory
    """
    try:
        log_root = _prepare_log_root()

        # Find all .log files (excluding rotated .log.N files for simplicity)
        log_files = sorted(
            log_root.glob("codex_agent_*.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True  # Most recent first
        )[:max_count]

        logs_info = []
        for log_file in log_files:
            stat = log_file.stat()
            logs_info.append({
                "path": str(log_file),
                "size_bytes": stat.st_size,
                "size_human": f"{stat.st_size / 1024:.1f} KB" if stat.st_size < 1024 * 1024 else f"{stat.st_size / (1024 * 1024):.1f} MB",
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })

        total_logs = len(list(log_root.glob("codex_agent_*.log")))

        return {
            "total": total_logs,
            "showing": len(logs_info),
            "log_dir": str(log_root),
            "logs": logs_info,
        }
    except Exception as e:
        raise ToolExecutionError(
            code="list_logs_failed",
            kind=ErrorCategory.IO,
            message="Failed to list agent logs",
            details={"error": str(e)},
        )


@mcp.tool()
async def cleanup_old_logs(days: int = 7, dry_run: bool = True) -> dict[str, Any]:
    """Delete agent log files older than specified number of days.

    Args:
        days: Delete logs older than this many days (default: 7)
        dry_run: If True, only report what would be deleted without deleting (default: True)

    Returns:
        dict with:
        - deleted_count: Number of files deleted (or would be deleted)
        - freed_bytes: Bytes freed (or would be freed)
        - files: List of deleted file paths
        - dry_run: Whether this was a dry run
    """
    if days < 1:
        raise ToolExecutionError(
            code="invalid_days",
            kind=ErrorCategory.VALIDATION,
            message="'days' must be at least 1",
        )

    try:
        log_root = _prepare_log_root()
        cutoff_time = time.time() - (days * 24 * 60 * 60)

        # Find all old log files (including rotated ones)
        old_logs = [
            p for p in log_root.glob("codex_agent_*")
            if p.is_file() and p.stat().st_mtime < cutoff_time
        ]

        deleted_files = []
        freed_bytes = 0

        for log_file in old_logs:
            size = log_file.stat().st_size
            freed_bytes += size
            deleted_files.append(str(log_file))

            if not dry_run:
                log_file.unlink()

        return {
            "deleted_count": len(deleted_files),
            "freed_bytes": freed_bytes,
            "freed_human": f"{freed_bytes / 1024:.1f} KB" if freed_bytes < 1024 * 1024 else f"{freed_bytes / (1024 * 1024):.1f} MB",
            "files": deleted_files,
            "dry_run": dry_run,
            "message": f"{'Would delete' if dry_run else 'Deleted'} {len(deleted_files)} log files older than {days} days",
        }
    except Exception as e:
        raise ToolExecutionError(
            code="cleanup_failed",
            kind=ErrorCategory.IO,
            message="Failed to cleanup old logs",
            details={"error": str(e)},
        )


def main() -> None:
    """Entry point for the MCP server v2."""
    mcp.run()


if __name__ == "__main__":
    main()
