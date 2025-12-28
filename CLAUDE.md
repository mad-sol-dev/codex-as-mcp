# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MCP (Model Context Protocol) server that exposes Codex CLI as tools. It allows Claude Desktop and other MCP clients to spawn autonomous Codex subagents that run `codex e --full-auto` with complete autonomy.

## Dependency Management

- Use `uv` for all dependency management and command execution
- Run commands with `uv run ...` (e.g., `uv run python -m codex_as_mcp`)
- Python 3.11+ required

## Key Commands

### Development & Testing

```bash
# Run the MCP server locally
uv run python -m codex_as_mcp

# Start MCP Inspector with local server (recommended for testing)
./test.sh

# Start Inspector only (manual server configuration)
./test.sh --ui-only

# Cleanup ports if Inspector is stuck
./test.sh --cleanup-ports [--yes]

# Run tests
uv run pytest
```

### Building & Publishing

```bash
# Source environment variables for PyPI credentials
source ~/.bashrc  # or ~/.zshrc

# Update version in pyproject.toml first (format: YYYY.MM.DD.N)

# Build package
rm -rf dist/
uv build

# Publish to PyPI
twine upload dist/* --username "$PYPI_USERNAME" --password "$PYPI_TOKEN" --non-interactive

# Verify: https://pypi.org/project/codex-as-mcp/
```

## Architecture

### Core Components

- **`src/codex_as_mcp/server.py`**: Main MCP server implementation using FastMCP
  - Exposes 7 tools: `spawn_agent`, `spawn_agent_async`, `get_agent_status`, `list_agent_tasks`, `spawn_agents_parallel`, `list_agent_logs`, `cleanup_old_logs`
  - Manages Codex CLI subprocess execution with stdout/stderr capture
  - Implements log rotation and persistent logging to `~/.cache/codex-as-mcp/logs`
  - Progress reporting via MCP context with heartbeats every 2 seconds
  - Async task tracking for background agent execution
  - **Output size protection**: Automatically truncates responses >4000 chars to prevent Claude Desktop sync issues

- **`src/codex_as_mcp/__main__.py`**: Entry point for `python -m codex_as_mcp`

### MCP Tools

#### 1. `spawn_agent` (Blocking) - For Claude Code CLI

**⚠️ NOT recommended for Claude Desktop** (60-second timeout limitation)

```python
spawn_agent(
    prompt: str,
    reasoning_effort: str | None = None,
    model: str | None = None,
    agent_index: int | None = None,
    agent_count: int | None = None
) -> str
```

- Spawns a single Codex agent with `codex e --full-auto` (BLOCKING - waits for completion)
- Runs in server's current working directory (override with `CODEX_AGENT_CWD`)
- Returns agent's final message and log file path
- Default timeout: 8 hours (28800 seconds) - configurable via `CODEX_TIMEOUT_SECONDS` env var
- **Parameters:**
  - `prompt`: Instructions for the agent
  - `reasoning_effort`: Optional. "low", "medium", "high", or "xhigh"
  - `model`: Optional. Override Codex model (e.g., "gpt-5.2-codex", "gpt-5.1-codex-mini")
  - `agent_index`, `agent_count`: For parallel execution tracking
- Command: `codex e --cd <work_dir> --skip-git-repo-check --full-auto [--model <model>] [-c model_reasoning_effort=<level>] --output-last-message <temp_file> "<prompt>"`

#### 2. `spawn_agent_async` (Non-blocking) - ✅ Recommended for Claude Desktop

```python
spawn_agent_async(
    prompt: str,
    reasoning_effort: str | None = None,
    model: str | None = None
) -> dict
```

- Spawns a Codex agent in the background and returns IMMEDIATELY
- **Avoids Claude Desktop's 60-second timeout** - tool call returns within milliseconds
- Returns: `{"task_id": "...", "status": "running", "log_file": "...", "started_at": "..."}`
- Automatically enhances prompt to prevent agent from waiting for user input
- Agent continues running in background; use `get_agent_status(task_id)` to check progress

**Workflow:**
```python
# 1. Start agent (returns immediately)
result = spawn_agent_async("Scan repository for sensitive data")
# Returns: {"task_id": "codex_20251228_123456_abc123", "status": "running", "log_file": "/home/.../.log", ...}

# 2. Check status (can be called multiple times)
status = get_agent_status(result["task_id"])
# Returns: {"status": "running|completed|failed", "output": "..." (if completed), ...}

# 3. Optionally read log file while running
Read(result["log_file"])  # See real-time progress
```

#### 3. `get_agent_status` - Check async agent status

```python
get_agent_status(task_id: str) -> dict
```

- Get status of an agent started with `spawn_agent_async`
- Returns:
  - `status`: "running", "completed", or "failed"
  - `output`: Agent's final response (if completed)
  - `error`: Error details (if failed)
  - `log_file`: Path to log file
  - `elapsed_seconds`: Runtime (if completed/failed)

#### 4. `list_agent_tasks` - List all tracked tasks

```python
list_agent_tasks() -> dict
```

- Returns summary of all agent tasks (running and completed)
- Useful for debugging and monitoring multiple agents

#### 5. `spawn_agents_parallel` - Run multiple agents concurrently

```python
spawn_agents_parallel(
    agents: list[dict],
    max_parallel: int | None = None
) -> list[dict]
```

- Spawns multiple Codex agents concurrently using `asyncio.gather`
- Optional `max_parallel` caps simultaneous agents via `asyncio.Semaphore` (default `None` = no limit)
- Each agent spec is a dict with `prompt` and optional `reasoning_effort`, `model`
- Returns list of results with `index`, `output`/`error`, and `log_file` fields
- **Example:**
  ```json
  [
    {"prompt": "Create math.md"},
    {"prompt": "Analyze binary", "reasoning_effort": "high"},
    {"prompt": "Debug issue", "model": "gpt-5.1-codex-mini"}
  ]
  ```

#### 6. `list_agent_logs` - List recent log files

```python
list_agent_logs(max_count: int = 20) -> dict
```

- Lists recent agent log files sorted by modification time (most recent first)
- Returns log file paths, sizes, and modification timestamps
- Useful for finding logs to inspect or cleanup
- **Returns:**
  - `total`: Total number of log files
  - `showing`: Number of logs returned
  - `log_dir`: Path to log directory
  - `logs`: List of log file info (path, size_bytes, size_human, modified)

#### 7. `cleanup_old_logs` - Delete old log files

```python
cleanup_old_logs(days: int = 7, dry_run: bool = True) -> dict
```

- Deletes agent log files older than specified number of days
- **Default is dry_run=True** - only reports what would be deleted without deleting
- Set `dry_run=False` to actually delete files
- Returns count of deleted files and bytes freed
- **Parameters:**
  - `days`: Delete logs older than this many days (default: 7, minimum: 1)
  - `dry_run`: If True, only report what would be deleted (default: True)
- **Returns:**
  - `deleted_count`: Number of files deleted (or would be deleted)
  - `freed_bytes` / `freed_human`: Bytes freed
  - `files`: List of deleted file paths
  - `dry_run`: Whether this was a dry run

### Logging System

- Logs stored in `~/.cache/codex-as-mcp/logs` (override with `CODEX_AS_MCP_LOG_DIR` env var)
- Log rotation at 512KB (`LOG_ROTATE_BYTES`)
- Filename format: `codex_agent_<timestamp>_<pid>_[agent_index]_<uuid>.log`
- Temporary output files cleaned up after agent completes
- Recent 50 lines kept in memory for error reporting

#### Log Format

**Human-Readable Text Format** (default for streaming output):
```
[2025-12-15 22:00:00] INFO    stdout: Analyzing codebase...
[2025-12-15 22:00:15] INFO    stdout: Found 42 files
[2025-12-15 22:00:30] INFO    progress: [agent] 30s elapsed; stdout=12; stderr=5
[2025-12-15 22:00:45] INFO    stdout: Creating analysis report
```

- **stdout/stderr**: Simple text format for real-time monitoring with `tail -f`
- **Progress updates**: Only logged when output changes OR every 10 seconds (reduces repetitiveness)
- **Important events**: JSON format for start, completion, errors, and warnings

**JSON Format** (for important events only):
```json
{"ts": "2025-12-15T22:00:00.000Z", "level": "info", "event": "agent_start", "message": "Launching Codex agent", "context": {...}}
{"ts": "2025-12-15T22:00:45.000Z", "level": "error", "event": "timeout", "message": "Codex agent timed out", "context": {...}}
```

This dual-format approach makes logs:
- **Readable**: Easy to monitor with `tail -f ~/.cache/codex-as-mcp/logs/codex_agent_*.log`
- **Compact**: Only logs changes, not repetitive progress updates
- **Structured**: JSON for machine-readable events when needed

### Output Size Protection

**Problem:** Claude Desktop has response size limits that cause `message_store_sync_loss` errors when exceeded, leading to conversation deletion.

**Solution:** Automatic output truncation to prevent sync issues.

- **Maximum output length**: 4000 characters
- **Behavior**: If Codex agent output exceeds 4000 chars, it's automatically truncated
- **User notification**: Truncation message shows original length and directs to log file
- **Log file**: Full untruncated output always available in log file
- **Applies to**: All tool responses (`spawn_agent`, `spawn_agent_async` results, fallback outputs)

**Example truncated response:**
```
Created analysis report with 50 findings...
[... output continues ...]

[Output truncated: 8542 chars total, showing first 4000. See full output in log file]
Log file: /home/user/.cache/codex-as-mcp/logs/codex_agent_20251228_151347_564689_bdf52cd9.log
```

**Recommendation:** For tasks that produce large outputs, use `spawn_agent_async` and read the log file directly with the Read tool while the task runs, or after completion.

### Error Handling

- Tool failures raise `ToolExecutionError`; FastMCP surfaces these as `CallToolResult.isError=true` with JSON text in the content block.
- Payload shape:
  ```json
  {
    "code": "agent_timeout",
    "kind": "timeout_error",
    "message": "Codex agent timed out after 300 seconds.",
    "log_file": "/home/user/.cache/codex-as-mcp/logs/codex_agent_*.log",
    "details": { "timeout_seconds": 300, "elapsed_seconds": 301.2 }
  }
  ```
- Error categories (`kind`): `validation_error`, `runtime_error`, `timeout_error`, `io_error`.
- Common codes: `invalid_prompt_type`, `empty_prompt`, `codex_not_found`, `process_start_failed`, `agent_timeout`, `agent_non_zero_exit`, `output_missing`, `output_read_failed`.
- `spawn_agents_parallel` aggregates per-agent errors using the same payload structure and still includes a `log_file` entry when available. Batch-level validation issues (empty list, invalid `max_parallel`) raise `ToolExecutionError`.

### Progress Reporting

- Initial ping when launching agent
- Heartbeat every 2 seconds during execution
- Reports: elapsed time, stdout/stderr line counts, bytes processed, last non-empty line
- Uses MCP context's `report_progress(index, total, message)` method

### Configuration

#### Tool Parameters (Recommended)

The MCP tools support per-call configuration via parameters. This is the **recommended approach** as it allows the calling agent (e.g., Claude Desktop) to choose appropriate settings per task.

**`reasoning_effort` Parameter:**
- **"low"**: Fast, simple tasks (file operations, basic edits, grep searches)
- **"medium"**: Balanced approach (most general-purpose tasks) - default if not specified
- **"high"**: Complex analysis (architecture design, debugging, code review)
- **"xhigh"**: Very complex (reverse engineering, binary analysis, security research)

**Note:** Higher reasoning effort increases latency and cost. Only use "high" or "xhigh" when task complexity justifies it.

**`model` Parameter:**
Examples: `"gpt-5.2-codex"`, `"gpt-5.1-codex-max"`, `"gpt-5.1-codex-mini"`

**Available Models:**
- `gpt-5.2-codex` - Latest, most advanced model
- `gpt-5.1-codex-max` - Based on improved foundational reasoning model
- `gpt-5.1-codex-mini` - Smaller, cost-effective (~4x more usage)
- `gpt-5.2`, `gpt-5.1`, `gpt-5.1-codex`, `gpt-5-codex`, `gpt-5-codex-mini`, `gpt-5` - Alternative models

**Note:** Models like `o3-mini` or `o1-preview` are NOT supported in Codex CLI. Use the gpt-5.x-codex variants instead.

**Example MCP tool call:**
```python
await mcp.call_tool("spawn_agent", {
    "prompt": "Analyze this binary for SD card write operations",
    "reasoning_effort": "high",
    "model": "gpt-5.2-codex"
})
```

#### Environment Variables (Fallback)

Environment variables serve as defaults when tool parameters are not specified:

**`CODEX_REASONING_EFFORT`**
Default reasoning effort level. Overridden by tool parameter.

**`CODEX_MODEL`**
Default model. Overridden by tool parameter.

**`CODEX_TIMEOUT_SECONDS`**
Maximum seconds for agent execution. Default: 28800 (8 hours).
**Warning:** LLMs cannot reliably estimate task duration. Only change this if you have specific requirements.

**`CODEX_AS_MCP_LOG_DIR`**
Override the default log directory (`~/.cache/codex-as-mcp/logs`).

**Example `.mcp.json` with env vars:**
```json
{
  "mcpServers": {
    "codex": {
      "command": "uv",
      "args": ["run", "python", "-m", "codex_as_mcp"],
      "env": {
        "CODEX_REASONING_EFFORT": "medium",
        "CODEX_MODEL": "gpt-5.1-codex-max"
      }
    }
  }
}
```

## Client Compatibility & Timeouts

### ⚠️ Important: Claude Desktop Limitation

**Claude Desktop has a fixed 60-second timeout** for MCP tool calls that **cannot be configured** and **does not reset on progress updates**. This means:

- ❌ Tasks longer than 60 seconds will fail with `MCP error -32001: Request timed out`
- ❌ Progress updates do NOT reset the timeout (known limitation)
- ❌ No configuration workaround available with blocking tools

### ✅ Solution: Use `spawn_agent_async` with Claude Desktop

**For Claude Desktop users, use the async workflow:**

```python
# 1. Start agent (returns immediately, no timeout)
result = spawn_agent_async("Long-running task...")
# Returns in <1 second with task_id

# 2. Poll for status (each call is <1 second)
status = get_agent_status(result["task_id"])
# Can call this multiple times until status == "completed"

# 3. Read log file to monitor progress
Read(result["log_file"])  # See what the agent is doing
```

**Benefits:**
- ✅ No 60-second timeout issues (each tool call returns quickly)
- ✅ Can monitor progress via log file
- ✅ Can check status as frequently as needed
- ✅ Agent runs autonomously in background

**For Claude Code CLI users:**
- ✅ Can use either `spawn_agent` (blocking) or `spawn_agent_async`
- ✅ Use `./test.sh` for testing (properly configured timeouts)
- ✅ Use MCP Inspector for development (supports timeout configuration)

### Claude Code CLI (Recommended for Long Tasks)

Claude Code CLI properly supports:
- ✅ Configurable timeouts via `MCP_TIMEOUT` and `MCP_TOOL_TIMEOUT`
- ✅ Timeout reset on progress updates (`MCP_REQUEST_TIMEOUT_RESET_ON_PROGRESS=true`)
- ✅ Long-running tasks up to 8 hours

Configure in `~/.claude/settings.json`:
```json
{
  "env": {
    "MCP_TIMEOUT": "60000",
    "MCP_TOOL_TIMEOUT": "28800000",
    "MCP_REQUEST_TIMEOUT_RESET_ON_PROGRESS": "true"
  }
}
```

### Related Issues

This is a known limitation tracked in:
- [Issue #470: resetTimeoutOnProgress](https://github.com/anthropics/claude-code/issues/470)
- [Issue #424: MCP Timeout needs to be configurable](https://github.com/anthropics/claude-code/issues/424)
- [Issue #5221: Make MCP tool timeouts configurable](https://github.com/anthropics/claude-code/issues/5221)

## Testing with MCP Inspector

Inspector runs at `http://localhost:6274` with proxy on port 6277.

### Configuration for Local Development

- **Transport**: STDIO
- **Command**: `uv`
- **Arguments**: `["run", "python", "-m", "codex_as_mcp"]`
- **Working Directory**: Repository root

### Timeout Configuration

For long-running agents, the `test.sh` script exports:
- `MCP_SERVER_REQUEST_TIMEOUT=300000` (5 minutes)
- `MCP_REQUEST_TIMEOUT_RESET_ON_PROGRESS=true` (resets on progress)
- `MCP_REQUEST_MAX_TOTAL_TIMEOUT=28800000` (8 hours max)

## Requirements

- Codex CLI >= 0.46.0 installed globally (`npm install -g @openai/codex@latest`)
- Codex must be in PATH and authenticated (`codex login`)
- Server validates Codex executable exists using `shutil.which("codex")`

## Package Metadata

- Package name: `codex-as-mcp`
- Entry point: `codex-as-mcp` command (runs `codex_as_mcp.server:main`)
- Build system: `hatchling`
- Main dependency: `mcp[cli]>=1.12.4`
- Dev dependencies: `twine`, `pytest`, `pytest-asyncio`
