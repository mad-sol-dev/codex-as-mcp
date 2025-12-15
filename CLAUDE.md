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
  - Exposes two tools: `spawn_agent` and `spawn_agents_parallel`
  - Manages Codex CLI subprocess execution with stdout/stderr capture
  - Implements log rotation and persistent logging to `~/.cache/codex-as-mcp/logs`
  - Progress reporting via MCP context with heartbeats every 2 seconds

- **`src/codex_as_mcp/__main__.py`**: Entry point for `python -m codex_as_mcp`

### MCP Tools

1. **`spawn_agent(prompt: str, reasoning_effort: str | None, model: str | None, agent_index: int | None, agent_count: int | None)`**
   - Spawns a single Codex agent with `codex e --full-auto`
   - Runs in server's current working directory (`os.getcwd()`)
   - Returns agent's final message and log file path
   - Default timeout: 8 hours (28800 seconds) - configurable via `CODEX_TIMEOUT_SECONDS` env var
   - **Parameters:**
     - `prompt`: Instructions for the agent
     - `reasoning_effort`: Optional. "low", "medium", "high", or "xhigh" (see recommendations below)
     - `model`: Optional. Override Codex model (e.g., "o3-mini", "o1-preview")
   - Command: `codex e --cd <work_dir> --skip-git-repo-check --full-auto [--model <model>] [-c model_reasoning_effort=<level>] --output-last-message <temp_file> "<prompt>"`

2. **`spawn_agents_parallel(agents: list[dict], max_parallel: int | None = None)`**
   - Spawns multiple Codex agents concurrently using `asyncio.gather`
   - Optional `max_parallel` caps simultaneous agents via `asyncio.Semaphore` (default `None` = no limit)
   - Each agent spec is a dict with `prompt` and optional `reasoning_effort`, `model`
   - Returns list of results with `index`, `output`/`error`, and `log_file` fields
   - **Example:**
     ```json
     [
       {"prompt": "Create math.md"},
       {"prompt": "Analyze binary", "reasoning_effort": "high"},
       {"prompt": "Debug issue", "model": "o3-mini"}
     ]
     ```

### Logging System

- Logs stored in `~/.cache/codex-as-mcp/logs` (override with `CODEX_AS_MCP_LOG_DIR` env var)
- Log rotation at 512KB (`LOG_ROTATE_BYTES`)
- Filename format: `codex_agent_<timestamp>_<pid>_[agent_index]_<uuid>.log`
- Temporary output files cleaned up after agent completes
- Recent 50 lines kept in memory for error reporting

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
Examples: `"o3-mini"`, `"o1-preview"`, `"gpt-5.1-codex-max"`

Use when specific model capabilities are needed (e.g., o3 models for extended reasoning).

**Example MCP tool call:**
```python
await mcp.call_tool("spawn_agent", {
    "prompt": "Analyze this binary for SD card write operations",
    "reasoning_effort": "high",
    "model": "gpt-5.1-codex-max"
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
