# codex-as-mcp

Minimal MCP server that lets you spawn Codex agents from any MCP client. Two tools are exposed:
- `spawn_agent` runs a single Codex agent in the server's working directory.
- `spawn_agents_parallel` launches multiple Codex agents concurrently in the same directory.

[中文版](./README.zh-CN.md)

**Use it in Claude Code**

There are two tools in codex-as-mcp
![tools](assets/tools.png)

You can spawn parallel codex subagents using prompt.
![alt text](assets/claude.png)

Here's a sample Codex session delegating two tasks in parallel.
![Codex use case](assets/codex.png)

## Setup

### 1. Install Codex CLI

**Requires Codex CLI >= 0.46.0**

```bash
npm install -g @openai/codex@latest
codex login

# Verify installation
codex --version
```

### 2. Configure MCP

Add to your `.mcp.json`:
```json
{
  "mcpServers": {
    "codex-subagent": {
      "type": "stdio",
      "command": "uvx",
      "args": ["codex-as-mcp@latest"]
    }
  }
}
```

Or use Claude Desktop commands:
```bash
claude mcp add codex-subagent -- uvx codex-as-mcp@latest
```

If you're configuring Codex CLI directly (for example `~/.config/codex/config.toml`), add:
```toml
[mcp_servers.subagents]
command = "uvx"
args = ["codex-as-mcp@latest"]
```

## Tools

- `spawn_agent(prompt: str, reasoning_effort?: str, model?: str)` – Spawns an autonomous Codex agent in the server's working directory and returns the final message. Optional `reasoning_effort` (`low`, `medium`, `high`, `xhigh`) and `model` override the Codex defaults.
- `spawn_agents_parallel(agents: list[dict])` – Runs multiple Codex agents in parallel. Each item must include a `prompt` and may include `reasoning_effort` and `model`. Results contain either an `output` or an `error` per agent plus the log path when available.

Codex stdout/stderr logs are persisted under `~/.cache/codex-as-mcp/logs` by default. Override the location by setting `CODEX_AS_MCP_LOG_DIR`, and use the `Log file:` line in tool responses to inspect the saved output.

## Run from a cloned repository

If you want to run the MCP server directly from a local checkout without installing the published package, use:

```bash
uv run python -m codex_as_mcp
```

This uses `uv` to resolve and install dependencies declared in `pyproject.toml` into an isolated environment. If you've already installed the dependencies into your active environment, you can invoke the module directly:

```bash
python -m codex_as_mcp
```

To point a `.mcp.json` configuration at your cloned source (using `uv` to handle dependencies), add:

```json
{
  "mcpServers": {
    "codex-subagent": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "python", "-m", "codex_as_mcp"]
    }
  }
}
```
