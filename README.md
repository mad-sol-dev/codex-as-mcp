# codex-as-mcp
The Actual Config Locations
Based on the documentation and community findings:

User scope (--scope user): Stored in ~/.claude.json
Project scope (--scope project): Stored in .mcp.json in the project root
Local scope (--scope local): Project-specific, private to you

The user scope is stored in ~/.claude.json and is recommended for tools you'll use regularly. Cloud Artisan
Your Actual Issue
Looking back at your error message, you had the server configured at:

/root/.claude.json (local config for project /mnt/e/A/az)
/mnt/e/A/az/.mcp.json (project config)

But you were in a different directory (/mnt/e/A/Carrace), so the project-level config wasn't being picked up.
The Fix
Try adding it with user scope so it works everywhere:
bashclaude mcp add codex-subagent --scope user -- uvx codex-as-mcp@latest
Then restart Claude Code and run /mcp to verify.

[中文版](./README.zh-CN.md)

**Spawn multiple subagents via Codex-as-MCP**

Each subagent runs `codex e --full-auto` with complete autonomy inside the MCP server's current working directory. Perfect for Plus/Pro/Team subscribers leveraging GPT-5 capabilities.

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

- `spawn_agent(prompt: str)` – Spawns an autonomous Codex subagent using the server's working directory and returns the agent's final message.
- `spawn_agents_parallel(agents: list[dict])` – Spawns multiple Codex subagents in parallel; each item must include a `prompt` key and results include either an `output` or an `error` per agent.

Codex stdout/stderr logs are persisted under `~/.cache/codex-as-mcp/logs` by default. Override the location by setting `CODEX_AS_MCP_LOG_DIR`, and use the `Log file:` line in tool responses to inspect the saved output.
