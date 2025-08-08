# codex-as-mcp

[中文版](./README.cn.md)

Enable Claude Code, Cursor and other AI tools to call Codex for task execution. Plus/Pro/Team subscribers can maximize GPT-5 usage without additional costs.

## Setup

### 1. Install Codex CLI
```bash
npm install -g @openai/codex
codex login
```

### 2. Configure MCP

Add to your `.mcp.json`:
```json
{
  "mcpServers": {
    "codex": {
      "type": "stdio",
      "command": "uvx",
      "args": [
        "codex-as-mcp@latest"
      ],
      "env": {}
    }
  }
}
```

Or use Claude Code command:
```bash
claude mcp add codex-as-mcp -- uvx codex-as-mcp@latest
```

### 3. Usage

The MCP tool exposes `codex_execute(prompt, work_dir)` function, allowing your AI assistant to call Codex for code tasks.

## ⚠️ Warning

Codex operates in YOLO mode under this MCP, which may pose environment risks. Use in test environments or projects with version control.