# codex-as-mcp

[中文版](./README.cn.md)

Enable Claude Code, Cursor and other AI tools to call Codex for task execution. Plus/Pro/Team subscribers can maximize GPT-5 usage without additional costs.

## Execution Modes

🔒 **Safe Mode (Default)**: Read-only operations, no file modifications
⚡ **Writable Mode**: Full codex capabilities with file/git operations

### Why Sequential Execution?
Codex is an agent that modifies files and system state. Running multiple instances in parallel could cause file conflicts, git race conditions, and conflicting system modifications. Sequential execution prevents these issues.

## Setup

### 1. Install Codex CLI
```bash
npm install -g @openai/codex
codex login
```

### 2. Configure MCP

Add to your `.mcp.json`:
**Safe Mode (Default):**
```json
{
  "mcpServers": {
    "codex": {
      "type": "stdio",
      "command": "uvx",
      "args": ["codex-as-mcp@latest"],
      "env": {}
    }
  }
}
```

**Writable Mode:**
```json
{
  "mcpServers": {
    "codex": {
      "type": "stdio",
      "command": "uvx",
      "args": ["codex-as-mcp@latest", "--yolo"],
      "env": {}
    }
  }
}
```

Or use Claude Code commands:
```bash
# Safe mode (default)
claude mcp add codex-as-mcp -- uvx codex-as-mcp@latest

# Writable mode  
claude mcp add codex-as-mcp -- uvx codex-as-mcp@latest --yolo

# Show detailed mode explanation
uvx codex-as-mcp@latest --help-modes
```

### 3. Usage

The MCP server exposes two tools:
- `codex_execute(prompt, work_dir)` - General purpose codex execution
- `codex_review(review_type, work_dir, target?, prompt?)` - Specialized code review

## Safety

- **Safe Mode**: Default read-only operations protect your environment
- **Writable Mode**: Use `--yolo` flag when you need full codex capabilities
- **Sequential Execution**: Prevents conflicts from parallel agent operations