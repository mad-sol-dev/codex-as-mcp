# codex-as-mcp

让 Claude Code、Cursor 等 AI 工具调用 Codex 执行任务。Plus/Pro/Team 订阅用户可在不增加额外费用的情况下最大化使用 GPT-5。

## 安装与配置

### 1. 安装 Codex CLI
```bash
npm install -g @openai/codex
codex login
```

### 2. 配置 MCP

在 `.mcp.json` 中添加：
【安全模式（默认）】
```json
{
  "mcpServers": {
    "codex": {
      "type": "stdio",
      "command": "uvx",
      "args": ["codex-as-mcp@latest"]
    }
  }
}
```

【可写模式】
```json
{
  "mcpServers": {
    "codex": {
      "type": "stdio",
      "command": "uvx",
      "args": ["codex-as-mcp@latest", "--yolo"]
    }
  }
}
```

或者使用 Claude Code 命令：
```bash
# 安全模式（默认）
claude mcp add codex-as-mcp -- uvx codex-as-mcp@latest

# 可写模式
claude mcp add codex-as-mcp -- uvx codex-as-mcp@latest --yolo
```

## 工具

MCP 服务器暴露两个工具：
- `codex_execute(prompt, work_dir)`：通用的 Codex 执行
- `codex_review(review_type, work_dir, target?, prompt?)`：专项代码审查

如有其他使用场景需求，欢迎提交 issue。

## 安全性

- 安全模式：默认只读操作，保护你的环境
- 可写模式：需要完整能力时使用 `--yolo` 标志
- 顺序执行：避免多代理并行操作产生冲突
