# codex-as-mcp

让你的 Claude Code / Cursor 等工具调用 Codex 执行任务，Plus/Pro/Team 订阅用户不需要额外支付费用，最大化利用 GPT-5。

## 使用方法

### 1. 安装 Codex CLI
```bash
npm install -g @openai/codex
codex login
```

### 2. 配置 MCP

在 `.mcp.json` 中添加：
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

或者在 Claude Code 中使用命令：
```bash
claude mcp add codex-as-mcp -- uvx codex-as-mcp@latest
```

### 3. 开始使用

MCP 工具会暴露 `codex_execute(prompt, work_dir)` 函数，让你的 AI 助手可以调用 Codex 执行代码任务。

## ⚠️ 风险提示

Codex 在该 MCP 下处于 YOLO 模式，可能存在环境风险。请在测试环境或有版本控制的项目中使用。

