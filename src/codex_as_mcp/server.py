from mcp.server.fastmcp import FastMCP, Context
import subprocess
import re
from typing import List, Dict, Optional, Sequence

mcp = FastMCP("codex-as-mcp")

HEADER_RE = re.compile(
    r'^'
    r'\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\]'   # 1: timestamp
    r'\s+'
    r'([^\n]+)'                                    # 2: tag (整行，允许包含空格/冒号)
    r'\n',
    flags=re.M
)

BLOCK_RE = re.compile(
    r'^'
    r'\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\]\s+([^\n]+)\n'  # 1: ts, 2: tag
    r'(.*?)'                                                   # 3: body
    r'(?=^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\]\s+[^\n]+\n|\Z)',
    flags=re.M | re.S
)

def run_and_extract_codex_blocks(
    cmd: Sequence[str],
    tags: Optional[Sequence[str]] = ("codex",),
    last_n: int = 1
) -> List[Dict[str, str]]:
    """
    运行命令并抽取日志块。每个块由形如
    [YYYY-MM-DDTHH:MM:SS] <tag>
    <正文...直到下一个时间戳头或文件结束>
    组成。

    :param cmd: 完整命令（列表形式）
    :param tags: 需要过滤的 tag 列表（大小写不敏感）。None 表示不过滤。
    :param last_n: 返回最后 N 个匹配块
    :return: [{timestamp, tag, body, raw}] 按时间顺序（旧->新）
    """
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    out = proc.stdout

    blocks = []
    for m in BLOCK_RE.finditer(out):
        ts, tag, body = m.group(1), m.group(2).strip(), m.group(3)
        if tags is None or tag.lower() in {t.lower() for t in tags}:
            raw = f'[{ts}] {tag}\n{body}'
            blocks.append({"timestamp": ts, "tag": tag, "body": body, "raw": raw})

    # 只取最后 1 个
    return blocks[-last_n:]


@mcp.tool()
async def codex_execute(prompt: str, work_dir: str, ctx: Context) -> str:
    """
    Execute prompt using codex for general purpose.

    Args:
        prompt (str): The prompt for codex
        work_dir (str): The working directory, e.g. /Users/kevin/Projects/demo_project
        ctx (Context): MCP context for logging
    """
    cmd = [
        "codex", "exec",
        "--full-auto", "--skip-git-repo-check",
        "--cd", work_dir,
        prompt,
    ]
    return run_and_extract_codex_blocks(cmd)[-1]["raw"]


def main():
    """Entry point for the MCP server"""
    import asyncio
    from mcp.server.stdio import stdio_server
    
    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await mcp.run(read_stream, write_stream, mcp.create_initialization_options())
    
    asyncio.run(run())


if __name__ == "__main__":
    main()
