import asyncio
import sys

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def test_lm_mcp_speaks_stdio_and_lists_the_lm_tools():
    async def go():
        params = StdioServerParameters(command=sys.executable, args=["-c", "import sys; from lm.cli import main; sys.exit(main(['mcp']))"])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = {t.name: t for t in (await session.list_tools()).tools}
                assert set(tools) == {"status", "load", "unload", "up", "down", "generate"}
                assert "model" in tools["load"].input_schema["required"]
    asyncio.run(asyncio.wait_for(go(), 30))


def test_tool_failures_carry_the_cli_message(monkeypatch):
    from lm import cli, mcp_server

    def fail(argv):
        print("拒绝：pika 需要先卸载 inkling")
        return 1
    monkeypatch.setattr(cli, "main", fail)
    try:
        mcp_server.load("pika")
    except mcp_server.ToolError as e:
        assert "先卸载 inkling" in str(e)
    else:
        raise AssertionError("expected a failure")
