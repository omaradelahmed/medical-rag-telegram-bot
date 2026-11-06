"""End-to-end MCP test: start medical_server.py over stdio exactly like agent.py
does, list the tools, and call the new ones (no LLM, no embeddings needed).
Run:  uv run test_mcp.py
"""
import asyncio
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def show(name, r):
    txt = r.content[0].text if r.content else ""
    print(f"\n=== {name}{'  [ERROR]' if r.is_error else ''}")
    print(txt[:600])


async def main():
    async with AsyncExitStack() as stack:
        reader, writer = await stack.enter_async_context(stdio_client(
            StdioServerParameters(command="uv", args=["run", "medical_server.py"])))
        session = await stack.enter_async_context(ClientSession(reader, writer))
        await session.initialize()

        tools = await session.list_tools()
        print("TOOLS:", [t.name for t in tools.tools])

        show("calc_gfr(creatinine 1.0, 65, female)",
             await session.call_tool("calc_gfr",
                                     {"creatinine_mg_dl": 1.0, "age": 65, "sex": "female"}))
        show("calc_bmi(80, 175)",
             await session.call_tool("calc_bmi", {"weight_kg": 80, "height_cm": 175}))
        show("calc_chadsvat(80F, all risk factors)",
             await session.call_tool("calc_chadsvat",
                                     {"age": 80, "hypertension": True, "diabetes": True,
                                      "heart_failure": True,
                                      "stroke_tia_or_thromboembolism": False,
                                      "prior_mh_or_atherosclerosis": False, "female": True}))
        show("get_page(2nd indexed source, p.1)",
             await session.call_tool("get_page", {"source": "Study_Smarter_IM_Paper1.pdf", "page": 1}))
        show("get_page bad source",
             await session.call_tool("get_page", {"source": "nope.pdf", "page": 1}))

        src = tools.tools  # keep linters quiet
        print("\nOK - all tools callable over stdio")


asyncio.run(main())
