import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

QUERIES = [
    "What are the clinical features of chronic kidney disease?",
    "What is the capital of France?",
]


async def main():
    params = StdioServerParameters(command="uv", args=["run", "medical_server.py"])
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as s:
            await s.initialize()
            print("Tools:", [t.name for t in (await s.list_tools()).tools])

            res = await s.read_resource("resource://documents")
            print("Documents:", res.contents[0].text)

            for q in QUERIES:
                r = await s.call_tool("search_documents", {"query": q})
                print(f"\n=== {q}\n{r.content[0].text[:400]}")

            p = await s.get_prompt("medical_assistant", {"question": "ما هي أعراض الفشل الكلوي؟"})
            print("\n=== PROMPT (first 300 chars)\n", p.messages[0].content.text[:300])


if __name__ == "__main__":
    asyncio.run(main())
