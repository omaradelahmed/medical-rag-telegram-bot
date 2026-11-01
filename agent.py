"""Step 5: medical agent = MCP server (search + prompt) + LLM (see llm.py).
CLI test:  uv run agent.py
The MedicalAgent class is reused by the Telegram bot in the next step.

Flow (the code always searches first, the LLM only writes the answer):
  question -> [Arabic? glossary / translate to English] -> MCP tool search_documents
           -> NO_RELEVANT_CONTEXT? -> "I don't know."
           -> MCP prompt medical_answer(question, context) -> LLM -> answer + sources
"""
import asyncio
import re
import time
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import llm
from ar_terms import arabic_to_query

NO_ANSWER = "I don't know."
NO_CONTEXT = "NO_RELEVANT_CONTEXT"
DISCLAIMER = "This is general information and does not replace consulting a doctor."
SOURCE_RE = re.compile(r"\[Source: (.+?), page (\d+)")
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
LEADING_JUNK_RE = re.compile(r"^[\W_]+")  # bullets, dashes, quotes before the first word
IDK_PREFIXES = ("i don't know", "i don\u2019t know", "i do not know")


class MedicalAgent:
    def __init__(self):
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None

    async def start(self):
        params = StdioServerParameters(command="uv", args=["run", "medical_server.py"])
        reader, writer = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(ClientSession(reader, writer))
        await self._session.initialize()
        await llm.warm_up()
        await self._session.call_tool("search_documents", {"query": "hypertension"})  # load bge-m3

    async def close(self):
        await self._stack.aclose()

    async def _to_english(self, question: str) -> str:
        """English passes through. Arabic: glossary first (reliable), LLM translation as fallback."""
        if not ARABIC_RE.search(question):
            return question
        glossary_query = arabic_to_query(question)
        if glossary_query:
            return glossary_query
        translated = await llm.generate(
            "Translate this medical question into English. "
            f"Reply with the English translation only.\n\n{question}",
            max_tokens=400,
        )
        return translated.strip("\"'") or question

    async def ask(self, question: str, verbose: bool = False) -> str:
        t0 = time.perf_counter()
        english_q = await self._to_english(question)

        # 1) search through the MCP tool (always done by the code, never skipped)
        result = await self._session.call_tool("search_documents", {"query": english_q})
        text = result.content[0].text if result.content else ""
        if verbose:
            status = "NO_CONTEXT" if text.strip() == NO_CONTEXT else f"{len(text)} chars"
            print(f"  [search] query={english_q!r} -> {status}")
        if result.is_error or not text.strip() or text.strip() == NO_CONTEXT:
            return NO_ANSWER

        sources: list[tuple[str, str]] = []
        for f, p in SOURCE_RE.findall(text):
            if (f, p) not in sources:
                sources.append((f, p))

        # 2) one LLM call, using the MCP prompt template
        prompt = await self._session.get_prompt(
            "medical_answer", {"question": english_q, "context": text}
        )
        answer = await llm.generate(prompt.messages[0].content.text)
        if verbose:
            print(f"  [time] {time.perf_counter() - t0:.1f}s")

        cleaned = LEADING_JUNK_RE.sub("", answer).lower()  # "- I don't know" must count too
        if not cleaned or cleaned.startswith(IDK_PREFIXES):
            return NO_ANSWER

        refs = "\n".join(f"- {f}, p.{p}" for f, p in sources)
        return f"{answer}\n\nSources:\n{refs}\n\n{DISCLAIMER}"


async def main():
    agent = MedicalAgent()
    print(f"LLM backend: {llm.describe()}")
    print("Starting ...")
    await agent.start()
    print("Medical agent ready.")
    try:
        while True:
            q = input("\nQuestion (q to quit): ").strip()
            if q.lower() in ("q", "quit"):
                break
            if not q:
                continue
            try:
                print("\n" + await agent.ask(q, verbose=True))
            except Exception as e:  # API down, rate limit, ...
                print(f"\nError: {type(e).__name__}: {e}")
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
