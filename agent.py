"""Step 5: medical agent = MCP server (search + get_page + calculators) + LLM (see llm.py).
CLI test:  uv run agent.py
The MedicalAgent class is reused by the Telegram bot in the next step.

Flow (the code always searches first, the LLM only writes the answer):
  question -> [Arabic? glossary / translate to English] -> MCP tool search_documents
           -> NO_RELEVANT_CONTEXT? -> "I don't know."
           -> LLM (medical_answer prompt + the user's question) may call MCP tools:
              get_page (truncated tables) or calc_* (exact formula math, done by code)
           -> final answer + sources
"""
import asyncio
import json
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
SOURCE_RE = re.compile(r"\[Source: (.+?)[,](?: page|, page) (\d+)")
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
LEADING_JUNK_RE = re.compile(r"^[\W_]+")  # bullets, dashes, quotes before the first word
IDK_PREFIXES = ("i don't know", "i don\u2019t know", "i do not know")
MAX_TOOL_ROUNDS = 4   # one question can fetch pages / run at most this many tools

# The calculators and get_page are exposed to the LLM as OpenAI-style tools.
# search_documents is intentionally NOT here: the code always searches first.
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_page",
        "description": "Fetch the FULL text of one PDF page. Use when a search "
                       "result's table or list seems cut off.",
        "parameters": {"type": "object", "properties": {
            "source": {"type": "string",
                       "description": "File name after 'Source:' in search results"},
            "page": {"type": "integer", "description": "Page number, 1-based"}},
            "required": ["source", "page"]}}},
    {"type": "function", "function": {
        "name": "calc_gfr",
        "description": "eGFR (mL/min/1.73m2), CKD-EPI 2021, from serum creatinine "
                       "in mg/dL, age and sex - for kidney function questions.",
        "parameters": {"type": "object", "properties": {
            "creatinine_mg_dl": {"type": "number"}, "age": {"type": "integer"},
            "sex": {"enum": ["male", "female"]}},
            "required": ["creatinine_mg_dl", "age", "sex"]}}},
    {"type": "function", "function": {
        "name": "calc_bmi",
        "description": "BMI from weight (kg) and height (cm) with WHO category; "
                       "for weight/nutrition classification.",
        "parameters": {"type": "object", "properties": {
            "weight_kg": {"type": "number"}, "height_cm": {"type": "number"}},
            "required": ["weight_kg", "height_cm"]}}},
    {"type": "function", "function": {
        "name": "calc_bsa",
        "description": "Body surface area in m2 (Mosteller) from weight (kg) and "
                       "height (cm) - for dosing questions.",
        "parameters": {"type": "object", "properties": {
            "weight_kg": {"type": "number"}, "height_cm": {"type": "number"}},
            "required": ["weight_kg", "height_cm"]}}},
    {"type": "function", "function": {
        "name": "calc_chadsvat",
        "description": "CHA2DS2-VASc score + anticoagulation recommendation in "
                       "atrial fibrillation.",
        "parameters": {"type": "object", "properties": {
            "age": {"type": "integer"}, "hypertension": {"type": "boolean"},
            "diabetes": {"type": "boolean"}, "heart_failure": {"type": "boolean"},
            "stroke_tia_or_thromboembolism": {"type": "boolean"},
            "prior_mh_or_atherosclerosis": {"type": "boolean"},
            "female": {"type": "boolean"}},
            "required": ["age", "hypertension", "diabetes", "heart_failure",
                         "stroke_tia_or_thromboembolism", "prior_mh_or_atherosclerosis"]}}},
    {"type": "function", "function": {
        "name": "calc_crb65",
        "description": "CRB-65 score for pneumonia severity from confusion, "
                       "respiratory rate, blood pressure and age.",
        "parameters": {"type": "object", "properties": {
            "confusion": {"type": "boolean"}, "resp_rate_per_min": {"type": "integer"},
            "sbp_mmhg": {"type": "integer"}, "dbp_mmhg": {"type": "integer"},
            "age": {"type": "integer"}},
            "required": ["confusion", "resp_rate_per_min", "sbp_mmhg", "dbp_mmhg", "age"]}}},
]


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

        # 2) LLM with tools. The user message comes from the MCP prompt template
        #    (question + passages), so the rules stay in one place (medical_server.py).
        #    The LLM may request full pages (truncated tables) or run a calculator;
        #    the code computes - the LLM never does the math.
        prompt = await self._session.get_prompt(
            "medical_answer", {"question": english_q, "context": text})
        messages = [
            {"role": "system", "content": (
                "Tools note: tools are available but OPTIONAL - use them ONLY when "
                "they apply:\n"
                "- get_page: a passage's table or numbered list appears cut off "
                "mid-way; fetch the full page (source + page number are given in "
                "each passage header).\n"
                "- the calc_* tools: the question contains concrete numbers "
                "(creatinine, weight/height, BP, age, AF risk factors) and asks to "
                "compute a value; pass the numbers EXACTLY as the user gave them.\n"
                "After using a tool, incorporate its result into the final answer. "
                "Do not call a tool just because one is available.")
            },
            {"role": "user", "content": prompt.messages[0].content.text},
        ]
        answer = ""
        for _ in range(MAX_TOOL_ROUNDS):
            answer, calls = await llm.generate_chat(messages, TOOL_SCHEMAS)
            if not calls:
                break
            messages.append({"role": "assistant", "content": answer or None,
                             "tool_calls": [
                                 {"id": c["id"], "type": "function",
                                  "function": {"name": c["name"],
                                               "arguments": json.dumps(c["args"])}}
                                 for c in calls]})
            for c in calls:
                if verbose:
                    print(f"  [tool] {c['name']}({c['args']})")
                try:
                    r = await self._session.call_tool(c["name"], c["args"])
                    out = r.content[0].text if r.content else ""
                    if r.is_error:
                        out = f"ERROR: tool failed: {out}"
                except Exception as e:  # unknown tool, transport hiccup, ...
                    out = f"ERROR: {type(e).__name__}: {e}"
                if c["name"] == "get_page":
                    m = SOURCE_RE.search(out)
                    if m and (m.group(1), m.group(2)) not in sources:
                        sources.append((m.group(1), m.group(2)))
                if verbose:
                    print(f"         -> {out[:100]}")
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": out})
        else:
            # still calling tools after the last round: force a final answer
            # from whatever context we already have
            answer, _ = await llm.generate_chat(messages + [
                {"role": "system", "content": "Answer the question now. Do not call any tools."}
            ])

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
