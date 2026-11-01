"""Step 4: MCP server exposing the medical PDFs (via retriever.py).
Do NOT print() anything here: stdout is used by the stdio transport.
"""
import json
from collections import Counter

from mcp.server.mcpserver import MCPServer

import retriever

mcp = MCPServer("medical-server")

THRESHOLD = 0.5   # chunks scoring below this are considered irrelevant
TOP_K = 3         # fewer chunks = shorter prompt = faster answers
TEXT_LIMIT = 1200  # max characters returned per chunk
NO_CONTEXT = "NO_RELEVANT_CONTEXT"


@mcp.resource("resource://documents")
def documents() -> str:
    """The reference PDFs available for answering, with chunk counts."""
    counts = Counter(r["source"] for r in retriever._records)
    return json.dumps(counts, ensure_ascii=False)


@mcp.tool()
def search_documents(query: str) -> str:
    """Search the medical reference books (internal medicine) for passages
    relevant to a question.

    IMPORTANT: `query` MUST be in English. If the user's question is in another
    language, translate it to English first. Use medical keywords
    (e.g. "clinical features of chronic kidney disease").

    Returns passages with their source file and page, or NO_RELEVANT_CONTEXT
    if nothing relevant was found.
    """
    results = [r for r in retriever.search(query, top_k=TOP_K) if r["score"] >= THRESHOLD]
    if not results:
        return NO_CONTEXT
    parts = [
        f"[Source: {r['source']}, page {r['page']}, score {r['score']:.2f}]\n{r['text'][:TEXT_LIMIT]}"
        for r in results
    ]
    return "\n\n---\n\n".join(parts)


@mcp.prompt()
def medical_answer(question: str, context: str) -> str:
    """Prompt template: answer a medical question from the given passages only."""
    return f"""You are a medical information assistant. Answer the question using ONLY the reference passages below.

Rules:
1. Use the passages that address the question's topic; ignore passages about a different disease. The books may call "clinical features" the "clinical picture", and the text can be messy (list numbers stuck to words, like "lead9"): read through that.
2. NEVER add medical knowledge that is not in the passages.
3. Copy numbers, doses and units EXACTLY as written; if a unit looks unclear, write "check the source".
4. If the passages do not answer the question, reply exactly: I don't know.
5. Reply in English only, as a bullet list. Include ALL relevant points from the passages (do not drop items) and mention each point once.

Question: {question}

Passages:
{context}"""


@mcp.prompt()
def medical_assistant(question: str) -> str:
    """(Older template, tool-calling style. Not used by agent.py any more.)"""
    return f"""You are a medical information assistant. You answer ONLY from the reference documents, which you can access through the search_documents tool.

Rules:
1. The question may be in Arabic or English. Understand it, then call search_documents with an ENGLISH query (translate it if needed).
2. Answer ONLY with information found in the tool results. NEVER use your own medical knowledge.
3. If the tool returns NO_RELEVANT_CONTEXT, or the returned text does not actually answer the question, reply exactly: I don't know.
4. Always reply in English only, even if the question is in Arabic.
5. Be concise: use a short bullet list, mention each point once, no repetition.

Question: {question}"""


if __name__ == "__main__":
    mcp.run(transport="stdio")
