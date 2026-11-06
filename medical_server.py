"""Step 4: MCP server exposing the medical PDFs (via retriever.py).
Do NOT print() anything here: stdout is used by the stdio transport.
"""
import json
from collections import Counter

from mcp.server.mcpserver import MCPServer

import calc
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


@mcp.tool()
def get_page(source: str, page: int) -> str:
    """Fetch the FULL text of one PDF page (search chunks are limited to TEXT_LIMIT
    characters, so big tables and long lists can get cut off).

    Use it when a search result mentions a page and its passage seems truncated
    (a table or a list continues past the chunk). `source` is the file name shown
    after "Source:" in search results. Returns an ERROR message if the page does
    not exist.
    """
    try:
        text = retriever.get_page(source, page)
    except FileNotFoundError as e:
        return f"ERROR: {e}"
    except ValueError as e:
        return f"ERROR: {e}"
    if text is None:
        return f"ERROR: page {page} does not exist in {source}"
    return f"[Source: {source}, page {page} (full page)]\n{text}"


# ---------------------------------------------------------------------------
# Medical calculators (exact formula math, done by this code - never by the LLM)
# ---------------------------------------------------------------------------

def _result(func, *args) -> str:
    try:
        return func(*args)
    except TypeError as e:  # wrong argument names/types from the LLM
        return f"ERROR: bad arguments for this calculator: {e}"


@mcp.tool()
def calc_bmi(weight_kg: float, height_cm: float) -> str:
    """Calculate BMI from weight (kg) and height (cm), with WHO category."""
    return _result(calc.bmi, weight_kg, height_cm)


@mcp.tool()
def calc_bsa(weight_kg: float, height_cm: float) -> str:
    """Calculate body surface area in m2 (Mosteller formula) from weight (kg) and height (cm)."""
    return _result(calc.body_surface_area, weight_kg, height_cm)


@mcp.tool()
def calc_gfr(creatinine_mg_dl: float, age: int, sex: str) -> str:
    """Calculate estimated GFR (mL/min/1.73m2) with the CKD-EPI 2021 creatinine
    formula. Use when the question is about kidney function or eGFR for a patient
    with known serum creatinine (mg/dL), age and sex ('male' or 'female')."""
    return _result(calc.calc_gfr, creatinine_mg_dl, age, sex)


@mcp.tool()
def calc_chadsvat(age: int, hypertension: bool, diabetes: bool, heart_failure: bool,
                 stroke_tia_or_thromboembolism: bool, prior_mh_or_atherosclerosis: bool,
                 female: bool = False) -> str:
    """Calculate the CHA2DS2-VASc score (stroke risk in atrial fibrillation, with
    the sex-category point) and the usual anticoagulation recommendation."""
    return _result(calc.chadsvat_score, age, hypertension, diabetes, heart_failure,
                   stroke_tia_or_thromboembolism, prior_mh_or_atherosclerosis, female)


@mcp.tool()
def calc_crb65(confusion: bool, resp_rate_per_min: int, sbp_mmhg: int, dbp_mmhg: int, age: int) -> str:
    """Calculate the CRB-65 score for community-acquired pneumonia severity
    (CURB-65 without urea) from mental status, respiratory rate, blood pressure
    and age."""
    return _result(calc.crb65, confusion, resp_rate_per_min, sbp_mmhg, dbp_mmhg, age)


@mcp.prompt()
def medical_answer(question: str, context: str) -> str:
    """Prompt template: answer a medical question from the given passages only."""
    return f"""You are a medical information assistant. Answer the question using ONLY the reference passages below.

Tools: you may be given additional tool results after this prompt (e.g. a full PDF page
for a truncated table, or exact calculator output). Treat them as part of the reference
and quote calculator results VERBATIM (they are computed by code and are exact).

Rules:
1. Use the passages that address the question's topic; ignore passages about a different disease. The books may call "clinical features" the "clinical picture", and the text can be messy (list numbers stuck to words, like "lead9"): read through that.
2. NEVER add medical knowledge that is not in the passages.
3. Copy numbers, doses and units EXACTLY as written; if a unit looks unclear, write "check the source".
4. If the passages do not answer the question, reply exactly: I don't know.
5. Reply in English only, as a bullet list. Include ALL relevant points from the passages (do not drop items) and mention each point once.
6. If a tool result contains the answer to a computation, put it in the answer with the formula name (e.g. "eGFR (CKD-EPI): 63 mL/min/1.73m2").

Question: {question}

Passages:
{context}"""


if __name__ == "__main__":
    mcp.run(transport="stdio")
