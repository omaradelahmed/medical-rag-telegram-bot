# Medical RAG Telegram Bot

A Telegram bot that answers medical questions **only from your own reference PDFs**, shows which pages it used, and replies **"I don't know."** when the books do not contain the answer.

Built with the Model Context Protocol (MCP), hybrid retrieval (embeddings + BM25) and an OpenAI-compatible LLM API (Groq by default) or a local Ollama model.

> ⚠️ **Medical disclaimer.** This is an educational project. Answers are general information taken from the supplied documents and are **not medical advice**. Text extracted from PDFs can contain errors (for example lost symbols that change a unit), so always check doses and numbers in the original source.

## Features

- **Grounded answers only:** a relevance threshold plus code-level guards stop the bot from answering out of the model's own memory.
- **Hybrid retrieval:** `bge-m3` embeddings (via Ollama) + BM25 keywords, fused with reciprocal rank fusion; query expansion maps everyday wording to the books' wording (e.g. "clinical features" to "clinical picture", "chronic kidney disease" to "CKD").
- **Arabic questions, English answers:** an Arabic medical glossary builds the English search query without relying on the LLM.
- **MCP server** exposing a search tool, a prompt template and a documents resource.
- **Incremental indexing:** add, update or remove PDFs; the running bot sees the change without a restart.
- **Telegram bot** with a `/quiz` command (random exam questions, no answers).
- **LLM backend is a setting:** Groq / any OpenAI-compatible API, or local Ollama.

## Architecture

```
Telegram --> bot.py --> agent.py --> medical_server.py   (MCP server, stdio)
                          |               '-- retriever.py --> index/  (one .npz per PDF)
                          '--> llm.py   (API or Ollama)

docs/*.pdf --> ingest.py --> index/
```

Flow of one question:
`question` -> (Arabic? glossary to English query) -> MCP tool `search_documents` -> nothing relevant? **"I don't know."** -> MCP prompt `medical_answer` -> LLM -> answer + sources + disclaimer.

## Quick start

Requirements: Python 3.10+ (developed on 3.13), [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com/) (for embeddings), and either a free Groq API key or a local Ollama chat model.

```bash
uv sync
ollama pull bge-m3

cp .env.example .env        # Windows: copy .env.example .env
# edit .env: TELEGRAM_BOT_TOKEN, LLM_API_KEY, LLM_MODEL
```

Add reference PDFs and build the index. To try it **without your own books**, use the demo document:

```bash
cp examples/demo_docs/demo_reference.pdf docs/      # Windows: copy examples\demo_docs\demo_reference.pdf docs\
uv run ingest.py
```

Then:

```bash
uv run check_llm.py   # optional: verifies the API key and lists models
uv run agent.py       # chat in the terminal
uv run bot.py         # start the Telegram bot (Ctrl+C to stop)
```

Questions to try are in [`examples/demo_questions.md`](examples/demo_questions.md).

Answer format:

```
- <point>
- <point>

Sources:
- demo_reference.pdf, p.1

This is general information and does not replace consulting a doctor.
```

## Telegram commands

| Command | What it does |
|---------|--------------|
| any text | asks a question (English or Arabic) |
| `/start` | welcome message |
| `/quiz [n] [section]` | `n` random exam questions without answers, e.g. `/quiz 5 cardio` |

`/quiz` is optional and needs a "past questions" PDF in the style it was built for (bold 14.5 pt question titles). Build its question bank once with `uv run build_quiz.py`.

## Managing documents

| Command | Effect |
|---------|--------|
| `uv run ingest.py` | adds new PDFs, re-indexes changed ones, removes deleted ones |
| `uv run ingest.py --list` | shows what is indexed |
| `uv run ingest.py --remove NAME` | removes one document |
| `uv run ingest.py --force` | re-embeds everything |

PDFs are never committed to Git (see `.gitignore`); they are usually copyrighted.

## Project layout

| File | Role |
|------|------|
| `bot.py` | Telegram bot |
| `agent.py` | question -> search -> prompt -> LLM -> answer |
| `medical_server.py` | MCP server (tool, prompt, resource) |
| `retriever.py` | hybrid search, query expansion, automatic index reload |
| `store.py`, `ingest.py` | per-document index storage and incremental indexing |
| `llm.py` | LLM access (API or Ollama) |
| `ar_terms.py` | Arabic to English medical glossary |
| `quiz.py`, `build_quiz.py` | `/quiz` command |
| `check_llm.py`, `debug_search.py`, `test_server.py` | small diagnostic scripts |

## Limitations

- **Not for clinical use.** Verify anything important (especially doses and units) in the source.
- **PDF text extraction is imperfect:** lost symbols (for example µ) can change a unit; scanned PDFs have no text (no OCR).
- **Answers can be incomplete:** the LLM only sees the top few retrieved passages.
- **"Sources" lists the retrieved pages**, not necessarily the ones the model actually used.
- **Arabic support is glossary-based** (about 130 terms); unknown terms fall back to LLM translation, which is less reliable.
- **No formal evaluation yet.** The relevance threshold and query expansion were tuned by hand on two books.
- **Privacy:** questions and the retrieved passages are sent to the LLM provider unless you use a local model.
- **No access control:** anyone who finds the bot can use it (and use your API quota).

## Roadmap

- Evaluation set (questions with expected pages) and a hit-rate score
- Reranking and conversation memory
- Docker image and deployment
- Allowed-users list and per-user documents
