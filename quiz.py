"""Quiz questions (no answers) for the bot's /quiz command.
The bank is built by build_quiz.py and re-read automatically when the file changes.
"""
import json
import random
from pathlib import Path

BANK_FILE = Path(__file__).parent / "quiz_bank.json"
DEFAULT_N = 5
MAX_N = 10

_cache = {"mtime": None, "items": []}


def load() -> list[dict]:
    """All questions. Raises FileNotFoundError if build_quiz.py was not run yet."""
    mtime = BANK_FILE.stat().st_mtime_ns  # FileNotFoundError if missing
    if _cache["mtime"] != mtime:
        _cache["items"] = json.loads(BANK_FILE.read_text(encoding="utf-8"))
        _cache["mtime"] = mtime
    return _cache["items"]


def sections() -> list[str]:
    return sorted({q["section"] for q in load() if q["section"]})


def match_section(text: str) -> str | None:
    """'cardio' -> 'CARDIOLOGY' (prefix match, case-insensitive)."""
    text = text.strip().lower()
    for s in sections():
        if s.lower().startswith(text):
            return s
    return None


def pick(n: int = DEFAULT_N, section: str | None = None, exclude_ids=()) -> list[dict]:
    """n random questions. Prefers ones not in exclude_ids; fills from the rest if needed."""
    items = [q for q in load() if not section or q["section"] == section]
    n = max(1, min(n, MAX_N, len(items)))
    unseen = [q for q in items if q["id"] not in exclude_ids]
    chosen = random.sample(unseen, min(n, len(unseen)))
    if len(chosen) < n:  # not enough unseen questions: allow repeats
        chosen_ids = {q["id"] for q in chosen}
        rest = [q for q in items if q["id"] not in chosen_ids]
        chosen += random.sample(rest, n - len(chosen))
    return chosen


def _pretty(section: str) -> str:
    return section if len(section) <= 3 else section.title()  # GI stays GI


def format_quiz(questions: list[dict], section: str | None = None) -> str:
    head = f"📝 Quiz: {len(questions)} questions" + (f" ({_pretty(section)})" if section else "")
    lines = [head, ""]
    for i, q in enumerate(questions, start=1):
        lines.append(f"{i}. [{_pretty(q['section'])}] {q['question']}")
    lines += ["", "Another set: /quiz   |   One section: /quiz cardio   |   More: /quiz 10"]
    return "\n".join(lines)
