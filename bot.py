"""Step 7: Telegram bot on top of MedicalAgent.
Run:  uv run bot.py      (stop with Ctrl+C)
Needs in .env:  TELEGRAM_BOT_TOKEN=...   (plus the LLM_* lines used by llm.py)
"""
import asyncio
import logging
import os

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

import llm
import quiz
from agent import MedicalAgent

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip().strip("\"'")

MAX_CONCURRENT = 2     # questions processed at the same time (protects the API rate limit)
MAX_QUESTION_CHARS = 500
TELEGRAM_LIMIT = 4000  # Telegram's hard limit is 4096 characters per message

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# httpx logs full request URLs, and those URLs contain the bot token: keep them quiet.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger("bot")

WELCOME = (
    "Hi! Ask me a medical question, in English or Arabic.\n"
    "I answer ONLY from my reference books, in English, and I show the pages used. "
    "If I can't find the answer there, I say \"I don't know.\"\n\n"
    "مرحباً! اسألني سؤالاً طبياً بالعربية أو الإنجليزية. أجيب فقط من الكتب المرجعية "
    "التي لدي (والإجابة بالإنجليزية)، وإذا لم أجد الإجابة أقول \"I don't know.\"\n\n"
    "📝 Want to practise? Send /quiz for 5 random exam questions (/quiz cardio for one section).\n\n"
    "⚠️ General information only. It does not replace consulting a doctor."
)


def split_message(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    """Split a long answer on paragraph/line boundaries so each part fits in one message."""
    parts, current = [], ""
    for block in text.split("\n"):
        while len(block) > limit:  # a single huge line
            parts.append(block[:limit])
            block = block[limit:]
        if len(current) + len(block) + 1 > limit:
            parts.append(current)
            current = ""
        current += block + "\n"
    if current.strip():
        parts.append(current)
    return [p.rstrip() for p in parts if p.strip()]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(WELCOME)


async def quiz_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/quiz [n] [section]  ->  random exam questions from the book, WITHOUT answers."""
    n, section_text = quiz.DEFAULT_N, ""
    for arg in context.args or []:
        if arg.isdigit():
            n = max(1, min(int(arg), quiz.MAX_N))
        else:
            section_text += (" " if section_text else "") + arg
    try:
        section = None
        if section_text:
            section = quiz.match_section(section_text)
            if section is None:
                await update.message.reply_text("Unknown section. Available: " + ", ".join(quiz.sections()))
                return
        seen = context.user_data.setdefault("quiz_seen", set())  # avoid repeats for this user
        questions = quiz.pick(n, section, exclude_ids=seen)
    except FileNotFoundError:
        log.warning("quiz_bank.json not found: run build_quiz.py")
        await update.message.reply_text("The quiz is not available yet.")
        return
    seen.update(q["id"] for q in questions)
    await update.message.reply_text(quiz.format_quiz(questions, section))


async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    text = update.message.text.strip()
    if len(text) > MAX_QUESTION_CHARS:
        await update.message.reply_text(f"Please keep your question under {MAX_QUESTION_CHARS} characters.")
        return

    agent: MedicalAgent = context.bot_data["agent"]
    sem: asyncio.Semaphore = context.bot_data["sem"]
    log.info("question received (%d chars)", len(text))  # content is not logged

    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)
    try:
        async with sem:
            answer = await agent.ask(text)
    except Exception:
        log.exception("ask() failed")
        answer = "Sorry, something went wrong. Please try again in a moment."

    for part in split_message(answer):
        await update.message.reply_text(part)


async def main():
    if not TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN not found. Add it to .env (no quotes): TELEGRAM_BOT_TOKEN=123456:ABC...")

    log.info("LLM backend: %s", llm.describe())
    agent = MedicalAgent()
    await agent.start()  # started and closed in THIS task (the MCP connection requires it)

    app = Application.builder().token(TOKEN).build()
    app.bot_data["agent"] = agent
    app.bot_data["sem"] = asyncio.Semaphore(MAX_CONCURRENT)
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(CommandHandler("quiz", quiz_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

    try:
        async with app:
            await app.start()
            await app.updater.start_polling()
            log.info("Bot is running. Press Ctrl+C to stop.")
            await asyncio.Event().wait()  # run until cancelled
    finally:
        await agent.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Stopped.")
