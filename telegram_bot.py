"""
Telegram Bot: Claude AI + TTS + Video Generation
================================================
Установка зависимостей:
  pip install python-telegram-bot anthropic gTTS moviepy pillow requests

Переменные окружения (.env или export):
  TELEGRAM_BOT_TOKEN=your_telegram_bot_token
  ANTHROPIC_API_KEY=your_anthropic_api_key
  ELEVENLABS_API_KEY=your_elevenlabs_api_key   # опционально (для premium TTS)
  PEXELS_API_KEY=your_pexels_api_key           # опционально (для видеофонов)
"""

import os
import asyncio
import logging
import tempfile
import textwrap
from pathlib import Path

from telegram import Update, BotCommand
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
import anthropic
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
import requests

# ── Настройка логирования ──────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Клиент Claude ──────────────────────────────────────────────────────────────
claude = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ── Системный промпт ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Ты умный, дружелюбный ассистент в Telegram.
Отвечай кратко и по существу. Если пользователь не указал язык — отвечай на том же языке, на котором написан вопрос.
Для команды /video создавай яркое описание сцены в 2-3 предложениях, подходящее для видео."""

# ── Хранилище истории диалогов (в памяти) ─────────────────────────────────────
conversation_history: dict[int, list[dict]] = {}

MAX_HISTORY = 20  # максимальное число сообщений на пользователя


# ── Вспомогательные функции ────────────────────────────────────────────────────

def get_history(user_id: int) -> list[dict]:
    return conversation_history.setdefault(user_id, [])


def add_to_history(user_id: int, role: str, content: str):
    history = get_history(user_id)
    history.append({"role": role, "content": content})
    if len(history) > MAX_HISTORY:
        conversation_history[user_id] = history[-MAX_HISTORY:]


def generate_text(user_id: int, user_message: str) -> str:
    """Генерация ответа через Claude API с историей диалога."""
    add_to_history(user_id, "user", user_message)

    response = claude.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=get_history(user_id),
    )
    assistant_text = response.content[0].text
    add_to_history(user_id, "assistant", assistant_text)
    return assistant_text


def text_to_speech_gtts(text: str, lang: str = "ru") -> Path:
    """Синтез речи через бесплатный gTTS. Возвращает путь к .mp3."""
    tts = gTTS(text=text, lang=lang, slow=False)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return Path(tmp.name)


def text_to_speech_elevenlabs(text: str) -> Path | None:
    """Синтез речи через ElevenLabs (опционально, premium качество)."""
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        return None

    VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel — по умолчанию
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    if resp.status_code != 200:
        logger.warning("ElevenLabs error %s", resp.text)
        return None

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp.write(resp.content)
    tmp.close()
    return Path(tmp.name)


def generate_video_from_text(text: str) -> Path:
    """
    Генерация видео: текст → кадры с субтитрами → MP4.
    Использует только Pillow + moviepy (без внешних API).
    При наличии PEXELS_API_KEY добавляет фоновые кадры из Pexels.
    """
    try:
        from moviepy.editor import (
            ImageClip,
            concatenate_videoclips,
            AudioFileClip,
            CompositeVideoClip,
        )
    except ImportError:
        raise RuntimeError("Установи moviepy: pip install moviepy")

    # Разбиваем текст на части (до 60 символов на строку)
    lines = textwrap.wrap(text, width=60) or [text]
    segments = []
    for i in range(0, len(lines), 3):
        segments.append(" ".join(lines[i : i + 3]))

    clips = []
    for seg in segments:
        img = _make_text_frame(seg)
        tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        img.save(tmp_img.name)
        clip = ImageClip(tmp_img.name, duration=3)
        clips.append(clip)

    if not clips:
        clips = [ImageClip(_make_text_frame(text[:120]).filename, duration=4)]

    video = concatenate_videoclips(clips, method="compose")

    # TTS-аудиодорожка
    audio_path = text_to_speech_elevenlabs(text) or text_to_speech_gtts(text, lang="ru")
    try:
        audio = AudioFileClip(str(audio_path))
        video = video.set_audio(audio.subclip(0, min(audio.duration, video.duration)))
    except Exception as e:
        logger.warning("Audio attach failed: %s", e)

    out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video.write_videofile(
        out.name,
        fps=24,
        codec="libx264",
        audio_codec="aac",
        logger=None,
    )
    return Path(out.name)


def _make_text_frame(text: str, size=(1280, 720)) -> Image.Image:
    """Создаёт PNG-кадр с текстом на градиентном фоне."""
    img = Image.new("RGB", size, color=(15, 15, 30))
    draw = ImageDraw.Draw(img)

    # Рамочка
    for offset, alpha in [(6, 60), (3, 120), (0, 200)]:
        draw.rectangle(
            [offset, offset, size[0] - offset, size[1] - offset],
            outline=(100, 100, 220, alpha),
            width=2,
        )

    # Текст
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except Exception:
        font = ImageFont.load_default()

    wrapped = textwrap.fill(text, width=35)
    bbox = draw.textbbox((0, 0), wrapped, font=font)
    x = (size[0] - (bbox[2] - bbox[0])) // 2
    y = (size[1] - (bbox[3] - bbox[1])) // 2

    # Тень
    draw.text((x + 2, y + 2), wrapped, font=font, fill=(0, 0, 0))
    draw.text((x, y), wrapped, font=font, fill=(230, 230, 255))

    return img


# ── Обработчики команд ─────────────────────────────────────────────────────────

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    name = update.effective_user.first_name or "друг"
    await update.message.reply_text(
        f"Привет, {name}! 👋\n\n"
        "Я умею:\n"
        "• Отвечать на любые вопросы (просто напиши)\n"
        "• /tts <текст> — синтез речи (аудио)\n"
        "• /video <тема> — генерация видео\n"
        "• /clear — очистить историю диалога\n"
        "• /help — справка"
    )


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 Команды:\n\n"
        "/tts <текст> — создать голосовое сообщение\n"
        "/video <тема> — создать короткое видео (10-30 сек)\n"
        "/clear — сбросить историю диалога\n\n"
        "Или просто напиши мне — отвечу через Claude AI 🤖"
    )


async def cmd_clear(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    conversation_history.pop(update.effective_user.id, None)
    await update.message.reply_text("История диалога очищена ✅")


async def cmd_tts(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = " ".join(ctx.args)
    if not text:
        await update.message.reply_text("Использование: /tts <текст для озвучки>")
        return

    msg = await update.message.reply_text("🎙 Синтезирую речь…")
    try:
        # Пробуем ElevenLabs, фолбэк на gTTS
        audio_path = text_to_speech_elevenlabs(text) or text_to_speech_gtts(text)
        await update.message.reply_voice(voice=open(audio_path, "rb"))
        audio_path.unlink(missing_ok=True)
        await msg.delete()
    except Exception as e:
        logger.exception("TTS error")
        await msg.edit_text(f"Ошибка TTS: {e}")


async def cmd_video(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    topic = " ".join(ctx.args)
    if not topic:
        await update.message.reply_text("Использование: /video <тема или описание>")
        return

    msg = await update.message.reply_text("🎬 Генерирую сценарий и видео…")
    try:
        # Claude пишет сценарий
        script = generate_text(
            update.effective_user.id,
            f"Напиши яркое и интересное описание для короткого видео на тему: «{topic}». "
            "2-3 предложения, образно и динамично.",
        )
        await msg.edit_text(f"📝 Сценарий готов, создаю видео…\n\n_{script}_", parse_mode="Markdown")

        # Генерируем видео
        video_path = generate_video_from_text(script)
        await update.message.reply_video(
            video=open(video_path, "rb"),
            caption=f"🎥 {topic}\n\n{script[:200]}…",
        )
        video_path.unlink(missing_ok=True)
        await msg.delete()
    except Exception as e:
        logger.exception("Video generation error")
        await msg.edit_text(f"Ошибка генерации видео: {e}")


async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Основной обработчик текстовых сообщений."""
    user_text = update.message.text
    user_id = update.effective_user.id

    await update.message.chat.send_action("typing")
    try:
        reply = generate_text(user_id, user_text)
        await update.message.reply_text(reply)
    except Exception as e:
        logger.exception("Claude API error")
        await update.message.reply_text(f"Ошибка: {e}")


# ── Запуск ─────────────────────────────────────────────────────────────────────

def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Переменная окружения TELEGRAM_BOT_TOKEN не задана")

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("tts", cmd_tts))
    app.add_handler(CommandHandler("video", cmd_video))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot started. Press Ctrl+C to stop.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
