# translate.py
# Usage:
#   python translate.py --input path/to/book.txt --spacer "$$$$" --language "English" --model "aya-expanse"
# Optional:
#   --max-chars 12000   (approx context limit per request, in characters)
#   --mode chat|generate  (chat recommended for better instruction-following)
#   --stream             (stream translation tokens to stdout)

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple

from ollama_wrapper import OllamaClient, OllamaOptions


# -----------------------------
# Sentence splitting & chunking
# -----------------------------

SENT_SPLIT_REGEX = re.compile(
    # split on ., !, ?, … followed by whitespace and a new sentence start (letter/quote/number)
    r'(?<=[\.\!\?\…])\s+(?=[A-ZÀ-ÖØ-Þ0-9"“])'
)


def split_into_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    # quick normalization of whitespace
    text = re.sub(r"\s+", " ", text)
    parts = SENT_SPLIT_REGEX.split(text)
    # merge tiny trailing pieces if any weird split
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences


def pack_sentences(sentences: List[str], max_chars: int) -> List[str]:
    """
    Greedy pack sentences into chunks <= max_chars.
    """
    chunks: List[str] = []
    buf: List[str] = []
    size = 0
    for s in sentences:
        add = (1 if size > 0 else 0) + len(s)  # account for space joiner
        if size + add <= max_chars or not buf:
            buf.append(s)
            size += add
        else:
            chunks.append(" ".join(buf))
            buf = [s]
            size = len(s)
    if buf:
        chunks.append(" ".join(buf))
    return chunks


# -----------------------------
# Translation via Ollama
# -----------------------------

DEFAULT_SYSTEM = (
    "You are a professional literary translator. Translate the user's text into the target language.\n"
    "- Preserve meaning, tone, and style.\n"
    "- Keep paragraph breaks and inline punctuation.\n"
    "- Do NOT add commentary, notes, or explanations—return ONLY the translation."
)


def build_user_prompt(text: str, language: str) -> str:
    return f"Translate the following text into {language}.\n\n" f"--- BEGIN TEXT ---\n{text}\n--- END TEXT ---"


def translate_block(
    client: OllamaClient,
    model: str,
    text: str,
    language: str,
    *,
    mode: str = "chat",
    stream: bool = False,
    temperature: float = 0.2,
) -> str:
    opts = OllamaOptions(temperature=temperature)
    user = build_user_prompt(text, language)

    if mode == "generate":
        # single prompt mode: inline system guidance at top of prompt
        prompt = f"{DEFAULT_SYSTEM}\n\n{user}"
        if stream:
            out = []
            for t in client.generate(prompt, model, options=opts, stream=True):
                sys.stdout.write(t)
                sys.stdout.flush()
                out.append(t)
            print()
            return "".join(out)
        return client.generate(prompt, model, options=opts, stream=False)

    # default: chat with explicit system message
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM},
        {"role": "user", "content": user},
    ]
    if stream:
        out = []
        for t in client.chat(messages, model, options=opts, stream=True):
            sys.stdout.write(t)
            sys.stdout.flush()
            out.append(t)
        print()
        return "".join(out)
    return client.chat(messages, model, options=opts, stream=False)


# -----------------------------
# Chapter processing
# -----------------------------


def normalize_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def split_chapters(raw_text: str, spacer: str) -> List[str]:
    # keep empty chapters out; strip whitespace around each chapter
    parts = [p.strip() for p in raw_text.split(spacer)]
    return [p for p in parts if p]


def translate_chapter(
    client: OllamaClient,
    model: str,
    chapter_text: str,
    language: str,
    max_chars: int,
    *,
    mode: str = "chat",
    stream: bool = False,
) -> str:
    """
    If the chapter exceeds max_chars, split on sentence boundaries and translate in chunks.
    Reassemble translated chunks with double newlines between chunks to encourage paragraph separation.
    """
    chapter_text = chapter_text.strip()
    if not chapter_text:
        return ""

    if len(chapter_text) <= max_chars:
        return translate_block(client, model, chapter_text, language, mode=mode, stream=stream)

    sentences = split_into_sentences(chapter_text)
    if not sentences:
        # fallback: hard split
        return _fallback_chunk_translate(client, model, chapter_text, language, max_chars, mode=mode, stream=stream)

    chunks = pack_sentences(sentences, max_chars)
    translated_parts: List[str] = []
    for idx, ch in enumerate(chunks, 1):
        print(f"    - translating chunk {idx}/{len(chunks)} (chars={len(ch)})")
        translated = translate_block(client, model, ch, language, mode=mode, stream=False if not stream else False)
        translated_parts.append(translated.strip())
        # small pause to be gentle on local server
        time.sleep(0.05)
    return "\n\n".join(translated_parts)


def _fallback_chunk_translate(
    client: OllamaClient,
    model: str,
    text: str,
    language: str,
    max_chars: int,
    *,
    mode: str = "chat",
    stream: bool = False,
) -> str:
    parts: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        parts.append(text[start:end])
        start = end
    out: List[str] = []
    for idx, p in enumerate(parts, 1):
        print(f"    - translating raw chunk {idx}/{len(parts)} (chars={len(p)})")
        out.append(
            translate_block(client, model, p, language, mode=mode, stream=False if not stream else False).strip()
        )
        time.sleep(0.05)
    return "\n\n".join(out)


# -----------------------------
# Files & CLI
# -----------------------------


def save_chapter(out_fname: str, content: str) -> Path:
    with open(out_fname, "w", encoding="utf-8") as f:
        f.write(content)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Translate a text file split by a spacer into per-chapter files using Ollama."
    )
    parser.add_argument("--input", required=True, help="Path to the input .txt file")
    parser.add_argument("--spacer", required=True, help='Chapter separator string (e.g., "$$$$")')
    parser.add_argument("--language", required=True, help="Target language (e.g., English, Italian, Spanish)")
    parser.add_argument("--model", default="aya-expanse", help="Ollama model name (default: aya-expanse)")
    parser.add_argument("--url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--timeout", type=int, default=200, help="HTTP timeout seconds")
    parser.add_argument("--retries", type=int, default=5, help="HTTP retries")
    parser.add_argument("--mode", choices=["chat", "generate"], default="chat", help="Use /chat or /generate")
    parser.add_argument(
        "--max-chars", type=int, default=2500, help="Approx max characters per request (tune to model context)."
    )
    parser.add_argument("--stream", action="store_true", help="Stream tokens to stdout during translation")
    args = parser.parse_args(argv)

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Input file not found: {in_path}")
        return 1
    raw = in_path.read_text(encoding="utf-16")
    raw = normalize_newlines(raw)

    print(f"Reading: {in_path} ({len(raw)} chars)")
    chapters = split_chapters(raw, args.spacer)
    if not chapters:
        print("No chapters found. Check your --spacer.")
        return 1

    # Output directory: translated/<bookname>
    bookname = in_path.stem
    out_dir = Path("translated") / bookname
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {out_dir.resolve()}")

    client = OllamaClient(base_url=args.url, timeout=args.timeout, retries=args.retries)

    # Best-effort check for model availability
    available = set(client.list_models())
    if args.model.split("@")[0] not in available:
        print(f"⚠️  Model '{args.model}' is not available. Install with: ollama pull {args.model}")
        # continue anyway—Ollama will error if missing

    # Process each chapter
    for i, ch in enumerate(chapters, start=1):
        print(f"\nChapter {i}: {len(ch)} chars")

        out_file = out_dir / f"chapter_{i:03d}.txt"

        if out_file.exists():
            print(f"  ✓ Already exists: {out_file.name} (skipping)")
            continue

        translated = translate_chapter(
            client,
            args.model,
            ch,
            args.language,
            args.max_chars,
            mode=args.mode,
            stream=args.stream,
        )
        save_chapter(out_file, translated)
        print(f"  ✓ Saved: {out_file.name}")

    print("\nAll done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
