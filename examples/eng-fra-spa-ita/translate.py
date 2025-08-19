# translate.py
# Usage:
#   python translate.py --input path/to/book.txt --spacer "$$$$" --language "English" --model "aya-expanse"
# Optional:
#   --max-chars 12000   (approx context limit per request, in characters)
#   --mode chat|generate
#   --stream
#   --url http://remote-host:11434

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import List

from ollama_wrapper import OllamaClient, OllamaOptions, OllamaError


# -----------------------------
# Sentence splitting & chunking
# -----------------------------

SENT_SPLIT_REGEX = re.compile(r'(?<=[\.\!\?\…])\s+(?=[A-ZÀ-ÖØ-Þ0-9"“])')


def split_into_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    text = re.sub(r"\s+", " ", text)
    parts = SENT_SPLIT_REGEX.split(text)
    return [p.strip() for p in parts if p.strip()]


def pack_sentences(sentences: List[str], max_chars: int) -> List[str]:
    chunks: List[str] = []
    buf: List[str] = []
    size = 0
    for s in sentences:
        add = (1 if size > 0 else 0) + len(s)
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
    return f"Translate the following text into {language}.\n\n--- BEGIN TEXT ---\n{text}\n--- END TEXT ---"


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

    try:
        if mode == "generate":
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

    except OllamaError as e:
        raise RuntimeError(f"Ollama request failed (model='{model}', mode='{mode}'): {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error while translating block (model='{model}'): {e}") from e


# -----------------------------
# Chapter processing
# -----------------------------


def normalize_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def split_chapters(raw_text: str, spacer: str) -> List[str]:
    parts = [p.strip() for p in raw_text.split(spacer)]
    return [p for p in parts if p]


def _fallback_chunk_translate(
    client: OllamaClient,
    model: str,
    text: str,
    language: str,
    max_chars: int,
    *,
    mode: str = "chat",
    stream: bool = False,
) -> List[str]:
    parts: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        parts.append(text[start:end])
        start = end
    out: List[str] = []
    for idx, p in enumerate(parts, 1):
        print(f"    - translating raw chunk {idx}/{len(parts)} (chars={len(p)})")
        out.append(translate_block(client, model, p, language, mode=mode, stream=False).strip())
        time.sleep(0.05)
    return out


def chunk_chapter(chapter_text: str, max_chars: int) -> List[str]:
    if len(chapter_text) <= max_chars:
        return [chapter_text]
    sentences = split_into_sentences(chapter_text)
    if not sentences:
        return _fallback_chunk_translate  # type: ignore[return-value]
    return pack_sentences(sentences, max_chars)


# -----------------------------
# Files & utilities
# -----------------------------

# def try_read_text(path: Path) -> str:
#     # Helpful, robust file reading with clear messages
#     for enc in ("utf-8", "utf-16", "utf-8-sig"):
#         try:
#             return path.read_text(encoding=enc)
#         except UnicodeError:
#             continue
#         except FileNotFoundError:
#             raise FileNotFoundError(f"Input file not found: {path}")
#         except Exception as e:
#             raise RuntimeError(f"Error reading input file '{path}': {e}") from e
#     raise UnicodeError(
#         f"Could not decode '{path}'. Try re-saving as UTF-8, or pass a different encoding."
#     )


def try_read_text(path: Path) -> str:
    """
    Always try to read the file as UTF-16 first.
    If it fails, raise a clear error.
    """
    try:
        return path.read_text(encoding="utf-16")
    except UnicodeError as e:
        raise UnicodeError(
            f"❌ Failed to decode '{path}' as UTF-16. "
            "The file encoding might be different. "
            "Ensure it is saved as UTF-16 or re-save it with UTF-8."
        ) from e
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Input file not found: {path}")
    except Exception as e:
        raise RuntimeError(f"❌ Error reading input file '{path}': {e}") from e


def save_text(path: Path, content: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to save file '{path}': {e}") from e


def load_text_if_exists(path: Path) -> str | None:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to read existing chunk '{path}': {e}") from e
    return None


def merge_files_in_order(paths: List[Path]) -> str:
    merged_parts: List[str] = []
    for p in paths:
        try:
            merged_parts.append(p.read_text(encoding="utf-8").strip())
        except Exception as e:
            raise RuntimeError(f"Failed to read chunk for merge '{p}': {e}") from e
    return "\n\n".join(merged_parts).strip()


# -----------------------------
# CLI main
# -----------------------------


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Translate a text file split by a spacer into per-chapter files using Ollama."
    )
    parser.add_argument("--input", required=True, help="Path to the input .txt file")
    parser.add_argument("--spacer", required=True, help='Chapter separator string (e.g., "$$$$")')
    parser.add_argument("--language", required=True, default="Italian", help="Target language (e.g., English, Italian, Spanish)")
    parser.add_argument("--model", default="aya-expanse", help="Ollama model name (default: aya-expanse)")
    parser.add_argument("--url", default="http://localhost:11434", help="Ollama base URL (remote ok)")
    parser.add_argument("--timeout", type=int, default=200, help="HTTP timeout seconds")
    parser.add_argument("--retries", type=int, default=5, help="HTTP retries")
    parser.add_argument("--mode", choices=["chat", "generate"], default="chat", help="Use /chat or /generate")
    parser.add_argument("--max-chars", type=int, default=2500, help="Approx max characters per request.")
    parser.add_argument("--stream", action="store_true", help="Stream tokens to stdout during translation")
    args = parser.parse_args(argv)

    in_path = Path(args.input)
    try:
        raw = try_read_text(in_path)
    except Exception as e:
        print(f"❌ Failed to read input: {e}")
        return 1

    raw = normalize_newlines(raw)
    print(f"Reading: {in_path} ({len(raw)} chars)")
    chapters = split_chapters(raw, args.spacer)
    if not chapters:
        print("❌ No chapters found. Check your --spacer string.")
        return 1

    # Output directories
    bookname = in_path.stem
    base_out_dir = Path("translated") / bookname
    chunks_root = base_out_dir / "chunks"
    final_root = base_out_dir
    try:
        base_out_dir.mkdir(parents=True, exist_ok=True)
        chunks_root.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"❌ Cannot create output folders: {e}")
        return 1

    print(f"✅ Output folder: {final_root.resolve()}")
    print(f"🔗 Using Ollama at: {args.url}")

    # Client
    try:
        client = OllamaClient(base_url=args.url, timeout=args.timeout, retries=args.retries)
        # Best-effort model existence check
        try:
            available = set(client.list_models())
            if args.model.split("@")[0] not in available:
                print(
                    f"⚠️  Model '{args.model}' not reported by server. If missing, install with:\n    ollama pull {args.model}"
                )
        except Exception as e:
            print(f"⚠️  Could not list models from Ollama ({e}). Continuing anyway.")
    except Exception as e:
        print(f"❌ Failed to initialize Ollama client: {e}")
        return 1

    # Process chapters
    for chap_idx, chapter_text in enumerate(chapters, start=1):
        print(f"\n=== Chapter {chap_idx} ===")
        chapter_len = len(chapter_text)
        print(f"Chars: {chapter_len}")

        chapter_dir = chunks_root / f"chapter_{chap_idx:03d}"
        final_chapter_file = final_root / f"chapter_{chap_idx:03d}.txt"

        # If entire chapter fits and final exists, skip
        if chapter_len <= args.max_chars and final_chapter_file.exists():
            print(f"  ✓ Final chapter already exists: {final_chapter_file.name} (skipping)")
            continue

        # Build chunks (by sentences)
        try:
            if chapter_len <= args.max_chars:
                chunks = [chapter_text]
            else:
                sentences = split_into_sentences(chapter_text)
                if sentences:
                    chunks = pack_sentences(sentences, args.max_chars)
                else:
                    # fallback to raw slicing
                    chunks = []
                    start = 0
                    while start < chapter_len:
                        end = min(start + args.max_chars, chapter_len)
                        chunks.append(chapter_text[start:end])
                        start = end
            print(f"  → {len(chunks)} chunk(s)")
        except Exception as e:
            print(f"❌ Failed to split chapter {chap_idx}: {e}")
            continue

        # Ensure chapter chunk dir
        try:
            chapter_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"❌ Cannot create chunk folder '{chapter_dir}': {e}")
            continue

        # Translate each chunk, saving individually (skip existing)
        chunk_files: List[Path] = []
        for ci, chunk_text in enumerate(chunks, start=1):
            chunk_file = chapter_dir / f"chapter_{chap_idx:03d}_chunk_{ci:03d}.txt"
            chunk_files.append(chunk_file)

            existing = load_text_if_exists(chunk_file)
            if existing is not None and existing.strip():
                print(f"    ✓ Chunk {ci}/{len(chunks)} already exists (skipping)")
                continue

            print(f"    … Translating chunk {ci}/{len(chunks)} (chars={len(chunk_text)})")
            try:
                translated = translate_block(
                    client=client,
                    model=args.model,
                    text=chunk_text,
                    language=args.language,
                    mode=args.mode,
                    stream=False,  # per-chunk streaming is noisy; keep False
                ).strip()
            except Exception as e:
                print(f"    ❌ Chunk {ci} failed (chapter {chap_idx}): {e}")
                # Save an error marker so we know it failed (optional)
                try:
                    save_text(chunk_file.with_suffix(".error.txt"), f"ERROR: {e}\n")
                except Exception as se:
                    print(f"    ⚠️  Also failed to record error for chunk {ci}: {se}")
                continue

            # Save chunk
            try:
                save_text(chunk_file, translated)
                print(f"    ✓ Saved chunk {ci} -> {chunk_file.name}")
            except Exception as e:
                print(f"    ❌ Failed to save chunk {ci}: {e}")
                continue

            time.sleep(0.05)

        # Merge chunk files into final chapter if all chunks exist and are non-empty
        try:
            missing = [p for p in chunk_files if (not p.exists()) or (not p.read_text(encoding="utf-8").strip())]
            if missing:
                print(f"  ⚠️  Skipping merge for chapter {chap_idx}: {len(missing)} chunk(s) missing/empty.")
            else:
                merged = merge_files_in_order(chunk_files)
                save_text(final_chapter_file, merged)
                print(f"  ✅ Merged -> {final_chapter_file.name}")
        except Exception as e:
            print(f"  ❌ Failed to merge chapter {chap_idx}: {e}")

    print("\nAll done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
