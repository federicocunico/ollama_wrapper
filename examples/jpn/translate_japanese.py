# translate_japanese.py
# Usage examples:
#   python translate_japanese.py "It's incredible how the sun keeps going up and down every day."
#   echo "It's incredible..." | python translate_japanese.py --stream
#   python translate_japanese.py --text "It's incredible..." --temperature 0.2
#   python translate_japanese.py --url http://remote-host:11434 "Some text"

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional, List

from ollama_wrapper import OllamaClient, OllamaOptions, OllamaError


# -------- Easily tweakable defaults --------
MODEL_NAME = "7shi/gemma-2-jpn-translate:2b-instruct-q8_0"
TEMPERATURE = 0.2  # it affects the randomness of the output; lower values make output more focused and deterministic

SYSTEM_PROMPT = (
    "You are a professional Japanese literary translator.\n"
    "Translate the user's text into natural, fluent Japanese.\n"
    "- Preserve meaning, tone, and style.\n"
    "- Keep paragraph and line breaks as in the source.\n"
    "- Do NOT add commentary, notes, tags, or explanations — return ONLY the translation."
)


def init_client(base_url: str, timeout: int, retries: int) -> OllamaClient:
    return OllamaClient(base_url=base_url, timeout=timeout, retries=retries)


def build_messages(text: str, system_prompt: str) -> List[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]


def translate_japanese(
    client: OllamaClient,
    model: str,
    text: str,
    *,
    temperature: float = TEMPERATURE,
    stream: bool = False,
    system_prompt: str = SYSTEM_PROMPT,
) -> Optional[str]:
    """
    If stream=False, returns the full translated string.
    If stream=True, prints tokens as they arrive and returns None.
    """
    opts = OllamaOptions(temperature=temperature)
    messages = build_messages(text, system_prompt)

    try:
        if stream:
            out_parts: List[str] = []
            for token in client.chat(messages=messages, model=model, options=opts, stream=True):
                sys.stdout.write(token)
                sys.stdout.flush()
                out_parts.append(token)
            if not out_parts or (out_parts and out_parts[-1] != "\n"):
                sys.stdout.write("\n")
            return "".join(out_parts).strip()
        else:
            result = client.chat(messages=messages, model=model, options=opts, stream=False)
            return result.strip()
    except OllamaError as e:
        print(f"❌ Ollama error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
    return None


def read_stdin_text() -> str:
    data = input("Reading text from stdin. Press [Enter] to finish input:\n")
    if not data:
        raise ValueError("No input received on stdin.")
    return data


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Translate English (or any language) text into Japanese using an Ollama model."
    )
    parser.add_argument("text", nargs="?", help="Text to translate. If omitted, reads from stdin.")
    parser.add_argument("--url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--timeout", type=int, default=200, help="HTTP timeout in seconds")
    parser.add_argument("--retries", type=int, default=5, help="HTTP retries")
    parser.add_argument("--model", default=MODEL_NAME, help="Ollama model name")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--stream", action="store_true", help="Stream tokens to stdout")
    parser.add_argument(
        "--system",
        default=SYSTEM_PROMPT,
        help="Override the system instruction (ensure it still returns ONLY the translation).",
    )
    args = parser.parse_args(argv)

    text = args.text if args.text is not None else read_stdin_text()

    print("Initializing Ollama client...")
    client = init_client(base_url=args.url, timeout=args.timeout, retries=args.retries)

    print("Translating text...")
    start = time.time()
    translated = translate_japanese(
        client,
        model=args.model,
        text=text,
        temperature=args.temperature,
        stream=args.stream,
        system_prompt=args.system,
    )
    elapsed = time.time() - start
    print(f"Translation completed in {elapsed:.2f} seconds.")

    if translated is not None:
        print(translated)
    else:
        print("[W] No translation available, some error occurred.")


if __name__ == "__main__":
    raise SystemExit(main())
