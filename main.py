import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ollama_wrapper import (
    OllamaClient,
    OllamaOptions,
    ChatSession,
)

# -----------------------------
# CONFIGURE YOUR RUN HERE
# -----------------------------
BASE_URL = "http://localhost:11434"
TIMEOUT = 90
RETRIES = 3

# Models to compare (must be available in `ollama list`)
MODELS: List[str] = [
    "llama3.2:3b",
    "phi3:mini",
    "gemma2:2b",
]

# The single prompt to send to all models
PROMPT = "Analyze the sentiment of: 'The update is great, but settings are still confusing.' Return a short JSON."

# Choose interaction mode: "generate" (single-shot) or "chat" (multi-turn with optional system prompt)
MODE: str = "generate"  # or "chat"

SYSTEM_PROMPT: Optional[str] = "You are a precise analyst. Reply with valid JSON only."
JSON_MODE: bool = True         # ask Ollama for format='json' when possible
STREAM: bool = False           # set True to stream tokens (prints live)
TEMPERATURE: float = 0.2

# Output files (optional)
OUTPUT_DIR = Path("./runs")
SAVE_JSON = True
SAVE_CSV = True
RUN_TAG = time.strftime("%Y%m%d_%H%M%S")
# -----------------------------


def run_generate(client: OllamaClient, model: str, prompt: str, json_mode: bool, stream: bool) -> Dict[str, Any]:
    opts = OllamaOptions(temperature=TEMPERATURE)
    t0 = time.time()
    result_text = ""
    if stream:
        print(f"[{model}] streaming:")
        for chunk in client.generate(prompt, model, options=opts, format=("json" if json_mode else None), stream=True):
            print(chunk, end="", flush=True)
            result_text += chunk
        print()
    else:
        result_text = client.generate(prompt, model, options=opts, format=("json" if json_mode else None), stream=False)
    dt = time.time() - t0

    parsed: Optional[Any] = None
    if json_mode:
        parsed = _best_effort_json(result_text)

    return {
        "model": model,
        "mode": "generate",
        "latency_s": round(dt, 3),
        "raw": result_text,
        "json": parsed,
        "ok": True,
        "error": None,
    }


def run_chat(client: OllamaClient, model: str, prompt: str, system_prompt: Optional[str], json_mode: bool, stream: bool) -> Dict[str, Any]:
    opts = OllamaOptions(temperature=TEMPERATURE)
    session = client.start_chat(model, system=system_prompt, options=opts)
    t0 = time.time()
    result_text = ""
    if stream:
        print(f"[{model}] streaming (chat):")
        for chunk in session.send(prompt, stream=True):
            print(chunk, end="", flush=True)
            result_text += chunk
        print()
    else:
        result_text = session.send(prompt, stream=False)  # type: ignore[assignment]
    dt = time.time() - t0

    parsed: Optional[Any] = None
    if json_mode:
        parsed = _best_effort_json(result_text)

    return {
        "model": model,
        "mode": "chat",
        "latency_s": round(dt, 3),
        "raw": result_text,
        "json": parsed,
        "ok": True,
        "error": None,
    }


def _best_effort_json(text: str) -> Optional[Any]:
    """Try to parse strict JSON; if mixed text/JSON, extract the outermost object/array."""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # try to find first {...} or [...]
        start_obj, end_obj = text.find("{"), text.rfind("}")
        start_arr, end_arr = text.find("["), text.rfind("]")
        spans = []
        if start_obj != -1 and end_obj > start_obj:
            spans.append((start_obj, end_obj + 1))
        if start_arr != -1 and end_arr > start_arr:
            spans.append((start_arr, end_arr + 1))
        for s, e in sorted(spans, key=lambda p: p[0]):
            try:
                return json.loads(text[s:e])
            except Exception:
                continue
    return None


def ensure_models(client: OllamaClient, models: List[str]) -> List[str]:
    available = set(client.list_models())
    missing = [m for m in models if m.split("@")[0] not in available]
    if missing:
        print("⚠️  Missing models:", ", ".join(missing))
        print("    Install with:  ", " && ".join(f"ollama pull {m}" for m in missing))
    return [m for m in models if m.split("@")[0] in available]


def save_outputs(results: List[Dict[str, Any]], outdir: Path, tag: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if SAVE_JSON:
        jf = outdir / f"results_{tag}.json"
        with jf.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 JSON saved -> {jf}")

    if SAVE_CSV:
        # lightweight CSV writer to avoid pandas dependency
        import csv
        cf = outdir / f"results_{tag}.csv"
        fields = ["model", "mode", "latency_s", "ok", "error", "raw"]
        with cf.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in results:
                w.writerow({k: r.get(k, "") for k in fields})
        print(f"💾 CSV saved  -> {cf}")


def summarize(results: List[Dict[str, Any]]) -> None:
    print("\n=== SUMMARY ===")
    for r in results:
        model = r["model"]
        mode = r["mode"]
        ok = r["ok"]
        t = r["latency_s"]
        js = r.get("json")
        brief = ""
        if isinstance(js, dict):
            # show a couple of common keys if present
            sent = js.get("sentiment")
            summ = js.get("summary")
            brief = f" | sentiment={sent!r}, summary={summ!r}"
        print(f"- {model} [{mode}] ok={ok} time={t}s{brief}")


def main() -> int:
    client = OllamaClient(base_url=BASE_URL, timeout=TIMEOUT, retries=RETRIES)

    models = ensure_models(client, MODELS)
    if not models:
        print("No available models to run. Exiting.")
        return 1

    results: List[Dict[str, Any]] = []
    for model in models:
        try:
            if MODE == "generate":
                res = run_generate(client, model, PROMPT, JSON_MODE, STREAM)
            elif MODE == "chat":
                res = run_chat(client, model, PROMPT, SYSTEM_PROMPT, JSON_MODE, STREAM)
            else:
                raise ValueError(f"Unknown MODE: {MODE}")
            results.append(res)
        except Exception as e:
            results.append({
                "model": model,
                "mode": MODE,
                "latency_s": None,
                "raw": "",
                "json": None,
                "ok": False,
                "error": repr(e),
            })
            print(f"❌ {model} failed: {e}")

    summarize(results)
    save_outputs(results, OUTPUT_DIR, RUN_TAG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
