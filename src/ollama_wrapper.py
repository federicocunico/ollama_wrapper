import sys
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Iterable, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ---------------------------
# Core HTTP Client for Ollama
# ---------------------------


class OllamaError(RuntimeError):
    pass


def _build_session(total_retries: int = 3, backoff: float = 0.5) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=total_retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


@dataclass
class OllamaOptions:
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 40
    seed: Optional[int] = 7
    num_ctx: Optional[int] = None  # context window if supported
    # You can add more here (num_gpu, stop tokens, etc.)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
        if self.seed is not None:
            d["seed"] = self.seed
        if self.num_ctx is not None:
            d["num_ctx"] = self.num_ctx
        return d


class OllamaClient:
    """
    Thin wrapper over Ollama's HTTP API:
      - /api/tags
      - /api/generate
      - /api/chat
    """

    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 60, retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = _build_session(total_retries=retries)

    # --- discovery ---
    def list_models(self) -> List[str]:
        r = self.session.get(f"{self.base_url}/api/tags", timeout=self.timeout)
        if r.status_code != 200:
            raise OllamaError(f"tags failed: HTTP {r.status_code}: {r.text}")
        models = r.json().get("models", [])
        # Normalize names to ignore digests/aliases like name@sha
        names = sorted({m.get("name", "").split("@")[0] for m in models if m.get("name")})
        return names

    def has_model(self, model: str) -> bool:
        name = model.split("@")[0]
        return name in self.list_models()

    # --- single prompt ---
    def generate(
        self,
        prompt: str,
        model: str,
        *,
        options: Optional[OllamaOptions] = None,
        format: Optional[str] = None,  # e.g. "json"
        stream: bool = False,
    ) -> str | Generator[str, None, str]:
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": (options or OllamaOptions()).to_dict(),
        }
        if format:
            payload["format"] = format

        url = f"{self.base_url}/api/generate"
        if not stream:
            r = self.session.post(url, json=payload, timeout=self.timeout)
            if r.status_code != 200:
                raise OllamaError(f"generate failed: HTTP {r.status_code}: {r.text}")
            return r.json().get("response", "")
        # streaming
        return self._stream_generate(payload)

    def _stream_generate(self, payload: Dict[str, Any]) -> Generator[str, None, str]:
        url = f"{self.base_url}/api/generate"
        with self.session.post(url, json=payload, timeout=self.timeout, stream=True) as r:
            if r.status_code != 200:
                raise OllamaError(f"generate(stream) failed: HTTP {r.status_code}: {r.text}")
            final_text = []
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                token = chunk.get("response", "")
                if token:
                    final_text.append(token)
                    yield token
                if chunk.get("done"):
                    break
            return "".join(final_text)

    # --- chat (multi-turn) ---
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        *,
        options: Optional[OllamaOptions] = None,
        format: Optional[str] = None,
        stream: bool = False,
    ) -> str | Generator[str, None, str]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": (options or OllamaOptions()).to_dict(),
        }
        if format:
            payload["format"] = format

        url = f"{self.base_url}/api/chat"
        if not stream:
            r = self.session.post(url, json=payload, timeout=self.timeout)
            if r.status_code != 200:
                raise OllamaError(f"chat failed: HTTP {r.status_code}: {r.text}")
            return r.json().get("message", {}).get("content", "")
        # streaming
        return self._stream_chat(payload)

    def _stream_chat(self, payload: Dict[str, Any]) -> Generator[str, None, str]:
        url = f"{self.base_url}/api/chat"
        with self.session.post(url, json=payload, timeout=self.timeout, stream=True) as r:
            if r.status_code != 200:
                raise OllamaError(f"chat(stream) failed: HTTP {r.status_code}: {r.text}")
            final_text = []
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                msg = chunk.get("message", {}).get("content", "")
                if msg:
                    final_text.append(msg)
                    yield msg
                if chunk.get("done"):
                    break
            return "".join(final_text)

    # --- convenience for sessions ---
    def start_chat(
        self, model: str, *, system: Optional[str] = None, options: Optional[OllamaOptions] = None
    ) -> "ChatSession":
        return ChatSession(client=self, model=model, system=system, options=options or OllamaOptions())


# ---------------------------
# Chat session helper
# ---------------------------


@dataclass
class ChatSession:
    client: OllamaClient
    model: str
    system: Optional[str] = None
    options: OllamaOptions = field(default_factory=OllamaOptions)
    messages: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self):
        if self.system:
            self.messages.append({"role": "system", "content": self.system})

    def send(self, content: str, *, stream: bool = False) -> str | Generator[str, None, str]:
        """Send a USER turn and get ASSISTANT reply (optionally stream)."""
        self.messages.append({"role": "user", "content": content})
        if not stream:
            reply = self.client.chat(self.messages, self.model, options=self.options, stream=False)
            self.messages.append({"role": "assistant", "content": reply})
            return reply

        # streaming
        def generator():
            acc = []
            for chunk in self.client.chat(self.messages, self.model, options=self.options, stream=True):
                acc.append(chunk)
                yield chunk
            full = "".join(acc)
            self.messages.append({"role": "assistant", "content": full})

        return generator()

    def add_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def reset(self, *, keep_system: bool = True) -> None:
        sys_msg = next((m for m in self.messages if m["role"] == "system"), None)
        self.messages.clear()
        if keep_system and (sys_msg or self.system):
            self.messages.append(sys_msg or {"role": "system", "content": self.system or ""})


# ---------------------------
# Simple “Analyze” helper
# ---------------------------

DEFAULT_ANALYZE_SYSTEM = "You are a strict analyzer. Respond ONLY with valid JSON following exactly the schema."


def analysis_prompt(comment_text: str) -> str:
    return (
        "Analyze the following text and return a SINGLE JSON object with keys:\n"
        '{"sentiment":"positive|negative|neutral",'
        '"confidence":0.0-1.0,'
        '"topics":["t1",... max 5],'
        '"toxicity":"low|medium|high",'
        '"emotion":"anger|joy|fear|sadness|surprise|disgust|neutral",'
        '"summary":"brief one-sentence"}\n'
        "Text:\n"
        f'"""{comment_text.strip()}"""'
    )


def analyze_text(
    client: OllamaClient,
    text: str,
    model: str,
    *,
    options: Optional[OllamaOptions] = None,
    json_mode: bool = True,
    retries: int = 2,
) -> Dict[str, Any]:
    """
    Run a single-turn analysis using either /generate or /chat with a system prompt.
    If json_mode=True and the model supports it, we pass format='json' for stricter outputs.
    """
    opts = options or OllamaOptions()
    sys_prompt = DEFAULT_ANALYZE_SYSTEM
    user_prompt = analysis_prompt(text)

    # Prefer chat (lets us include a system message)
    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat(messages, model, options=opts, format=("json" if json_mode else None), stream=False)
            if json_mode:
                # If Ollama returns JSON string already, use it directly.
                if isinstance(resp, str):
                    try:
                        return json.loads(resp)
                    except json.JSONDecodeError:
                        # Some models still wrap JSON in text; try to extract a JSON object
                        start = resp.find("{")
                        end = resp.rfind("}")
                        if start != -1 and end != -1 and end > start:
                            return json.loads(resp[start : end + 1])
                        raise
            # Non-JSON path: attempt to parse; otherwise return raw
            try:
                return json.loads(resp)
            except Exception:
                return {"raw": resp}
        except Exception as e:
            last_err = e
            time.sleep(0.4 * attempt if attempt else 0)
    raise OllamaError(f"analyze_text failed after retries: {last_err}")


# ---------------------------
# Minimal CLI
# ---------------------------


def _print_stream(gen: Iterable[str]) -> None:
    try:
        for chunk in gen:
            sys.stdout.write(chunk)
            sys.stdout.flush()
        print()
    except KeyboardInterrupt:
        print("\n[stream interrupted]")


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="ollama-wrapper", description="Ollama multi-model prompt/chat tool")
    parser.add_argument("--url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout (s)")
    parser.add_argument("--retries", type=int, default=3, help="HTTP retries")

    sub = parser.add_subparsers(dest="cmd", required=True)

    # models
    p_models = sub.add_parser("models", help="List available Ollama models")

    # prompt
    p_prompt = sub.add_parser("prompt", help="Single prompt via /generate")
    p_prompt.add_argument("-m", "--model", required=True)
    p_prompt.add_argument("-p", "--prompt", required=True)
    p_prompt.add_argument("--json", action="store_true", help="Ask for JSON output if supported")
    p_prompt.add_argument("--stream", action="store_true")
    p_prompt.add_argument("--temperature", type=float, default=0.2)

    # chat (interactive)
    p_chat = sub.add_parser("chat", help="Interactive chat session")
    p_chat.add_argument("-m", "--model", required=True)
    p_chat.add_argument("--system", default=None, help="Optional system prompt")
    p_chat.add_argument("--temperature", type=float, default=0.2)

    # analyze
    p_an = sub.add_parser("analyze", help="Analyze text using default JSON schema")
    p_an.add_argument("-m", "--model", required=True)
    p_an.add_argument("-t", "--text", required=True)
    p_an.add_argument("--no-json", action="store_true", help="Disable format=json")

    args = parser.parse_args(argv)

    client = OllamaClient(base_url=args.url, timeout=args.timeout, retries=args.retries)

    if args.cmd == "models":
        for name in client.list_models():
            print(name)
        return 0

    if args.cmd == "prompt":
        opts = OllamaOptions(temperature=args.temperature)
        if args.stream:
            gen = client.generate(
                args.prompt, args.model, options=opts, format=("json" if args.json else None), stream=True
            )
            _print_stream(gen)  # stream to stdout
        else:
            out = client.generate(
                args.prompt, args.model, options=opts, format=("json" if args.json else None), stream=False
            )
            print(out)
        return 0

    if args.cmd == "chat":
        sess = client.start_chat(args.model, system=args.system, options=OllamaOptions(temperature=args.temperature))
        print(f"Chat started with model '{args.model}'. Type '/exit' to quit.")
        try:
            while True:
                user = input("> ").strip()
                if user.lower() in {"/exit", "/quit"}:
                    break
                gen = sess.send(user, stream=True)
                _print_stream(gen)
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
        return 0

    if args.cmd == "analyze":
        data = analyze_text(client, args.text, args.model, json_mode=not args.no_json)
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
