"""Measures the models you have a key for (design.md section 11, phase 2.6).

For each `provider:model` named on the command line - or the one in
`config.toml` when none is - three things are tried against the real
provider, with the key from the Credential Manager:

    1. the tool-use probe of `llm/probe.py`: does the model call
       `get_current_time` when asked the time in Istanbul, and how long
       does its first token take;
    2. a plain question with no tools, timed to the first token and to the
       end, with the token counts the provider reported;
    3. the answer itself, printed, so that its quality in the language of
       the question can be judged by eye.

    uv run python scripts/bench_llm.py
    uv run python scripts/bench_llm.py gemini:gemini-2.5-flash groq:llama-3.3-70b-versatile
    uv run python scripts/bench_llm.py --ask "Türkiye'nin başkenti neresidir? Bir cümleyle."

The probe is the one the setup wizard runs (`llm/probe.py`), called rather
than copied: what this script measures is what the wizard will decide on.
Nothing here writes to the database or to the settings. Every run costs a
few hundred tokens per model.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from assistant import locales
from assistant.config import load_settings
from assistant.llm import probe
from assistant.llm.base import Message, ProviderError, Usage
from assistant.llm.registry import RegistryError, create_provider

# The plain question, when `--ask` names none. In the language of the
# pack in `config.toml`, so that the answer's quality is judged in the
# language the assistant will actually speak; the probe's own question
# comes from the same pack.
SAMPLE = {
    "tr": "Türkiye'nin başkenti neresidir? Bir cümleyle cevap ver.",
    "en": "What is the capital of Türkiye? Answer in one sentence.",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "models",
        nargs="*",
        metavar="provider:model",
        help="which models to measure; the configured one when none is given",
    )
    parser.add_argument("--ask", default=None, help="the plain question to time and print")
    args = parser.parse_args(argv)

    settings = load_settings()
    pack = locales.load(settings.locale.code)
    names = args.models or ([settings.llm.primary] if settings.llm.primary else [])
    if not names:
        print("nothing to measure: run 'assistant setup' or name a provider:model")
        return 2

    question = args.ask or SAMPLE.get(pack.code, SAMPLE["en"])
    probe_question = pack.probe_question or probe.QUESTION
    print(f"probe question : {probe_question}")
    print(f"plain question : {question}\n")

    for name in names:
        asyncio.run(measure(name, question=question, probe_question=probe_question))
        print()
    return 0


async def measure(name: str, *, question: str, probe_question: str) -> None:
    provider_id, _, model = name.partition(":")
    print(f"== {name}")
    if not model:
        print("   expected provider:model")
        return

    try:
        provider = create_provider(provider_id)
    except RegistryError as problem:
        print(f"   {problem}")
        return

    try:
        verdict = await probe.probe_tool_support(provider, model, question=probe_question)
    except ProviderError as refusal:
        print(f"   probe   : refused - {refusal}")
        return
    first = "-" if verdict.first_token_ms is None else f"{verdict.first_token_ms:.0f} ms"
    outcome = "calls the tool" if verdict.ok else f"NO tool call ({verdict.reason})"
    print(f"   probe   : {outcome}; first token {first}")

    started = time.perf_counter()
    first_token: float | None = None
    words: list[str] = []
    usage = Usage()
    try:
        async for delta in provider.stream([Message.user(question)], [], model=model):
            if delta.text:
                if first_token is None:
                    first_token = time.perf_counter() - started
                words.append(delta.text)
            if delta.usage is not None:
                usage = delta.usage
    except ProviderError as refusal:
        print(f"   answer  : refused - {refusal}")
        return
    total = time.perf_counter() - started

    first_ms = "-" if first_token is None else f"{first_token * 1000:.0f} ms"
    print(
        f"   answer  : first token {first_ms}, whole answer {total:.2f} s,"
        f" {usage.input_tokens} in / {usage.output_tokens} out"
    )
    print(f"   said    : {''.join(words).strip()}")


if __name__ == "__main__":
    raise SystemExit(main())
