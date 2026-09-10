"""Measures a whole turn on the real pieces (design.md section 4, 2.8): a
recorded question in, the first sound out.

    uv run python scripts/bench_e2e.py                      # every fixture, the configured model
    uv run python scripts/bench_e2e.py --fixtures some/dir
    uv run python scripts/bench_e2e.py --runs 2             # each fixture twice

Each fixture goes through what `assistant run` runs: the real Whisper
(`small`, int8, the pack's vocabulary hint), the real provider with the key
from the Credential Manager, the real gate with the clock as its only tool,
the real Windows voice - and a sound card that writes down when each buffer
reached it instead of playing it. The number section 4 budgets is the time
to the first sound, and playing the answer out would only add its own length
to every clock after it. No app is opened and nothing is written to the
settings; the audit rows go to a database held in memory. Every run costs
tokens.

Three clocks per turn, all from the moment the recording is handed over:
transcription done, first sound, turn over. Reported as p50 and p95, and
separately for turns that ran a tool and turns that did not, because section
4 budgets the two apart and the tool turn is the one that matters. What the
model was asked comes from the recording; whether it reaches for the clock
is its own choice.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_stt import FIXTURES, load_fixtures, percentile

from assistant import locales
from assistant.__main__ import _gate
from assistant.agent.core import Agent
from assistant.agent.limits import Limits
from assistant.app import Assistant, State, Turn
from assistant.config import is_configured, load_settings
from assistant.llm.registry import create_provider
from assistant.store.db import open_database
from assistant.store.repos import AuditRepo
from assistant.stt.base import Audio
from assistant.stt.local_whisper import LocalWhisper
from assistant.tools.registry import ToolRegistry
from assistant.tools.system import get_current_time
from assistant.tts.sapi import SapiTTS


class NoMicrophone:
    """The turns are driven by hand; nobody presses anything."""

    on_listening: Callable[[], None] | None = None

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def mute(self) -> None:
        pass

    def unmute(self) -> None:
        pass

    async def utterance(self) -> Audio:
        raise NotImplementedError

    async def listen_for(self, seconds: float) -> Audio | None:
        return None


class Clock:
    """A sound card that notes when the first buffer arrived and plays nothing."""

    def __init__(self) -> None:
        self.buffers = 0

    async def play(self, buffers: AsyncIterator[bytes], *, sample_rate: int) -> None:
        async for _ in buffers:
            self.buffers += 1

    def stop(self) -> None:
        pass


@dataclass
class Measured:
    """One turn's three clocks, in milliseconds."""

    name: str
    transcribed_ms: float
    first_sound_ms: float | None
    total_ms: float
    turn: Turn


@dataclass
class Bench:
    """The assistant with its clocks attached."""

    assistant: Assistant
    states: list[tuple[float, State]] = field(default_factory=list)
    started: float = 0.0

    def noted(self, state: State) -> None:
        self.states.append((time.perf_counter() - self.started, state))

    def when(self, state: State) -> float | None:
        return next((at for at, seen in self.states if seen is state), None)


async def build(model: str | None) -> Bench:
    settings = load_settings()
    if not is_configured():
        raise SystemExit("nothing is set up: run 'assistant setup' first")
    provider_id, _, model_id = (model or settings.llm.primary).partition(":")
    provider = create_provider(provider_id, base_url=settings.llm.base_url or None)
    pack = locales.load(settings.locale.code)
    limits = Limits.from_settings(settings.limits)
    print(f"model {provider_id}:{model_id}, locale {pack.code}")

    speech = LocalWhisper(vocabulary=[pack.stt_vocabulary])
    started = time.perf_counter()
    await speech.load()
    print(f"whisper loaded in {time.perf_counter() - started:.1f} s")

    tools = ToolRegistry([get_current_time])
    gate = _gate(settings, tools, AuditRepo(open_database(":memory:")), limits=limits, pack=pack)
    bench = Bench(assistant=None)  # type: ignore[arg-type]  # filled just below
    bench.assistant = Assistant(
        capture=NoMicrophone(),
        stt=speech,
        agent=Agent(provider, model=model_id, tools=tools, dispatch=gate, limits=limits),
        tts=SapiTTS(),
        speaker=Clock(),
        locale=pack,
        thinking_timeout=limits.turn_seconds,
        on_state=bench.noted,
        dispatch=gate,
    )
    await bench.assistant.begin()
    return bench


async def one(bench: Bench, name: str, pcm: Audio) -> Measured:
    bench.states.clear()
    bench.started = time.perf_counter()
    turn = await bench.assistant.turn(pcm)
    total = time.perf_counter() - bench.started
    # Transcription is over when the state machine leaves `TRANSCRIBING`:
    # for `THINKING`, or straight to speaking on the fast path.
    after = [at for at, state in bench.states if state is not State.TRANSCRIBING]
    transcribed = after[0] if after else total
    return Measured(
        name=name,
        transcribed_ms=transcribed * 1000,
        first_sound_ms=turn.first_sound_ms,
        total_ms=total * 1000,
        turn=turn,
    )


def show(measured: Measured) -> None:
    turn = measured.turn
    first = "-" if measured.first_sound_ms is None else f"{measured.first_sound_ms:.0f}"
    print(
        f"  {measured.name}: stt {measured.transcribed_ms:.0f} ms, first sound {first} ms, "
        f"over {measured.total_ms:.0f} ms; {turn.tool_calls} tools, "
        f"{turn.usage.input_tokens}/{turn.usage.output_tokens} tokens"
        + (f", intent {turn.intent}" if turn.intent else "")
        + (f", FAILED {turn.failure}" if turn.failure else "")
    )
    print(f"     heard: {turn.heard}")
    print(f"     said : {turn.said}")


def summary(title: str, rows: list[Measured]) -> None:
    if not rows:
        print(f"\n{title}: none")
        return
    print(f"\n{title}: {len(rows)} turns")
    print(f"  {'':<14} {'p50':>9} {'p95':>9}")
    for label, values in (
        ("transcription", [row.transcribed_ms for row in rows]),
        ("first sound", [row.first_sound_ms for row in rows if row.first_sound_ms is not None]),
        ("whole turn", [row.total_ms for row in rows]),
    ):
        if not values:
            continue
        p50, p95 = percentile(values, 0.5), percentile(values, 0.95)
        print(f"  {label:<14} {p50:>7.0f} ms {p95:>7.0f} ms")


async def run(fixtures_dir: Path, *, runs: int, model: str | None) -> int:
    fixtures = load_fixtures(fixtures_dir)
    if not fixtures:
        print(f"nothing to measure: no .wav with a .txt beside it in {fixtures_dir}")
        return 2

    bench = await build(model)
    measured: list[Measured] = []
    for _ in range(runs):
        for fixture in fixtures:
            result = await one(bench, fixture.wav.name, fixture.pcm)
            show(result)
            measured.append(result)

    answered = [row for row in measured if row.turn.failure is None and row.turn.intent is None]
    summary("without a tool", [row for row in answered if row.turn.tool_calls == 0])
    summary("with a tool", [row for row in answered if row.turn.tool_calls > 0])
    fast = [row for row in measured if row.turn.intent is not None]
    if fast:
        summary("fast path, no model", fast)
    failed = [row for row in measured if row.turn.failure is not None]
    if failed:
        print(f"\n{len(failed)} turns failed and are not in the numbers above")
    spent = sum(row.turn.usage.input_tokens + row.turn.usage.output_tokens for row in measured)
    speech = statistics.mean(fixture.seconds for fixture in fixtures)
    print(f"\n{len(measured)} turns, {spent} tokens; mean speech {speech:.1f} s")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fixtures", type=Path, default=FIXTURES, help="where the .wav files are")
    parser.add_argument("--runs", type=int, default=1, help="how many times each fixture")
    parser.add_argument("--model", default=None, help="provider:model; default: the configured one")
    args = parser.parse_args(argv)
    return asyncio.run(run(args.fixtures, runs=args.runs, model=args.model))


if __name__ == "__main__":
    raise SystemExit(main())
