# byok-voice-assistant

A Windows voice assistant that runs on **your** API key, **your** model, and speaks **your** language.

> **v0.1.0 — the voice loop.** Hold a hotkey, say something, hear the model answer out loud.
> That is all it does, and it does it end to end: your speech is transcribed on your own
> machine, only text leaves it, and the answer is spoken by a Windows voice.

<!-- TODO(v0.1.0): 20-second screen recording goes here, above everything else. -->

```
    you  What is the capital of Turkey?
assistant  Ankara is the capital of Turkey.   297 in, 8 out
● ready    Hold Ctrl+Alt+Space to talk. Ctrl+C stops.
```

## Two things are the point

**Bring your own key.** You choose the provider and the model; the application hardcodes
neither. Today one adapter ships — Google Gemini — but nothing above it knows that: the
agent loop never learns which service is behind it, and one contract suite runs every
adapter through the same tests, so adding the next one changes no line of the loop. The
OpenAI-compatible adapter (OpenAI, OpenRouter, Groq, DeepSeek, Ollama, anything with a
`base_url`) lands in v0.2.0, Anthropic in v0.4.0.

**No language constant anywhere in the code.** The assistant replies in whatever language
you speak to it and switches mid-conversation when you do. Everything else that depends
on language — the speech recogniser's hint, the Windows voice, every sentence the program
shows or says — lives in one TOML file per language. Turkish and English ship complete;
adding a third is copying a template and translating the right-hand side.

## What it does not do yet

| | Arrives in |
|---|---|
| Tools — "open Spotify", "summarise this page" | v0.2.0 |
| The permission gate that asks before it acts | v0.2.0 |
| Sentence-by-sentence speech, which roughly halves the wait | v0.2.0 |
| Endpointing — for now the key decides when you stopped talking | v0.2.0 |
| Reading web pages and mail | v0.3.0 |
| Notes, reminders, a tray icon | v0.4.0 |
| A wake word, and a window | v0.5.0 |

There is no database in v0.1.0 and nothing you say is written to disk.

## Requirements

- Windows 10 or 11
- Python 3.13 and [uv](https://docs.astral.sh/uv/)
- A microphone and a speaker
- A [Google AI Studio](https://aistudio.google.com/apikey) key — the free tier is enough
- **No GPU.** Whisper runs the `small` model in int8 on four CPU threads.

A Windows voice for your language makes the answer intelligible rather than merely
audible. Turkish needs *Microsoft Tolga*, which Windows installs under Settings →
Time & language → Speech.

## Install

```bash
git clone https://github.com/emreux/byok-voice-assistant.git
cd byok-voice-assistant
uv sync
```

## Set it up

```bash
uv run assistant setup
```

Three questions: the language the assistant speaks, your API key, and which model answers.
The key is checked against the provider before anything is written down, and it goes to the
**Windows Credential Manager** — never into a file. Settings land in
`%APPDATA%\assistant\config.toml`, which is plain TOML you can edit by hand.

## Talk to it

```bash
uv run assistant run
```

Wait for the line to say `ready`, then **hold `Ctrl+Alt+Space`, speak, and let go.** The
hotkey works wherever you are; the terminal does not need to be in front. About four
seconds later you hear the answer, and what was said scrolls past above the status line.
`Ctrl+C` stops.

Three things worth knowing:

- **Speak while you hold.** Releasing the key ends the recording. Anything shorter than a
  third of a second is treated as a key touched by accident.
- **Press again to cut it off.** If the assistant is still talking and you press the key,
  it stops immediately and listens — otherwise it would be talking into your microphone.
- **Silence is not answered.** A speech recogniser hands back confident-looking words for a
  recording of nothing; those turns are dropped rather than sent to a model.

## What a turn costs

Measured end to end on the development machine — a four-core laptop with no discrete GPU —
and never estimated.

| | Turkish | English |
|---|---|---|
| Transcription, for about 2.5 s of speech | 2.8 s | 2.1 s |
| The model's answer, in full | 1.3 s | 0.9 s |
| **From the key coming up to the first sound** | **4.1 s** | **3.0 s** |
| Tokens for a one-sentence question | 300 in / 10 out | 297 in / 8 out |

Whisper is loaded once while the program starts, about six seconds, so the first press
never waits for it. Sentence-by-sentence speech in v0.2.0 roughly halves the time to the
first sound.

## Where your data goes

- **Your voice stays on the machine.** Whisper runs locally; only the transcript is sent to
  the provider you chose.
- **Your API key is never written to a file.** It lives in the Windows Credential Manager,
  reached through `keyring`.
- **Nothing you say is written to disk.** The log records what each turn cost — tokens in,
  tokens out — and never what was heard or answered. The conversation is the last twelve
  turns, held in memory, gone when the program ends.
- Settings: `%APPDATA%\assistant\config.toml`. Log:
  `%LOCALAPPDATA%\assistant\Logs\assistant.log`.

## Adding a language

Copy `src/assistant/locales/_template.toml` to `<code>.toml` — `de.toml`, `es.toml`,
`ja.toml` — and translate the right-hand side. That is the whole procedure; no code
changes. A key you leave out is answered in English, so a half-finished pack is usable
rather than broken.

## Adding a provider

Add a row to `src/assistant/defaults/providers.toml`, write an adapter that satisfies the
`LLMProvider` protocol, and give your adapter's test file a `build` function plus one line
in `ADAPTERS` in `tests/test_llm_adapters.py`. The contract suite then asks your adapter
every question it asks the others, unchanged. Nothing else in the project changes — and a
test fails if an adapter is registered without being put through it.

## Development

```bash
uv run ruff check .            # lint
uv run ruff format .           # format
uv run mypy src tests --strict # type check
uv run pytest                  # tests
```

## License

MIT — see [LICENSE](LICENSE).

---

Türkçe: [README.tr.md](README.tr.md)
