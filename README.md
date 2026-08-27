# byok-voice-assistant

A Windows desktop voice assistant that runs on **your** API key, **your** model, and speaks **your** language.

> **Status: phase 1 — the voice loop is being built.** The package installs and the
> provider layer is in place; `setup` and `run` are not finished yet.

<!-- TODO(phase-1): 30-second demo video goes here, above everything else. -->

## Why another voice assistant

Most open-source voice assistants pin you to one provider and one language. This one does neither.

- **Bring your own key.** Three adapters cover 15+ providers — Anthropic, OpenAI, Google Gemini, OpenRouter, Groq, DeepSeek, xAI, Mistral, Together, Ollama, and any OpenAI-compatible endpoint. You choose provider and model in a first-run wizard; the application never hardcodes one.
- **Speaks your language.** The assistant replies in whatever language you speak to it and switches mid-conversation when you do. The interface language is a configuration value, not a constant in the code — adding one is a single TOML file.
- **Local first.** With local Whisper, your audio never leaves the machine; only text reaches the cloud. `assistant doctor` reports exactly which data goes to which provider.
- **Asks before it acts.** Every tool call passes through a single permission gate that defaults to "confirm". Sending mail, deleting files, and changing your calendar are read aloud and wait for a spoken yes.

## Architecture

<!-- TODO(phase-1): architecture diagram (SVG) goes here. -->

Speech is transcribed locally, the text goes to whichever model you configured, and the
reply is spoken back. Provider-specific code is confined to one adapter per vendor, so the
agent loop never learns which service is behind it.

## Latency

Measured end to end on the development machine, never estimated.

<!-- TODO(phase-0.3 onward): fill from scripts/bench_e2e.py at the end of each phase. -->

| Turn type | p50 | p95 |
|---|---|---|
| *not measured yet* | — | — |

## Requirements

- Windows 10 or 11
- Python 3.13+
- An API key for at least one supported provider

<!-- TODO(phase-0.4): recommended microphone, decided by scripts/bench_mic.py. -->

## Privacy

Notes, conversation history, and tool results are stored as plaintext SQLite under `%LOCALAPPDATA%\assistant\`, readable by your Windows account only. Full-disk encryption (BitLocker) is your responsibility. `assistant purge --all` deletes the database and every stored credential.

API keys go to Windows Credential Manager through `keyring`, never to disk in plaintext.

## Installation

<!-- TODO(phase-1): installation instructions once the package builds. -->

Not yet installable.

## License

MIT — see [LICENSE](LICENSE).

---

Türkçe: [README.tr.md](README.tr.md)
