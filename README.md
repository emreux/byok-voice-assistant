# byok-voice-assistant

A Windows voice assistant that runs on **your** API key, **your** model, and speaks **your** language.

> **v0.2.0 — the first version that does things.** Say "open Spotify" and Spotify opens;
> ask for the time and you hear it before the model is even asked; say "remember that I
> take my coffee black" and it still knows next week. Every action goes through one
> permission gate, the risky ones ask out loud first, and every call and every cent is
> written down. The first sentence of an answer is spoken while the model is still
> writing the second. And the same program talks to Google Gemini or to any
> OpenAI-compatible server - OpenAI, OpenRouter, Groq, DeepSeek, a local Ollama, your own.

<!-- TODO(v0.2.0): 20-second screen recording goes here, above everything else. -->

```
    you  Bluetooth ayarlarını aç.
assistant  Bluetooth ayarları açılıyor.   443 in, 6 out
    you  Saat kaç?
assistant  Saat 12 13.
● ready    Hold Ctrl+Alt+Space to talk, or press Ctrl+Alt+H to keep listening. Ctrl+C stops.
```

## Two things are the point

**Bring your own key.** You choose the provider and the model; the application hardcodes
neither. Two adapters ship: one for Google Gemini, one for everything that speaks the
OpenAI chat API - OpenAI, OpenRouter, Groq, DeepSeek, Ollama without a key, or any server
whose address you type in. Nothing above them knows which is in use: the agent loop, the
tools and the gate never learn which service is behind the words, and one contract suite
runs every adapter through the same tests. Verified end to end against Gemini and against
Google's OpenAI-compatible endpoint; the other servers share that adapter and are tried at
release. Anthropic arrives in v0.4.0.

**No language constant anywhere in the code.** The assistant replies in whatever language
you speak to it and switches mid-conversation when you do. Everything else that depends
on language - the recogniser's hint, the Windows voice, the words that count as yes and
no, the short commands answered without the model, every sentence the program shows or
says - lives in one TOML file per language. Turkish and English ship complete; adding a
third is copying a template and translating the right-hand side.

## What it does

- **Opens things.** An application by name ("Spotify'ı aç" - case, accents and the
  recogniser's spelling are forgiven, and Windows' English names are tried before it
  gives up), a web address, a page of Windows Settings (Bluetooth, Wi-Fi, display,
  sound...). The media keys: play, pause, next, previous, volume.
- **Tells the time without asking anybody.** "Saat kaç", "dur", "iptal" and the other
  short commands the locale pack lists never reach the model: no wait, no tokens. The
  time still comes through the gate, from the same tool the model would call.
- **Asks before it acts, and only then.** Every tool declares its risk. `safe` runs,
  `confirm` is read out to you with the real argument values - "'...' will be
  forgotten. Say yes or no." - and runs only on a clear yes within six seconds; silence,
  a "no" anywhere in the answer, or a key press is a no. `blocked` never runs unless you
  name it in `config.toml`, and even then it asks. There is exactly one path from the
  model's request to a running tool, and `tests/test_policy.py` proves a risky tool cannot
  run unconfirmed.
- **Remembers what you ask it to.** "Bana Emre de", "adın Ada" - kept in
  `%APPDATA%\assistant\memory.toml`, plain text you can edit, read into every request,
  still there after a restart. Forty facts at most, and it says so rather than dropping
  one; forgetting asks first.
- **Speaks as it thinks.** The first sentence is spoken while the model writes the
  second; a tool that takes more than a second gets "bir saniye, bakıyorum" instead of
  silence; a key press cuts the answer *and* the request.
- **Keeps the books.** Every turn is priced from `pricing.toml` and written to
  `usage_log`; `uv run assistant cost` shows today and this month by model. Past $2 a day
  or $30 a month every answer starts with a warning, and `hard_stop = true` stops asking
  the model at all. A turn stops at eight tool calls, the same call three times in a row
  is refused, and an answer cut by the token limit says so at its end.
- **Refuses a model that cannot call tools.** At setup the chosen model is sent one
  question and one tool; a model that answers in prose instead of calling it is not
  accepted. The verdict is kept a week and checked again at startup.

## What it does not do yet

| | Arrives in |
|---|---|
| Reading web pages and mail, a cloud recogniser as an option | v0.3.0 |
| Notes, reminders, a tray icon, the full wizard, Anthropic | v0.4.0 |
| A wake word, a window, MCP servers, packaging | v0.5.0 |

## Requirements

- Windows 10 or 11
- Python 3.13 and [uv](https://docs.astral.sh/uv/)
- A microphone and a speaker
- An API key: [Google AI Studio](https://aistudio.google.com/apikey) (the free tier is
  enough), or one for any OpenAI-compatible service - or a local Ollama, which needs none
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

The provider, the language the assistant speaks, your API key (or the server's address
for your own), and which model answers - from the live list your key can reach. The key
is checked against the provider before anything is written down, and it goes to the
**Windows Credential Manager** - never into a file. Then the model is tested: it gets one
question with one tool, and is accepted only if it calls the tool. Settings land in
`%APPDATA%\assistant\config.toml`, plain TOML you can edit by hand.

Running setup again rewrites `config.toml` from scratch, including `[audio] input_device`
- put it back afterwards if you had set one.

## Talk to it

```bash
uv run assistant run
```

Wait for the line to say `ready`. There are two ways to be heard, and both keys work
wherever you are - the terminal does not need to be in front.

**Hold `Ctrl+Alt+Space`, speak, and let go.** Three to four seconds later you hear the
first sentence of the answer, and what was said scrolls past above the status line.
`Ctrl+C` stops.

**Or press `Ctrl+Alt+H` once and just talk.** The microphone stays live, and each time you
stop speaking for about half a second that sentence becomes a turn. Press it again to
switch back. The status line always says which of the two you are in.

Things worth knowing:

- **Speak while you hold.** Releasing the key ends the recording. Anything shorter than a
  third of a second is treated as a key touched by accident.
- **Press again to cut it off.** If the assistant is still talking and you press
  `Ctrl+Alt+Space`, it stops immediately, drops the request it was waiting on, and
  listens. `Ctrl+Alt+H` does not interrupt; it only switches the mode.
- **When it asks, answer.** A risky tool reads its question and listens for six seconds
  without a key. Say yes or no; when it caught neither it asks once more, and then takes
  silence for no.
- **Hands-free hears the whole room.** A television, a phone call, somebody else talking:
  each is a turn it will try to answer. In a room with other people in it, use the key.
  A wake word that answers only to its name is v0.5.0.
- **It does not hear itself.** The microphone is deaf for as long as the answer lasts, plus
  a quarter of a second for the room to stop repeating it.
- **Silence is not answered.** The recogniser is asked whether the recording held speech at
  all, never how sure it is of the words: a held key over a quiet room says nothing back, a
  sentence it could not read gets "I did not catch that", and words are answered however
  unsure the decoder was of them.

Which microphone all of this listens through is the system default unless you say otherwise.
`uv run python scripts/bench_mic.py --list-devices` prints every device it can see; name yours
by words from its line - `--device "Microphone Array 1"` - on `assistant run` for one
evening with a headset, or as `input_device` under `[audio]` in `config.toml` for good. Words
rather than the index: the indices shift every time a Bluetooth device connects. A device that
will not run at 16 kHz is opened at its own rate and resampled on the way in.

**Which microphone path, measured.** On the development laptop (an Intel Smart Sound array)
the default path with Windows' audio enhancements switched on garbled Whisper: no-speech
0.15-0.43, words wrong. With enhancements off, or on the raw kernel-streaming entry
"Microphone Array 1" that bypasses them, the same sentence came back verbatim at no-speech
0.01-0.04. The raw path is half the level and reads better - the problem was processing, not
volume. If hearing is bad, try the enhancements first, then the raw entry; the shared WASAPI
path was worst of all. Before relying on hands-free from across the room, measure what your
microphone actually picks up from there:

```bash
uv run python scripts/bench_mic.py --quiet        # the room, saying nothing
uv run python scripts/bench_mic.py --at "2 m"     # speaking, from where you sit
uv run python scripts/bench_mic.py --echo         # what the speakers put back in
uv run python scripts/bench_mic.py --fixtures     # the endpoint over your recordings
```

## What a turn costs

Measured end to end on the development machine - a four-core laptop with no discrete GPU -
with Gemini 3.5 Flash-Lite, over fourteen Turkish sentences of 2.6 s on average. The
sentences were synthesised with the Windows voice because the owner's recordings were not
made yet; the times do not depend on the voice, the recognition rate does.

| | p50 | p95 |
|---|---|---|
| Transcription (Whisper `small`, int8, 4 threads) | 2.8 s | 3.3 s |
| **From the recording to the first sound, no tool** | **3.6 s** | **4.2 s** |
| From the recording to the first sound, one tool | 4.7 s | (one turn) |
| "Saat kaç" answered without the model | 3.0 s | - |
| Hands-free adds, waiting for you to stop | +0.6 s | +0.6 s |

Transcription is three quarters of the wait. The three Whisper sizes on the same sentences:
`tiny` 0.56 s at p50 but 38 % of words wrong, `base` 0.96 s and 27 %, `small` 2.9 s and
18 % (half of it foreign app names). The design's gate for local speech recognition -
`small` under 1.2 s at p95 and under 15 % of words wrong - is missed on time, so a cloud
recogniser arrives as an option in v0.3.0; local Whisper stays the default and never leaves,
because your voice never leaves the machine with it. `scripts/bench_stt.py` and
`scripts/bench_e2e.py` measure both on your own recordings (see `fixtures/audio/`).

Whisper is loaded once while the program starts, about three seconds, so the first press
never waits for it.

## Where your data goes

- **Your voice stays on the machine.** Whisper runs locally; only the transcript is sent to
  the provider you chose.
- **Your API key is never written to a file.** It lives in the Windows Credential Manager,
  reached through `keyring`.
- **What you say is not written down; what the assistant did is.** The database at
  `%LOCALAPPDATA%\assistant\assistant.db` holds every tool call (which tool, which
  arguments, what came of it, when), every turn's token counts and price, and the verdict
  on your model. The conversation itself is the last twelve turns, held in memory, gone
  when the program ends. The log records numbers - tokens, tools, price, how long the
  first sound took - and never the words.
- **What you asked it to remember is plain text.** `%APPDATA%\assistant\memory.toml`
  follows you through a roaming profile; edit it, or delete it, by hand.
- Settings: `%APPDATA%\assistant\config.toml`. Prices: `pricing.toml` beside it overrides
  the shipped table. Log: `%LOCALAPPDATA%\assistant\Logs\assistant.log`.

## Adding a language

Copy `src/assistant/locales/_template.toml` to `<code>.toml` - `de.toml`, `es.toml`,
`ja.toml` - and translate the right-hand side: the sentences, the yes and no words, the
short commands, the filler, the tool-test question. That is the whole procedure; no code
changes. A key you leave out is answered in English, so a half-finished pack is usable
rather than broken.

## Adding a provider

If it speaks the OpenAI chat API, add a row to `src/assistant/defaults/providers.toml` with
its address and whether it needs a key, and you are done. If it does not, write an adapter
that satisfies the `LLMProvider` protocol, give its test file a `build` function, and add
one line to `ADAPTERS` in `tests/test_llm_adapters.py`. The contract suite then asks your
adapter every question it asks the others, unchanged - and a test fails if an adapter is
registered without being put through it.

## Development

```bash
uv run ruff check .            # lint
uv run ruff format .           # format
uv run mypy                    # type check, strict, src and tests
uv run pytest                  # tests
```

## License

MIT - see [LICENSE](LICENSE).

---

Türkçe: [README.tr.md](README.tr.md)
