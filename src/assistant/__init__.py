"""A provider-agnostic, language-agnostic Windows desktop voice assistant.

`v0.1.0` was the voice loop and nothing else: hold the hotkey, speak, and hear
the model answer out loud, with speech transcribed locally and every sentence
the product says coming from a locale pack. `v0.2.0` is the first version that
does things: tools behind one permission gate, the risky ones asking out loud
first, every call and every cent written down; short commands answered without
the model; the first sentence spoken while the model writes the second; what
the user asked to be remembered kept between runs; and a second adapter that
speaks to any OpenAI-compatible server (design.md section 8). `v0.4.0` is the
closing release, and the last: pages, the clipboard and mail read as content
inside a marked block; notes found in any spelling; reminders that fire
without the model; a tray icon, autostart, `doctor` and `purge`; Google's
voice as an option beside Windows'; and the third adapter, Anthropic's, so
that one loop and one gate speak to three vendors. Development continues in
a separate project built on live speech-to-speech models.
"""

__all__ = ["__version__"]

# Kept in step with `[project] version` in pyproject.toml by hand: two places,
# both read by a person, and neither worth a build plugin to reconcile.
__version__ = "0.4.0"
