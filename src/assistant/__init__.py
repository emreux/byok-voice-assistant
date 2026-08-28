"""A provider-agnostic, language-agnostic Windows desktop voice assistant.

`v0.1.0` is the voice loop and nothing else: hold the hotkey, speak, and hear
the model answer out loud. Speech is transcribed locally, one adapter talks to
one provider through a protocol that names none, and every sentence the product
says comes from a locale pack. Tools and the permission gate that guards them
arrive together in v0.2.0 (design.md section 8).
"""

__all__ = ["__version__"]

# Kept in step with `[project] version` in pyproject.toml by hand: two places,
# both read by a person, and neither worth a build plugin to reconcile.
__version__ = "0.1.0"
