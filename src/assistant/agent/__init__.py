"""The agent: the loop, the prompt, and later the gate every tool goes through.

Nothing in this package knows which provider is answering. It sees the protocol
of `llm/base.py` and nothing else, which is what lets the same loop run against
Gemini, Anthropic and any OpenAI-compatible endpoint (design.md section 3.2).

Phase 1 was `core.py` and `prompts.py` - a loop with no tools in it. Phase 2.1
added `policy.py`, the one permission gate of section 3.9, and gave the loop
its `while`; 2.4 added `limits.py`, the limits of section 3.11 the loop asks
before every call. `intents.py` and `memory.py` arrive with the steps that
need them; do not assume a module exists because section 7 lists it.
"""

__all__: list[str] = []
