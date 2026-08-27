"""The provider layer: one narrow protocol, one adapter per vendor protocol.

Nothing outside this package may import a provider SDK. `agent/`, `tools/` and
`policy.py` see only what `base.py` declares, which is what lets the same agent
loop run against Gemini, Anthropic and any OpenAI-compatible endpoint
(design.md section 3.2).
"""

__all__: list[str] = []
