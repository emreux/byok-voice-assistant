"""The fast path: short commands answered without the model (design.md section 4, 2.5).

About a fifth of what is said to an assistant is a command of a word or two
- "saat kaç", "dur", "iptal" - and until this module every one of them was
a request to the model: a second or two of waiting and a few hundred
tokens, to be told the time by a tool this program already has. Section 4
lists answering those locally among the free ways of cutting the delay, and
this is that: the phrase is matched here, the intent is acted on in
`app.py`, and the model never hears of it.

**A phrase is matched whole, never found inside a sentence.** "saat kaçta
toplantım var" contains "saat kaçta" and is a question about the calendar;
answered with the time, the assistant would look deaf. What was heard and
what the pack lists are folded the same way - case, accents and punctuation
off, one space between words, by the folding search uses
(`store/normalize.py`) - so "Saat kaç?", "SAAT KAÇ" and the "saat kac" the
recogniser sometimes writes are one phrase, and nothing more is done to
either side.

**The phrases live in the locale pack; the intents live here.** The name of
an intent is English and is the key of the pack's `[intents]` table; the
phrases under it are the language's own (section 3.12). This file carries
no Turkish: `INTENTS` below is the last link of the chain, the English
phrases for a pack that lists none, as `app.YES_WORDS` is for the
confirmation window. A pack replaces the English for the intents it lists
and falls back for the ones it leaves out, intent by intent; an intent a
pack lists and this file does not name is ignored, because nothing would
answer it.

**The fast path skips the model, not the gate.** Matching is this file's
whole job. What an intent does is `app.py`'s: `get_time` runs
`get_current_time` through `dispatch`, the one way any tool runs (invariant
1), so the call is judged and written down in `tool_audit` like the model's
own; `stop` and `cancel` are answered by silence.
"""

from __future__ import annotations

import re

from assistant.locales import Locale
from assistant.store.normalize import normalize_search

__all__ = ["CANCEL", "GET_TIME", "INTENTS", "STOP", "TIME_TOOL", "match_intent"]

# The intents the code can answer: what a pack's `[intents]` table is keyed
# by, and what `app.py` acts on.
GET_TIME = "get_time"
STOP = "stop"
CANCEL = "cancel"

# The tool `get_time` runs, by the name the model would call it by
# (`tools/system.py`). `test_intents.py` checks the registry knows it.
TIME_TOOL = "get_current_time"

# The last link of the chain of section 3.12: the English phrases, for an
# intent the pack lists nothing for. Per intent rather than per pack - a
# translator who wrote down "saat kaç" and forgot "dur" has not thereby
# switched "stop" off.
INTENTS: dict[str, tuple[str, ...]] = {
    GET_TIME: ("what time is it", "what is the time"),
    STOP: ("stop",),
    CANCEL: ("cancel", "never mind"),
}

# A word, for the purpose of comparing a phrase with what was heard: letters
# and digits in any script. Punctuation is where words end.
_WORD = re.compile(r"\w+")


def match_intent(text: str, locale: Locale) -> str | None:
    """The intent `text` is a phrase for, or `None` for a sentence the model
    should hear.

    Whole phrases only: what was heard has to be one of the phrases for the
    intent, folded as the phrase is, with nothing before or after it. The
    pack's phrases answer first, and `INTENTS` for an intent it has none
    for.
    """
    said = _phrase(text)
    if not said:
        return None

    for name, english in INTENTS.items():
        phrases = locale.intents.get(name) or english
        if any(_phrase(phrase) == said for phrase in phrases):
            return name
    return None


def _phrase(text: str) -> str:
    """`text` as one folded phrase: its words, one space apart, nothing else."""
    return " ".join(_WORD.findall(normalize_search(text)))
