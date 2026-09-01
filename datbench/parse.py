"""Pull candidate words out of a raw LLM response.

This is the module most able to corrupt the benchmark quietly: a word we fail to
extract is indistinguishable from a word the model never said, and the model gets
marked down for our bug. So every heuristic here leans toward keeping a token and
letting validate.py judge it. The only things deliberately discarded are
conversational scaffolding and chain-of-thought prose.

Deduplication is NOT done here -- a repeated word is real signal about mode
collapse, and validate.py flags it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

__all__ = ["parse_words"]


# Wrappers a model puts around a word. str.strip() with a char set handles every
# combination of them ("**'cat'**") without a regex per shape.
_EMPH_CHARS = "*_`~"
_QUOTE_CHARS = "\"'“”‘’«»„‟‹›′″"

_BULLET_CHARS = (
    "-*+>"
    "•"  # bullet
    "‣"  # triangular bullet
    "◦"  # white bullet
    "⁃"  # hyphen bullet
    "·"  # middle dot
    "▪●○◆"  # squares/circles/diamond
    "–—"  # en/em dash used as a dash-bullet
    "»→➤"  # guillemet / arrows
)

_FENCE_RE = re.compile(r"^\s*(?:```|~~~)")
_HR_RE = re.compile(r"^\s*([-*_=])(?:\s*\1){2,}\s*$")
_HEADING_RE = re.compile(r"^\s*#{1,6}\s*")

# Phrases that announce the answer. Deliberately narrow: a false positive splits a
# genuine list and costs words, so this matches announcements ("final answer",
# "here are the ten words") and not commentary about the items themselves.
_LEAD_IN_RE = re.compile(
    r"\b(?:final\s+(?:answer|list|words)"
    r"|(?:my|the)\s+(?:final\s+)?(?:answer|(?:ten|10)\s+words)"
    r"|here\s+(?:are|is)\b[^.]*\bwords"
    r"|answer\s*(?:is)?\s*$)",
    re.IGNORECASE,
)

_NUM_HEAD_RE = re.compile(r"^\s*[(\[#]?(?P<num>\d{1,3})(?P<rest>.*)$")
# Either a real delimiter (". " ") " " - " ":") or plain whitespace. Requiring one
# of the two is what stops "1985 saw the rise of..." from parsing as item 198.
_NUM_DELIM_RE = re.compile(r"^(?:\s*[)\].:–—-]+\s*|\s+)")
_BULLET_ITEM_RE = re.compile(
    rf"^\s*(?P<marker>[{re.escape(_BULLET_CHARS)}])\s*(?P<body>\S.*)$"
)

# Enumeration markers found mid-line. The trailing \s+ keeps "0.5" intact.
_INLINE_ENUM_RE = re.compile(r"(?:(?<=\s)|^)\(?(?P<num>\d{1,3})[.):\-]\s+")
_INLINE_BULLET_RE = re.compile(r"(?:(?<=\s)|^)[•‣◦▪●○⁃]\s*")

_LEAD_PUNCT_RE = re.compile(r"^[\s.,;:!?…\-–—]+")
_TRAIL_PUNCT_RE = re.compile(r"[\s.,;:!?…、。\-–—]+$")
_INLINE_SPLIT_RE = re.compile(r"[,;，；]")

# A trailing gloss: "cat - a small animal", "cat: feline", "cat (an animal)".
# Only the left side is kept, so the gloss cannot become a candidate of its own.
_ANNOTATION_RE = re.compile(r"\s+[-–—]\s+|\s*:\s+|\s+[(\[]")

# Letters, plus internal hyphen/apostrophe. "cul-de-sac" and "o'clock" pass; "H2O"
# and "1." do not.
_WORD_TOKEN_RE = re.compile(r"^[^\W\d_]+(?:['’-][^\W\d_]+)*$", re.UNICODE)

# Pure punctuation ("|---|---|" out of a markdown table) is structure, never an
# answer. A digit still counts as content, so "42" survives to be flagged
# not_alpha by validate.py rather than vanishing here.
_ALNUM_RE = re.compile(r"[^\W_]", re.UNICODE)

# A candidate from unstructured text may be at most this many whitespace tokens.
# "New York" and "ice cream" survive; a sentence of refusal prose does not.
_MAX_PLAIN_TOKENS = 3

_BOILERPLATE_EXACT = frozenset({
    "sure", "sure thing", "certainly", "absolutely", "of course", "okay", "ok",
    "alright", "all right", "got it", "understood", "no problem", "here you go",
    "here it is", "enjoy", "thanks", "thank you", "you're welcome", "voila",
    "voilà", "final answer", "final answers", "answer", "final", "words",
    "word list", "my words", "my list", "the words", "no", "none", "n/a", "na",
    "sorry",
})

_BOILERPLATE_PREFIX_RE = re.compile(
    r"^(?:"
    r"here (?:are|is)|here's|below (?:are|is)|these are|those are|"
    r"i (?:hope|will|am|can|cannot|can't|could|would|don't|do|need|think)|"
    r"i'll|i'm|i've|let me|let's|feel free|hope (?:this|that|it)|"
    r"as (?:an? )?(?:ai|language model|requested)|would you|do you|if you|"
    r"note that|please|sorry|unfortunately|thanks for|thank you for|"
    r"that's (?:all|it)|good luck|each of these|all of these"
    r")\b",
    re.IGNORECASE,
)

# Function words and refusal verbs. Applied ONLY to multi-token candidates from
# unstructured text, so a single word is never dropped for being on this list --
# "can" and "will" are nouns to validate.py, not ours to veto.
_PROSE_TOKENS = frozenset({
    "i", "i'm", "im", "i'll", "i've", "you", "your", "we", "he", "she", "they",
    "it", "is", "am", "are", "was", "were", "be", "been", "being", "do", "does",
    "did", "not", "no", "cannot", "can't", "won't", "don't", "doesn't", "the",
    "a", "an", "of", "to", "and", "but", "or", "as", "that", "this", "these",
    "those", "with", "for", "from", "in", "on", "at", "by", "sorry", "unable",
    "help", "please", "here", "there", "would", "could", "should", "will",
    "just", "very", "my", "me", "them", "us", "any", "some", "all", "so",
})

_JSON_WORD_KEYS = ("words", "word_list", "final_words", "final_answer",
                   "answer", "items", "list", "output", "result")
_JSON_ITEM_KEYS = ("word", "term", "text", "value", "name")


def parse_words(text: str, *, want: int = 10) -> list[str]:
    """Extract at most `want` candidate words from a raw response, order preserved.

    Strategies are tried in the CONTRACT order -- JSON, list markers, then
    unstructured text -- and the first one that finds anything wins.
    """
    if not isinstance(text, str) or not text.strip() or want <= 0:
        return []

    lines = _prepare_lines(text)
    for candidates in (_from_json(text),
                       _from_list_blocks(lines, want),
                       _from_plain(lines)):
        if candidates:
            return candidates[:want]
    return []


# --- shared cleaning ------------------------------------------------------------

def _strip_wrappers(s: str) -> str:
    prev = None
    while s != prev:
        prev = s
        s = s.strip().strip(_EMPH_CHARS).strip(_QUOTE_CHARS)
    return s


def _clean(s: str) -> str:
    """Wrappers, leading/trailing punctuation, collapsed whitespace. Never None."""
    prev = None
    while s != prev:
        prev = s
        s = _strip_wrappers(s)
        s = _TRAIL_PUNCT_RE.sub("", s)
        s = _LEAD_PUNCT_RE.sub("", s)
    return " ".join(s.split())


def _is_boilerplate(s: str) -> bool:
    norm = " ".join(s.lower().replace("’", "'").split())
    norm = norm.strip(" .,;:!?-–—\"'*_")
    if not norm:
        return True
    return norm in _BOILERPLATE_EXACT or bool(_BOILERPLATE_PREFIX_RE.match(norm))


def _keep(s: str) -> list[str]:
    cand = _clean(s)
    if not cand or not _ALNUM_RE.search(cand) or _is_boilerplate(cand):
        return []
    return [cand]


def _is_wordlike(s: str) -> bool:
    tokens = s.split()
    return (1 <= len(tokens) <= _MAX_PLAIN_TOKENS
            and all(_WORD_TOKEN_RE.match(t) for t in tokens))


def _strip_annotation(s: str) -> str:
    m = _ANNOTATION_RE.search(s)
    if m and s[:m.start()].strip():
        return s[:m.start()]
    return s


# --- 1. JSON --------------------------------------------------------------------

def _match_bracket(text: str, start: int) -> int | None:
    """Index of the bracket closing text[start], or None if it never closes."""
    want_close = "]" if text[start] == "[" else "}"
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 0:
                return i if ch == want_close else None
    return None


def _loads(span: str) -> object | None:
    try:
        return json.loads(span)
    except ValueError:
        pass
    relaxed = re.sub(r",\s*([\]}])", r"\1", span)  # trailing comma
    try:
        return json.loads(relaxed)
    except ValueError:
        pass
    # A Python literal (['cat', 'dog']) is common enough to be worth rescuing, but
    # only when no double quote can collide with the swap.
    if '"' not in relaxed and "'" in relaxed:
        try:
            return json.loads(relaxed.replace("'", '"'))
        except ValueError:
            return None
    return None


def _get_ci(obj: dict, key: str) -> object | None:
    for k, v in obj.items():
        if isinstance(k, str) and k.strip().lower().replace(" ", "_") == key:
            return v
    return None


def _json_candidates(obj: object) -> list[str]:
    if isinstance(obj, str):
        out: list[str] = []
        for piece in _INLINE_SPLIT_RE.split(obj):
            out.extend(_keep(piece))
        return out
    if isinstance(obj, list):
        out = []
        for item in obj:
            if isinstance(item, str):
                out.extend(_keep(item))
            elif isinstance(item, dict):
                for key in _JSON_ITEM_KEYS:
                    val = _get_ci(item, key)
                    if isinstance(val, str):
                        out.extend(_keep(val))
                        break
        return out
    if isinstance(obj, dict):
        for key in _JSON_WORD_KEYS:
            val = _get_ci(obj, key)
            if isinstance(val, (list, str)):
                found = _json_candidates(val)
                if found:
                    return found
    return []


def _from_json(text: str) -> list[str] | None:
    best: tuple[tuple[int, int], list[str]] | None = None
    i = 0
    while i < len(text):
        if text[i] not in "[{":
            i += 1
            continue
        end = _match_bracket(text, i)
        if end is None:
            i += 1
            continue
        obj = _loads(text[i:end + 1])
        found = _json_candidates(obj) if obj is not None else []
        if found:
            # Later wins, so a final answer beats an illustrative array in the
            # reasoning -- but a real list beats a one-element stray either way.
            key = (1 if len(found) >= 2 else 0, i)
            if best is None or key > best[0]:
                best = (key, found)
        i = end + 1
    return best[1] if best else None


# --- 2 & 3. numbered and bulleted lists -----------------------------------------

@dataclass(frozen=True)
class _Item:
    kind: str          # "num" | "bullet"
    num: int | None
    body: str


def _explode_inline_list(line: str) -> list[str]:
    """Split "1. cat 2. dog 3. bird" (and the bullet equivalent) into lines.

    Small models answer on one line often enough that not doing this loses the
    whole run. Two markers with rising numbers is the guard against splitting
    prose that merely mentions a number.
    """
    hits = list(_INLINE_ENUM_RE.finditer(line))
    nums = [int(m["num"]) for m in hits]
    if not (len(hits) >= 2 and all(b > a for a, b in zip(nums, nums[1:]))):
        hits = list(_INLINE_BULLET_RE.finditer(line))
        if len(hits) < 2:
            return [line]
    bounds = [m.start() for m in hits] + [len(line)]
    return [line[bounds[i]:bounds[i + 1]] for i in range(len(hits))]


def _prepare_lines(text: str) -> list[str]:
    lines = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        # A fence or a rule is a separator, not a list break: blanking it keeps
        # items on either side of it in one block.
        if _FENCE_RE.match(raw) or _HR_RE.match(raw):
            lines.append("")
            continue
        lines.extend(_explode_inline_list(raw))
    return lines


def _num_item(line: str) -> tuple[int, str] | None:
    m = _NUM_HEAD_RE.match(line)
    if not m:
        return None
    delim = _NUM_DELIM_RE.match(m["rest"])
    if not delim:
        return None
    body = m["rest"][delim.end():]
    if not body.strip():
        return None
    return int(m["num"]), body


def _classify(line: str) -> _Item | None:
    s = _HEADING_RE.sub("", line)
    if not s.strip():
        return None
    numbered = _num_item(s)
    if numbered is not None:
        return _Item("num", numbered[0], numbered[1])
    m = _BULLET_ITEM_RE.match(s)
    if m:
        marker, body = m["marker"], m["body"]
        # "*cat*" and "**cat**" are emphasis, not bullets. Misreading them as a
        # list builds a block that then hides every unmarked line around it.
        if marker in "*_" and (body.startswith(marker)
                               or body.rstrip().endswith(marker)):
            return None
        inner = _num_item(body)  # "- 1. cat"
        if inner is not None:
            body = inner[1]
        return _Item("bullet", None, body)
    return None


def _is_lead_in(line: str) -> bool:
    """Does this prose line announce that the answer follows?

    Needed because the numeric-restart signal below is structurally unavailable to
    bullets: _classify gives every bullet num=None, so bullet -> bullet can never
    trigger it. Without this, a brainstorm bulleted list, one lead-in line, and the
    answer bulleted list merge into a single block, and the reasoning words get
    scored while the tail of the real answer is truncated away. A lead-in nearly
    always ends in a colon or names the answer; a per-item annotation
    ("(all common nouns)") does neither, so annotation tolerance survives.
    """
    s = _HEADING_RE.sub("", line).strip()
    s = s.strip("*_` ")
    if not s:
        return False
    if s.endswith(":"):
        return True
    return bool(_LEAD_IN_RE.search(s))


def _blocks(lines: list[str]) -> list[list[str]]:
    """Group list items into blocks. Blank lines never split a block."""
    blocks: list[list[str]] = []
    current: list[str] | None = None
    kind: str | None = None
    last_num: int | None = None
    prose = 0
    lead_in = False
    for line in lines:
        if not line.strip():
            continue
        item = _classify(line)
        if item is None:
            prose += 1
            lead_in = lead_in or _is_lead_in(line)
            continue
        # Numbering back to 1 after a higher number is the strongest available
        # signal that a new list began -- an answer list always starts at 1, so a
        # CoT preamble list gets cut off here instead of merged with the answer.
        # A repeated number ("1. / 1. / 1.") is just a markdown habit and must not
        # split anything. One stray line is tolerated (models annotate items); a
        # paragraph is not. A lead-in line splits on its own, whatever the kind --
        # see _is_lead_in for why bullets need it.
        restart = (
            current is None
            or kind != item.kind
            or prose >= 2
            or lead_in
            or (item.num is not None and last_num is not None
                and item.num <= 1 < last_num)
        )
        if restart:
            current = []
            blocks.append(current)
            kind = item.kind
        current.append(item.body)
        last_num = item.num
        prose = 0
        lead_in = False
    return blocks


def _block_candidates(bodies: list[str]) -> list[str]:
    out: list[str] = []
    for body in bodies:
        out.extend(_keep(_strip_annotation(_strip_wrappers(body))))
    return out


def _from_list_blocks(lines: list[str], want: int) -> list[str] | None:
    best: tuple[tuple[int, int], list[str]] | None = None
    for idx, bodies in enumerate(_blocks(lines)):
        items = _block_candidates(bodies)
        # A lone marked line is not a list; leaving it to the plain path means a
        # response mixing "cat\ndog\n- bird" keeps all three.
        if len(items) < 2:
            continue
        wordlike = sum(1 for it in items if _is_wordlike(it)) / len(items)
        # Below this the block is reasoning or a restatement of the rules, not an
        # answer. Nothing is lost by falling through: the plain path reads the
        # same lines with the markers stripped and its own prose gate.
        if wordlike < 0.6:
            continue
        # Fuller list first, then the later one: that is what makes the final
        # list of a CoT response win over the domain list in its reasoning.
        key = (min(len(items), want), idx)
        if best is None or key > best[0]:
            best = (key, items)
    return best[1] if best else None


# --- 4 & 5. bare lines, comma/semicolon runs ------------------------------------

def _strip_markers(line: str) -> str:
    s = _HEADING_RE.sub("", line)
    numbered = _num_item(s)
    if numbered is not None:
        s = numbered[1]
    m = _BULLET_ITEM_RE.match(s)
    if m:
        s = m["body"]
    return s.strip()


def _apply_colon_rule(line: str) -> str:
    """CONTRACT: keep the part after the final colon when the part before is short.

    Two extra escapes, both of which exist to avoid *losing* a word: a long
    preamble ("Sure! Here is my list of ten words: cat, dog") still yields its
    tail, and a gloss ("cat: a small animal") yields its head instead.
    """
    if ":" not in line:
        return line
    head, _, tail = line.rpartition(":")
    head, tail = head.strip(), tail.strip()
    if not tail:
        return ""  # a line that ends in a colon is a header, never a word
    tail_split = bool(_INLINE_SPLIT_RE.search(tail))
    if (len(head.split()) == 1 and _is_wordlike(head) and not _is_boilerplate(head)
            and not tail_split and len(tail.split()) > 1):
        return head
    if (len(head.split()) <= 4
            or _is_boilerplate(head)
            or (tail_split and not _INLINE_SPLIT_RE.search(head))):
        return tail
    return line


def _from_plain(lines: list[str]) -> list[str]:
    out: list[str] = []
    for raw in lines:
        line = _strip_wrappers(_strip_markers(raw))
        if not line:
            continue
        line = _apply_colon_rule(line)
        if not line:
            continue
        # Boilerplate is judged per piece, not per line: "I need 10 words. cat,
        # dog" opens with a filler phrase and still ends with two real answers.
        for piece in _INLINE_SPLIT_RE.split(line):
            # Again per piece: "Sure, here is my list: cat, dog" only exposes its
            # colon once the commas are split off.
            cand = _clean(_apply_colon_rule(piece.strip()))
            if not cand or not _ALNUM_RE.search(cand) or _is_boilerplate(cand):
                continue
            tokens = cand.split()
            if len(tokens) > _MAX_PLAIN_TOKENS:
                continue  # prose, not a word
            if len(tokens) > 1 and any(
                t.lower().strip(_QUOTE_CHARS) in _PROSE_TOKENS for t in tokens
            ):
                continue
            out.append(cand)
    return out
