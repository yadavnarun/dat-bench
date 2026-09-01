"""Apply the DAT's five word rules to the candidates parse.py pulled out of a response.

Every check sits behind an optional dependency, and every one of them degrades to
"skipped" rather than to "passed" -- see capabilities(). A benchmark that quietly
stopped checking nouns would publish a validity rate it never measured.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WordCheck:
    word: str
    clean: str
    valid: bool
    flags: tuple[str, ...]
    zipf: float | None


FLAGS = ("empty", "multiword", "not_alpha", "not_in_dict",
         "not_noun", "proper_noun", "rare", "duplicate")

# Darwin ships web2 as /usr/share/dict/words; the rest are Linux package layouts.
_DICT_PATHS: tuple[Path, ...] = (Path("/usr/share/dict/words"),
                                 Path("/usr/share/dict/web2"),
                                 Path("/usr/dict/words"))
_PROPERNAME_PATHS: tuple[Path, ...] = (Path("/usr/share/dict/propernames"),)

_NON_ALPHA_RE = re.compile(r"[^a-z-]")
_WS_RE = re.compile(r"\s+")

try:  # nltk and wordfreq live in the [validity] extra, so both are optional
    from nltk.corpus import wordnet as _wn
except Exception:  # pragma: no cover - only on a stripped install
    _wn = None  # type: ignore[assignment]

try:
    from wordfreq import zipf_frequency as _zipf_frequency
except Exception:  # pragma: no cover
    _zipf_frequency = None  # type: ignore[assignment]


@dataclass(frozen=True)
class _Dictionary:
    lower: frozenset[str]        # attested in lowercase form
    cap_only: frozenset[str]     # attested only capitalised -- the proper-noun signal
    propernames: frozenset[str]


@functools.cache
def _wordnet():
    """The corpus reader, or None when nltk or the WordNet corpus is missing."""
    if _wn is None:
        return None
    try:
        # LazyCorpusLoader defers the LookupError to first real use, so probe now.
        _wn.synsets("cat", pos=_wn.NOUN)
    except Exception:
        return None
    return _wn


@functools.cache
def _wordfreq() -> Callable[..., float] | None:
    if _zipf_frequency is None:
        return None
    try:
        _zipf_frequency("cat", "en")
    except Exception:
        return None
    return _zipf_frequency


@functools.cache
def _dictionary() -> _Dictionary | None:
    """web2 + propernames, read once: validate_words runs thousands of times."""
    entries = _read_first(_DICT_PATHS)
    if entries is None:
        return None
    lower: set[str] = set()
    capitalised: set[str] = set()
    for entry in entries:
        folded = entry.lower()
        if entry == folded:
            lower.add(folded)
        elif entry.isupper() and len(entry) > 1:
            # An ALL-CAPS entry is an initialism ('TV', 'DNA'), not evidence of a
            # name -- the same rule the WordNet branch of _is_proper_noun already
            # applies. Without this the two sources disagree, and which one you
            # get depends on the host: macOS web2 lists lowercase 'tv' and 'dna'
            # so the bug stayed hidden, while Linux word lists carry only 'TV'
            # and 'DNA' and flagged both as proper nouns.
            # Treated as a lowercase attestation so it also vetoes propernames.
            lower.add(folded)
        else:
            capitalised.add(folded)
    return _Dictionary(
        lower=frozenset(lower),
        # Subtract the lowercase set: web2 lists both 'China' and 'china', and only
        # the words it never lists lowercase are evidence of a proper noun.
        cap_only=frozenset(capitalised - lower),
        propernames=frozenset(w.lower() for w in _read_first(_PROPERNAME_PATHS) or ()),
    )


def _read_first(paths: Sequence[Path]) -> list[str] | None:
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        return [line.strip() for line in text.splitlines() if line.strip()]
    return None


def _reset_caches() -> None:
    """Drop the loader caches. Only tests need this, after faking availability."""
    _wordnet.cache_clear()
    _wordfreq.cache_clear()
    _dictionary.cache_clear()


def _clean(word: str) -> tuple[str, bool]:
    """-> (cleaned form, whether the raw word had internal whitespace).

    Internal whitespace collapses to one hyphen because that is how the reference
    DAT implementation looks compounds up ('cul de sac' -> 'cul-de-sac'). The
    whitespace fact is returned separately: the word stays embeddable while still
    being reported as not-a-single-token.
    """
    stripped = _WS_RE.sub(" ", word.strip().lower())
    return stripped.replace(" ", "-"), " " in stripped


def _spellings(clean: str) -> tuple[str, ...]:
    if "-" not in clean:
        return (clean,)
    # WordNet joins compounds with underscores and web2 often writes them solid, so
    # a compound has to be probed all three ways or it looks like it is not English.
    return (clean, clean.replace("-", "_"), clean.replace("-", ""))


def validate_words(words: Sequence[str], *,
                   rare_zipf_threshold: float = 2.5) -> list[WordCheck]:
    wordnet = _wordnet()
    zipf_of = _wordfreq()
    dictionary = _dictionary()

    checks: list[WordCheck] = []
    seen: set[str] = set()
    for raw in words:
        clean, multiword = _clean(raw)
        flags: set[str] = set()

        if clean in seen:
            flags.add("duplicate")   # the later occurrence is the duplicate one
        seen.add(clean)

        zipf: float | None = None
        if not clean:
            flags.add("empty")
        else:
            if multiword:
                flags.add("multiword")
            if _NON_ALPHA_RE.search(clean):
                flags.add("not_alpha")

            spellings = _spellings(clean)
            if wordnet is not None or dictionary is not None:
                in_wordnet = wordnet is not None and any(
                    wordnet.synsets(s) for s in spellings)
                in_dict = dictionary is not None and any(
                    s in dictionary.lower or s in dictionary.cap_only
                    for s in spellings)
                if not (in_wordnet or in_dict):
                    flags.add("not_in_dict")

            if wordnet is not None and not any(
                    wordnet.synsets(s, pos=wordnet.NOUN) for s in spellings):
                flags.add("not_noun")

            if _is_proper_noun(spellings, wordnet, dictionary):
                flags.add("proper_noun")

            if zipf_of is not None:
                zipf = zipf_of(clean, "en")
                if zipf < rare_zipf_threshold:
                    flags.add("rare")

        checks.append(WordCheck(word=raw, clean=clean, valid=not flags,
                                flags=tuple(sorted(flags)), zipf=zipf))
    return checks


def _is_proper_noun(spellings: Sequence[str], wordnet,
                    dictionary: _Dictionary | None) -> bool:
    """Two independent votes; either is enough, neither can be faked when absent.

    web2 alone misses names it does not list at all ('tokyo'), and WordNet alone
    misses names it does not know, so the two sources cover different gaps. Each
    votes only when it has no lowercase attestation of the word.
    """
    if dictionary is not None:
        # The lowercase veto has to cover propernames too: that file is a list of
        # login nicknames and contains 'Root', 'Shadow', 'Plastic', 'Wolf'.
        if any(s not in dictionary.lower
               and (s in dictionary.cap_only or s in dictionary.propernames)
               for s in spellings):
            return True
    if wordnet is not None:
        for spelling in spellings:
            names = [lemma.name() for syn in wordnet.synsets(spelling)
                     for lemma in syn.lemmas()
                     if lemma.name().lower() == spelling]
            # An ALL-CAPS lemma is an initialism ('TV', 'CO2'), not evidence of a
            # name, and treating it as one would invalidate ordinary words.
            titled = [n for n in names if n[:1].isupper() and not n.isupper()]
            if titled and not any(n[:1].islower() for n in names):
                return True
    return False


def capabilities() -> dict[str, bool]:
    return {
        "dictionary": _dictionary() is not None,
        "wordnet": _wordnet() is not None,
        "wordfreq": _wordfreq() is not None,
    }
