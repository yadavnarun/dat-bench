from __future__ import annotations

from pathlib import Path

import pytest

from datbench import validate
from datbench.validate import FLAGS, WordCheck, capabilities, validate_words


@pytest.fixture(autouse=True)
def _clean_loader_caches():
    # The loaders are functools.cache'd on purpose (web2 is 236k lines), so a test
    # that fakes availability has to invalidate them on both sides.
    validate._reset_caches()
    yield
    validate._reset_caches()


def flags_for(word: str, **kw) -> tuple[str, ...]:
    return validate_words([word], **kw)[0].flags


def only(word: str, **kw) -> WordCheck:
    return validate_words([word], **kw)[0]


# --------------------------------------------------------------- environment

def test_capabilities_reports_what_is_actually_available():
    """capabilities() must agree with ground truth, not with a fixed machine.

    The previous version asserted all three are True, which encoded a macOS
    developer box: /usr/share/dict/words is a BSD/macOS file and Linux CI images
    do not ship it, so the suite failed on a correct report. The contract worth
    testing is that capabilities() never claims a check it cannot perform.
    """
    caps = capabilities()
    assert set(caps) == {"dictionary", "wordnet", "wordfreq"}
    assert all(isinstance(v, bool) for v in caps.values())

    assert caps["dictionary"] is bool(validate._dictionary())

    try:
        import wordfreq  # noqa: F401
        wf = True
    except Exception:
        wf = False
    assert caps["wordfreq"] is wf

    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("cat")
        has_wn = True
    except Exception:
        has_wn = False
    assert caps["wordnet"] is has_wn


def test_flags_constant_matches_contract():
    assert FLAGS == ("empty", "multiword", "not_alpha", "not_in_dict",
                     "not_noun", "proper_noun", "rare", "duplicate")


# --------------------------------------------------------------- clean words

@pytest.mark.parametrize("word", ["cat", "thimble", "quark", "table", "hammer",
                                  "river", "guitar", "china"])
def test_ordinary_nouns_have_no_flags(word):
    check = only(word)
    assert check.flags == ()
    assert check.valid is True


def test_thimble_regression():
    """The DAT paper's own example word: zipf 2.53 only just clears the 2.5 default."""
    check = only("thimble")
    assert check.valid is True
    assert check.flags == ()
    assert check.zipf is not None and check.zipf >= 2.5
    # ...and it is the margin, not luck, that keeps it: nudge the bar past it.
    assert flags_for("thimble", rare_zipf_threshold=2.6) == ("rare",)


# --------------------------------------------------------------- one per flag

def test_empty():
    assert flags_for("") == ("empty",)
    assert flags_for("   \t ") == ("empty",)
    assert only("").zipf is None


def test_multiword():
    check = only("cul de sac")
    assert "multiword" in check.flags
    # Cleaning still yields something embeddable -- that is the point of the hyphen.
    assert check.clean == "cul-de-sac"
    assert "not_in_dict" not in check.flags


def test_hyphenated_compound_is_not_multiword():
    assert "multiword" not in flags_for("cul-de-sac")


def test_not_alpha():
    assert "not_alpha" in flags_for("co2")
    assert "not_alpha" in flags_for("cat's")
    assert "not_alpha" in flags_for("café")
    assert "not_alpha" not in flags_for("cul-de-sac")


def test_not_in_dict():
    assert "not_in_dict" in flags_for("zzzqqxk")
    assert "not_in_dict" not in flags_for("cat")


def test_not_noun():
    assert "not_noun" in flags_for("quickly")
    assert "not_noun" in flags_for("hastily")
    assert "not_noun" not in flags_for("cat")


def test_proper_noun():
    assert "proper_noun" in flags_for("Paris")
    assert "proper_noun" in flags_for("tokyo")
    assert "proper_noun" in flags_for("New York")


def test_rare():
    check = only("xylophone")
    assert "rare" in check.flags
    assert check.zipf is not None and check.zipf < 2.5
    # The threshold is a knob, not a constant.
    assert "rare" not in flags_for("xylophone", rare_zipf_threshold=2.0)
    assert "rare" in flags_for("cat", rare_zipf_threshold=6.0)


def test_duplicate_is_on_the_later_occurrence():
    checks = validate_words(["cat", "thimble", "Cat", "  cat "])
    assert checks[0].flags == ()
    assert checks[1].flags == ()
    assert checks[2].flags == ("duplicate",)   # case-folded to the same clean form
    assert checks[3].flags == ("duplicate",)


# --------------------------------------------------- proper nouns, both ways

def test_lowercase_china_is_a_material_not_a_country():
    assert "proper_noun" not in flags_for("china")
    assert only("china").valid is True


@pytest.mark.parametrize("word", ["Paris", "tokyo", "mongolia", "shakespeare",
                                  "aaron", "jupiter"])
def test_real_proper_nouns_are_flagged(word):
    assert "proper_noun" in flags_for(word)


@pytest.mark.parametrize("word", ["wolf", "root", "shadow", "plastic", "clay",
                                  "cliff", "van"])
def test_propernames_nicknames_do_not_condemn_common_nouns(word):
    """/usr/share/dict/propernames is a login-name list: it contains 'Root',
    'Shadow', 'Plastic'. A lowercase web2 entry has to veto it."""
    assert "proper_noun" not in flags_for(word)


@pytest.mark.parametrize("word", ["tv", "dna"])
def test_initialisms_are_not_proper_nouns(word):
    # WordNet spells these 'TV'/'DNA'; an ALL-CAPS lemma is not evidence of a name.
    assert "proper_noun" not in flags_for(word)


def test_capitalised_only_dictionary_entry_is_a_proper_noun(tmp_path, monkeypatch):
    words = tmp_path / "words"
    words.write_text("china\nChina\nZzyzx\nthimble\nwolf\n", encoding="utf-8")
    propernames = tmp_path / "propernames"
    propernames.write_text("Wolf\nQuxling\n", encoding="utf-8")
    monkeypatch.setattr(validate, "_DICT_PATHS", (words,))
    monkeypatch.setattr(validate, "_PROPERNAME_PATHS", (propernames,))
    monkeypatch.setattr(validate, "_wn", None)   # isolate the dictionary rule
    validate._reset_caches()

    assert "proper_noun" in flags_for("zzyzx")      # capitalised-only entry
    assert "proper_noun" in flags_for("quxling")    # propernames only
    assert "proper_noun" not in flags_for("china")  # both cases listed
    assert "proper_noun" not in flags_for("wolf")   # propernames, but listed lower


# ----------------------------------------------------------- structural rules

def test_flags_are_sorted_and_drawn_from_FLAGS():
    checks = validate_words(["", "New York", "cat's", "zzzqqxk", "quickly",
                            "Paris", "xylophone", "cat", "cat", "  cul  de   sac "])
    for check in checks:
        assert list(check.flags) == sorted(check.flags)
        assert set(check.flags) <= set(FLAGS)
        assert check.valid == (not check.flags)


def test_word_is_preserved_verbatim_and_order_is_kept():
    raw = ["  Cat  ", "cul  de\tsac", "THIMBLE"]
    checks = validate_words(raw)
    assert [c.word for c in checks] == raw
    assert [c.clean for c in checks] == ["cat", "cul-de-sac", "thimble"]


def test_empty_input():
    assert validate_words([]) == []


def test_zipf_is_reported_for_known_words():
    assert only("cat").zipf == pytest.approx(4.78, abs=0.5)


# --------------------------------------------------------- degraded operation

def test_missing_wordnet_skips_noun_checks_and_says_so(monkeypatch):
    monkeypatch.setattr(validate, "_wn", None)
    validate._reset_caches()

    caps = capabilities()
    assert caps["wordnet"] is False
    check = only("quickly")
    assert "not_noun" not in check.flags   # skipped, not fabricated
    if caps["dictionary"]:
        # Where a system word list exists it still answers the English check;
        # without one that check is skipped too, which capabilities() reports.
        assert "not_in_dict" not in check.flags


def test_broken_wordnet_corpus_is_reported_not_raised(monkeypatch):
    class Exploding:
        NOUN = "n"

        def synsets(self, *a, **kw):
            raise LookupError("Resource wordnet not found")

    monkeypatch.setattr(validate, "_wn", Exploding())
    validate._reset_caches()

    assert capabilities()["wordnet"] is False
    assert only("cat").valid is True


def test_missing_wordfreq_skips_rare(monkeypatch):
    monkeypatch.setattr(validate, "_zipf_frequency", None)
    validate._reset_caches()

    assert capabilities()["wordfreq"] is False
    check = only("xylophone")
    assert check.zipf is None
    assert "rare" not in check.flags


def test_missing_dictionary_skips_its_checks(monkeypatch):
    monkeypatch.setattr(validate, "_DICT_PATHS", (Path("/nope/words"),))
    monkeypatch.setattr(validate, "_PROPERNAME_PATHS", (Path("/nope/pn"),))
    monkeypatch.setattr(validate, "_wn", None)
    validate._reset_caches()

    assert capabilities() == {"dictionary": False, "wordnet": False, "wordfreq": True}
    assert flags_for("zzzqqxk") == ("rare",)     # only wordfreq had anything to say
    assert flags_for("Paris") == ()


def test_fully_degraded_flags_nothing_it_cannot_check(monkeypatch):
    monkeypatch.setattr(validate, "_DICT_PATHS", ())
    monkeypatch.setattr(validate, "_PROPERNAME_PATHS", ())
    monkeypatch.setattr(validate, "_wn", None)
    monkeypatch.setattr(validate, "_zipf_frequency", None)
    validate._reset_caches()

    assert capabilities() == {"dictionary": False, "wordnet": False, "wordfreq": False}
    check = only("zzzqqxk")
    assert check.flags == ()
    assert check.valid is True    # nothing loaded => nothing measured; capabilities() tells
    # Structure-only checks still run: they need no corpus.
    assert flags_for("New York") == ("multiword",)
    assert flags_for("co2") == ("not_alpha",)
    assert validate_words(["cat", "cat"])[1].flags == ("duplicate",)


def test_dataclass_is_frozen():
    with pytest.raises(Exception):
        only("cat").valid = False


# --------------------------------------------------- initialisms, both hosts ----
# The proper-noun rule is "attested only capitalised". An ALL-CAPS word list entry
# breaks that: 'TV' and 'DNA' are initialisms for common nouns, not names. The
# WordNet branch always excluded all-caps lemmas; the dictionary branch did not,
# so behaviour depended on the host word list -- macOS web2 happens to list
# lowercase 'tv'/'dna' and hid it, Linux lists only the caps forms and flagged
# both as proper nouns. Found by CI on Ubuntu.

def _dict_from(entries, propernames=()):
    """Build a _Dictionary from a synthetic word list, host files bypassed."""
    import datbench.validate as v
    real_read = v._read_first

    def fake(paths):
        return list(entries) if paths is v._DICT_PATHS else list(propernames)

    v._read_first = fake
    v._reset_caches()
    try:
        return v._dictionary()
    finally:
        v._read_first = real_read
        v._reset_caches()


def test_all_caps_dictionary_entry_is_not_proper_noun_evidence():
    d = _dict_from(["TV", "DNA", "Tokyo", "china", "China", "table"])
    assert "tv" not in d.cap_only, "an initialism is not a name"
    assert "dna" not in d.cap_only
    assert "tokyo" in d.cap_only, "a title-case-only entry still signals a name"
    assert "china" not in d.cap_only, "listed lowercase too, so not a name"


@pytest.mark.parametrize("word", ["tv", "dna"])
def test_initialism_survives_a_caps_only_word_list(word, monkeypatch):
    """Reproduces the Linux list exactly: caps-only, no lowercase form present."""
    import datbench.validate as v
    d = _dict_from(["TV", "DNA", "Tokyo"])
    assert word not in d.cap_only
    assert not v._is_proper_noun([word], None, d), f"{word!r} flagged from a caps-only entry"
    # ...while a real name in the same list still is caught.
    assert v._is_proper_noun(["tokyo"], None, d)


def test_single_letter_caps_entry_still_counts_as_a_name():
    """'A' is one character; the initialism carve-out requires len > 1."""
    d = _dict_from(["A", "apple"])
    assert "a" in d.cap_only
