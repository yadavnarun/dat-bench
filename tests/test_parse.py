from __future__ import annotations

import pytest

from datbench.parse import parse_words

TEN = ["cat", "thimble", "avalanche", "grief", "tuba",
       "kelp", "pension", "quasar", "wrench", "mango"]


# --- 1. JSON ---------------------------------------------------------------------

def test_json_array_alone():
    assert parse_words('["cat", "dog", "bird"]') == ["cat", "dog", "bird"]


def test_json_array_inside_a_fenced_block():
    text = (
        "Sure! Here are ten words.\n\n"
        "```json\n"
        '["cat", "thimble", "avalanche", "grief", "tuba",\n'
        ' "kelp", "pension", "quasar", "wrench", "mango"]\n'
        "```\n\n"
        "Let me know if you want another set.\n"
    )
    assert parse_words(text) == TEN


def test_json_array_mid_sentence():
    text = 'My answer is ["cat", "dog", "bird"] -- hope that helps!'
    assert parse_words(text) == ["cat", "dog", "bird"]


def test_json_object_with_words_key():
    text = '{"reasoning": "picked far-apart domains", "words": ["cat", "dog"]}'
    assert parse_words(text) == ["cat", "dog"]


def test_json_object_words_key_is_case_insensitive_and_may_be_a_string():
    assert parse_words('{"Words": "cat, dog, bird"}') == ["cat", "dog", "bird"]


def test_json_survives_a_python_literal_and_a_trailing_comma():
    assert parse_words("['cat', 'dog']") == ["cat", "dog"]
    assert parse_words('["cat", "dog",]') == ["cat", "dog"]


def test_json_list_of_objects():
    text = '[{"word": "cat"}, {"word": "dog"}]'
    assert parse_words(text) == ["cat", "dog"]


def test_json_the_final_array_wins_over_one_quoted_in_the_reasoning():
    text = 'I first considered ["dog"], too close to cat. Final: ["cat", "tuba"]'
    assert parse_words(text) == ["cat", "tuba"]


def test_unparseable_brackets_are_not_json():
    # "[brackets]" is not JSON; the sentence must still yield its two words.
    assert parse_words("I thought about [brackets] but here: cat, dog") == ["cat", "dog"]


# --- 2. numbered lists -----------------------------------------------------------

@pytest.mark.parametrize("sep", [". ", ") ", " - ", ": ", ".", " ", "- ", " – "])
def test_numbered_list_delimiters(sep):
    text = "\n".join(f"{i}{sep}{w}" for i, w in enumerate(TEN, start=1))
    assert parse_words(text) == TEN


def test_numbered_list_with_blank_lines_between_items():
    text = "\n\n".join(f"{i}. {w}" for i, w in enumerate(TEN, start=1))
    assert parse_words(text) == TEN


def test_numbered_list_in_parentheses_form():
    assert parse_words("(1) cat\n(2) dog\n(3) bird") == ["cat", "dog", "bird"]


def test_a_year_is_not_a_list_item():
    # "1985 saw..." must not parse as item 198 with body "5 saw...".
    text = "1985 saw a lot of things. Anyway: cat, dog"
    assert parse_words(text) == ["cat", "dog"]


def test_repeated_item_number_does_not_split_the_list():
    # Models emit "1. / 1. / 1." (a markdown habit) and "1. / 2. / 2." (a slip).
    assert parse_words("1. cat\n1. dog\n1. bird") == ["cat", "dog", "bird"]
    assert parse_words("1. cat\n2. dog\n2. bird") == ["cat", "dog", "bird"]


# --- 3. bulleted lists -----------------------------------------------------------

@pytest.mark.parametrize("bullet", ["-", "*", "+", "•", "‣", "◦",
                                    "–", "→", ">"])
def test_bulleted_list_markers(bullet):
    text = "\n".join(f"{bullet} {w}" for w in TEN)
    assert parse_words(text) == TEN


def test_bulleted_items_with_emphasis_and_a_gloss():
    text = "- **Cat** - a domestic feline\n- **Tuba** (a brass instrument)\n- **Mango**: a fruit"
    assert parse_words(text) == ["Cat", "Tuba", "Mango"]


# --- 4. bare newline-separated words ---------------------------------------------

def test_bare_newline_separated_words():
    assert parse_words("\n".join(TEN)) == TEN


def test_bare_words_with_stray_blank_and_marker_only_lines():
    assert parse_words("cat\n\n   \n**\n---\ndog") == ["cat", "dog"]


# --- 5. comma / semicolon on one line --------------------------------------------

def test_comma_separated_on_one_line():
    assert parse_words(", ".join(TEN)) == TEN


def test_semicolon_separated_on_one_line():
    assert parse_words("; ".join(TEN)) == TEN


# --- cleaning --------------------------------------------------------------------

def test_markdown_emphasis_is_stripped():
    assert parse_words("**cat**\n*dog*\n`bird`\n__tuba__\n~~kelp~~") == [
        "cat", "dog", "bird", "tuba", "kelp"]


def test_surrounding_quotes_and_trailing_punctuation_are_stripped():
    text = '"cat"\n“dog”,\n‘bird’;\n«tuba».\nkelp!'
    assert parse_words(text) == ["cat", "dog", "bird", "tuba", "kelp"]


def test_case_and_internal_hyphens_are_preserved_for_validate():
    # parse.py reports the word as emitted; lowercasing is validate.py's job.
    assert parse_words("1. Cat\n2. cul-de-sac\n3. o'clock") == [
        "Cat", "cul-de-sac", "o'clock"]


# --- preamble / postamble --------------------------------------------------------

def test_preamble_and_postamble_lines_are_dropped():
    text = (
        "Sure!\n"
        "Here are 10 words:\n"
        + "\n".join(TEN) + "\n"
        "\nLet me know if you would like another set.\n"
        "I hope this helps!\n"
    )
    assert parse_words(text) == TEN


def test_a_bulleted_postamble_does_not_beat_the_answer_list():
    text = "1. cat\n2. tuba\n3. mango\n\nNotes:\n- common nouns\n- no proper nouns\n"
    assert parse_words(text) == ["cat", "tuba", "mango"]


def test_restated_rules_do_not_beat_the_answer_list():
    text = (
        "Rules:\n1. Only single words in English.\n2. Only nouns.\n"
        "3. No proper nouns.\n\nMy answer:\n\n- cat\n- tuba\n- mango\n"
    )
    assert parse_words(text) == ["cat", "tuba", "mango"]


def test_a_header_line_ending_in_a_colon_is_never_a_word():
    assert parse_words("### My ten words:\ncat\ndog") == ["cat", "dog"]
    assert parse_words("Here are 10 words:") == []


# --- the colon rule --------------------------------------------------------------

def test_short_colon_prefix_keeps_the_tail():
    assert parse_words("My words: cat, dog") == ["cat", "dog"]


def test_long_colon_prefix_still_keeps_the_tail_rather_than_losing_a_word():
    assert parse_words("Sure! Here is my list of ten words: cat, dog") == ["cat", "dog"]
    assert parse_words("Sure, here is my list: cat, dog") == ["cat", "dog"]


def test_a_glossed_line_keeps_the_word_not_the_gloss():
    assert parse_words("cat: a small animal\ntuba: a brass instrument") == ["cat", "tuba"]


# --- chain of thought ------------------------------------------------------------

COT = """Let me think about this step by step.

First I should identify unrelated semantic domains to draw from:
- animals
- tools
- weather
- emotions
- finance

Now checking pairs: "cat" and "dog" are far too close, both household pets, so I
will drop dog. "hammer" and "wrench" share a category too, so only one survives.

Candidates I rejected: 1. dog 2. kitten 3. puppy

Final answer:

1. cat
2. thimble
3. avalanche
4. grief
5. tuba
6. kelp
7. pension
8. quasar
9. wrench
10. mango
"""


def test_cot_final_numbered_list_wins_over_the_reasoning():
    assert parse_words(COT) == TEN


def test_cot_final_list_wins_over_an_equally_wordlike_earlier_list():
    text = (
        "Shortlist:\n1. animals\n2. tools\n3. weather\n\n"
        "On reflection those are categories, not words.\n\n"
        "Final:\n1. cat\n2. tuba\n3. mango\n"
    )
    assert parse_words(text) == ["cat", "tuba", "mango"]


def test_cot_reasoning_prose_never_supplies_candidates():
    text = (
        "I need ten nouns that share no category, no setting and no sensory\n"
        "quality with each other, so I will spread them across domains.\n\n"
        "1. cat\n2. tuba\n"
    )
    assert parse_words(text) == ["cat", "tuba"]


# --- want ------------------------------------------------------------------------

def test_more_words_than_wanted_are_truncated_to_the_first_want():
    text = "\n".join(f"{i}. w{i}" for i in range(1, 16))
    assert parse_words(text) == [f"w{i}" for i in range(1, 11)]
    assert parse_words(text, want=3) == ["w1", "w2", "w3"]


def test_fewer_words_than_wanted_are_not_padded():
    assert parse_words("cat\ndog\nbird") == ["cat", "dog", "bird"]
    assert parse_words("- cat") == ["cat"]


def test_want_is_keyword_only_and_non_positive_want_yields_nothing():
    assert parse_words("cat, dog", want=0) == []
    assert parse_words("cat, dog", want=-1) == []
    with pytest.raises(TypeError):
        parse_words("cat, dog", 5)  # type: ignore[misc]


# --- nothing there ---------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "",
    "   \n\t\n  ",
    "```\n```",
    "I'm sorry, but I can't help with that.",
    "As an AI language model, I cannot generate that list.",
    "I cannot comply with this request.",
    "Sure!",
    "Here you go:",
])
def test_no_candidates_returns_empty_list(text):
    assert parse_words(text) == []


def test_non_string_input_is_tolerated():
    assert parse_words(None) == []  # type: ignore[arg-type]


# --- invariants ------------------------------------------------------------------

def test_duplicates_are_preserved_because_they_are_signal():
    assert parse_words("1. cat\n2. dog\n3. cat\n4. cat") == ["cat", "dog", "cat", "cat"]


def test_order_is_preserved():
    text = "\n".join(f"- {w}" for w in reversed(TEN))
    assert parse_words(text) == list(reversed(TEN))


CORPUS = [
    "",
    "   ",
    "Sure!\n\n1. cat\n2. dog",
    '```json\n["cat", "", "  ", "dog"]\n```',
    "- \n- cat\n- **\n- dog",
    "1. \n2. cat\n3. .\n4. dog",
    "cat,,,dog,;,bird",
    "* *\n* cat",
    COT,
    "My words: cat, dog",
    "#### words\n\n> cat\n> dog",
    "•••\ncat",
]


@pytest.mark.parametrize("text", CORPUS)
def test_never_returns_empty_or_whitespace_padded_strings(text):
    out = parse_words(text)
    assert isinstance(out, list)
    for word in out:
        assert isinstance(word, str)
        assert word == word.strip()
        assert word != ""
        assert word.strip()


@pytest.mark.parametrize("text", CORPUS)
def test_respects_want_on_every_shape(text):
    assert len(parse_words(text, want=2)) <= 2


# --- one-line answers and non-word noise -----------------------------------------

def test_a_whole_list_on_one_line_is_exploded():
    assert parse_words("1. cat 2. dog 3. bird") == ["cat", "dog", "bird"]
    assert parse_words("Here are my words: 1. cat 2. dog 3. bird") == [
        "cat", "dog", "bird"]
    assert parse_words("• cat • dog • bird") == ["cat", "dog", "bird"]


def test_decimals_and_years_do_not_trigger_the_one_line_split():
    assert parse_words("The score was 0.5 and 1.2, so: cat, dog") == ["cat", "dog"]


def test_markdown_table_scaffolding_never_becomes_a_candidate():
    # A table is out of scope; yielding nothing leaves a visibly refused run,
    # which is better than half a row of column headers scored as words.
    assert parse_words("| Word | Domain |\n|---|---|\n| cat | animals |") == []


def test_a_number_is_kept_as_a_candidate_for_validate_to_flag():
    assert parse_words("1. cat\n2. 42\n3. dog") == ["cat", "42", "dog"]


def test_filler_sharing_a_line_with_answers_does_not_eat_them():
    assert parse_words("cat, dog\nLet me know if you want more, or another set.") == [
        "cat", "dog"]


# --------------------------------------------------- bullet -> bullet lead-in ----
# Regression: a bulleted brainstorm list and a bulleted answer list separated by a
# single prose line used to merge into one block, so parse_words returned the
# reasoning words and truncated the tail of the real answer. Bullets always carry
# num=None, so the numeric-restart signal could never fire for them.
# Found by adversarial review, confirmed 3/3.

_BRAINSTORM = ["animals", "tools", "emotions"]
_ANSWER = ["thimble", "galaxy", "otter", "treaty", "mango",
           "quartz", "lullaby", "anchor", "pepper", "vortex"]


def _bulleted(words):
    return "\n".join(f"- {w}" for w in words)


@pytest.mark.parametrize(
    "separator",
    [
        "Those are categories, not my answer. Here are the actual ten words:",
        "## Final answer",
        "**Final answer:**",
        "My final answer",
        "Here are my ten words:",
    ],
)
def test_bulleted_reasoning_list_does_not_merge_into_bulleted_answer(separator):
    text = f"I'll brainstorm first:\n{_bulleted(_BRAINSTORM)}\n{separator}\n{_bulleted(_ANSWER)}"
    assert parse_words(text) == _ANSWER


def test_lead_in_split_does_not_break_per_item_annotation():
    """A stray annotation between items must still be tolerated.

    The block threshold of 2 prose lines exists so models can annotate items; the
    lead-in rule must not undo that, or a genuine list gets cut and words are lost.
    """
    words = _ANSWER[:7]
    text = f"- {words[0]}\n(all common nouns, no proper names)\n" + _bulleted(words[1:])
    assert parse_words(text) == words


def test_lead_in_rule_is_not_triggered_by_ordinary_commentary():
    for line in ["(these are all nouns)", "I avoided proper nouns.", "Nothing technical."]:
        words = _ANSWER[:6]
        text = f"- {words[0]}\n{line}\n" + _bulleted(words[1:])
        assert parse_words(text) == words, f"{line!r} wrongly split the list"
