"""Turning a Hugging Face NER row into legal-ml-core's text and entities (#117)."""
from dataset_library.ner import entity_types, tokens_to_text

NAMES = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG"]


def test_tokens_are_joined_with_single_spaces():
    assert tokens_to_text(["John", "Smith", "works"]) == "John Smith works"


def test_a_string_is_kept_as_it_is():
    assert tokens_to_text("Already a sentence.") == "Already a sentence."


def test_missing_tokens_are_skipped_not_written_as_none():
    assert tokens_to_text([None, "Smith"]) == "Smith"


def test_an_empty_or_missing_token_list_is_empty_text():
    assert tokens_to_text([]) == ""
    assert tokens_to_text(None) == ""


def test_numbers_are_read_through_the_class_label_names():
    assert entity_types([0, 1, 2, 3, 4], NAMES) == "PER, ORG"


def test_every_position_prefix_is_stripped():
    assert entity_types(["B-PER", "I-PER", "E-PER", "S-PER", "L-PER", "U-PER"]) == "PER"


def test_underscore_prefixes_are_stripped_too():
    assert entity_types(["B_ORG", "I_ORG"]) == "ORG"


def test_outside_and_empty_tags_are_dropped():
    assert entity_types(["O", "", "  ", "B-DATE"]) == "DATE"


def test_types_keep_their_first_seen_order():
    assert entity_types(["B-ORG", "B-PER", "I-ORG", "B-DATE"]) == "ORG, PER, DATE"


def test_a_row_with_no_tags_has_no_entities():
    assert entity_types([]) == ""
    assert entity_types(None) == ""


def test_character_tokens_with_space_tokens_are_joined_as_written():
    # klue/klue's ner: one token per character, spaces included.
    assert tokens_to_text(["특", "히", " ", "영", "동"]) == "특히 영동"


def test_word_tokens_are_still_joined_by_spaces():
    assert tokens_to_text(["The", "court", "held"]) == "The court held"


def test_a_stray_space_or_newline_token_keeps_word_spacing():
    assert tokens_to_text(["The", "court", "\n", "held", "that"]) == "The court \n held that"
    assert tokens_to_text(["The", " ", "court", "held"]) == "The   court held"
    assert tokens_to_text(["Smith", "v.", "\xa0", "Jones"]) == "Smith v. \xa0 Jones"
