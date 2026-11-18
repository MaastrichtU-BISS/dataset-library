"""Turning a Hugging Face NER row into legal-ml-core's text and entity spans (#6)."""
from dataset_library.ner import spans, tokens_to_text

NAMES = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG"]


def check(text, found):
    """Every span is inside the text, in order, and never overlaps another."""
    starts = [span["start"] for span in found]
    assert starts == sorted(starts)
    previous_end = 0
    for span in found:
        assert span["start"] < span["end"]
        assert span["start"] >= previous_end
        assert text[span["start"]:span["end"]] == span["text"]
        assert span["text"] == span["text"].strip()
        previous_end = span["end"]


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
    text, found = spans(["John", "Smith", "works", "at", "ACME"], [1, 2, 0, 0, 3], NAMES)

    assert text == "John Smith works at ACME"
    assert found == [
        {"text": "John Smith", "start": 0, "end": 10, "type": "PER"},
        {"text": "ACME", "start": 20, "end": 24, "type": "ORG"},
    ]
    check(text, found)


def test_bio_tags_become_spans_in_text_order():
    text, found = spans(["John", "Smith", "works", "at", "ACME"], ["B-PER", "I-PER", "O", "O", "B-ORG"])

    assert text == "John Smith works at ACME"
    assert found == [
        {"text": "John Smith", "start": 0, "end": 10, "type": "PER"},
        {"text": "ACME", "start": 20, "end": 24, "type": "ORG"},
    ]
    check(text, found)


def test_an_i_tag_with_no_open_span_starts_one_iob1():
    text, found = spans(["John", "Smith", "met", "Mary"], ["I-PER", "I-PER", "O", "I-PER"])

    assert found == [
        {"text": "John Smith", "start": 0, "end": 10, "type": "PER"},
        {"text": "Mary", "start": 15, "end": 19, "type": "PER"},
    ]
    check(text, found)


def test_a_type_change_mid_run_starts_a_new_span():
    text, found = spans(["John", "Smith"], ["I-PER", "I-ORG"])

    assert found == [
        {"text": "John", "start": 0, "end": 4, "type": "PER"},
        {"text": "Smith", "start": 5, "end": 10, "type": "ORG"},
    ]
    check(text, found)


def test_bioes_tags_decode_to_closed_spans():
    text, found = spans(
        ["John", "Smith", "visited", "ACME", "Inc"],
        ["B-PER", "E-PER", "O", "B-ORG", "E-ORG"],
    )

    assert found == [
        {"text": "John Smith", "start": 0, "end": 10, "type": "PER"},
        {"text": "ACME Inc", "start": 19, "end": 27, "type": "ORG"},
    ]
    check(text, found)


def test_single_and_last_tags_end_their_spans():
    text, found = spans(["Solo", "and", "Taylor", "Ltd"], ["S-PER", "O", "B-ORG", "L-ORG"])

    assert found == [
        {"text": "Solo", "start": 0, "end": 4, "type": "PER"},
        {"text": "Taylor Ltd", "start": 9, "end": 19, "type": "ORG"},
    ]
    check(text, found)


def test_tags_without_prefixes_follow_the_io_scheme():
    text, found = spans(["John", "Smith", "at", "ACME"], ["PER", "PER", "O", "ORG"])

    assert found == [
        {"text": "John Smith", "start": 0, "end": 10, "type": "PER"},
        {"text": "ACME", "start": 14, "end": 18, "type": "ORG"},
    ]
    check(text, found)


def test_an_io_type_change_ends_the_open_span():
    text, found = spans(["John", "Smith"], ["PER", "ORG"])

    assert found == [
        {"text": "John", "start": 0, "end": 4, "type": "PER"},
        {"text": "Smith", "start": 5, "end": 10, "type": "ORG"},
    ]
    check(text, found)


def test_outside_only_rows_have_no_spans():
    text, found = spans(["just", "words"], ["O", "O"])

    assert text == "just words"
    assert found == []
    check(text, found)


def test_adjacent_spans_of_the_same_type_stay_separate():
    text, found = spans(["John", "Smith"], ["B-PER", "B-PER"])

    assert found == [
        {"text": "John", "start": 0, "end": 4, "type": "PER"},
        {"text": "Smith", "start": 5, "end": 10, "type": "PER"},
    ]
    check(text, found)


def test_underscore_prefixes_are_read_too():
    text, found = spans(["John", "Smith"], ["B_PER", "I_PER"])

    assert found == [{"text": "John Smith", "start": 0, "end": 10, "type": "PER"}]
    check(text, found)


def test_character_level_rows_include_the_space_inside_a_span():
    # klue/klue's ner: one token per character, spaces included.
    text, found = spans(
        ["특", "히", " ", "영", "동"],
        ["B-LOC", "I-LOC", "O", "I-LOC", "E-LOC"],
    )

    assert text == "특히 영동"
    assert found == [{"text": "특히 영동", "start": 0, "end": 5, "type": "LOC"}]
    check(text, found)


def test_trailing_space_tokens_are_trimmed_from_a_span():
    text, found = spans(["서", "울", " "], ["B-LOC", "I-LOC", "O"])

    assert text == "서울 "
    assert found == [{"text": "서울", "start": 0, "end": 2, "type": "LOC"}]
    check(text, found)


def test_a_space_token_never_starts_a_span():
    text, found = spans(["a", " ", "b"], ["O", "B-PER", "O"])

    assert text == "a b"
    assert found == []
    check(text, found)


def test_a_none_token_is_dropped_with_its_tag():
    text, found = spans(["John", None, "Smith"], ["B-PER", "B-ORG", "I-PER"])

    assert text == "John Smith"
    assert found == [{"text": "John Smith", "start": 0, "end": 10, "type": "PER"}]
    check(text, found)


def test_a_shorter_tag_list_decodes_only_the_tokens_it_names():
    text, found = spans(["John", "Smith", "works"], ["B-PER"])

    assert text == "John Smith works"
    assert found == [{"text": "John", "start": 0, "end": 4, "type": "PER"}]
    check(text, found)


def test_a_longer_tag_list_ignores_the_extra_tags():
    text, found = spans(["John"], ["B-PER", "B-ORG"])

    assert found == [{"text": "John", "start": 0, "end": 4, "type": "PER"}]
    check(text, found)


def test_unreadable_and_out_of_range_tags_are_outside():
    text, found = spans(["John", "Smith"], [7, None], NAMES)

    assert found == []
    check(text, found)
    for tags in ("", None):
        text, found = spans(["John"], [tags])
        assert found == []
        check(text, found)


def test_a_row_with_no_tags_has_no_spans():
    for text, found in (spans(["a"], None), spans([], []), spans(None, None)):
        assert found == []
        check(text, found)
    assert spans(["a"], None) == ("a", [])
    assert spans([], []) == ("", [])
    assert spans(None, None) == ("", [])


def test_a_string_row_is_its_own_text_and_has_no_spans():
    text, found = spans("Already a sentence.", None)

    assert (text, found) == ("Already a sentence.", [])
    check(text, found)


def _types(found):
    return [(span["text"], span["type"]) for span in found]


def test_an_o_on_a_newline_ends_a_span_in_a_word_level_row():
    for tags in (["I-PER", "O", "I-PER"], ["B-PER", "O", "I-PER"]):
        text, found = spans(["Smith", "\n", "Jones"], tags)
        check(text, found)
        assert _types(found) == [("Smith", "PER"), ("Jones", "PER")]


def test_a_character_level_row_keeps_a_multi_word_entity_across_an_o_space():
    # klue/klue's ner tags the spaces inside an entity O and carries on with I-.
    text, found = spans(["강", "원", " ", "영", "동"], ["B-LC", "I-LC", "O", "I-LC", "I-LC"])
    check(text, found)
    assert _types(found) == [("강원 영동", "LC")]


def test_an_o_space_separates_io_entities_in_a_character_level_row():
    text, found = spans(["김", " ", "이"], ["PER", "O", "PER"])
    check(text, found)
    assert _types(found) == [("김", "PER"), ("이", "PER")]


def test_a_tagged_space_with_another_type_or_a_new_start_closes_the_span():
    text, found = spans(["가", "나", " ", "다"], ["B-PER", "I-PER", "B-ORG", "I-PER"])
    check(text, found)
    assert _types(found) == [("가나", "PER"), ("다", "PER")]


def test_bmes_middle_tags_continue_the_entity():
    text, found = spans(["张", "三", "丰"], ["B-PER", "M-PER", "E-PER"])
    check(text, found)
    assert _types(found) == [("张 三 丰", "PER")]


def test_whitespace_at_a_tokens_edges_is_not_part_of_the_span():
    text, found = spans([" John", "Smith\xa0"], ["B-PER", "I-PER"])
    check(text, found)
    assert _types(found) == [("John Smith", "PER")]


def test_outside_and_placeholder_tags_are_never_types():
    for tag in ("B-O", "X", "PAD", "[PAD]", "<pad>", "B- "):
        assert spans(["a"], [tag])[1] == []


def test_a_space_after_the_prefix_is_not_part_of_the_type():
    assert _types(spans(["a", "b"], ["B- PER", "I-PER"])[1]) == [("a b", "PER")]


def test_lowercase_prefixes_and_outside_tags_are_read():
    text, found = spans(["John", "Smith", "works"], ["b-per", "i-per", "o"])
    check(text, found)
    assert _types(found) == [("John Smith", "per")]


def test_bert_special_tokens_are_not_types():
    for tag in ("[CLS]", "[SEP]"):
        assert spans(["a"], [tag])[1] == []
