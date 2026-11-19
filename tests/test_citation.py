"""The citation a Hub card gives a dataset (#120): what it is cited from, and the
card's own BibTeX kept exactly as written."""
from conftest import TASK_TYPES

from dataset_library.providers.huggingface import _card

SHA = "c23fdff1a6bf74e0e1a71cb86f1e781d37da888c"
LEXGLUE_BIBTEX = """@inproceedings{chalkidis-etal-2022-lexglue,
    title={{LexGLUE}: A Benchmark Dataset for Legal Language Understanding in English},
    author={Chalkidis, Ilias and Jana, Abhik and Hartung, Dirk and Bommarito, Michael and Androutsopoulos, Ion and Katz, Daniel Martin and Aletras, Nikolaos},
    year={2022}
}"""


def record(**overrides):
    """A Hub record for coastalcph/lex_glue with the fields a citation reads."""
    return {
        "id": "coastalcph/lex_glue", "sha": SHA,
        "createdAt": "2021-10-17T11:42:31.000Z", "lastModified": "2024-01-04T14:25:27.000Z",
        "tags": ["license:cc-by-4.0", "task_categories:text-classification"],
        "cardData": {"pretty_name": "LexGLUE"}, "downloads": 1, "description": "",
        **overrides,
    }


def cite(card_data, **overrides):
    return _card(record(cardData=card_data, **overrides), TASK_TYPES)["citation"]


def test_a_card_with_its_own_bibtex_cites_it_exactly_as_written():
    body = cite({"pretty_name": "LexGLUE", "citation_bibtex": LEXGLUE_BIBTEX})

    assert body["bibtex"] == LEXGLUE_BIBTEX
    assert body["title"] == "LexGLUE"
    assert body["authors"] == (
        "Ilias Chalkidis, Abhik Jana, Dirk Hartung, Michael Bommarito, Ion Androutsopoulos, "
        "Daniel Martin Katz and Nikolaos Aletras"
    )
    assert body["year"] == 2022
    assert body["url"] == "https://huggingface.co/datasets/coastalcph/lex_glue"
    assert body["revision"] == SHA
    assert body["accessedAt"] is None


def test_a_card_without_bibtex_cites_with_null_bibtex_and_the_repository_year():
    body = cite({"pretty_name": "LexGLUE"})

    assert body["bibtex"] is None
    assert body["title"] == "LexGLUE"
    assert body["authors"] == "coastalcph"
    assert body["year"] == 2021


def test_a_plain_text_citation_is_not_bibtex():
    assert cite({"citation": "Chalkidis et al. (2022). LexGLUE."})["bibtex"] is None


def test_the_citation_field_holds_bibtex_when_it_starts_with_an_at_sign():
    assert cite({"citation": LEXGLUE_BIBTEX})["bibtex"] == LEXGLUE_BIBTEX


def test_a_card_without_a_pretty_name_takes_the_dataset_name_as_its_title():
    assert cite({})["title"] == "lex_glue"


def test_the_card_names_its_authors_before_the_bibtex_names_them():
    body = cite({"authors": ["Ilias Chalkidis", "Nikolaos Aletras"], "citation_bibtex": LEXGLUE_BIBTEX})

    assert body["authors"] == "Ilias Chalkidis, Nikolaos Aletras"


def test_a_bibtex_with_others_reads_as_others():
    bibtex = "@misc{x, author = {Chalkidis, Ilias and others}, year = 2020}"

    assert cite({"citation_bibtex": bibtex})["authors"] == "Ilias Chalkidis and others"


def test_the_year_comes_from_the_card_then_its_bibtex_then_the_repository():
    assert cite({"year": 2020, "citation_bibtex": LEXGLUE_BIBTEX})["year"] == 2020
    assert cite({"citation_bibtex": LEXGLUE_BIBTEX})["year"] == 2022
    assert cite({})["year"] == 2021


def test_a_record_with_no_creation_date_and_no_year_has_a_null_year():
    assert cite({}, createdAt=None)["year"] is None
