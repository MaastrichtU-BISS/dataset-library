"""The citation a Hub dataset gives (#120): its title, structured authors, year
and BibTeX, read from the Hub's top-level `citation`, the card, or the README,
and the BibTeX scanner that reads them. Expectations are written out by hand."""
import pytest
import requests

from conftest import TASK_TYPES

from dataset_library.providers import huggingface
from dataset_library.providers.huggingface import (
    _bibtex_field, _card, _entry, _names, _plain, _readme_bibtex, _readme_entry,
)

SHA = "c23fdff1a6bf74e0e1a71cb86f1e781d37da888c"
SWISS_SHA = "29806f87bba4f23d0707d3b6d9ea5432afefbe2f"
LEXGLUE_BIBTEX = """@inproceedings{chalkidis-etal-2022-lexglue,
    title={{LexGLUE}: A Benchmark Dataset for Legal Language Understanding in English},
    author={Chalkidis, Ilias and Jana, Abhik and Hartung, Dirk and Bommarito, Michael and Androutsopoulos, Ion and Katz, Daniel Martin and Aletras, Nikolaos},
    year={2022}
}"""
LEXGLUE_AUTHORS = [
    {"family": "Chalkidis", "given": "Ilias"},
    {"family": "Jana", "given": "Abhik"},
    {"family": "Hartung", "given": "Dirk"},
    {"family": "Bommarito", "given": "Michael"},
    {"family": "Androutsopoulos", "given": "Ion"},
    {"family": "Katz", "given": "Daniel Martin"},
    {"family": "Aletras", "given": "Nikolaos"},
]
# The BibTeX in lex_glue's README, in its "Citation Information" section, as the README writes it.
LEX_README_BIBTEX = "\n".join([
    "@inproceedings{chalkidis-etal-2021-lexglue,",
    "        title={LexGLUE: A Benchmark Dataset for Legal Language Understanding in English}, ",
    "        author={Chalkidis, Ilias and Jana, Abhik and Hartung, Dirk and",
    "        Bommarito, Michael and Androutsopoulos, Ion and Katz, Daniel Martin and",
    "        Aletras, Nikolaos},",
    "        year={2022},",
    "        booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},",
    "        address={Dubln, Ireland},",
    "}",
])
# The Hub's top-level `citation` for rcds/swiss_judgment_prediction: two entries, verbatim.
SWISS_BIBTEX = "\n".join([
    "@InProceedings{niklaus-etal-2021-swiss,",
    "  author = {Niklaus, Joel",
    "                and Chalkidis, Ilias",
    "                and Stürmer, Matthias},",
    "  title = {Swiss-Court-Predict: A Multilingual Legal Judgment Prediction Benchmark},",
    "  booktitle = {Proceedings of the 2021 Natural Legal Language Processing Workshop},",
    "  year = {2021},",
    "  location = {Punta Cana, Dominican Republic},",
    "}",
    "@misc{niklaus2022empirical,",
    "    title={An Empirical Study on Cross-X Transfer for Legal Judgment Prediction},",
    "    author={Joel Niklaus and Matthias Stürmer and Ilias Chalkidis},",
    "    year={2022},",
    "    eprint={2209.12325},",
    "    archivePrefix={arXiv},",
    "    primaryClass={cs.CL}",
    "}",
])
SWISS_AUTHORS = [
    {"family": "Niklaus", "given": "Joel"},
    {"family": "Chalkidis", "given": "Ilias"},
    {"family": "Stürmer", "given": "Matthias"},
]


def record(**overrides):
    """A Hub record for coastalcph/lex_glue with the fields a citation reads."""
    return {
        "id": "coastalcph/lex_glue", "sha": SHA,
        "createdAt": "2021-10-17T11:42:31.000Z", "lastModified": "2024-01-04T14:25:27.000Z",
        "tags": ["license:cc-by-4.0", "task_categories:text-classification"],
        "cardData": {"pretty_name": "LexGLUE"}, "downloads": 1, "description": "",
        **overrides,
    }


def cite(card_data, readme=None, **overrides):
    return _card(record(cardData=card_data, **overrides), TASK_TYPES, readme=readme)["citation"]


class Stream:
    """A README's answer, streamed in chunks, which counts the bytes read."""

    def __init__(self, body, status=200):
        self.body, self.status_code, self.served, self.closed = body, status, 0, False

    def iter_content(self, chunk_size=1):
        for start in range(0, len(self.body), chunk_size):
            chunk = self.body[start:start + chunk_size]
            self.served += len(chunk)
            yield chunk

    def close(self):
        self.closed = True


# --- Real Hub shapes ----------------------------------------------------------


def test_a_top_level_citation_is_the_bibtex_kept_verbatim_and_names_the_year(hub):
    body = huggingface.dataset("rcds/swiss_judgment_prediction", TASK_TYPES)["citation"]

    assert body == {
        "title": "Swiss-Judgment-Prediction",
        "authors": SWISS_AUTHORS,
        # The first entry's year: not the 2022 the repository was created (its createdAt).
        "year": 2021,
        "url": "https://huggingface.co/datasets/rcds/swiss_judgment_prediction",
        "revision": SWISS_SHA,
        "accessedAt": None,
        "bibtex": SWISS_BIBTEX,
    }
    assert not any("/resolve/" in call["url"] for call in hub.calls), "no README is needed"


def test_a_bibtex_only_in_the_readme_is_read_there_at_the_cards_revision(hub):
    body = huggingface.dataset("coastalcph/lex_glue", TASK_TYPES)["citation"]

    assert body["bibtex"] == LEX_README_BIBTEX
    assert body["authors"] == LEXGLUE_AUTHORS
    assert body["year"] == 2022
    assert body["revision"] == SHA
    readme_urls = [call["url"] for call in hub.calls if "/resolve/" in call["url"]]
    assert readme_urls == [f"https://huggingface.co/datasets/coastalcph/lex_glue/resolve/{SHA}/README.md"]


def test_a_dataset_with_no_bibtex_anywhere_is_cited_without_one(hub):
    # The Hub renamed this dataset: its record names the new owner, who is the literal author.
    body = huggingface.dataset("coastalcph/moved_corpus", TASK_TYPES)["citation"]

    assert body["bibtex"] is None
    assert body["authors"] == [{"literal": "elsewhere"}]
    assert body["year"] is None


def test_a_search_result_never_reads_a_readme(hub):
    results = huggingface.search("lex_glue", None, None, None, TASK_TYPES)
    lex_glue = next(card for card in results if card["externalId"] == "coastalcph/lex_glue")

    assert lex_glue["citation"]["bibtex"] is None
    assert not any("/resolve/" in call["url"] for call in hub.calls)


# --- Where the BibTeX comes from ----------------------------------------------


def test_a_card_with_its_own_bibtex_cites_it_exactly_as_written():
    body = cite({"pretty_name": "LexGLUE", "citation_bibtex": LEXGLUE_BIBTEX})

    assert body["bibtex"] == LEXGLUE_BIBTEX
    assert body["title"] == "LexGLUE"
    assert body["authors"] == LEXGLUE_AUTHORS
    assert body["year"] == 2022
    assert body["url"] == "https://huggingface.co/datasets/coastalcph/lex_glue"
    assert body["revision"] == SHA
    assert body["accessedAt"] is None


def test_the_top_level_citation_wins_over_the_card():
    top = "@misc{top, author = {Ortiz, Ana}, year = {2019}}"

    body = _card(record(citation=top, cardData={"pretty_name": "LexGLUE", "citation_bibtex": LEXGLUE_BIBTEX}), TASK_TYPES)["citation"]

    assert body["bibtex"] == top


def test_a_plain_text_top_level_citation_is_skipped_for_the_card_bibtex():
    body = _card(record(citation="Chalkidis et al. (2022). LexGLUE.", cardData={"citation_bibtex": LEXGLUE_BIBTEX}), TASK_TYPES)["citation"]

    assert body["bibtex"] == LEXGLUE_BIBTEX


def test_the_card_bibtex_is_used_before_the_readme_is_read():
    def no_readme():
        raise AssertionError("the README must not be read when the card has BibTeX")

    assert cite({"citation_bibtex": LEXGLUE_BIBTEX}, readme=no_readme)["bibtex"] == LEXGLUE_BIBTEX


def test_a_plain_text_citation_is_not_bibtex():
    assert cite({"citation": "Chalkidis et al. (2022). LexGLUE."})["bibtex"] is None


def test_the_card_citation_field_holds_bibtex_when_it_starts_with_an_at_sign():
    assert cite({"citation": LEXGLUE_BIBTEX})["bibtex"] == LEXGLUE_BIBTEX


def test_a_card_without_a_pretty_name_takes_the_dataset_name_as_its_title():
    assert cite({})["title"] == "lex_glue"


# --- Year ----------------------------------------------------------------------


def test_the_year_is_the_bibtex_years_then_the_cards_then_none():
    assert cite({"year": 2020, "citation_bibtex": LEXGLUE_BIBTEX})["year"] == 2022
    assert cite({"year": 2020})["year"] == 2020
    assert cite({})["year"] is None


def test_the_repositorys_creation_date_is_never_the_year():
    # Older datasets were all created on the Hub's 2022-03-02 migration date.
    assert cite({}, createdAt="2022-03-02T23:29:22.000Z")["year"] is None
    assert cite({}, createdAt="2021-10-17T11:42:31.000Z")["year"] is None


def test_a_record_with_no_creation_date_and_no_year_has_a_null_year():
    assert cite({}, createdAt=None)["year"] is None


def test_a_year_inside_another_field_is_not_the_year():
    bibtex = "@misc{x, note = {Prior work uses year=1999}, year = {2025}}"
    assert cite({"citation_bibtex": bibtex})["year"] == 2025
    assert cite({"citation_bibtex": "@misc{x, booktitle = {Proc year = 1999}}"})["year"] is None


# --- Authors -------------------------------------------------------------------


def test_the_card_names_its_authors_in_order_before_the_bibtex_does():
    body = cite({"authors": ["Ilias Chalkidis", "Nikolaos Aletras"], "citation_bibtex": LEXGLUE_BIBTEX})

    assert body["authors"] == [{"family": "Chalkidis", "given": "Ilias"}, {"family": "Aletras", "given": "Nikolaos"}]


def test_the_bibtex_authors_are_read_when_the_card_names_none():
    body = cite({"citation_bibtex": "@misc{x, author = {Chalkidis, Ilias and others}, year = 2020}"})

    assert body["authors"] == [{"family": "Chalkidis", "given": "Ilias"}, {"literal": "others"}]


def test_a_braced_organisation_is_a_literal_author():
    bibtex = "@misc{x, author = {{Research and Development Council} and Chalkidis, Ilias}, year = {2020}}"

    assert cite({"citation_bibtex": bibtex})["authors"] == [
        {"literal": "Research and Development Council"},
        {"family": "Chalkidis", "given": "Ilias"},
    ]


def test_the_owner_is_the_literal_author_when_nothing_names_them():
    assert cite({})["authors"] == [{"literal": "coastalcph"}]


def test_several_entries_keep_all_their_text_and_the_first_names_the_authors_and_year():
    bibtex = "@misc{first, author = {Ortiz, Ana}, year = {2019}}\n@misc{second, author = {Bell, Ben}, year = {2024}}"

    body = cite({"citation_bibtex": bibtex})

    assert body["bibtex"] == bibtex
    assert body["authors"] == [{"family": "Ortiz", "given": "Ana"}]
    assert body["year"] == 2019


# --- The BibTeX scanner ---------------------------------------------------------


def test_only_top_level_fields_are_read():
    entry = _entry("@misc{x, note = {Prior work uses year=1999}, year = {2025}, booktitle = {Proc year = 1999}}")

    assert entry == {"note": "Prior work uses year=1999", "year": "2025", "booktitle": "Proc year = 1999"}


def test_a_string_macro_before_the_entry_is_not_the_entry():
    assert _bibtex_field('@string{pub = "Nature"}\n@misc{x, year = {2019}}', "year") == "2019"


def test_a_value_in_nested_braces_reads_to_the_close_of_its_outer_brace():
    assert _bibtex_field("@misc{x, title = {A {B {C}} D}}", "title") == "A B C D"


def test_a_quoted_value_may_hold_braces_and_escaped_quotes():
    entry = _entry('@misc{x, title = "Set {X} of \\"y\\"", year = 2020}')

    assert entry["title"] == 'Set {X} of \\"y\\"'
    assert _bibtex_field('@misc{x, title = "Set {X}"}', "title") == "Set X"
    # The escaped quotes did not end the title, so the field after it is still read.
    assert entry["year"] == "2020"


def test_a_bare_number_and_joined_values_are_read():
    assert _bibtex_field("@misc{x, year = 2021}", "year") == "2021"
    assert _bibtex_field('@misc{x, title = "Part " # {One}}', "title") == "Part One"


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("Niklaus, Joel", [{"family": "Niklaus", "given": "Joel"}]),
        ("Niklaus, Joel,", [{"family": "Niklaus", "given": "Joel"}]),
        ("King, Jr, Martin Luther", [{"family": "King", "given": "Martin Luther"}]),
        ("Martin Luther King", [{"family": "King", "given": "Martin Luther"}]),
        ("Ludwig van Beethoven", [{"family": "van Beethoven", "given": "Ludwig"}]),
        ("Anonymous", [{"literal": "Anonymous"}]),
        ("{Research and Development Council}", [{"literal": "Research and Development Council"}]),
    ],
)
def test_each_name_form_is_read(written, expected):
    assert _names(written) == expected


def test_and_splits_names_in_any_case_but_not_inside_braces():
    assert _names("Niklaus, Joel AND Chalkidis, Ilias") == [
        {"family": "Niklaus", "given": "Joel"},
        {"family": "Chalkidis", "given": "Ilias"},
    ]
    assert _names("{Research and Development Council} and Chalkidis, Ilias") == [
        {"literal": "Research and Development Council"},
        {"family": "Chalkidis", "given": "Ilias"},
    ]


@pytest.mark.parametrize(
    ("latex", "plain"),
    [
        (r'{\"u}ber', "über"),
        (r'\"uber', "über"),
        (r"\'e", "é"),
        (r"\`a", "à"),
        (r"\^o", "ô"),
        (r"\~n", "ñ"),
        (r"Fran\c{c}ais", "Français"),
        (r"Stra{\ss}e", "Straße"),
        (r"{\v{S}}koda", "Škoda"),
        (r"{\o}ster", "øster"),
        ("Stürmer", "Stürmer"),
    ],
)
def test_latex_accents_and_letters_read_as_their_letters(latex, plain):
    assert _plain(latex) == plain


def test_stray_braces_are_removed_from_a_name():
    assert _names("{Stürmer}, Matthias") == [{"family": "Stürmer", "given": "Matthias"}]


def test_latex_accented_authors_read_as_plain_names():
    assert _names(r'{\"U}nal, Ahmet and Fran\c{c}ois Ozdemir') == [
        {"family": "Ünal", "given": "Ahmet"},
        {"family": "Ozdemir", "given": "François"},
    ]


# --- The README -----------------------------------------------------------------


def test_the_first_fenced_block_holding_an_entry_is_the_readme_bibtex():
    readme = "Intro\n\n```python\nprint('hi')\n```\n\n```bibtex\n@misc{x, year = {2020}}\n```\n"

    assert _readme_entry(readme) == "@misc{x, year = {2020}}"


def test_the_citation_section_wins_over_an_upstream_entry_quoted_earlier():
    readme = (
        "## Source data\n\nBuilt from:\n\n```bibtex\n@article{other2018, year={2018}}\n```\n\n"
        "## Citation Information\n\n```bibtex\n@misc{mine2024, year={2024}}\n```\n"
    )

    assert _readme_entry(readme) == "@misc{mine2024, year={2024}}"


def test_a_bare_entry_under_a_citation_heading_wins_over_an_earlier_fence():
    readme = "## Source\n\n```bibtex\n@article{other2018, year={2018}}\n```\n\n## Citation\n\n@misc{mine2024, year={2024}}\n"

    assert _readme_entry(readme) == "@misc{mine2024, year={2024}}"


def test_without_a_fence_the_entry_under_a_citation_heading_is_read():
    readme = "## Citation\n\nSee below.\n\n@article{y, title={Z}, year={2019}}\n\n## Notes\n\n@misc{other, year={2001}}\n"

    assert _readme_entry(readme) == "@article{y, title={Z}, year={2019}}"


def test_a_hash_line_inside_a_code_block_is_not_a_heading():
    readme = "```\n# Citation\n```\n\n@misc{x, year={2020}}\n"

    assert _readme_entry(readme) is None


def test_an_entry_with_unbalanced_braces_is_no_bibtex():
    assert _readme_entry("## Citation\n\n@misc{x, title={oops\n") is None


def test_a_readme_without_an_entry_has_no_bibtex():
    assert _readme_entry("# Title\n\nNothing to cite here.\n") is None


def test_the_readme_is_read_up_to_the_cap_and_no_further(monkeypatch):
    # The entry sits past the cap, under its heading: read in full, it would be found.
    body = b"## Citation\n\n" + b"x" * (huggingface.README_BYTES + 100_000) + b"@misc{late, year={2020}}\n"
    streams = []

    def get(url, **kwargs):
        streams.append(Stream(body))
        return streams[-1]

    monkeypatch.setattr(huggingface.requests, "get", get)

    assert _readme_bibtex("coastalcph/lex_glue", SHA) is None
    assert streams[0].served <= huggingface.README_BYTES + 16_384
    assert streams[0].closed


def test_an_entry_within_the_cap_is_read(monkeypatch):
    body = b"## Citation\n\n@misc{early, year={2020}}\n" + b"x" * (huggingface.README_BYTES * 2)
    monkeypatch.setattr(huggingface.requests, "get", lambda url, **kwargs: Stream(body))

    assert _readme_bibtex("coastalcph/lex_glue", SHA) == "@misc{early, year={2020}}"


@pytest.mark.parametrize("status", [404, 500])
def test_a_readme_that_does_not_answer_is_no_bibtex(monkeypatch, status):
    monkeypatch.setattr(huggingface.requests, "get", lambda url, **kwargs: Stream(LEX_README_BIBTEX.encode(), status))

    assert _readme_bibtex("coastalcph/lex_glue", SHA) is None


def test_a_readme_that_cannot_be_reached_still_builds_the_card(monkeypatch, hub):
    real = hub.get

    def get(url, **kwargs):
        if "/resolve/" in url:
            raise requests.ConnectionError("the Hub went away")
        return real(url, **kwargs)

    monkeypatch.setattr(huggingface.requests, "get", get)

    body = huggingface.dataset("coastalcph/lex_glue", TASK_TYPES)
    assert body["title"] == "LexGLUE"
    assert body["citation"]["bibtex"] is None
    assert body["citation"]["authors"] == [{"literal": "coastalcph"}]


def test_a_path_like_id_or_a_branch_name_is_never_fetched(monkeypatch):
    def get(url, **kwargs):
        raise AssertionError(f"nothing should be fetched, but got {url}")

    monkeypatch.setattr(huggingface.requests, "get", get)

    assert _readme_bibtex("coastalcph/../secret", SHA) is None
    assert _readme_bibtex("coastalcph/lex_glue", "main") is None
