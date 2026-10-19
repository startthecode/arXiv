from rag.clean import clean_inline, is_bib_like


def test_clean_inline_strips_markup_and_normalizes_unicode():
    raw = "**Bold** and _it_ x<sup>2</sup> a<br>b � ﬁne 𝑝"
    assert clean_inline(raw) == "Bold and it x2 a b fine p"


def test_clean_inline_keeps_snake_case():
    assert clean_inline("set max_tokens and _emph_") == "set max_tokens and emph"


def test_bib_like():
    assert is_bib_like("- [1] A. Author. 2020. Title. In Proceedings of X.")
    assert is_bib_like("Smith, J. et al. Deep things. Journal of Stuff, 2019.")
    assert not is_bib_like("We train the model for three epochs on the full dataset.")
