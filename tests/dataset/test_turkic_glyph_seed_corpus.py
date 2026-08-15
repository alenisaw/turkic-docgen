import importlib.resources
from pathlib import Path


def test_seed_corpus_contains_required_kazakh_glyphs():
    corpus_dir = importlib.resources.files("turkicdocgen") / "data" / "corpus"
    text = (
        Path(str(corpus_dir / "kk_words.txt")).read_text(encoding="utf-8")
        + "\n"
        + Path(str(corpus_dir / "kk_phrases.txt")).read_text(encoding="utf-8")
    )
    for ch in "ӘәҒғҚқҢңӨөҰұҮүҺһІі":
        assert ch in text


def test_seed_corpus_contains_required_kyrgyz_glyphs():
    corpus_dir = importlib.resources.files("turkicdocgen") / "data" / "corpus"
    text = (
        Path(str(corpus_dir / "ky_words.txt")).read_text(encoding="utf-8")
        + "\n"
        + Path(str(corpus_dir / "ky_phrases.txt")).read_text(encoding="utf-8")
    )
    for ch in "ҢңӨөҮү":
        assert ch in text
