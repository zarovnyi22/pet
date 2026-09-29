import re
from itertools import pairwise

import pytest

from app.chunking import chunk_text


def word_spans(text: str) -> list[tuple[int, int]]:
    """Fake tokenizer: one token per whitespace-separated word."""
    return [m.span() for m in re.finditer(r"\S+", text)]


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_empty_and_blank_text_give_no_chunks():
    assert chunk_text("", word_spans, size=10, overlap=2) == []
    assert chunk_text("  \n\t ", word_spans, size=10, overlap=2) == []


def test_short_text_is_one_chunk():
    assert chunk_text("hello world", word_spans, size=10, overlap=2) == ["hello world"]


def test_exact_size_is_one_chunk():
    assert chunk_text(words(10), word_spans, size=10, overlap=2) == [words(10)]


def test_chunk_size_and_overlap():
    chunks = chunk_text(words(25), word_spans, size=10, overlap=3)
    tokens = [c.split() for c in chunks]

    # windows start at 0, 7, 14, 21 (step = size - overlap)
    assert [t[0] for t in tokens] == ["w0", "w7", "w14", "w21"]
    assert all(len(t) <= 10 for t in tokens)
    for prev, nxt in pairwise(tokens):
        assert prev[-3:] == nxt[:3]


def test_every_token_is_covered_and_last_chunk_ends_at_text_end():
    text = words(57)
    chunks = chunk_text(text, word_spans, size=10, overlap=4)
    covered = {w for c in chunks for w in c.split()}
    assert covered == set(text.split())
    assert chunks[-1].endswith("w56")


def test_no_trailing_chunk_fully_inside_previous_one():
    # 12 tokens, size 10, overlap 2: second window starts at 8 and reaches the end — stop.
    chunks = chunk_text(words(12), word_spans, size=10, overlap=2)
    assert len(chunks) == 2


def test_chunks_are_slices_of_original_text():
    text = "Oat drink:  4.0 g sugars\nper 100 g.  Replace milk 1:1."
    for chunk in chunk_text(text, word_spans, size=3, overlap=1):
        assert chunk in text


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (10, 10), (10, -1), (10, 11)])
def test_invalid_params(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("a b c", word_spans, size=size, overlap=overlap)
