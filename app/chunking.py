"""Split text into overlapping token windows.

Pure function: the tokenizer is injected as `token_spans(text) -> [(start, end), ...]`
(character offsets of each token), so tests run without the embedding model. Chunk text
is sliced from the original string by those offsets, not decoded from token ids — the
MiniLM tokenizer is uncased, and decoding would lowercase the text we later cite.
"""

from collections.abc import Callable

TokenSpans = Callable[[str], list[tuple[int, int]]]


def chunk_text(text: str, token_spans: TokenSpans, size: int, overlap: int) -> list[str]:
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be in [0, size)")

    spans = token_spans(text)
    if not spans:
        return []

    chunks = []
    step = size - overlap
    for start in range(0, len(spans), step):
        window = spans[start : start + size]
        chunks.append(text[window[0][0] : window[-1][1]])
        if start + size >= len(spans):
            break
    return chunks
