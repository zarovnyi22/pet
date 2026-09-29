from sentence_transformers import SentenceTransformer

EMBEDDING_DIM = 384


class Embedder:
    """sentence-transformers wrapper: embeddings plus the model's own tokenizer for chunking."""

    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name, device="cpu")
        dim = self._model.get_embedding_dimension()
        if dim != EMBEDDING_DIM:
            raise ValueError(f"{model_name} produces {dim}-dim vectors, schema expects 384")

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        encoded = self._model.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True, verbose=False
        )
        return [tuple(span) for span in encoded["offset_mapping"]]

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Normalized vectors: cosine distance in pgvector then equals 1 - dot product.
        vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vectors.tolist()
