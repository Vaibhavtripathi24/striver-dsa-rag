"""Pluggable embedder with ultra-lightweight ONNX support for low RAM cloud hosting."""

import contextlib
import io
import re
import sys
from typing import Protocol

from ytrag.config import EMBED_BATCH, EMBED_MODEL, EMBED_QUERY_PREFIX

_BENIGN = re.compile(
    r"unauthenticated requests to the HF Hub|Loading weights:|^\s*$"
)


@contextlib.contextmanager
def _quiet_load():
    """Swallow the known-benign loader chatter, re-emit everything else."""
    captured = io.StringIO()
    try:
        with contextlib.redirect_stderr(captured):
            yield
    finally:
        for line in captured.getvalue().splitlines():
            if not _BENIGN.search(line):
                print(line, file=sys.stderr)


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Local embeddings fallback using sentence-transformers."""

    def __init__(self, model_name: str = EMBED_MODEL, batch_size: int = EMBED_BATCH):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.batch_size = batch_size
        with _quiet_load():
            self.model = SentenceTransformer(model_name)
        get_dim = getattr(self.model, "get_embedding_dimension", None) or (
            self.model.get_sentence_embedding_dimension
        )
        self.dim = int(get_dim())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            EMBED_QUERY_PREFIX + text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()


class FastEmbedder:
    """Ultra-lightweight ONNX embedder for memory-constrained cloud environments (Render 512MB RAM).

    Uses ~60MB RAM instead of ~500MB PyTorch footprint.
    """

    def __init__(self, model_name: str = EMBED_MODEL, batch_size: int = EMBED_BATCH):
        from fastembed import TextEmbedding

        self.name = model_name
        self.batch_size = batch_size
        model_id = "BAAI/bge-small-en-v1.5" if "bge" in model_name.lower() else "sentence-transformers/all-MiniLM-L6-v2"
        self.model = TextEmbedding(model_name=model_id)
        self.dim = 384

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = list(self.model.embed(texts, batch_size=self.batch_size))
        return [v.tolist() for v in embeddings]

    def embed_query(self, text: str) -> list[float]:
        query = EMBED_QUERY_PREFIX + text if EMBED_QUERY_PREFIX else text
        embedding = list(self.model.embed([query]))[0]
        return embedding.tolist()


_EMBEDDER: Embedder | None = None


def get_embedder() -> Embedder:
    """Load the embedder once per process. Prefers FastEmbedder (~60MB RAM) for low memory."""
    global _EMBEDDER
    if _EMBEDDER is None:
        try:
            _EMBEDDER = FastEmbedder()
            print("Loaded FastEmbedder (ONNX lightweight embedder: ~60MB RAM)", flush=True)
        except Exception as exc:
            print(f"FastEmbedder unavailable ({exc}), fallback to SentenceTransformer", flush=True)
            _EMBEDDER = SentenceTransformerEmbedder()
    return _EMBEDDER
