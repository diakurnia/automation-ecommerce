import os
from functools import lru_cache

# xet-based fast transfer can hang indefinitely in sandboxed/restricted
# network environments; force the plain HTTP download path instead.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from sentence_transformers import SentenceTransformer
from src import config

_cfg = config.load()


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(_cfg.embedding_model)


def embed(texts: list[str]) -> list[list[float]]:
    """Ubah daftar teks jadi daftar vektor 384-dim (float)."""
    cleaned = [(t or "").strip()[:2000] for t in texts]
    vectors = _model().encode(cleaned, normalize_embeddings=True)
    return [[float(x) for x in v] for v in vectors]
