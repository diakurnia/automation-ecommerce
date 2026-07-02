from src import embeddings


def test_embed_returns_384_dim_vectors():
    vecs = embeddings.embed(["kopi arabika premium", "sepatu lari pria"])
    assert len(vecs) == 2
    assert all(len(v) == 384 for v in vecs)
    assert all(isinstance(x, float) for x in vecs[0])


def test_embed_is_deterministic():
    a = embeddings.embed(["tas kulit coklat"])[0]
    b = embeddings.embed(["tas kulit coklat"])[0]
    assert a == b
