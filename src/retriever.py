from src import embeddings, db, config

_cfg = config.load()


def search_products(query: str, *, min_price=None, max_price=None,
                    category=None, only_in_stock: bool = False,
                    top_k: int | None = None) -> list[dict]:
    """Hybrid search: embed query lalu match di pgvector dengan filter metadata."""
    vec = embeddings.embed([query])[0]
    return db.match_products(
        vec,
        match_count=top_k or _cfg.top_k,
        min_price=min_price,
        max_price=max_price,
        category=category,
        only_in_stock=only_in_stock,
    )
