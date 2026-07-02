from functools import lru_cache
from supabase import create_client, Client
from src import config

_cfg = config.load()


@lru_cache(maxsize=1)
def get_client() -> Client:
    return create_client(_cfg.supabase_url, _cfg.supabase_service_key)


def upsert_products(records: list[dict]) -> None:
    """Upsert baris produk (termasuk kolom embedding) ke tabel products."""
    if not records:
        return
    get_client().table("products").upsert(records).execute()


def match_products(query_embedding: list[float], match_count: int = 6,
                   min_price=None, max_price=None,
                   category=None, only_in_stock: bool = False) -> list[dict]:
    """Panggil RPC match_products (hybrid search)."""
    resp = get_client().rpc("match_products", {
        "query_embedding": query_embedding,
        "match_count": match_count,
        "min_price": min_price,
        "max_price": max_price,
        "category": category,
        "only_in_stock": only_in_stock,
    }).execute()
    return resp.data or []
