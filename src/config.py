import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    anthropic_api_key: str
    anthropic_model: str
    supabase_url: str
    supabase_service_key: str
    shopify_domain: str
    shopify_token: str
    shopify_api_version: str
    embedding_model: str
    embedding_dim: int
    top_k: int


def load() -> Config:
    return Config(
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        anthropic_model=os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8"),
        supabase_url=os.environ.get("SUPABASE_URL", ""),
        supabase_service_key=os.environ.get("SUPABASE_SERVICE_KEY", ""),
        shopify_domain=os.environ.get("SHOPIFY_STORE_DOMAIN", ""),
        shopify_token=os.environ.get("SHOPIFY_ADMIN_TOKEN", ""),
        shopify_api_version=os.environ.get("SHOPIFY_API_VERSION", "2024-10"),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"),
        embedding_dim=int(os.environ.get("EMBEDDING_DIM", "384")),
        top_k=int(os.environ.get("TOP_K", "6")),
    )
