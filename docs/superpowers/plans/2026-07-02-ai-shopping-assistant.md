# AI Shopping Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bangun AI Shopping Assistant di mana pembeli mengobrol natural, sistem menemukan produk lewat hybrid semantic search (pgvector), dan Claude (via LangChain agent + tools + memory) menyusun rekomendasi.

**Architecture:** Dua jalur terpisah — *ingest* (Shopify → embed lokal → pgvector) dan *query* (Streamlit → LangChain agent Claude → tool search_products/check_stock/get_price → pgvector → rekomendasi). Tiap komponen adalah unit terpisah dengan interface bersih.

**Tech Stack:** Python 3.11+, LangChain + langchain-anthropic, Claude API, sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`, 384 dim), Supabase Postgres + pgvector, Shopify Admin GraphQL API, Streamlit, pytest.

---

## Catatan Embedding Model

Spec menyebut "BGE-small (384 dim)". Saat perencanaan dipilih `paraphrase-multilingual-MiniLM-L12-v2` — **tetap 384 dimensi** tapi multilingual, agar query bahasa Indonesia cocok dengan teks produk. Kolom `vector(384)` di spec tidak berubah.

## File Structure

```
automation-ecommerce/
├── .env.example                 # template env vars
├── requirements.txt             # dependencies
├── pytest.ini                   # config test
├── sql/
│   └── 001_init.sql             # extension pgvector + tabel products + RPC match_products
├── src/
│   ├── __init__.py
│   ├── config.py                # baca env vars (keys, model name, top_k)
│   ├── embeddings.py            # embed(texts) -> list[list[float]]
│   ├── shopify_client.py        # fetch_products(), get_product(id) via Admin GraphQL
│   ├── normalize.py             # normalize_product(raw) -> ProductRecord + build_embedding_text
│   ├── db.py                    # klien Supabase + upsert_products() + match_products()
│   ├── ingest.py                # pipeline: shopify -> normalize -> embed -> upsert
│   ├── retriever.py             # search_products(query, filters) -> list[ProductHit]
│   ├── tools.py                 # LangChain tools: search_products, check_stock, get_price
│   └── agent.py                 # build_agent() -> AgentExecutor
├── app.py                       # Streamlit chat UI
└── tests/
    ├── __init__.py
    ├── test_embeddings.py
    ├── test_normalize.py
    ├── test_shopify_client.py
    ├── test_db.py
    ├── test_retriever.py
    └── test_tools.py
```

---

## Phase 0 — Setup

### Task 1: Project scaffold & config

**Files:**
- Create: `requirements.txt`, `.env.example`, `pytest.ini`, `src/__init__.py`, `tests/__init__.py`, `src/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write `requirements.txt`**

```
langchain>=0.3,<0.4
langchain-anthropic>=0.3,<0.4
langchain-community>=0.3,<0.4
sentence-transformers>=3.0
supabase>=2.7
requests>=2.32
streamlit>=1.38
python-dotenv>=1.0
pytest>=8.0
```

- [ ] **Step 2: Write `.env.example`**

```
ANTHROPIC_API_KEY=sk-ant-xxx
ANTHROPIC_MODEL=claude-opus-4-8
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_SERVICE_KEY=eyJ...
SHOPIFY_STORE_DOMAIN=your-store.myshopify.com
SHOPIFY_ADMIN_TOKEN=shpat_xxx
SHOPIFY_API_VERSION=2024-10
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DIM=384
TOP_K=6
```

- [ ] **Step 3: Write `pytest.ini`**

```ini
[pytest]
pythonpath = .
testpaths = tests
```

- [ ] **Step 4: Create empty `src/__init__.py` and `tests/__init__.py`**

Both files empty.

- [ ] **Step 5: Write the failing test** `tests/test_config.py`

```python
import os
from src import config

def test_config_reads_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("SUPABASE_URL", "u")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "s")
    monkeypatch.setenv("TOP_K", "9")
    cfg = config.load()
    assert cfg.anthropic_api_key == "k"
    assert cfg.supabase_url == "u"
    assert cfg.top_k == 9
    assert cfg.embedding_dim == 384
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL (`AttributeError: module 'src.config' has no attribute 'load'`)

- [ ] **Step 7: Write `src/config.py`**

```python
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
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add requirements.txt .env.example pytest.ini src tests
git commit -m "chore: project scaffold and config loader"
```

---

### Task 2: Supabase schema (pgvector + products + RPC)

**Files:**
- Create: `sql/001_init.sql`

- [ ] **Step 1: Write `sql/001_init.sql`**

```sql
-- Enable pgvector
create extension if not exists vector;

-- Products table
create table if not exists products (
    id            text primary key,
    title         text not null,
    description   text default '',
    product_type  text default '',
    tags          text[] default '{}',
    price         numeric default 0,
    currency      text default 'IDR',
    inventory_qty integer default 0,
    in_stock      boolean default false,
    image_url     text default '',
    url           text default '',
    embedding     vector(384),
    updated_at    timestamptz default now()
);

-- HNSW index for cosine similarity
create index if not exists products_embedding_idx
    on products using hnsw (embedding vector_cosine_ops);

-- Hybrid search RPC: cosine similarity + metadata filters
create or replace function match_products(
    query_embedding vector(384),
    match_count int default 6,
    min_price numeric default null,
    max_price numeric default null,
    category text default null,
    only_in_stock boolean default false
)
returns table (
    id text,
    title text,
    description text,
    product_type text,
    price numeric,
    currency text,
    in_stock boolean,
    image_url text,
    url text,
    similarity float
)
language sql stable
as $$
    select
        p.id, p.title, p.description, p.product_type,
        p.price, p.currency, p.in_stock, p.image_url, p.url,
        1 - (p.embedding <=> query_embedding) as similarity
    from products p
    where (min_price is null or p.price >= min_price)
      and (max_price is null or p.price <= max_price)
      and (category is null or p.product_type ilike '%' || category || '%')
      and (only_in_stock = false or p.in_stock = true)
    order by p.embedding <=> query_embedding
    limit match_count;
$$;
```

- [ ] **Step 2: Apply the migration to Supabase**

Jalankan isi `sql/001_init.sql` di Supabase (SQL Editor di dashboard, atau via MCP `apply_migration` dengan name `init` dan query dari file). Verifikasi tabel muncul.

Run (verifikasi lewat SQL Editor): `select count(*) from products;`
Expected: `0` (tabel ada, kosong)

- [ ] **Step 3: Commit**

```bash
git add sql/001_init.sql
git commit -m "feat: supabase pgvector schema and match_products RPC"
```

---

### Task 3: Shopify store readiness

**Files:** none (langkah operasional)

- [ ] **Step 1: Pastikan ada store + produk**

Jika sudah punya store Shopify dengan produk: lewati ke Step 2. Jika belum: buat store contoh (mis. lewat Shopify MCP `get-new-store-previews` dari deskripsi toko, sign up, produk contoh ikut terpasang), lalu tambahkan beberapa produk bervariasi (kategori, harga berbeda) agar semantic search terlihat bekerja.

- [ ] **Step 2: Buat Admin API access token**

Di Shopify admin → Settings → Apps and sales channels → Develop apps → buat app → Admin API access scopes: aktifkan `read_products`, `read_inventory`. Install app, salin Admin API access token (`shpat_...`).

- [ ] **Step 3: Isi `.env`**

Salin `.env.example` menjadi `.env`, isi `ANTHROPIC_API_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `SHOPIFY_STORE_DOMAIN`, `SHOPIFY_ADMIN_TOKEN`.

```bash
cp .env.example .env
# edit .env dengan nilai asli
```

- [ ] **Step 4: Verifikasi `.env` tidak ke-commit**

```bash
echo ".env" >> .gitignore
git add .gitignore
git commit -m "chore: gitignore .env"
```

---

## Phase 1 — Ingestion + Embeddings

### Task 4: Embedding module

**Files:**
- Create: `src/embeddings.py`
- Test: `tests/test_embeddings.py`

- [ ] **Step 1: Write the failing test** `tests/test_embeddings.py`

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_embeddings.py -v`
Expected: FAIL (`AttributeError: module 'src.embeddings' has no attribute 'embed'`)

- [ ] **Step 3: Write `src/embeddings.py`**

```python
from functools import lru_cache
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_embeddings.py -v`
Expected: PASS (download model ~120MB pada run pertama)

- [ ] **Step 5: Commit**

```bash
git add src/embeddings.py tests/test_embeddings.py
git commit -m "feat: local multilingual embedding module"
```

---

### Task 5: Shopify client

**Files:**
- Create: `src/shopify_client.py`
- Test: `tests/test_shopify_client.py`

- [ ] **Step 1: Write the failing test** `tests/test_shopify_client.py`

```python
from unittest.mock import patch, MagicMock
from src import shopify_client

FAKE_GRAPHQL_RESPONSE = {
    "data": {
        "products": {
            "pageInfo": {"hasNextPage": False, "endCursor": "c1"},
            "edges": [
                {"node": {
                    "id": "gid://shopify/Product/1",
                    "title": "Kopi Arabika Gayo 250g",
                    "description": "Biji kopi arabika single origin.",
                    "productType": "Kopi",
                    "tags": ["kopi", "arabika"],
                    "onlineStoreUrl": "https://shop/products/kopi",
                    "featuredImage": {"url": "https://img/kopi.jpg"},
                    "variants": {"edges": [
                        {"node": {"price": "85000", "inventoryQuantity": 12}}
                    ]},
                }}
            ],
        }
    }
}

def test_fetch_products_parses_nodes():
    with patch("src.shopify_client._post_graphql", return_value=FAKE_GRAPHQL_RESPONSE):
        products = shopify_client.fetch_products()
    assert len(products) == 1
    p = products[0]
    assert p["id"] == "gid://shopify/Product/1"
    assert p["title"] == "Kopi Arabika Gayo 250g"
    assert p["variants"][0]["price"] == "85000"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_shopify_client.py -v`
Expected: FAIL (`AttributeError: ... 'fetch_products'`)

- [ ] **Step 3: Write `src/shopify_client.py`**

```python
import requests
from src import config

_cfg = config.load()

_PRODUCTS_QUERY = """
query($cursor: String) {
  products(first: 50, after: $cursor) {
    pageInfo { hasNextPage endCursor }
    edges {
      node {
        id title description productType tags onlineStoreUrl
        featuredImage { url }
        variants(first: 1) {
          edges { node { price inventoryQuantity } }
        }
      }
    }
  }
}
"""


def _post_graphql(query: str, variables: dict) -> dict:
    url = f"https://{_cfg.shopify_domain}/admin/api/{_cfg.shopify_api_version}/graphql.json"
    resp = requests.post(
        url,
        json={"query": query, "variables": variables},
        headers={"X-Shopify-Access-Token": _cfg.shopify_token,
                 "Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_products() -> list[dict]:
    """Ambil semua produk (paginasi) sebagai list dict mentah."""
    out, cursor = [], None
    while True:
        data = _post_graphql(_PRODUCTS_QUERY, {"cursor": cursor})
        conn = data["data"]["products"]
        for edge in conn["edges"]:
            node = edge["node"]
            node["variants"] = [v["node"] for v in node["variants"]["edges"]]
            out.append(node)
        if not conn["pageInfo"]["hasNextPage"]:
            break
        cursor = conn["pageInfo"]["endCursor"]
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_shopify_client.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/shopify_client.py tests/test_shopify_client.py
git commit -m "feat: shopify admin graphql product fetcher"
```

---

### Task 6: Product normalization

**Files:**
- Create: `src/normalize.py`
- Test: `tests/test_normalize.py`

- [ ] **Step 1: Write the failing test** `tests/test_normalize.py`

```python
from src import normalize

RAW = {
    "id": "gid://shopify/Product/1",
    "title": "Kopi Arabika Gayo 250g",
    "description": "Biji kopi arabika single origin.",
    "productType": "Kopi",
    "tags": ["kopi", "arabika"],
    "onlineStoreUrl": "https://shop/products/kopi",
    "featuredImage": {"url": "https://img/kopi.jpg"},
    "variants": [{"price": "85000", "inventoryQuantity": 12}],
}

def test_normalize_maps_fields():
    rec = normalize.normalize_product(RAW)
    assert rec["id"] == "gid://shopify/Product/1"
    assert rec["price"] == 85000.0
    assert rec["inventory_qty"] == 12
    assert rec["in_stock"] is True
    assert rec["image_url"] == "https://img/kopi.jpg"

def test_out_of_stock_when_qty_zero():
    raw = {**RAW, "variants": [{"price": "1000", "inventoryQuantity": 0}]}
    rec = normalize.normalize_product(raw)
    assert rec["in_stock"] is False

def test_embedding_text_includes_title_and_type():
    text = normalize.build_embedding_text(normalize.normalize_product(RAW))
    assert "Kopi Arabika Gayo 250g" in text
    assert "Kopi" in text
    assert "arabika" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_normalize.py -v`
Expected: FAIL (`AttributeError: ... 'normalize_product'`)

- [ ] **Step 3: Write `src/normalize.py`**

```python
def normalize_product(raw: dict) -> dict:
    variants = raw.get("variants") or [{}]
    v0 = variants[0]
    qty = int(v0.get("inventoryQuantity") or 0)
    price = float(v0.get("price") or 0)
    image = (raw.get("featuredImage") or {}).get("url", "") or ""
    return {
        "id": raw["id"],
        "title": raw.get("title", ""),
        "description": raw.get("description", "") or "",
        "product_type": raw.get("productType", "") or "",
        "tags": raw.get("tags", []) or [],
        "price": price,
        "currency": "IDR",
        "inventory_qty": qty,
        "in_stock": qty > 0,
        "image_url": image,
        "url": raw.get("onlineStoreUrl", "") or "",
    }


def build_embedding_text(rec: dict) -> str:
    """Gabungkan field yang bermakna jadi satu teks untuk di-embed."""
    parts = [
        rec.get("title", ""),
        rec.get("product_type", ""),
        " ".join(rec.get("tags", [])),
        rec.get("description", ""),
    ]
    return " | ".join(p for p in parts if p).strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_normalize.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/normalize.py tests/test_normalize.py
git commit -m "feat: product normalization and embedding text builder"
```

---

### Task 7: DB layer (Supabase upsert + match)

**Files:**
- Create: `src/db.py`
- Test: `tests/test_db.py`

- [ ] **Step 1: Write the failing test** `tests/test_db.py`

```python
from unittest.mock import MagicMock, patch
from src import db

def test_upsert_products_calls_supabase():
    fake_client = MagicMock()
    with patch("src.db.get_client", return_value=fake_client):
        db.upsert_products([{"id": "1", "title": "x"}])
    fake_client.table.assert_called_with("products")
    fake_client.table().upsert.assert_called()

def test_match_products_calls_rpc_and_returns_data():
    fake_client = MagicMock()
    fake_client.rpc().execute.return_value.data = [{"id": "1", "similarity": 0.9}]
    with patch("src.db.get_client", return_value=fake_client):
        rows = db.match_products([0.0] * 384, match_count=3)
    assert rows == [{"id": "1", "similarity": 0.9}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_db.py -v`
Expected: FAIL (`AttributeError: ... 'upsert_products'`)

- [ ] **Step 3: Write `src/db.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_db.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/db.py tests/test_db.py
git commit -m "feat: supabase db layer (upsert + match RPC)"
```

---

### Task 8: Ingestion pipeline + first real run

**Files:**
- Create: `src/ingest.py`

- [ ] **Step 1: Write `src/ingest.py`**

```python
from src import shopify_client, normalize, embeddings, db


def run() -> int:
    """Tarik produk Shopify, embed, upsert ke pgvector. Kembalikan jumlah produk."""
    raw = shopify_client.fetch_products()
    records, texts = [], []
    for r in raw:
        try:
            rec = normalize.normalize_product(r)
        except Exception as e:  # noqa: BLE001
            print(f"skip {r.get('id')}: {e}")
            continue
        records.append(rec)
        texts.append(normalize.build_embedding_text(rec))

    vectors = embeddings.embed(texts)
    for rec, vec in zip(records, vectors):
        rec["embedding"] = vec

    # upsert per batch 50 agar tahan payload besar
    for i in range(0, len(records), 50):
        db.upsert_products(records[i:i + 50])

    print(f"ingested {len(records)} products")
    return len(records)


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Jalankan ingest sungguhan**

Run: `python -m src.ingest`
Expected: cetak `ingested N products` (N = jumlah produk store).

- [ ] **Step 3: Verifikasi data + embedding di Supabase**

Jalankan di Supabase SQL Editor:
```sql
select count(*) filter (where embedding is not null) as embedded, count(*) as total from products;
```
Expected: `embedded` = `total` > 0.

- [ ] **Step 4: Commit**

```bash
git add src/ingest.py
git commit -m "feat: ingestion pipeline shopify -> embed -> pgvector"
```

---

## Phase 2 — Retriever (Hybrid Search)

### Task 9: Retriever

**Files:**
- Create: `src/retriever.py`
- Test: `tests/test_retriever.py`

- [ ] **Step 1: Write the failing test** `tests/test_retriever.py`

```python
from unittest.mock import patch
from src import retriever

def test_search_embeds_query_and_calls_match():
    with patch("src.retriever.embeddings.embed", return_value=[[0.1] * 384]) as m_embed, \
         patch("src.retriever.db.match_products",
               return_value=[{"id": "1", "title": "Kopi", "price": 85000,
                              "in_stock": True, "similarity": 0.8,
                              "image_url": "", "url": "", "description": "",
                              "product_type": "Kopi", "currency": "IDR"}]) as m_match:
        hits = retriever.search_products("kopi enak murah", max_price=100000)
    m_embed.assert_called_once()
    assert m_match.call_args.kwargs["max_price"] == 100000
    assert hits[0]["title"] == "Kopi"

def test_search_passes_filters_through():
    with patch("src.retriever.embeddings.embed", return_value=[[0.0] * 384]), \
         patch("src.retriever.db.match_products", return_value=[]) as m_match:
        retriever.search_products("x", min_price=10, category="Kopi", only_in_stock=True)
    kwargs = m_match.call_args.kwargs
    assert kwargs["min_price"] == 10
    assert kwargs["category"] == "Kopi"
    assert kwargs["only_in_stock"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_retriever.py -v`
Expected: FAIL (`AttributeError: ... 'search_products'`)

- [ ] **Step 3: Write `src/retriever.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_retriever.py -v`
Expected: PASS

- [ ] **Step 5: Smoke test manual terhadap data nyata**

```bash
python -c "from src import retriever; import json; print(json.dumps([h['title'] for h in retriever.search_products('sesuatu yang relevan dgn tokomu')], ensure_ascii=False))"
```
Expected: daftar judul produk yang masuk akal secara makna.

- [ ] **Step 6: Commit**

```bash
git add src/retriever.py tests/test_retriever.py
git commit -m "feat: hybrid search retriever"
```

---

## Phase 3 — RAG MVP + Streamlit

### Task 10: Streamlit RAG MVP (retrieve → Claude merangkai)

**Files:**
- Create: `app.py`

Pada fase ini belum agentic: kita ambil hasil retriever lalu berikan ke Claude untuk merangkai rekomendasi. Ini memberi UI yang bisa dilihat sebelum menambah kompleksitas agent.

- [ ] **Step 1: Write `app.py`**

```python
import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from src import retriever, config

_cfg = config.load()

st.set_page_config(page_title="AI Shopping Assistant", page_icon="🛍️")
st.title("🛍️ AI Shopping Assistant")

llm = ChatAnthropic(model=_cfg.anthropic_model, api_key=_cfg.anthropic_api_key,
                    temperature=0.3)

SYSTEM = (
    "Kamu asisten belanja yang ramah. Berdasarkan daftar produk yang diberikan, "
    "rekomendasikan yang paling cocok dengan kebutuhan user beserta alasan singkat. "
    "Sebutkan nama & harga. Jika tidak ada yang cocok, katakan jujur dan sarankan "
    "melonggarkan kriteria. Jawab dalam bahasa yang dipakai user."
)


def _format_products(hits: list[dict]) -> str:
    if not hits:
        return "(tidak ada produk ditemukan)"
    lines = []
    for h in hits:
        stok = "tersedia" if h["in_stock"] else "habis"
        lines.append(f"- {h['title']} | {h['currency']} {h['price']:.0f} | {h['product_type']} | stok: {stok}")
    return "\n".join(lines)


query = st.text_input("Cari produk (mis. 'kado kopi budget 300rb'):")
if query:
    hits = retriever.search_products(query)
    with st.spinner("Menyusun rekomendasi..."):
        resp = llm.invoke([
            SystemMessage(content=SYSTEM),
            HumanMessage(content=f"Kebutuhan user: {query}\n\nProduk kandidat:\n{_format_products(hits)}"),
        ])
    st.markdown(resp.content)
    with st.expander("Produk kandidat (raw)"):
        for h in hits:
            st.write(f"**{h['title']}** — {h['currency']} {h['price']:.0f}")
```

- [ ] **Step 2: Run the app**

Run: `streamlit run app.py`
Expected: browser terbuka; ketik query → muncul rekomendasi dari Claude berbasis produk nyata.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: streamlit RAG MVP (retrieve + claude recommend)"
```

---

## Phase 4 — Agent + Tools

### Task 11: LangChain tools

**Files:**
- Create: `src/tools.py`
- Test: `tests/test_tools.py`

- [ ] **Step 1: Write the failing test** `tests/test_tools.py`

```python
from unittest.mock import patch
from src import tools

def test_search_products_tool_returns_readable_text():
    fake = [{"id": "1", "title": "Kopi Gayo", "price": 85000, "currency": "IDR",
             "in_stock": True, "product_type": "Kopi", "url": "http://x",
             "image_url": "", "description": "", "similarity": 0.9}]
    with patch("src.tools.retriever.search_products", return_value=fake):
        out = tools.search_products_tool.invoke({"query": "kopi", "max_price": 100000})
    assert "Kopi Gayo" in out
    assert "85000" in out

def test_check_stock_tool():
    with patch("src.tools.shopify_client.get_product",
               return_value={"inventory_qty": 5}):
        out = tools.check_stock_tool.invoke({"product_id": "gid://shopify/Product/1"})
    assert "5" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tools.py -v`
Expected: FAIL (`AttributeError: ... 'search_products_tool'` / `get_product`)

- [ ] **Step 3: Add `get_product` to `src/shopify_client.py`**

Tambahkan di akhir `src/shopify_client.py`:
```python
_PRODUCT_QUERY = """
query($id: ID!) {
  product(id: $id) {
    id title
    variants(first: 1) { edges { node { price inventoryQuantity } } }
  }
}
"""


def get_product(product_id: str) -> dict:
    data = _post_graphql(_PRODUCT_QUERY, {"id": product_id})
    node = data["data"]["product"]
    v = node["variants"]["edges"][0]["node"] if node["variants"]["edges"] else {}
    return {
        "id": node["id"],
        "title": node["title"],
        "price": float(v.get("price") or 0),
        "inventory_qty": int(v.get("inventoryQuantity") or 0),
    }
```

- [ ] **Step 4: Write `src/tools.py`**

```python
from langchain_core.tools import tool
from src import retriever, shopify_client


@tool
def search_products_tool(query: str, min_price: float | None = None,
                         max_price: float | None = None,
                         category: str | None = None,
                         only_in_stock: bool = False) -> str:
    """Cari produk toko berdasarkan makna query + filter harga/kategori/stok.
    Gunakan saat user ingin menemukan atau membandingkan produk."""
    hits = retriever.search_products(
        query, min_price=min_price, max_price=max_price,
        category=category, only_in_stock=only_in_stock)
    if not hits:
        return "Tidak ada produk yang cocok."
    lines = []
    for h in hits:
        stok = "tersedia" if h["in_stock"] else "habis"
        lines.append(
            f"[{h['id']}] {h['title']} | {h['currency']} {h['price']:.0f} | "
            f"{h['product_type']} | stok: {stok} | {h['url']}")
    return "\n".join(lines)


@tool
def check_stock_tool(product_id: str) -> str:
    """Cek jumlah stok terkini sebuah produk berdasarkan product_id (gid)."""
    p = shopify_client.get_product(product_id)
    return f"{p['title']}: stok {p['inventory_qty']} unit."


@tool
def get_price_tool(product_id: str) -> str:
    """Cek harga terkini sebuah produk berdasarkan product_id (gid)."""
    p = shopify_client.get_product(product_id)
    return f"{p['title']}: harga {p['price']:.0f}."


ALL_TOOLS = [search_products_tool, check_stock_tool, get_price_tool]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_tools.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/tools.py src/shopify_client.py tests/test_tools.py
git commit -m "feat: langchain tools (search, check_stock, get_price)"
```

---

### Task 12: Agent assembly

**Files:**
- Create: `src/agent.py`

- [ ] **Step 1: Write `src/agent.py`**

```python
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src import config
from src.tools import ALL_TOOLS

_cfg = config.load()

SYSTEM = (
    "Kamu asisten belanja yang ramah dan jujur untuk sebuah toko online. "
    "Gunakan tool search_products_tool untuk menemukan produk sesuai kebutuhan user "
    "(termasuk filter harga/kategori/stok bila disebut). Gunakan check_stock_tool / "
    "get_price_tool untuk memastikan stok & harga terkini sebelum merekomendasikan. "
    "Rekomendasikan produk dengan alasan singkat, sebutkan nama & harga. "
    "Ingat preferensi dan budget yang sudah disebut user di percakapan. "
    "Jika tak ada yang cocok, katakan jujur dan sarankan melonggarkan kriteria. "
    "Jawab dalam bahasa yang dipakai user."
)


def build_agent() -> AgentExecutor:
    llm = ChatAnthropic(model=_cfg.anthropic_model,
                        api_key=_cfg.anthropic_api_key, temperature=0.3)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, ALL_TOOLS, prompt)
    return AgentExecutor(agent=agent, tools=ALL_TOOLS, verbose=True)
```

- [ ] **Step 2: Smoke test agent dari CLI**

```bash
python -c "from src.agent import build_agent; a=build_agent(); print(a.invoke({'input':'cari kopi murah di bawah 100rb','chat_history':[]})['output'])"
```
Expected: agent memanggil tool search lalu mengembalikan rekomendasi teks.

- [ ] **Step 3: Commit**

```bash
git add src/agent.py
git commit -m "feat: langchain tool-calling agent with claude"
```

---

### Task 13: Wire agent into Streamlit (ganti RAG MVP)

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Replace `app.py` dengan versi agent + chat**

```python
import streamlit as st
from src.agent import build_agent

st.set_page_config(page_title="AI Shopping Assistant", page_icon="🛍️")
st.title("🛍️ AI Shopping Assistant")

if "agent" not in st.session_state:
    st.session_state.agent = build_agent()
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of (role, content)

for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

if prompt := st.chat_input("Tanya apa saja soal produk..."):
    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Berpikir..."):
            result = st.session_state.agent.invoke({
                "input": prompt,
                "chat_history": [],  # diisi di Task 14 (memory)
            })
        st.markdown(result["output"])
    st.session_state.messages.append(("assistant", result["output"]))
```

- [ ] **Step 2: Run the app**

Run: `streamlit run app.py`
Expected: UI chat; tanya produk → agent memakai tool → rekomendasi muncul.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: streamlit chat UI backed by agent"
```

---

## Phase 5 — Conversation Memory

### Task 14: Multi-turn memory

**Files:**
- Modify: `app.py`

Sekarang isi `chat_history` dari pesan sebelumnya agar agent ingat budget & preferensi lintas pesan.

- [ ] **Step 1: Update `app.py` untuk mengirim chat_history**

Ganti blok `if prompt := st.chat_input(...)` menjadi:
```python
from langchain_core.messages import HumanMessage, AIMessage

if prompt := st.chat_input("Tanya apa saja soal produk..."):
    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    history = []
    for role, content in st.session_state.messages[:-1]:
        history.append(HumanMessage(content=content) if role == "user"
                       else AIMessage(content=content))

    with st.chat_message("assistant"):
        with st.spinner("Berpikir..."):
            result = st.session_state.agent.invoke({
                "input": prompt,
                "chat_history": history,
            })
        st.markdown(result["output"])
    st.session_state.messages.append(("assistant", result["output"]))
```

- [ ] **Step 2: Uji multi-turn manual**

Run: `streamlit run app.py`
Uji: pesan 1 "budget saya 200rb", pesan 2 "cari kopi" → rekomendasi harus menghormati budget dari pesan 1.
Expected: agent mengingat budget lintas pesan.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: multi-turn conversation memory in chat UI"
```

---

## Phase 6 — Polish + Deploy

### Task 15: Product cards di UI

**Files:**
- Modify: `app.py`
- Modify: `src/agent.py`

Agar UI menampilkan kartu produk (gambar+harga+link), agent mengembalikan produk terstruktur lewat intermediate steps.

- [ ] **Step 1: Aktifkan intermediate steps di `src/agent.py`**

Ubah baris return di `build_agent()`:
```python
    return AgentExecutor(agent=agent, tools=ALL_TOOLS, verbose=True,
                         return_intermediate_steps=True)
```

- [ ] **Step 2: Render kartu dari hasil tool search di `app.py`**

Tambahkan helper dan render setelah `st.markdown(result["output"])`:
```python
def _render_product_cards(result):
    for action, observation in result.get("intermediate_steps", []):
        if action.tool != "search_products_tool":
            continue
        for line in observation.splitlines():
            if not line.startswith("["):
                continue
            # format: [id] title | CUR price | type | stok: x | url
            try:
                _, rest = line.split("] ", 1)
                title, cur_price, _type, _stok, url = [s.strip() for s in rest.split("|")]
            except ValueError:
                continue
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(cur_price)
                st.markdown(f"[Lihat produk]({url})")
```
Lalu panggil `_render_product_cards(result)` tepat setelah menampilkan `result["output"]`.

- [ ] **Step 3: Run the app & verifikasi kartu**

Run: `streamlit run app.py`
Expected: setelah rekomendasi, muncul kartu produk dengan judul, harga, dan link.

- [ ] **Step 4: Commit**

```bash
git add app.py src/agent.py
git commit -m "feat: render product cards from agent tool output"
```

---

### Task 16: README + deploy ke Streamlit Cloud

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# AI Shopping Assistant

Asisten belanja berbasis LangChain + Claude + Supabase pgvector + Shopify.

## Setup
1. `pip install -r requirements.txt`
2. `cp .env.example .env` lalu isi kredensial.
3. Jalankan `sql/001_init.sql` di Supabase.
4. `python -m src.ingest` untuk mengisi katalog.
5. `streamlit run app.py`

## Arsitektur
- Ingest: Shopify → embedding lokal (MiniLM 384d) → pgvector.
- Query: Streamlit → LangChain agent (Claude) → tools (search/stock/price) → pgvector.

## Test
`pytest`
```

- [ ] **Step 2: Jalankan seluruh test**

Run: `pytest -v`
Expected: semua PASS.

- [ ] **Step 3: Deploy ke Streamlit Cloud**

Push repo ke GitHub (private OK). Di share.streamlit.io: New app → pilih repo → main file `app.py`. Di Settings → Secrets, tempel isi `.env` dalam format TOML:
```toml
ANTHROPIC_API_KEY="sk-ant-..."
SUPABASE_URL="https://..."
SUPABASE_SERVICE_KEY="..."
SHOPIFY_STORE_DOMAIN="...myshopify.com"
SHOPIFY_ADMIN_TOKEN="shpat_..."
```
Expected: app live di URL publik untuk portfolio.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: readme and deploy instructions"
```

---

## Self-Review (dilakukan penulis rencana)

**Spec coverage:** Semua komponen spec tercakup —
ingestion (Task 8), embedding module (Task 4), retriever hybrid search (Task 9),
shopify tools (Task 11), agent (Task 12), UI Streamlit (Task 10/13/15), config/env
(Task 1). Data model + RPC (Task 2). 6 fase spec = Phase 0-6 di plan. Error handling
(ingest per-produk Task 8, retrieval kosong Task 11 tool, fallback UI). Testing (unit
di tiap task, integration smoke test Task 8/9/12, deploy Task 16). Non-goals dihormati
(tidak ada multi-tenant/checkout/widget produksi).

**Placeholder scan:** Tidak ada TBD/TODO; semua step berisi kode/perintah nyata.

**Type consistency:** `embed()` konsisten dikembalikan `list[list[float]]`; `normalize_product` dict dipakai konsisten di ingest & build_embedding_text; `match_products` params (min_price/max_price/category/only_in_stock/match_count) konsisten antara SQL RPC (Task 2), db.py (Task 7), retriever (Task 9), tools (Task 11); `get_product` dipakai di tools (Task 11) didefinisikan di Task 11 Step 3; format baris output `search_products_tool` (`[id] title | CUR price | type | stok | url`) konsisten dengan parser kartu di Task 15.
