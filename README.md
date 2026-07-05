# AI Shopping Assistant

An AI shopping assistant built with LangChain, Claude, Supabase pgvector, and Shopify.

Shoppers chat naturally, the agent finds products through hybrid semantic
search (pgvector), checks live stock/price against Shopify, then puts
together a recommendation with reasoning — remembering preferences and
budget across messages.

## Architecture

- **Ingest** (batch): Shopify Admin API → local multilingual embeddings
  (`paraphrase-multilingual-MiniLM-L12-v2`, 384 dim) → Supabase pgvector.
- **Query** (runtime): Streamlit chat → LangChain tool-calling agent (Claude)
  → tools (`search_products`, `check_stock`, `get_price`) → pgvector hybrid
  search (RPC `match_products`) → recommendation + product cards.

## Setup

1. Python **3.12** is recommended (Python 3.14 currently has compatibility
   issues with pydantic/langchain).

   ```bash
   python3.12 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in credentials:
   - `ANTHROPIC_API_KEY` — from https://console.anthropic.com/settings/keys
   - `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` — from Supabase project settings
     (use the **legacy service_role key**, not the newer `sb_secret_...` key,
     for compatibility with `supabase-py`)
   - `SHOPIFY_STORE_DOMAIN` / `SHOPIFY_ADMIN_TOKEN` — an Admin API access
     token with `read_products` + `read_inventory` scopes. For apps created
     through Shopify's newer **Dev Dashboard** (not the legacy custom app
     flow), the token isn't shown directly in the UI — you need to complete
     an OAuth code exchange manually (open the authorize URL, approve, copy
     the `code` from the redirect URL, exchange it for a token via
     `POST /admin/oauth/access_token`).

3. Run the SQL schema (`sql/001_init.sql`) in the Supabase SQL Editor (enables
   the pgvector extension, creates the `products` table + `match_products`
   RPC).

4. Ingest the catalog:

   ```bash
   .venv/bin/python -m src.ingest
   ```

5. Run the app:

   ```bash
   .venv/bin/streamlit run app.py
   ```

   Change the port if `8501` is already taken by another process:
   `--server.port 8502`.

## Switching the Claude model

Change `ANTHROPIC_MODEL` in `.env` (e.g. `claude-opus-4-8`, `claude-sonnet-4-6`).
Restart Streamlit after changing it.

## Re-syncing the catalog

Re-run `python -m src.ingest` whenever products change in Shopify — the
pipeline is idempotent (upserts by product id).

## Tests

```bash
.venv/bin/python -m pytest tests/ -v
```

## Retrieval eval

`scripts/eval_retrieval.py` is a small "eval-lite" for the hybrid search
retriever: a golden set of ~10 queries with an expected product or category,
checked against the current live catalog.

```bash
.venv/bin/python scripts/eval_retrieval.py
```

Reports a hit-rate@3 and flags any query whose expected result didn't
appear in the top 3. Useful as a quick regression check after changing the
embedding model, prompt, or retriever filters.
