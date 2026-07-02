# AI Shopping Assistant

Asisten belanja berbasis LangChain + Claude + Supabase pgvector + Shopify.

Pembeli mengobrol secara natural, agent mencari produk lewat hybrid semantic
search (pgvector), mengecek stok/harga live ke Shopify, lalu menyusun
rekomendasi dengan alasan — mengingat preferensi & budget lintas pesan.

## Arsitektur

- **Ingest** (batch): Shopify Admin API → embedding lokal multilingual
  (`paraphrase-multilingual-MiniLM-L12-v2`, 384 dim) → Supabase pgvector.
- **Query** (runtime): Streamlit chat → LangChain tool-calling agent (Claude)
  → tools (`search_products`, `check_stock`, `get_price`) → pgvector hybrid
  search (RPC `match_products`) → rekomendasi + kartu produk.

## Setup

1. Python **3.12** direkomendasikan (Python 3.14 punya masalah kompatibilitas
   dengan pydantic/langchain saat ini).

   ```bash
   python3.12 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

2. Salin `.env.example` ke `.env`, isi kredensial:
   - `ANTHROPIC_API_KEY` — dari https://console.anthropic.com/settings/keys
   - `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` — dari Supabase project settings
     (pakai **legacy service_role key**, bukan `sb_secret_...` yang baru,
     untuk kompatibilitas dengan `supabase-py`)
   - `SHOPIFY_STORE_DOMAIN` / `SHOPIFY_ADMIN_TOKEN` — Admin API access token
     dengan scope `read_products` + `read_inventory`. Untuk app yang dibuat
     lewat Shopify **Dev Dashboard** baru (bukan custom app legacy), token
     tidak ditampilkan langsung di UI — perlu OAuth code exchange manual
     (buka authorize URL, approve, tukar `code` jadi token via
     `POST /admin/oauth/access_token`).

3. Jalankan schema SQL (`sql/001_init.sql`) di Supabase SQL Editor (aktifkan
   ekstensi pgvector, buat tabel `products` + RPC `match_products`).

4. Ingest katalog:

   ```bash
   .venv/bin/python -m src.ingest
   ```

5. Jalankan app:

   ```bash
   .venv/bin/streamlit run app.py
   ```

   Ganti port kalau `8501` sudah dipakai proses lain:
   `--server.port 8502`.

## Ganti model Claude

Ubah `ANTHROPIC_MODEL` di `.env` (mis. `claude-opus-4-8`, `claude-sonnet-4-6`).
Restart Streamlit setelah mengubahnya.

## Sinkronisasi ulang katalog

Jalankan ulang `python -m src.ingest` kapan pun produk di Shopify berubah —
pipeline idempotent (upsert berdasarkan product id).

## Test

```bash
.venv/bin/python -m pytest tests/ -v
```
