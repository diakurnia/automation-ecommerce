# AI Shopping Assistant — Design Spec

**Date:** 2026-07-02
**Status:** Approved (design), pending implementation plan
**Owner:** Dia (freelance automation specialist)

## Tujuan & Konteks

Membangun **AI Shopping Assistant** untuk toko ecommerce: pembeli mengobrol
secara natural (mis. "cari kado ultah cowok umur 25 hobi kopi, budget 500rb"),
sistem menemukan produk relevan lewat *semantic + hybrid search* lalu Claude
menyusun rekomendasi dengan alasan, mengingat preferensi & budget lintas pesan,
dan bisa mengecek stok/harga live.

Prioritas: **belajar vector database + LangChain secara mendalam**, hasil cukup
kuat untuk portfolio, dengan jalur komersil yang jelas untuk dikembangkan nanti
(dijual per-klien atau jadi produk).

## Keputusan Teknologi

| Komponen | Pilihan | Alasan |
|---|---|---|
| Bahasa | Python | Natural untuk LangChain + embedding lokal |
| Orkestrasi | LangChain | Merangkai RAG, agent, tools, memory |
| Chat/reasoning | Claude API (Anthropic) | Kualitas penalaran & bahasa Indonesia bagus |
| Embedding | Open-source lokal — BGE-small / sentence-transformers (384 dim) | Gratis, tanpa API key tambahan, memahami cara kerja embedding; Claude tidak menyediakan embedding |
| Vector DB | Supabase Postgres + pgvector | Satu DB untuk data produk + vektor; sudah tersedia; belajar SQL + vector |
| Sumber data | Shopify API (store nyata / preview) | Paling komersil & impresif untuk portfolio |
| UI | Streamlit chat app | Cepat dgn Python murni, demo portfolio interaktif, deploy gratis |

## Arsitektur

Dua jalur terpisah:

- **Ingest (batch, berkala):** Shopify → normalisasi → susun teks embedding →
  embed lokal → `upsert` ke pgvector. Idempotent, bisa dijalankan ulang untuk sinkron.
- **Query (runtime):** pembeli ngobrol di Streamlit → LangChain agent (Claude)
  memutuskan memanggil tool → `search_products` (embed query → hybrid search
  pgvector) mengembalikan top-k → Claude boleh memanggil `check_stock`/`get_price`
  → menyusun rekomendasi → streaming ke UI + kartu produk. Memory menyimpan
  preferensi & budget lintas pesan.

## Komponen (tiap unit satu tanggung jawab)

1. **Ingestion pipeline** (`ingest/`) — tarik produk Shopify, normalisasi, susun
   teks untuk embedding, embed, `upsert` ke pgvector. Idempotent (upsert by
   product id). Tahan gagal per-produk.
2. **Embedding module** (`embeddings.py`) — bungkus sentence-transformers; satu
   fungsi `embed(texts) -> vectors`. Dimensi tetap (384). Dipakai bersama oleh
   ingest & query agar konsisten.
3. **Retriever / hybrid search** (`retriever.py`) — cari di pgvector: kemiripan
   vektor + filter SQL (rentang harga, kategori, stok). Lewat SQL RPC
   `match_products(...)`.
4. **Shopify tools** (`tools/shopify.py`) — tool live untuk agent: `check_stock`,
   `get_price`/varian.
5. **Agent** (`agent.py`) — LangChain agent pakai Claude, tools
   `[search_products, check_stock, get_price]` + memory. System prompt:
   asisten belanja yang merekomendasikan dengan alasan, menghormati budget &
   preferensi.
6. **UI** (`app.py`) — Streamlit chat; session state simpan memory; render kartu
   produk (gambar, nama, harga, link); streaming jawaban.
7. **Config** (`config.py` / `.env`) — semua key & setting via env / Streamlit
   secrets; tidak pernah hardcode.

## Data Model (Supabase)

Tabel `products`:

| Kolom | Tipe |
|---|---|
| `id` | text (Shopify GID) — primary key |
| `title` | text |
| `description` | text |
| `product_type` | text |
| `tags` | text[] |
| `price` | numeric |
| `currency` | text |
| `inventory_qty` | integer |
| `in_stock` | boolean |
| `image_url` | text |
| `url` | text |
| `embedding` | vector(384) |
| `updated_at` | timestamptz |

- Index HNSW pada `embedding`.
- RPC `match_products(query_embedding, match_count, min_price, max_price, category, only_in_stock)`
  → mengembalikan baris ter-ranking (similarity + filter metadata).

## Build Order (bertahap; tiap fase menghasilkan sesuatu yang bisa dites)

- **Fase 0** — Setup project, env, aktifkan ekstensi pgvector, siapkan store
  Shopify + produk (jika belum ada, buat store preview berisi produk contoh).
- **Fase 1** — Ingestion + embedding → produk masuk pgvector. Verifikasi lewat
  query similarity mentah.
- **Fase 2** — Retriever hybrid search (RPC). Tes via script: query → produk
  relevan + filter harga/stok berfungsi.
- **Fase 3** — **RAG MVP**: Claude menyusun rekomendasi dari hasil retrieval +
  Streamlit chat dasar.
- **Fase 4** — **Agent + tools**: ubah jadi LangChain agent, `search_products`
  jadi tool + tool Shopify live.
- **Fase 5** — **Memory**: multi-turn, ingat budget & preferensi lintas pesan.
- **Fase 6** — Poles UI (kartu produk, streaming) + deploy ke Streamlit Cloud.

## Error Handling

- Ingest tahan gagal per-produk (retry + skip + log, tidak abort semua run).
- Teks kosong/kepanjangan → truncate sebelum embed.
- Retrieval kosong → agent menjawab santun ("belum ada yang cocok, coba
  longgarkan budget/ubah kata kunci").
- Error Claude/timeout → fallback ramah di UI.
- Error tool (mis. cek stok gagal) → agent tetap jalan (degrade gracefully).
- Secrets selalu via env / Streamlit secrets, tidak pernah hardcode.

## Testing

- **Unit:** konsistensi dimensi embedding; logika filter retriever (harga/stok);
  parsing Shopify tool (mock API).
- **Integration:** end-to-end query → rekomendasi dengan katalog kecil ter-seed.
- **Eval-lite:** ~5-10 query contoh dengan kategori yang diharapkan, untuk cek
  kualitas retrieval secara cepat.

## Catatan Komersil (untuk nanti, bukan scope sekarang)

Retriever + agent + tools berada di balik interface bersih, sehingga:

- Ganti **Streamlit → FastAPI + widget embeddable** = ganti lapisan UI saja.
- Ganti **store contoh → store klien** = ganti config/adapter Shopify.

Ini jalur untuk mengkomersilkan (jual per-klien atau produk SaaS).

## Non-Goals (YAGNI untuk versi ini)

- Multi-tenant / manajemen banyak store sekaligus.
- Pembayaran / checkout dalam chat.
- Rekomendasi berbasis riwayat pembelian personal per-user.
- Widget embeddable produksi (disiapkan jalurnya, tidak dibangun sekarang).
