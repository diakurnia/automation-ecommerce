from src import shopify_client, normalize, embeddings, db


def run() -> int:
    """Tarik produk Shopify, embed, upsert ke pgvector. Kembalikan jumlah produk."""
    currency = shopify_client.fetch_shop_currency()
    raw = shopify_client.fetch_products()
    records, texts = [], []
    for r in raw:
        try:
            rec = normalize.normalize_product(r, currency=currency)
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
