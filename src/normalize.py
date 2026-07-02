def normalize_product(raw: dict, currency: str = "IDR") -> dict:
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
        "currency": currency,
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
