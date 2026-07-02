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
