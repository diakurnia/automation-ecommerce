from unittest.mock import patch
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


def test_get_product_parses_variant():
    resp = {"data": {"product": {
        "id": "gid://shopify/Product/1", "title": "Kopi",
        "variants": {"edges": [{"node": {"price": "85000", "inventoryQuantity": 5}}]}}}}
    with patch("src.shopify_client._post_graphql", return_value=resp):
        p = shopify_client.get_product("gid://shopify/Product/1")
    assert p["price"] == 85000.0
    assert p["inventory_qty"] == 5
