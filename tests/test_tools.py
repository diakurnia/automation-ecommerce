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
               return_value={"title": "Kopi", "inventory_qty": 5}):
        out = tools.check_stock_tool.invoke({"product_id": "gid://shopify/Product/1"})
    assert "5" in out
