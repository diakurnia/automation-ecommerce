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
