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
