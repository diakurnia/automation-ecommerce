import os
from src import config

def test_config_reads_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("SUPABASE_URL", "u")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "s")
    monkeypatch.setenv("TOP_K", "9")
    cfg = config.load()
    assert cfg.anthropic_api_key == "k"
    assert cfg.supabase_url == "u"
    assert cfg.top_k == 9
    assert cfg.embedding_dim == 384
