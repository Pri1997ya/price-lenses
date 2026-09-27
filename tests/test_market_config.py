import pytest

from tools.market_config import ConfigurationError, MarketSettings


def _base_environment(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/pricelens")
    monkeypatch.setenv("MARKET_COUNTRY", "in")
    monkeypatch.setenv("MARKET_GOOGLE_DOMAIN", "google.co.in")
    monkeypatch.setenv("MARKET_AMAZON_DOMAIN", "amazon.in")


def test_settings_default_to_india_and_search_api(monkeypatch):
    _base_environment(monkeypatch)
    monkeypatch.delenv("SERPAPI_ENRICH_AMAZON", raising=False)

    settings = MarketSettings.from_env()

    assert settings.country == "in"
    assert settings.google_domain == "google.co.in"
    assert settings.amazon_domain == "amazon.in"
    assert settings.serpapi_enrich_amazon is False
    assert settings.market_agent_llm_enabled is True
    assert settings.market_provider_policy == "api_first"


def test_settings_accept_existing_pl_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("PL_DATABASE_URL", "postgresql://localhost/pricelens")
    monkeypatch.setenv("MARKET_COUNTRY", "in")
    monkeypatch.setenv("MARKET_GOOGLE_DOMAIN", "google.co.in")
    monkeypatch.setenv("MARKET_AMAZON_DOMAIN", "amazon.in")

    settings = MarketSettings.from_env()

    assert settings.database_url == "postgresql://localhost/pricelens"


def test_settings_reject_non_india_market(monkeypatch):
    _base_environment(monkeypatch)
    monkeypatch.setenv("MARKET_COUNTRY", "us")

    with pytest.raises(ConfigurationError, match="India only"):
        MarketSettings.from_env()


def test_settings_validate_apify_json(monkeypatch):
    _base_environment(monkeypatch)
    monkeypatch.setenv("APIFY_EXTRA_INPUT_JSON", "not-json")

    with pytest.raises(ConfigurationError, match="valid JSON"):
        MarketSettings.from_env()


def test_settings_reject_unknown_market_provider_policy(monkeypatch):
    _base_environment(monkeypatch)
    monkeypatch.setenv("MARKET_PROVIDER_POLICY", "cache_magic")

    with pytest.raises(ConfigurationError, match="MARKET_PROVIDER_POLICY"):
        MarketSettings.from_env()
