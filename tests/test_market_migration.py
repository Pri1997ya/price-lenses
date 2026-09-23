from pathlib import Path


MIGRATION = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "db"
    / "migrations"
    / "001_market_investigator.sql"
)


def test_market_migration_extends_existing_product_catalog():
    sql = MIGRATION.read_text().lower()

    assert "create table if not exists products" not in sql
    assert "references products(canonical_id)" in sql
    assert "create table if not exists market_search_runs" in sql
    assert "create table if not exists market_offers" in sql
    assert "create table if not exists market_offer_details" in sql
    assert "create table if not exists market_offer_promotions" in sql
    assert "create or replace view latest_market_offers" in sql
