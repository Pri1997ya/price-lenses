"""PostgreSQL repository for live market offers.

This repository extends the schema inherited from ``main``. It uses the existing
``products.canonical_id`` catalog and stores append-only provider responses in
market-specific tables so historical price analysis remains unchanged.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Iterable

import psycopg2
from psycopg2.extras import Json, RealDictCursor

from .market_models import Offer
from .market_normalize import normalize_title


class MarketSchemaError(RuntimeError):
    """Raised when the market database migration has not been applied."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _native(value):
    return float(value) if isinstance(value, Decimal) else value


def _canonical_id(resolved_id: str, offer: Offer) -> str:
    if offer.asin:
        return offer.asin.upper()
    if resolved_id.lower().startswith("asin:"):
        return resolved_id.split(":", 1)[1].upper()
    return resolved_id[:50]


class MarketDatabase:
    def __init__(self, database_url: str, *, verify_schema: bool = True):
        if not database_url:
            raise ValueError("database_url is required")
        self.database_url = database_url
        self.connection = psycopg2.connect(database_url, connect_timeout=10)
        if verify_schema:
            self._verify_schema()

    def __enter__(self) -> "MarketDatabase":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def _verify_schema(self) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT to_regclass('public.market_search_runs')")
            ready = cursor.fetchone()[0]
        self.connection.rollback()
        if ready is None:
            self.close()
            raise MarketSchemaError(
                "Market database tables are missing. Run: python scripts/db/migrate.py"
            )

    def known_products(self) -> dict[str, str]:
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT canonical_id, title FROM products")
            rows = cursor.fetchall()
        self.connection.rollback()
        return {product_id: normalize_title(title or "") for product_id, title in rows}

    def start_run(self, query: str, provider: str) -> str:
        run_id = uuid.uuid4().hex
        with self.connection:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO market_search_runs
                        (run_id, query, provider, status, result_count, started_at)
                    VALUES (%s, %s, %s, 'running', 0, %s)
                    """,
                    (run_id, query, provider, _now()),
                )
        return run_id

    def finish_run(
        self,
        run_id: str,
        count: int,
        *,
        status: str,
        error: str | None = None,
    ) -> None:
        with self.connection:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE market_search_runs
                    SET status=%s, error=%s, result_count=%s, finished_at=%s
                    WHERE run_id=%s
                    """,
                    (status, error, count, _now(), run_id),
                )

    def upsert_product(self, resolved_id: str, offer: Offer) -> str:
        canonical_id = _canonical_id(resolved_id, offer)
        with self.connection:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO products (
                        canonical_id, title, brand, current_price, customer_rating,
                        review_count, seller_name, created_at
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (canonical_id) DO UPDATE SET
                        brand=COALESCE(products.brand, EXCLUDED.brand),
                        current_price=COALESCE(products.current_price, EXCLUDED.current_price),
                        customer_rating=COALESCE(products.customer_rating, EXCLUDED.customer_rating),
                        review_count=COALESCE(products.review_count, EXCLUDED.review_count),
                        seller_name=COALESCE(products.seller_name, EXCLUDED.seller_name)
                    """,
                    (
                        canonical_id,
                        offer.title or canonical_id,
                        offer.brand,
                        offer.price,
                        offer.rating,
                        offer.review_count,
                        offer.seller_name,
                        _now(),
                    ),
                )
        return canonical_id

    def insert_offers(
        self,
        run_id: str,
        rows: Iterable[tuple[str, Offer]],
    ) -> int:
        count = 0
        with self.connection:
            with self.connection.cursor() as cursor:
                for canonical_id, offer in rows:
                    offer_id = uuid.uuid4().hex
                    cursor.execute(
                        """
                        INSERT INTO market_offers (
                            offer_id, run_id, canonical_id, provider, marketplace,
                            external_id, title, url, price, currency, original_price,
                            rating, review_count, seller_name, seller_rating,
                            availability, shipping, offers_json, raw_json, fetched_at
                        ) VALUES (
                            %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                            %s,%s,%s,%s,%s,%s,%s,%s,%s,%s
                        )
                        """,
                        (
                            offer_id,
                            run_id,
                            canonical_id,
                            offer.provider,
                            offer.marketplace,
                            offer.external_id,
                            offer.title,
                            offer.url,
                            offer.price,
                            offer.currency or "INR",
                            offer.original_price,
                            offer.rating,
                            offer.review_count,
                            offer.seller_name,
                            offer.seller_rating,
                            offer.availability,
                            offer.shipping,
                            Json(offer.offers),
                            Json(offer.raw, dumps=lambda value: json.dumps(value, default=str)),
                            _now(),
                        ),
                    )
                    if any(
                        value is not None
                        for value in (
                            offer.seller_id,
                            offer.price_with_offers,
                            offer.is_assured,
                            offer.cod_available,
                            offer.no_cost_emi,
                            offer.return_policy,
                            offer.delivery_by,
                            offer.warranty,
                            offer.condition,
                        )
                    ):
                        cursor.execute(
                            """
                            INSERT INTO market_offer_details (
                                offer_id, seller_id, price_with_offers, is_assured,
                                cod_available, no_cost_emi, return_policy, delivery_by,
                                warranty, item_condition
                            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            """,
                            (
                                offer_id,
                                offer.seller_id,
                                offer.price_with_offers,
                                offer.is_assured,
                                offer.cod_available,
                                offer.no_cost_emi,
                                offer.return_policy,
                                offer.delivery_by,
                                offer.warranty,
                                offer.condition,
                            ),
                        )
                    for promotion in offer.promos:
                        cursor.execute(
                            """
                            INSERT INTO market_offer_promotions (
                                promotion_id, offer_id, promotion_type, bank,
                                card_type, description, amount, percent, is_emi, source
                            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            """,
                            (
                                uuid.uuid4().hex,
                                offer_id,
                                promotion.promo_type,
                                promotion.bank,
                                promotion.card_type,
                                promotion.description,
                                promotion.amount,
                                promotion.percent,
                                promotion.is_emi,
                                promotion.source,
                            ),
                        )
                    count += 1
        return count

    def _dict_rows(self, statement: str, params: tuple = ()) -> list[dict]:
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(statement, params)
            rows = cursor.fetchall()
        self.connection.rollback()
        return [{key: _native(value) for key, value in row.items()} for row in rows]

    def offers_for_query(self, query: str, limit: int = 100) -> list[dict]:
        return self._dict_rows(
            """
            SELECT o.*, p.title AS product_title,
                   d.seller_id, d.price_with_offers, d.is_assured,
                   d.cod_available, d.no_cost_emi, d.return_policy,
                   d.delivery_by, d.warranty, d.item_condition
            FROM latest_market_offers o
            JOIN products p ON p.canonical_id = o.canonical_id
            LEFT JOIN market_offer_details d ON d.offer_id = o.offer_id
            WHERE o.canonical_id IN (
                SELECT DISTINCT found.canonical_id
                FROM market_offers found
                JOIN market_search_runs run ON run.run_id = found.run_id
                WHERE lower(run.query) = lower(%s)
            )
            ORDER BY COALESCE(d.price_with_offers, o.price) NULLS LAST
            LIMIT %s
            """,
            (query, limit),
        )

    def promotions_for_query(self, query: str) -> list[dict]:
        return self._dict_rows(
            """
            SELECT o.offer_id, o.canonical_id, o.external_id, o.url, o.provider,
                   o.marketplace, o.seller_name, o.price, p.title AS product_title,
                   promo.promotion_type, promo.bank, promo.card_type,
                   promo.description, promo.amount, promo.percent,
                   promo.is_emi, promo.source
            FROM latest_market_offers o
            JOIN products p ON p.canonical_id = o.canonical_id
            JOIN market_offer_promotions promo ON promo.offer_id = o.offer_id
            WHERE o.canonical_id IN (
                SELECT DISTINCT found.canonical_id
                FROM market_offers found
                JOIN market_search_runs run ON run.run_id = found.run_id
                WHERE lower(run.query) = lower(%s)
            )
            ORDER BY o.price NULLS LAST, promo.promotion_type
            """,
            (query,),
        )

    def recent_runs(self, limit: int = 25) -> list[dict]:
        return self._dict_rows(
            """
            SELECT run_id, query, provider, status, error, result_count,
                   started_at, finished_at
            FROM market_search_runs
            ORDER BY started_at DESC
            LIMIT %s
            """,
            (limit,),
        )
