#!/usr/bin/env python3
"""Apply versioned PriceLens PostgreSQL migrations to ``DATABASE_URL``."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv


MIGRATIONS = Path(__file__).with_name("migrations")


def _checksum(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def main() -> None:
    load_dotenv()
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise SystemExit("DATABASE_URL is required")

    files = sorted(MIGRATIONS.glob("*.sql"))
    if not files:
        raise SystemExit(f"No SQL migrations found in {MIGRATIONS}")

    connection = psycopg2.connect(database_url, connect_timeout=10)
    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version VARCHAR(255) PRIMARY KEY,
                        checksum VARCHAR(64) NOT NULL,
                        applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtext('price_lenses_migrations'))"
                )
                cursor.execute("SELECT to_regclass('public.products')")
                if cursor.fetchone()[0] is None:
                    raise RuntimeError(
                        "Base schema is missing. Run python scripts/db/init_db.py first."
                    )

                for path in files:
                    contents = path.read_bytes()
                    checksum = _checksum(contents)
                    cursor.execute(
                        "SELECT checksum FROM schema_migrations WHERE version=%s",
                        (path.name,),
                    )
                    existing = cursor.fetchone()
                    if existing:
                        if existing[0] != checksum:
                            raise RuntimeError(
                                f"Applied migration was modified: {path.name}"
                            )
                        print(f"Already applied: {path.name}")
                        continue

                    cursor.execute(contents.decode("utf-8"))
                    cursor.execute(
                        "INSERT INTO schema_migrations(version, checksum) VALUES (%s, %s)",
                        (path.name, checksum),
                    )
                    print(f"Applied: {path.name}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
