from tools.market_models import Offer
from tools.market_service import MarketInvestigatorService


class FakeDatabase:
    def __init__(self):
        self.runs = []
        self.finished = []
        self.rows = []

    def known_products(self):
        return {}

    def start_run(self, query, provider):
        run_id = f"run-{provider}"
        self.runs.append((run_id, query, provider))
        return run_id

    def upsert_product(self, resolved_id, offer):
        return resolved_id.removeprefix("asin:")

    def insert_offers(self, run_id, rows):
        rows = list(rows)
        self.rows.extend((run_id, *row) for row in rows)
        return len(rows)

    def finish_run(self, run_id, count, *, status, error=None):
        self.finished.append((run_id, count, status, error))


class StubProvider:
    name = "serpapi"
    warnings = []

    def search(self, query, limit=20):
        return [
            Offer(
                "serpapi",
                "amazon.in",
                "Apple iPhone 16 128GB",
                asin="B0EXAMPLE1",
                price=69900,
            )
        ]


class FailingProvider:
    name = "apify"
    warnings = []

    def search(self, query, limit=20):
        raise RuntimeError("actor unavailable")


def test_service_persists_success_and_isolates_provider_failure():
    database = FakeDatabase()
    service = MarketInvestigatorService(database, [StubProvider(), FailingProvider()])

    results = service.search("iPhone 16", limit=5)

    assert [(result.provider, result.status, result.count) for result in results] == [
        ("serpapi", "ok", 1),
        ("apify", "error", 0),
    ]
    assert database.rows[0][1] == "B0EXAMPLE1"
    assert database.finished[1][3] == "actor unavailable"


def test_service_rejects_empty_query_and_invalid_limit():
    service = MarketInvestigatorService(FakeDatabase(), [StubProvider()])

    for query, limit in (("", 10), ("phone", 0), ("phone", 101)):
        try:
            service.search(query, limit)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid search input was accepted")


def test_validation_keeps_phone_and_rejects_accessories_invalid_rows():
    offers = [
        Offer("serpapi", "amazon.in", "S2 5G 8GB RAM 128GB Storage", price=39999, currency="INR"),
        Offer("serpapi", "flipkart.com", "Flip Cover For Vivo S2", price=300, currency="INR"),
        Offer("apify", "croma.com", "Croma", price=None, currency="INR"),
        Offer("serpapi", "example.com", "S2 5G 8GB RAM 128GB Storage", price=39000, currency="INR"),
    ]

    accepted, warnings = MarketInvestigatorService._validated_offers("vivo s2", offers)

    assert [offer.price for offer in accepted] == [39999]
    assert any("irrelevant product or accessory" in warning for warning in warnings)
    assert any("missing price" in warning for warning in warnings)
    assert any("unsupported retailer" in warning for warning in warnings)
