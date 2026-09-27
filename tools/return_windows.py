"""Reviewed return windows per retailer and product type (Agent 3).

``data/policies/return_windows.json`` holds the few rules buyers ask about most,
copied by hand from the retailer's policy page ("Flipkart, Mobiles: 7 days,
replacement only"). They give an exact answer where search over passages may
paraphrase or miss a table row; the policy RAG still handles everything else.

Every row cites the page and retrieval date it was copied from::

    {"retailer": "flipkart", "category": "mobile phones", "window_days": 7,
     "action": "replacement", "conditions": "Only if delivered defective or damaged",
     "source_id": "flipkart-return-policy",
     "source_url": "https://www.flipkart.com/pages/returnpolicy",
     "retrieved_at": "2026-09-27"}

``category`` is a product type from ``tools.eligibility_agent.CATEGORY_PATTERNS``
or ``"*"`` for a retailer-wide default.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from .policy_corpus import DEFAULT_POLICY_DIR, REGULATION, RETAILER_LABELS, CorpusError

DEFAULT_RETURN_WINDOWS_FILE = DEFAULT_POLICY_DIR / "return_windows.json"
ANY_CATEGORY = "*"
ACTIONS = {
    "return": "return for a refund",
    "replacement": "replacement only",
    "return_or_replacement": "return or replacement",
    "service_center": "service-centre repair or replacement only",
    "not_returnable": "not returnable",
}
REQUIRED_FIELDS = ("retailer", "category", "window_days", "action", "source_id", "source_url", "retrieved_at")


@dataclass(frozen=True)
class ReturnWindow:
    retailer: str
    category: str
    window_days: int
    action: str
    source_id: str
    source_url: str
    retrieved_at: str
    conditions: str = ""

    @property
    def summary(self) -> str:
        label = RETAILER_LABELS.get(self.retailer, self.retailer)
        what = "all products" if self.category == ANY_CATEGORY else self.category
        if self.action == "not_returnable":
            text = f"{label}, {what}: {ACTIONS[self.action]}"
        else:
            text = f"{label}, {what}: {self.window_days} days, {ACTIONS[self.action]}"
        return f"{text} ({self.conditions})" if self.conditions else text

    def to_dict(self) -> dict:
        return {**asdict(self), "summary": self.summary}


def _row(entry: dict, number: int, categories: set[str] | None) -> ReturnWindow:
    where = f"return_windows.json row {number}"
    missing = [key for key in REQUIRED_FIELDS if entry.get(key) in (None, "")]
    if missing:
        raise CorpusError(f"{where} is missing {', '.join(missing)}")
    retailer = entry["retailer"]
    if retailer not in RETAILER_LABELS or retailer == REGULATION:
        raise CorpusError(f"{where}: unknown retailer {retailer!r}")
    category = entry["category"]
    if categories is not None and category != ANY_CATEGORY and category not in categories:
        raise CorpusError(f"{where}: unknown category {category!r}")
    if entry["action"] not in ACTIONS:
        raise CorpusError(f"{where}: action must be one of {', '.join(sorted(ACTIONS))}")
    window = entry["window_days"]
    if not isinstance(window, int) or isinstance(window, bool) or window < 0:
        raise CorpusError(f"{where}: window_days must be a whole number of days")
    try:
        date.fromisoformat(entry["retrieved_at"])
    except (TypeError, ValueError):
        raise CorpusError(f"{where}: retrieved_at must be a YYYY-MM-DD date") from None
    return ReturnWindow(
        retailer=retailer,
        category=category,
        window_days=window,
        action=entry["action"],
        source_id=entry["source_id"],
        source_url=entry["source_url"],
        retrieved_at=entry["retrieved_at"],
        conditions=(entry.get("conditions") or "").strip(),
    )


def load_return_windows(
    path: Path = DEFAULT_RETURN_WINDOWS_FILE, categories: set[str] | None = None
) -> list[ReturnWindow]:
    """Load and validate the table. A missing file means no reviewed rows yet."""
    path = Path(path)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [_row(entry, number, categories) for number, entry in enumerate(data.get("rows", []), start=1)]
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row.retailer, row.category)
        if key in seen:
            raise CorpusError(f"return_windows.json has two rows for {row.retailer} / {row.category}")
        seen.add(key)
    return rows


def find_return_window(rows: list[ReturnWindow], retailer: str, category: str) -> ReturnWindow | None:
    """The row for this product type at this retailer, else the retailer-wide default."""
    exact = next((r for r in rows if r.retailer == retailer and r.category == category), None)
    return exact or next((r for r in rows if r.retailer == retailer and r.category == ANY_CATEGORY), None)
