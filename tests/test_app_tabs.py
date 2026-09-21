from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_renders_history_and_market_tabs(monkeypatch):
    # Initial rendering does not connect or call providers; it only validates config.
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/pricelens")

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=20).run()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "📈 History & Timing",
        "🛒 Market Investigator",
    ]
    assert [title.value for title in app.title] == [
        "🔍 PriceLens: Autonomous Deal Advisor"
    ]
