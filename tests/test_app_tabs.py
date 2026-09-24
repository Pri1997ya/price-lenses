from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_renders_history_and_market_tabs(monkeypatch, tmp_path):
    # Initial rendering does not connect or call providers; it only validates config.
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/pricelens")
    # Keep the policy tab offline and away from the real data/chroma folder.
    monkeypatch.setenv("POLICY_EMBEDDINGS", "hash")
    monkeypatch.setenv("POLICY_CHROMA_DIR", str(tmp_path / "chroma"))

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    # A clean CI environment may need extra time for first-time LangGraph imports.
    app = AppTest.from_file(app_path, default_timeout=60).run()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "📈 History & Timing",
        "🛒 Market Investigator",
        "🛡️ Policy & Seller Check",
    ]
    assert [title.value for title in app.title] == [
        "🔍 PriceLens: Autonomous Deal Advisor"
    ]
