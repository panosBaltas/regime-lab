import json

import pandas as pd
import pytest

from regime_lab import capture
from regime_lab.execution import ExecutionEngine, ExecutionSpec


@pytest.fixture
def public_api(monkeypatch):
    now = "2026-10-20T13:00:00+00:00"
    engine = ExecutionEngine(ExecutionSpec())
    model = {"mode": "prospective_paper", "created_at": "2026-10-20T12:00:00+00:00",
             "execution": {"contract_faces": [100., 10.]}}
    monkeypatch.setattr(capture, "replay_journal", lambda *a: (model, [], engine))
    monkeypatch.setattr(capture, "now_utc", lambda: now)
    event_ms = int(pd.Timestamp("2026-10-20T12:30:00Z").timestamp()*1000)
    settings = {"multiplier": 100., "missing_mark": False, "duplicate": False}
    class Response:
        def __init__(self, url, body):
            self.url, self.body = url, body
        def raise_for_status(self):
            return None
        def json(self):
            return self.body
    def get(url, params=None, **kwargs):
        if "exchangeInfo" in url:
            body = {"symbols": [{"symbol": "BTCUSD_PERP", "contractSize": settings["multiplier"], "contractStatus": "TRADING"},
                                 {"symbol": "ETHUSD_PERP", "contractSize": 10., "contractStatus": "TRADING"}]}
        elif "bookTicker" in url:
            body = {"bidPrice": "99.9", "askPrice": "100.1"}
            if "dapi" in url:
                body = [body]
        elif "premiumIndex" in url:
            body = [{"markPrice": "100"}]
        elif "fundingRate" in url:
            row = {"fundingTime": event_ms, "fundingRate": ".001" if params["symbol"].startswith("BTC") else "-.001"}
            if not settings["missing_mark"]:
                row["markPrice"] = "100"
            body = [row, row.copy()] if settings["duplicate"] else [row]
        elif "markPriceKlines" in url:
            body = [[event_ms, "100"]]
        else:
            raise AssertionError(url)
        return Response(url+str(params), body)
    monkeypatch.setattr(capture.requests, "get", get)
    return settings


def test_public_capture_preserves_actual_funding_and_raw_sources(public_api, tmp_path):
    result = capture.capture_public("unused", tmp_path/"snapshot.json", quote_usd=1.)
    value = json.loads((tmp_path/"snapshot.json").read_text())
    assert result["funding_events"] == 2
    assert value["funding_coin_per_face"] == pytest.approx([.00001, -.00001])
    assert value["funding_from"].startswith("2026-10-20T12:00")
    assert len(value["raw"]) >= 9
    assert value["fx_source"] == "explicit operator FX assumption"


def test_missing_settlement_mark_uses_its_own_historical_bar(public_api, tmp_path):
    public_api["missing_mark"] = True
    capture.capture_public("unused", tmp_path/"snapshot.json", quote_usd=1.)
    value = json.loads((tmp_path/"snapshot.json").read_text())
    assert value["funding_coin_per_face"][0] == pytest.approx(.00001)
    assert any("markPriceKlines" in key for key in value["raw"])


def test_contract_changes_or_duplicate_funding_stop_capture_without_writing(public_api, tmp_path):
    public_api["multiplier"] = 1000.
    with pytest.raises(ValueError, match="multiplier/status"):
        capture.capture_public("unused", tmp_path/"snapshot.json", quote_usd=1.)
    assert not (tmp_path/"snapshot.json").exists()
    public_api.update(multiplier=100., duplicate=True)
    with pytest.raises(ValueError, match="Duplicate funding"):
        capture.capture_public("unused", tmp_path/"snapshot.json", quote_usd=1.)
    assert not (tmp_path/"snapshot.json").exists()
