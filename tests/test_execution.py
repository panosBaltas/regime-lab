from dataclasses import replace
import copy

import numpy as np
import pandas as pd
import pytest

from regime_lab.execution import Account, ExecutionEngine, ExecutionSpec
from regime_lab.project import exchange_time


def observation(time, price=100., **extra):
    return {"time": time, "prices": {"spot": [price, price], "future": [price, price], "mark": [price, price],
            "quote_usd": 1., "etf": {"SPY": price, "BIL": 100.}}, "crypto_trade": True, **extra}


def funded_inverse():
    spec = ExecutionSpec(capital=100., spot_cost=0, future_cost=0, etf_cost=0, transfer_cost=0)
    engine = ExecutionEngine(spec)
    engine.advance(observation("2024-01-01T00:00:00Z"))
    engine.account = Account(bank=0., wallets=[1., 0.], faces=[100., 0.], entries=[1., 0.])
    return engine


def test_native_dollar_pnl_and_old_contract_funding_reconcile_independently():
    engine = funded_inverse()
    row = engine.advance(observation("2024-01-01T01:00:00Z", 200., funding_coin_per_face=[.001, 0.]))
    assert row["pnl_spot"] == pytest.approx(100.)
    assert row["pnl_inverse"] == pytest.approx(-100.)
    assert row["pnl_funding"] == pytest.approx(20.)
    assert row["nav"] == pytest.approx(120.)
    assert row["reconciliation_error"] == pytest.approx(0., abs=1e-12)
    paid = funded_inverse().advance(observation("2024-01-01T01:00:00Z", funding_coin_per_face=[-.001, 0.]))
    assert paid["nav"] == pytest.approx(90.)


def test_business_day_transfers_and_signal_delay_cannot_earn_earlier_funding():
    engine = ExecutionEngine()
    engine.advance(observation("2024-01-05T00:00:00Z"))  # Friday
    engine.plan({"time": "2024-01-05T00:00:00Z", "latest_input_time": "2024-01-04T00:00:00Z", "allocation": [1., 0.]})
    early = engine.advance(observation("2024-01-05T00:30:00Z", funding_coin_per_face=[1., 1.]))
    assert early["bank"] == 100_000 and early["pnl_funding"] == 0
    sent = engine.advance(observation("2024-01-05T01:00:00Z"))
    assert sent["transit"] == pytest.approx(99_900.)
    assert engine.account.transfers[0]["due"].startswith("2024-01-09T01")
    weekend = engine.advance(observation("2024-01-07T01:00:00Z", funding_coin_per_face=[1., 1.]))
    assert weekend["crypto"] == 0 and weekend["pnl_funding"] == 0
    arrival = engine.advance(observation("2024-01-09T01:00:00Z", funding_coin_per_face=[1., 1.]))
    assert arrival["pnl_funding"] == 0  # old position at settlement, then fills
    assert all(face/lot == pytest.approx(round(face/lot)) for face, lot in zip(engine.account.faces, engine.spec.contract_faces))
    assert engine.account.crypto_quote >= 0
    assert arrival["nav"] < 99_900  # actual entry fees, independently of transfer fee


def test_stale_etf_marks_never_fill_broker_orders_and_bid_ask_costs_are_real():
    engine = ExecutionEngine(ExecutionSpec(capital=1000.))
    engine.advance(observation("2024-01-01T00:00:00Z"))
    engine.account = Account(bank=0, broker_cash=1000.)
    engine.plan({"time": "2024-01-01T00:00:00Z", "latest_input_time": "2023-12-31T00:00:00Z", "macro": {"SPY": 1.}})
    engine.advance(observation("2024-01-01T02:00:00Z"))
    assert engine.account.etfs == {}
    quote = observation("2024-01-02T14:30:00Z", etf_trade=True)
    quote["prices"].update(etf_bid={"SPY": 99., "BIL": 99.}, etf_ask={"SPY": 101., "BIL": 101.})
    row = engine.advance(quote)
    assert 9 < engine.account.etfs["SPY"] < 10
    assert engine.account.broker_cash >= -1e-8
    assert row["nav"] < 990
    assert row["pnl_execution"] < 0


def test_etf_dividend_is_a_receivable_and_split_preserves_nav():
    engine = ExecutionEngine(ExecutionSpec(capital=100.))
    engine.advance(observation("2024-01-01T00:00:00Z"))
    engine.account = Account(bank=0, etfs={"SPY": 1.})
    event = observation("2024-01-02T14:30:00Z", 49.)
    event["corporate_actions"] = [{"symbol": "SPY", "effective_at": "2024-01-02T14:30:00Z",
                                    "split_ratio": 2., "cash_dividend_per_unit": 1.}]
    row = engine.advance(event)
    assert row["nav"] == pytest.approx(100.)
    assert engine.account.etfs["SPY"] == 2
    assert engine.account.broker_cash == 0
    assert engine.account.broker_receivables == 2
    assert row["pnl_dividend"] == 2


def test_fills_future_prices_and_source_timing_are_causal():
    first = funded_inverse()
    second = ExecutionEngine(first.spec, first.state())
    same = observation("2024-01-01T01:00:00Z", 110.)
    assert first.advance(same) == second.advance(copy.deepcopy(same))
    saved = first.state()
    first.advance(observation("2024-01-02T00:00:00Z", 50.))
    second.advance(observation("2024-01-02T00:00:00Z", 200.))
    assert saved["account"]["faces"] == [100., 0.]
    with pytest.raises(ValueError, match="future"):
        first.plan({"time": "2024-01-03", "latest_input_time": "2024-01-04", "allocation": [.5, .5]})


def test_maintenance_failure_is_explicit_and_us_open_observes_dst():
    engine = funded_inverse()
    with pytest.raises(ValueError, match="Maintenance breach"):
        engine.advance(observation("2024-01-01T01:00:00Z", funding_coin_per_face=[-.0095, 0.]))
    assert exchange_time("2024-01-02", 9, 30).hour == 14
    assert exchange_time("2024-07-02", 9, 30).hour == 13


def test_monthly_sharpe_uses_complete_nav_boundaries_not_a_partial_final_month():
    from regime_lab.project import account_metrics, complete_monthly_excess
    dates = pd.date_range("2024-01-01", "2024-03-15", freq="D")
    nav = pd.Series({pd.Timestamp("2024-01-01"): 100., pd.Timestamp("2024-02-01"): 110.,
                     pd.Timestamp("2024-03-01"): 107.8, pd.Timestamp("2024-03-15"): 120.})
    ledger = pd.DataFrame({"nav": nav.reindex(dates).interpolate(), "benchmark": 1.})
    monthly = complete_monthly_excess(ledger)
    np.testing.assert_allclose(monthly, [.10, -.02], atol=1e-12)
    result = account_metrics(ledger)
    assert result["complete_months"] == 2
    assert result["days"] == len(dates)-1
    assert result["monthly_excess_sharpe"] == pytest.approx(np.mean([.10, -.02])/np.std([.10, -.02], ddof=1)*np.sqrt(12))
