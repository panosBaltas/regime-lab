from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from regime_lab.stock_data import choose_universe, synthetic_stock_panel, validate_stock_prices
from regime_lab.stock_features import StockConfig, build_stock_features, make_stock_panel
from regime_lab.stock_models import pooled_rows, stock_forecasts
from regime_lab.stock_portfolio import StockOptimizer, simulate_stock_portfolio, stock_weights
from regime_lab.stock_report import render_saved_stock_report
from regime_lab.stock_study import run_stock_study


@pytest.fixture(scope="module")
def research_fixture():
    prices, universe = synthetic_stock_panel(n_sessions=250, n_stocks=20)
    config = StockConfig(factor_lookback=63, n_components=2, min_stocks=12,
                         minimum_training_dates=90, training_window=126,
                         minimum_inner_dates=30, validation_block=21,
                         position_limit=0.1)
    panel = make_stock_panel(prices, universe)
    features = build_stock_features(panel, config)
    forecasts, decisions = stock_forecasts(panel, features, config)
    return prices, universe, config, panel, features, forecasts, decisions


def flat_panel():
    dates = pd.bdate_range("2021-01-07", periods=5)
    universe = pd.DataFrame({"symbol": ["A", "B"], "sector": ["X", "X"]})
    rows = []
    for t, date in enumerate(dates):
        for symbol in ("A", "B", "SPY", "BIL"):
            price = 100 * 1.001**t if symbol == "BIL" else 100
            rows.append(dict(date=date, symbol=symbol, open=price, close=price,
                             raw_open=price, raw_close=price, volume=1_000_000))
    return make_stock_panel(pd.DataFrame(rows), universe)


def test_universe_is_deterministic_balanced_and_issuer_unique():
    frame = pd.DataFrame({"symbol": [f"S{i}" for i in range(30)],
                          "sector": [f"Sector {i % 3}" for i in range(30)],
                          "issuer_id": range(30)})
    frame.loc[29, "issuer_id"] = 0
    first = choose_universe(frame, 12)
    second = choose_universe(frame.sample(frac=1, random_state=5), 12)
    pd.testing.assert_frame_equal(first, second)
    assert first["sector"].value_counts().tolist() == [4, 4, 4]
    assert first["issuer_id"].is_unique
    assert not first["symbol"].equals(choose_universe(frame, 12, seed=7)["symbol"])


def test_prices_reject_inconsistent_adjustments_and_duplicates(research_fixture):
    prices = research_fixture[0].copy()
    prices.loc[0, "open"] *= 1.1
    with pytest.raises(ValueError, match="adjustment factor"):
        validate_stock_prices(prices)
    with pytest.raises(ValueError, match="duplicate"):
        validate_stock_prices(pd.concat([prices.iloc[:1], prices.iloc[:1]]))


def test_short_histories_and_failed_symbols_are_not_filled(research_fixture):
    prices, universe, config, *_ = research_fixture
    debut = pd.Timestamp(prices["date"].unique()[100])
    prices = prices.loc[~((prices["symbol"] == "S000") & (prices["date"] < debut))]
    universe = pd.concat([universe, pd.DataFrame({"symbol": ["MISSING"], "sector": ["Sector 0"]})], ignore_index=True)
    panel = make_stock_panel(prices, universe)
    assert np.isnan(panel.opens[:100, 0]).all()
    assert np.isnan(panel.opens[:, -1]).all()
    features = build_stock_features(panel, config)
    assert not features.eligible[:100 + config.factor_lookback, 0].any()
    assert features.eligible[100 + config.factor_lookback:, 0].any()
    assert not features.eligible[:, -1].any()


def test_factor_hedges_preserve_actual_neutral_portfolio_pnl(research_fixture):
    _, _, _, panel, features, *_ = research_fixture
    snapshot = features.snapshots[0]
    loadings, hedge = np.asarray(snapshot["loadings"]), np.asarray(snapshot["hedge"])
    constraints = np.asarray(snapshot["constraints"])
    np.testing.assert_allclose(hedge @ loadings, np.eye(len(hedge)), atol=1e-12)
    rng = np.random.default_rng(4)
    weights = rng.normal(size=len(snapshot["indices"]))
    weights -= constraints.T @ (constraints @ weights)
    returns = rng.normal(0.002, 0.01, len(weights))
    residual = returns - loadings @ (hedge @ returns)
    assert weights @ returns == pytest.approx(weights @ residual, abs=1e-12)
    assert weights.sum() == pytest.approx(0, abs=1e-12)
    for sector in set(snapshot["sectors"]):
        assert weights[np.asarray(snapshot["sectors"]) == sector].sum() == pytest.approx(0, abs=1e-12)


def test_every_model_and_inner_split_purges_unavailable_labels(research_fixture):
    _, _, _, _, features, _, decisions = research_fixture
    assert len(decisions) >= 3
    for decision in decisions:
        assert pd.Timestamp(decision["latest_label_exit"]) < pd.Timestamp(decision["signal_date"])
        for split in decision["splits"]:
            assert pd.Timestamp(split["latest_training_exit"]) < pd.Timestamp(split["validation_start"])
    dates = np.array([63, 64, 65])
    _, _, weights = pooled_rows(features, dates)
    assert weights.sum() == pytest.approx(len(dates))


def test_future_prices_cannot_change_old_features_forecasts_or_holdings(research_fixture):
    prices, universe, config, panel, features, forecasts, _ = research_fixture
    cutoff = 205
    changed = prices.copy()
    future = changed["date"] > panel.dates[cutoff]
    rng = np.random.default_rng(3)
    multipliers = np.exp(rng.normal(0, 0.15, future.sum()))
    for column in ("open", "close", "raw_open", "raw_close"):
        changed.loc[future, column] *= multipliers
    new_panel = make_stock_panel(changed, universe)
    new_features = build_stock_features(new_panel, config)
    new_forecasts, _ = stock_forecasts(new_panel, new_features, config)
    np.testing.assert_allclose(features.x[:cutoff + 1], new_features.x[:cutoff + 1], equal_nan=True)
    mature = features.exit_positions <= cutoff
    np.testing.assert_allclose(features.y[mature], new_features.y[mature], equal_nan=True)
    np.testing.assert_allclose(forecasts[:cutoff + 1], new_forecasts[:cutoff + 1], equal_nan=True)
    original, start, _ = stock_weights(panel, features, forecasts, config)
    mutated, new_start, _ = stock_weights(new_panel, new_features, new_forecasts, config)
    assert start == new_start
    np.testing.assert_allclose(original[:cutoff + 1], mutated[:cutoff + 1], atol=1e-9)


def test_optimizer_enforces_factor_sector_beta_gross_and_capacity_limits(research_fixture):
    _, _, config, _, features, *_ = research_fixture
    snapshot = features.snapshots[0]
    constraints = np.asarray(snapshot["constraints"])
    solver = StockOptimizer(constraints, np.asarray(snapshot["variance"]), config)
    n = len(snapshot["indices"])
    caps = np.full(n, 0.08)
    caps[0] = 0
    weights, details = solver.solve(np.linspace(-0.003, 0.003, n), np.zeros(n), caps)
    assert np.abs(weights).sum() <= 1 + 2e-6
    assert np.all(np.abs(weights) <= caps + 2e-6)
    np.testing.assert_allclose(constraints @ weights, 0, atol=2e-6)
    assert np.asarray(snapshot["beta"]) @ weights == pytest.approx(0, abs=2e-6)
    assert details["constraint_error"] < 2e-6
    cash, _ = solver.solve(np.zeros(n), np.zeros(n), caps)
    np.testing.assert_allclose(cash, 0, atol=2e-6)


def test_stock_accounting_is_self_financing_and_charges_weekend_borrow():
    panel = flat_panel()
    config = StockConfig(cost_bps=10, borrow_rate=0.1)
    weights = np.tile([0.25, -0.25], (5, 1))
    ledger = simulate_stock_portfolio(panel, weights, 0, config)
    # First holding period is Friday to Monday: all three days incur borrow.
    borrow = 0.25 * 0.1 * 3 / 365.25
    assert ledger["borrow_fee"].iloc[0] == pytest.approx(borrow)
    assert ledger["cash_income"].iloc[0] == pytest.approx(0.75 * 0.001)
    assert ledger["net_return"].iloc[0] == pytest.approx((1 + 0.75 * 0.001 - borrow) / 1.0005 - 1)
    assert ledger["turnover"].iloc[0] == pytest.approx(0.5 / 1.0005)
    assert ledger["liquidation_turnover"].iloc[-1] == pytest.approx(0.5)
    assert ledger["turnover"].iloc[1] > 0  # Prior holdings drift relative to NAV.


def test_cash_is_exactly_benchmark_and_cost_stresses_reduce_wealth():
    panel = flat_panel()
    config = StockConfig()
    cash = simulate_stock_portfolio(panel, np.zeros((5, 2)), 0, config)
    np.testing.assert_allclose(cash["excess_return"], 0, atol=1e-15)
    assert cash["turnover"].sum() == 0
    weights = np.tile([0.25, -0.25], (5, 1))
    base = simulate_stock_portfolio(panel, weights, 0, config)
    fees = simulate_stock_portfolio(panel, weights, 0, config, cost_bps=20)
    borrow = simulate_stock_portfolio(panel, weights, 0, config, borrow_rate=0.1)
    assert fees["equity_net"].iloc[-1] < base["equity_net"].iloc[-1]
    assert borrow["equity_net"].iloc[-1] < base["equity_net"].iloc[-1]


def test_missing_held_prices_fail_instead_of_creating_zero_pnl():
    panel = flat_panel()
    panel.opens[2, 0] = np.nan
    with pytest.raises(ValueError, match="Held-position prices unavailable"):
        simulate_stock_portfolio(panel, np.tile([0.25, -0.25], (5, 1)), 0, StockConfig())


def test_offline_study_outputs_cache_validation_and_report_regeneration(tmp_path, research_fixture):
    prices, universe, config, *_ = research_fixture
    data_path, universe_path = tmp_path / "prices.csv", tmp_path / "universe.csv"
    prices.to_csv(data_path, index=False)
    universe.to_csv(universe_path, index=False)
    output = tmp_path / "output"
    metrics, manifest = run_stock_study(data_path, universe_path, output, config,
                                      synthetic=True, progress=lambda _: None)
    assert set(metrics["method"]) == {"residual_reversion", "pooled_ridge", "forecast_blend", "buy_hold"}
    assert manifest["synthetic"]
    assert not manifest["point_in_time_universe"]
    assert manifest["max_constraint_error"] < 2e-6
    assert (output / "report.html").exists()
    assert "synthetic" in (output / "report.html").read_text(encoding="utf-8").lower()
    assert len(pd.read_csv(output / "latest_research_proposal.csv")) == 60
    with np.load(output / "holdings.npz", allow_pickle=False) as archive:
        assert archive["weights"].shape == (250, 20, 3)
    render_saved_stock_report(output)
    with pytest.raises(ValueError, match="stale"):
        run_stock_study(data_path, universe_path, output, replace(config, n_components=3),
                        reuse_cache=True, progress=lambda _: None)
    saved = json.loads((output / "manifest.json").read_text())
    assert saved["data_sha256"] == manifest["data_sha256"]
