from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from regime_lab.backtest import MODEL_NAMES, walk_forward
from regime_lab.config import Config
from regime_lab.data import synthetic_prices
from regime_lab.features import build_dataset
from regime_lab.portfolio import paired_sharpe_intervals, portfolio_metrics, simulate_portfolio
from regime_lab.sharpe_models import select_ridge_alpha
from regime_lab.sharpe_study import ASSETS, construct_weights, minimum_variance_weights, period_masks


def dates(n):
    sessions = pd.bdate_range("2015-01-01", periods=5 * n + 2)
    return sessions[:5 * n:5], sessions[1:5 * n + 1:5], sessions[6:5 * n + 2:5]


def simulate(weights, returns, cost=5):
    signal, entry, exit_ = dates(len(weights))
    return simulate_portfolio(weights, returns, returns[:, -1], signal, entry, exit_, cost)


def test_self_financing_costs_include_initial_purchase_and_liquidation():
    weights = np.array([[1.0, 0.0]])
    returns = np.array([[0.10, 0.0]])
    ledger = simulate(weights, returns, cost=10)
    assert ledger["equity_net"].iloc[-1] == pytest.approx(1.1 * (1 - 0.001) / (1 + 0.001))
    assert ledger["turnover"].iloc[0] == pytest.approx(1 / 1.001)
    assert ledger["liquidation_turnover"].iloc[-1] == 1


def test_fractional_rebalancing_uses_drifted_holdings_and_charges_cash_sleeve():
    weights = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    returns = np.array([[0.10, 0.0], [0.0, 0.0], [0.0, 0.0]])
    ledger = simulate(weights, returns, cost=10)
    # Even unchanged target weights require a sale of appreciated equity and purchase of cash.
    assert ledger["turnover"].iloc[1] == pytest.approx(2 * (0.55 / 1.05 - 0.5), abs=1e-9)
    assert ledger["net_return"].iloc[1] < 0
    assert ledger["turnover"].iloc[2] == pytest.approx(0, abs=1e-10)


def test_constant_exposure_does_not_artificially_improve_excess_sharpe():
    rng = np.random.default_rng(42)
    equity = rng.normal(0.003, 0.02, 100)
    cash = rng.normal(0.0005, 0.00002, 100)
    returns = np.column_stack([equity, cash])
    full = simulate(np.tile([1.0, 0.0], (100, 1)), returns, cost=0)
    half = simulate(np.tile([0.4, 0.6], (100, 1)), returns, cost=0)
    assert portfolio_metrics(full)["sharpe_excess"] == pytest.approx(portfolio_metrics(half)["sharpe_excess"], abs=1e-10)
    np.testing.assert_allclose(half["excess_return"], 0.4 * (equity - cash), atol=1e-12)


@pytest.mark.parametrize("weights", [np.array([[1.1, -0.1]]), np.array([[0.5, 0.4]])])
def test_leverage_shorting_and_unfunded_weights_are_rejected(weights):
    with pytest.raises(ValueError, match="Long-only"):
        simulate(weights, np.zeros_like(weights))


def test_inner_ridge_validation_purges_unavailable_labels():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(450, 3))
    y = x[:, 0] * 0.002 + rng.normal(0, 0.02, len(x))
    sessions = pd.bdate_range("2010-01-01", periods=456)
    labels = pd.DataFrame({"exit_date": sessions[6:]}, index=sessions[:450])
    model, diagnostics = select_ridge_alpha(x, y, labels)
    assert diagnostics["selected_alpha"] in (100, 1000, 10000)
    assert len(diagnostics["inner_boundaries"]) == 4
    for boundary in diagnostics["inner_boundaries"]:
        assert pd.Timestamp(boundary["latest_train_exit"]) < pd.Timestamp(boundary["validation_start"])
    assert np.isfinite(model.predict(x[:3])).all()


def test_minimum_variance_constraints_and_redundant_assets():
    rng = np.random.default_rng(3)
    low = rng.normal(0, 0.002, 252)
    returns = np.column_stack([low, low, rng.normal(0, 0.02, 252), rng.normal(0, 0.03, 252)])
    weights = minimum_variance_weights(returns)
    assert weights.sum() == pytest.approx(1)
    assert np.all(weights >= -1e-12)
    assert np.all(weights <= 0.6 + 1e-12)


def test_bootstrap_is_paired_and_identical_strategies_have_zero_difference():
    rng = np.random.default_rng(5)
    returns = np.column_stack([rng.normal(0.002, 0.02, 100), np.full(100, 0.0004)])
    ledger = simulate(np.tile([0.8, 0.2], (100, 1)), returns)
    intervals = paired_sharpe_intervals({"buy_hold": ledger, "copy": ledger.copy()},
                                       references=("buy_hold",), replications=100)
    assert intervals["sharpe_difference"].iloc[0] == 0
    assert intervals["ci_95_lower"].iloc[0] == 0
    assert intervals["ci_95_upper"].iloc[0] == 0


def test_development_selection_excludes_trade_crossing_cutoff():
    ledger = pd.DataFrame({"exit_date": pd.to_datetime(["2018-12-28", "2019-01-03", "2019-01-10"])},
                          index=pd.to_datetime(["2018-12-20", "2018-12-27", "2019-01-04"]))
    np.testing.assert_array_equal(period_masks(ledger)["development"], [True, False, False])
    np.testing.assert_array_equal(period_masks(ledger)["later"], [False, False, True])


def test_allocation_decisions_are_unchanged_by_future_prices():
    prices = synthetic_prices(430, seed=42)
    config = Config(train_size=300, calibration_size=150, n_components=2, n_states=2)
    dataset = build_dataset(prices)
    predictions = dataset.labels.iloc[200:].copy()
    for name in MODEL_NAMES:
        predictions[name] = 0.001
    predictions["small_ridge_excess"] = 0.0008
    predictions["small_hmm_ridge_excess"] = 0.0007
    predictions["fold"] = 0
    predictions["p_state_0"] = 0.7
    predictions["p_state_1"] = 0.3
    states = pd.DataFrame({"fold": [0, 0], "risk_rank": [0, 1], "calibration_target_volatility": [0.1, 0.3]})
    folds = [{"fold": 0, "transition_matrix_risk_ranked": [[0.95, 0.05], [0.1, 0.9]]}]
    sessions = pd.DatetimeIndex(sorted(prices["date"].unique()))
    cash = pd.DataFrame({"date": sessions, "symbol": "BIL", "open": 100 * np.exp(np.arange(len(sessions)) * 0.00005),
                         "close": 100 * np.exp(np.arange(len(sessions)) * 0.00005)})
    weights, _, _ = construct_weights(prices, cash, predictions, folds, states, config)
    cutoff = predictions.index[15]
    revised = prices.copy()
    revised.loc[revised["date"] > cutoff, ["open", "close"]] *= 1.1
    changed, _, _ = construct_weights(revised, cash, predictions, folds, states, config)
    for name in weights:
        np.testing.assert_allclose(weights[name][:16], changed[name][:16], atol=1e-12)
        np.testing.assert_allclose(weights[name].sum(axis=1), 1, atol=1e-10)
        assert weights[name].shape[1] == len(ASSETS)

