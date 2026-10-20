from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd
import pytest
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm

from regime_lab.backtest import MODEL_NAMES, walk_forward
from regime_lab.config import Config
from regime_lab.data import synthetic_prices, validate_prices
from regime_lab.evaluation import forecast_metrics, strategy_results
from regime_lab.features import build_dataset
from regime_lab.models import RegimeEncoder, filtered_probabilities


def known_hmm():
    model = GaussianHMM(n_components=2, covariance_type="full", init_params="")
    model.startprob_ = np.array([0.6, 0.4])
    model.transmat_ = np.array([[0.9, 0.1], [0.2, 0.8]])
    model.means_ = np.array([[-1.0], [1.0]])
    model.covars_ = np.array([[[0.5]], [[0.8]]])
    model.n_features = 1
    return model


def small_config():
    return Config(train_size=300, calibration_size=150, test_size=50, n_components=2,
                  n_states=2, hmm_restarts=2, hmm_iterations=100, seed=19)


def test_filter_matches_brute_force_and_ignores_future_observations():
    model = known_hmm()
    observations = np.array([[-0.6], [0.2], [1.3], [-0.8]])
    actual = filtered_probabilities(model, observations)
    for t in range(len(observations)):
        mass = np.zeros(2)
        for path in product(range(2), repeat=t + 1):
            probability = model.startprob_[path[0]]
            for j, state in enumerate(path):
                probability *= norm.pdf(observations[j, 0], model.means_[state, 0], np.sqrt(model.covars_[state, 0, 0]))
                if j:
                    probability *= model.transmat_[path[j - 1], state]
            mass[path[-1]] += probability
        np.testing.assert_allclose(actual[t], mass / mass.sum(), atol=1e-12)
    changed = observations.copy()
    changed[2:] = 100
    np.testing.assert_allclose(filtered_probabilities(model, changed)[:2], actual[:2])


def test_filter_continues_across_batch_boundary_and_matches_terminal_posterior():
    model = known_hmm()
    observations = np.array([[-1.2], [-0.8], [0.7], [1.1]])
    whole = filtered_probabilities(model, observations)
    first = filtered_probabilities(model, observations[:2])
    second = filtered_probabilities(model, observations[2:], previous=first[-1])
    np.testing.assert_allclose(second, whole[2:], atol=1e-12)
    np.testing.assert_allclose(whole[-1], model.predict_proba(observations)[-1], atol=1e-12)


def test_features_are_causal_and_target_uses_next_open():
    prices = synthetic_prices(220)
    dataset = build_dataset(prices, horizon=5)
    signal = dataset.features.index[20]
    opens = prices.loc[prices["symbol"] == "SPY"].set_index("date")["open"]
    position = opens.index.get_loc(signal)
    label = dataset.labels.loc[signal]
    assert label["entry_date"] == opens.index[position + 1]
    assert label["exit_date"] == opens.index[position + 6]
    assert label["target_return"] == pytest.approx(opens.iloc[position + 6] / opens.iloc[position + 1] - 1)
    changed = prices.copy()
    changed.loc[changed["date"] > signal, ["open", "close"]] *= 7
    revised = build_dataset(changed)
    pd.testing.assert_frame_equal(dataset.features.loc[:signal], revised.features.loc[:signal])


@pytest.mark.parametrize("problem", ["duplicate", "missing", "nonpositive", "infinite"])
def test_invalid_price_panels_are_rejected(problem):
    prices = synthetic_prices(120)
    if problem == "duplicate":
        prices = pd.concat([prices, prices.iloc[[0]]])
    elif problem == "missing":
        prices = prices.iloc[1:]
    elif problem == "nonpositive":
        prices.loc[0, "open"] = 0
    else:
        prices.loc[0, "close"] = np.inf
    with pytest.raises(ValueError):
        validate_prices(prices)


def test_walk_forward_has_purged_labels_and_no_future_influence():
    config = small_config()
    prices = synthetic_prices(560, seed=8)
    dataset = build_dataset(prices)
    baseline = walk_forward(dataset, config)
    for fold in baseline.folds:
        assert pd.Timestamp(fold["latest_train_label_exit"]) < pd.Timestamp(fold["test_start"])
        assert pd.Timestamp(fold["calibration_end"]) < pd.Timestamp(fold["return_train_start"])
        assert pd.Timestamp(fold["return_train_end"]) < pd.Timestamp(fold["test_start"])
    cutoff = baseline.predictions.index[20]
    changed = prices.copy()
    changed.loc[changed["date"] > cutoff, ["open", "close"]] *= 1.3
    revised = walk_forward(build_dataset(changed), config)
    forecast_columns = list(MODEL_NAMES) + ["p_state_0", "p_state_1"]
    np.testing.assert_allclose(baseline.predictions.loc[:cutoff, forecast_columns],
                               revised.predictions.loc[:cutoff, forecast_columns], atol=1e-12)
    np.testing.assert_allclose(baseline.predictions[["p_state_0", "p_state_1"]].sum(axis=1), 1, atol=1e-12)


def test_encoder_scaling_and_pca_fit_on_calibration_only():
    config = small_config()
    features = build_dataset(synthetic_prices(400)).features
    calibration = features.iloc[:config.calibration_size]
    encoder = RegimeEncoder(config).fit(calibration.to_numpy(), calibration["SPY_ret_1"].to_numpy())
    np.testing.assert_allclose(encoder.scaler.mean_, calibration.mean().to_numpy())
    before = encoder.pca.components_.copy()
    encoder.transform(features.iloc[config.calibration_size:].to_numpy() * 100)
    np.testing.assert_array_equal(before, encoder.pca.components_)
    assert np.all(np.diff(encoder.calibration_volatility) >= 0)


def mock_predictions():
    dates = pd.bdate_range("2020-01-01", periods=16)
    frame = pd.DataFrame(index=dates[:10])
    frame.index.name = "signal_date"
    frame["entry_date"] = dates[1:11]
    frame["exit_date"] = dates[6:16]
    frame["target_return"] = 0.10
    for name in MODEL_NAMES:
        frame[name] = 0.01
    return frame


def test_nonoverlapping_trades_costs_and_final_liquidation():
    config = replace(small_config(), cost_bps=10)
    frame = mock_predictions()
    metrics, ledger = strategy_results(frame, config)
    buy_hold = ledger.loc[ledger["model"] == "buy_and_hold"]
    assert len(buy_hold) == 2
    assert buy_hold["exit_date"].iloc[0] == buy_hold["entry_date"].iloc[1]
    assert buy_hold["equity_net"].iloc[-1] == pytest.approx((1 - 0.001) ** 2 * 1.1**2)
    assert metrics.set_index("model").loc["buy_and_hold", "one_way_turnover"] == 2
    scores = forecast_metrics(frame, config.horizon)
    assert scores.loc[scores["sample"] == "nonoverlapping", "n"].eq(2).all()
    assert scores.loc[scores["model"] == "historical_mean", "oos_r2_vs_mean"].eq(0).all()


def test_cash_has_no_trading_costs_and_first_loss_counts_in_drawdown():
    config = replace(small_config(), cost_bps=0)
    frame = mock_predictions()
    frame["target_return"] = -0.1
    frame["ridge"] = -0.1
    metrics, ledger = strategy_results(frame, config)
    by_model = metrics.set_index("model")
    assert by_model.loc["ridge", "total_return_net"] == 0
    assert by_model.loc["ridge", "one_way_turnover"] == 0
    assert by_model.loc["buy_and_hold", "max_drawdown_net"] == pytest.approx(0.9**2 - 1)


def test_insufficient_history_and_invalid_config_fail_clearly():
    with pytest.raises(ValueError, match="Need more than"):
        walk_forward(build_dataset(synthetic_prices(200)), small_config())
    with pytest.raises(ValueError, match="mature"):
        Config(train_size=800, calibration_size=756)


def test_experiment_artifacts_and_saved_report_regeneration(tmp_path):
    import json
    from regime_lab.cli import run_experiment
    from regime_lab.report import render_saved_report

    result = run_experiment(synthetic_prices(430, seed=8), small_config(), tmp_path,
                            {"source": "Synthetic test fixture", "synthetic": True, "sha256": "fixture"})
    persisted = pd.read_csv(tmp_path / "predictions.csv")
    assert len(persisted) == len(result.predictions)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["data"]["synthetic"] is True
    assert manifest["n_forecasts"] == len(result.predictions)
    (tmp_path / "report.html").unlink()
    report = render_saved_report(tmp_path)
    assert "Synthetic demonstration" in report
    assert "What the factors represent" in report
    assert "<svg" in report
    assert (tmp_path / "regime_probabilities.png").stat().st_size > 1000
    assert (tmp_path / "strategy_equity.png").stat().st_size > 1000

