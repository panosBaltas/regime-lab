"""Fixed-catalog Sharpe research; all portfolio decisions use past information."""

from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from threadpoolctl import threadpool_limits

from .backtest import MODEL_NAMES, walk_forward
from .config import Config
from .data import file_digest, load_prices
from .features import build_dataset
from .portfolio import paired_sharpe_intervals, portfolio_metrics, simulate_portfolio
from .sharpe_models import SmallForecastExtension

PRIMARY = ("buy_hold", "original_hmm", "original_hmm_vol", "vol_target", "inverse_variance",
           "trend_12m", "trend_multiscale", "trend_vol", "hmm_risk_target", "shrunk_hmm",
           "shrunk_hmm_vol", "small_ridge", "small_hmm_ridge", "fixed_ensemble")
DIVERSIFIED = ("div_equal", "div_inverse_vol", "div_min_variance", "div_trend")
ASSETS = ("SPY", "TLT", "GLD", "DBC", "BIL")
CUTOFF = pd.Timestamp("2019-01-01")


def minimum_variance_weights(returns, cap=0.6):
    covariance = LedoitWolf().fit(returns).covariance_
    n = covariance.shape[0]
    scaled = covariance / max(float(np.trace(covariance) / n), 1e-12)
    result = minimize(lambda w: float(w @ scaled @ w), np.full(n, 1 / n),
                      jac=lambda w: 2 * scaled @ w, method="SLSQP",
                      bounds=[(0, cap)] * n,
                      constraints={"type": "eq", "fun": lambda w: w.sum() - 1,
                                   "jac": lambda w: np.ones(n)},
                      options={"ftol": 1e-10, "maxiter": 200})
    if not result.success or abs(result.x.sum() - 1) > 1e-8:
        raise ValueError(f"Minimum-variance optimization failed: {result.message}")
    return result.x


def construct_weights(prices, cash_prices, predictions, folds, states, config, progress=None):
    if config.target != "SPY" or config.horizon != 5:
        raise ValueError("This fixed research protocol requires SPY and a five-session horizon")
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index()
    opens = prices.pivot(index="date", columns="symbol", values="open").sort_index()
    cash = cash_prices.set_index("date")
    if set(cash_prices["symbol"].unique()) != {"BIL"} or not cash.index.equals(close.index):
        raise ValueError("Cash input must contain only BIL and exactly match the market sessions")
    close["BIL"], opens["BIL"] = cash["close"], cash["open"]
    missing = set(ASSETS) - set(close.columns)
    if missing:
        raise ValueError(f"Missing study assets: {sorted(missing)}")
    dates = predictions.index
    daily = np.log(close).diff()
    variance = daily.ewm(halflife=21, adjust=False, min_periods=63).var(bias=False) * 252
    variance = variance.clip(lower=0.05**2)
    vol = np.sqrt(variance).reindex(dates)
    if not np.isfinite(vol[list(ASSETS)].to_numpy()).all():
        raise ValueError("Insufficient volatility warmup")
    trend = {}
    for symbol in ASSETS[:-1]:
        scores = []
        for lookback in (63, 126, 252):
            difference = np.log(close[symbol]).diff(lookback) - np.log(close["BIL"]).diff(lookback)
            if difference.reindex(dates).isna().any():
                raise ValueError("Insufficient trend warmup")
            scores.append((difference.reindex(dates).to_numpy() > 0).astype(float))
        trend[symbol] = np.column_stack(scores)
    asset_returns = (opens.shift(-(config.horizon + 1)) / opens.shift(-1) - 1).reindex(dates)[list(ASSETS)]
    if not np.isfinite(asset_returns.to_numpy()).all():
        raise ValueError("Incomplete holding-period returns")
    hmm_variance = np.empty(len(dates))
    for fold in folds:
        mask = predictions["fold"].to_numpy() == fold["fold"]
        transition = np.asarray(fold["transition_matrix_risk_ranked"])
        state_variance = states.loc[states["fold"] == fold["fold"]].sort_values("risk_rank")[
            "calibration_target_volatility"].to_numpy() ** 2
        expected = np.zeros_like(state_variance)
        power = np.eye(config.n_states)
        for _ in range(config.horizon):
            power = power @ transition
            expected += power @ state_variance / config.horizon
        probabilities = predictions.loc[mask, [f"p_state_{k}" for k in range(config.n_states)]].to_numpy()
        hmm_variance[mask] = probabilities @ expected
    recent_variance = vol["SPY"].to_numpy() ** 2
    blend_variance = 0.5 * recent_variance + 0.5 * hmm_variance
    vol_weight = np.minimum(1, 0.10 / np.sqrt(recent_variance))
    cash_expectation = np.expm1(daily["BIL"].rolling(63).mean().reindex(dates).to_numpy() * config.horizon)
    shrunk_excess = (0.75 * predictions["historical_mean"].to_numpy()
                     + 0.25 * predictions["pca_hmm_ridge"].to_numpy() - cash_expectation)

    def allocation(expected_excess):
        return np.clip(expected_excess / (5 * recent_variance * config.horizon / 252), 0, 1)

    exposures = {
        "buy_hold": np.ones(len(dates)),
        "original_hmm": (predictions["pca_hmm_ridge"].to_numpy() > 0).astype(float),
        "original_hmm_vol": (predictions["pca_hmm_ridge"].to_numpy() > 0) * vol_weight,
        "vol_target": vol_weight,
        "inverse_variance": np.minimum(1, 0.10**2 / recent_variance),
        "trend_12m": trend["SPY"][:, -1],
        "trend_multiscale": trend["SPY"].mean(axis=1),
        "trend_vol": trend["SPY"].mean(axis=1) * vol_weight,
        "hmm_risk_target": np.minimum(1, 0.10 / np.sqrt(blend_variance)),
        "shrunk_hmm": allocation(shrunk_excess),
        "shrunk_hmm_vol": allocation(shrunk_excess) * vol_weight,
        "small_ridge": allocation(predictions["small_ridge_excess"].to_numpy()),
        "small_hmm_ridge": allocation(predictions["small_hmm_ridge_excess"].to_numpy()),
    }
    exposures["fixed_ensemble"] = np.mean([exposures[name] for name in
                                           ("vol_target", "trend_vol", "shrunk_hmm_vol", "small_hmm_ridge")], axis=0)
    weights = {}
    for name in PRIMARY:
        matrix = np.zeros((len(dates), len(ASSETS)))
        matrix[:, 0], matrix[:, -1] = exposures[name], 1 - exposures[name]
        weights[name] = matrix
    inverse_vol = 1 / vol[list(ASSETS[:-1])].to_numpy()
    inverse_vol /= inverse_vol.sum(axis=1, keepdims=True)
    weights["div_equal"] = np.column_stack([np.full((len(dates), 4), 0.25), np.zeros(len(dates))])
    weights["div_inverse_vol"] = np.column_stack([inverse_vol, np.zeros(len(dates))])
    diversified_trend = inverse_vol * np.column_stack([trend[symbol].mean(axis=1) for symbol in ASSETS[:-1]])
    weights["div_trend"] = np.column_stack([diversified_trend, 1 - diversified_trend.sum(axis=1)])
    min_variance = []
    with threadpool_limits(limits=1):
        for row, signal in enumerate(dates):
            end = close.index.get_loc(signal) + 1
            history = daily.iloc[end - 252:end][list(ASSETS[:-1])].to_numpy()
            if len(history) != 252 or not np.isfinite(history).all():
                raise ValueError("Insufficient minimum-variance covariance history")
            min_variance.append(minimum_variance_weights(history))
            if progress and row % 700 == 0:
                progress(f"Covariance portfolios: {row + 1}/{len(dates)} signal dates")
    weights["div_min_variance"] = np.column_stack([np.asarray(min_variance), np.zeros(len(dates))])
    diagnostics = pd.DataFrame({"ewma_volatility": np.sqrt(recent_variance),
                                "hmm_volatility": np.sqrt(hmm_variance),
                                "blended_volatility": np.sqrt(blend_variance),
                                "cash_expected_return": cash_expectation,
                                "shrunk_hmm_excess_forecast": shrunk_excess}, index=dates)
    return weights, asset_returns, diagnostics


def period_masks(ledger):
    return {
        "full": np.ones(len(ledger), dtype=bool),
        "development": (ledger.index < CUTOFF) & (ledger["exit_date"] < CUTOFF),
        "later": ledger.index >= CUTOFF,
        "2019_2022": (ledger.index >= CUTOFF) & (ledger["exit_date"] < pd.Timestamp("2023-01-01")),
        "2023_plus": ledger.index >= pd.Timestamp("2023-01-01"),
    }


def evaluate_catalog(weights, asset_returns, predictions, horizon=5, progress=None):
    rows, primary_ledgers = [], {}
    for cost in (5.0, 10.0, 25.0):
        for phase in range(horizon):
            indices = np.arange(phase, len(predictions), horizon)
            labels = predictions.iloc[indices]
            for name in (*PRIMARY, *DIVERSIFIED):
                ledger = simulate_portfolio(weights[name][indices], asset_returns.iloc[indices].to_numpy(),
                                            asset_returns.iloc[indices]["BIL"].to_numpy(), labels.index,
                                            labels["entry_date"], labels["exit_date"], cost, horizon)
                for period, mask in period_masks(ledger).items():
                    if np.count_nonzero(mask) < 2:
                        continue
                    rows.append({"strategy": name, "opportunity_set": "SPY_BIL" if name in PRIMARY else "diversified",
                                 "cost_bps": cost, "phase": phase, "period": period,
                                 **portfolio_metrics(ledger.loc[mask], horizon)})
                if cost == 5 and phase == 0:
                    primary_ledgers[name] = ledger
            if progress:
                progress(f"Evaluated {len(weights)} strategies | phase {phase} | {cost:g} bps")
    return pd.DataFrame(rows), primary_ledgers


def run_sharpe_study(data_path: Path, cash_path: Path, baseline: Path, output: Path,
                      reuse_forecasts=False, progress=print):
    baseline_manifest = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    config = Config(**baseline_manifest["config"])
    data_hash, cash_hash = file_digest(data_path), file_digest(cash_path)
    if data_hash != baseline_manifest["data"]["sha256"]:
        raise ValueError("Market CSV differs from the preserved baseline input")
    output.mkdir(parents=True, exist_ok=True)
    if output.resolve() == baseline.resolve():
        raise ValueError("Research output must differ from the preserved baseline directory")
    prices, cash = load_prices(data_path), load_prices(cash_path, "BIL")
    baseline_predictions = pd.read_csv(baseline / "predictions.csv", index_col="signal_date",
                                       parse_dates=["signal_date", "entry_date", "exit_date"])
    cache_key = {"data_sha256": data_hash, "cash_sha256": cash_hash, "config": config.to_dict(),
                 "source_hashes": {name: file_digest(Path(__file__).with_name(name)) for name in
                                   ("models.py", "backtest.py", "sharpe_models.py", "features.py")}}
    cache_path = output / "forecast_cache.json"
    if reuse_forecasts:
        if not cache_path.exists() or json.loads(cache_path.read_text()) != cache_key:
            raise ValueError("Forecast cache is absent or stale; rerun without --reuse-forecasts")
        predictions = pd.read_csv(output / "enhanced_predictions.csv", index_col="signal_date",
                                   parse_dates=["signal_date", "entry_date", "exit_date"])
        folds = json.loads((output / "enhanced_folds.json").read_text())
        states = pd.read_csv(output / "state_diagnostics.csv")
    else:
        dataset = build_dataset(prices, config.target, config.horizon)
        cash_labels = build_dataset(cash, "BIL", config.horizon).labels["target_return"]
        experiment = walk_forward(dataset, config, progress, SmallForecastExtension(cash_labels))
        predictions, folds, states = experiment.predictions, experiment.folds, experiment.states
        predictions.to_csv(output / "enhanced_predictions.csv", float_format="%.12g")
        states.to_csv(output / "state_diagnostics.csv", index=False, float_format="%.12g")
        (output / "enhanced_folds.json").write_text(json.dumps(folds, indent=2, allow_nan=False), encoding="utf-8")
        cache_path.write_text(json.dumps(cache_key, indent=2), encoding="utf-8")
    if not predictions.index.equals(baseline_predictions.index):
        raise ValueError("Enhanced forecasts do not match the preserved baseline dates")
    baseline_error = float(np.max(np.abs(predictions[list(MODEL_NAMES)].to_numpy()
                                         - baseline_predictions[list(MODEL_NAMES)].to_numpy())))
    if baseline_error > 1e-9:
        raise ValueError(f"Baseline forecast reproduction failed: max difference {baseline_error}")
    weights, asset_returns, diagnostics = construct_weights(prices, cash, predictions, folds, states, config, progress)
    metrics, ledgers = evaluate_catalog(weights, asset_returns, predictions, config.horizon, progress)
    development = metrics.loc[(metrics["cost_bps"] == 5) & (metrics["phase"] == 0)
                              & (metrics["period"] == "development") & (metrics["strategy"].isin(PRIMARY))]
    if len(development) != len(PRIMARY):
        raise ValueError("Insufficient development history for the fixed selection protocol")
    selected = max(PRIMARY, key=lambda name: float(development.set_index("strategy").loc[name, "sharpe_excess"]))
    later_ledgers = {name: ledger.loc[period_masks(ledger)["later"]] for name, ledger in ledgers.items()}
    intervals = paired_sharpe_intervals(later_ledgers, horizon=config.horizon)
    # Explicitly ex post exposure matching is a diagnostic, never a candidate used for selection.
    exposure = float(later_ledgers[selected]["risky_weight"].mean())
    matched = np.zeros((len(predictions), len(ASSETS)))
    matched[:, 0], matched[:, -1] = exposure, 1 - exposure
    indices = np.arange(0, len(predictions), config.horizon)
    exposure_ledger = simulate_portfolio(matched[indices], asset_returns.iloc[indices].to_numpy(),
                                         asset_returns.iloc[indices]["BIL"].to_numpy(), predictions.index[indices],
                                         predictions.iloc[indices]["entry_date"], predictions.iloc[indices]["exit_date"])
    exposure_metrics = portfolio_metrics(exposure_ledger.loc[period_masks(exposure_ledger)["later"]])
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "baseline": str(baseline.resolve()),
        "data_sha256": data_hash, "cash_sha256": cash_hash,
        "baseline_predictions_sha256": file_digest(baseline / "predictions.csv"),
        "baseline_reproduction_max_absolute_error": baseline_error,
        "selected_on_development_only": selected, "selection_cutoff": str(CUTOFF.date()),
        "holdout_status": "All historical data were previously examined in aggregate; later period is reused research history",
        "primary_candidates": list(PRIMARY), "diversified_candidates": list(DIVERSIFIED),
        "candidate_count_this_family": len(weights), "prior_project_trials": "Not included in this count",
        "config": config.to_dict(), "cash_proxy": "BIL, traded and charged transaction costs",
        "costs_bps": [5, 10, 25], "phases": list(range(config.horizon)),
        "bootstrap": {"replications": 2000, "block_size_periods": 13, "selection_adjusted": False},
        "ex_post_exposure_matched_later": {"risky_weight": exposure, **exposure_metrics},
        "versions": {name: importlib.metadata.version(name) for name in
                     ("numpy", "pandas", "scipy", "scikit-learn", "hmmlearn", "matplotlib")},
        "source_hashes": {path.name: file_digest(path) for path in Path(__file__).parent.glob("*.py")},
    }
    protocol = Path(__file__).resolve().parents[2] / "docs" / "SHARPE_PROTOCOL.md"
    if protocol.exists():
        shutil.copyfile(protocol, output / "protocol_snapshot.md")
        manifest["protocol_sha256"] = file_digest(protocol)
    metrics.to_csv(output / "metrics.csv", index=False, float_format="%.12g")
    intervals.to_csv(output / "sharpe_intervals.csv", index=False, float_format="%.12g")
    diagnostics.to_csv(output / "risk_diagnostics.csv", float_format="%.12g")
    pd.concat(ledgers, names=["strategy"]).to_csv(output / "trades.csv", float_format="%.12g")
    pd.concat({name: pd.DataFrame(matrix, index=predictions.index, columns=ASSETS)
               for name, matrix in weights.items()}, names=["strategy", "signal_date"]).to_csv(
                   output / "weights.csv", float_format="%.12g")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    from .sharpe_report import create_sharpe_report
    create_sharpe_report(output, metrics, intervals, ledgers, manifest)
    progress(f"Development-selected strategy: {selected}")
    progress(metrics.loc[(metrics["cost_bps"] == 5) & (metrics["phase"] == 0)
                          & (metrics["period"] == "later"),
                          ["strategy", "sharpe_excess", "cagr_net", "max_drawdown"]].to_string(index=False))
    progress(f"Report: {(output / 'report.html').resolve()}")
    return metrics, manifest

