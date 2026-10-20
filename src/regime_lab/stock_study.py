"""Reproducible stock-study orchestration; no brokers or order submission."""

from datetime import datetime, timezone
from pathlib import Path
import importlib.metadata
import json
import shutil

import numpy as np
import pandas as pd

from .data import file_digest
from .portfolio import portfolio_metrics
from .stock_features import (StockConfig, build_stock_features, load_stock_features,
                             make_stock_panel, save_stock_features)
from .stock_models import STOCK_METHODS, stock_forecasts
from .stock_portfolio import simulate_stock_portfolio, stock_weights


def stock_periods(ledger):
    return {"full": np.ones(len(ledger), dtype=bool),
            "before_2021": ledger["exit_date"] < pd.Timestamp("2021-01-01"),
            "2021_2023": (ledger.index >= pd.Timestamp("2021-01-01")) & (ledger["exit_date"] < pd.Timestamp("2024-01-01")),
            "2024_plus": ledger.index >= pd.Timestamp("2024-01-01")}


def stock_metrics(ledger):
    result = portfolio_metrics(ledger.assign(risky_weight=ledger["gross_exposure"]), horizon=1)
    result["average_gross_exposure"] = result.pop("average_risky_weight")
    result["average_daily_turnover"] = float(ledger["turnover"].mean())
    result["average_long_positions"] = float(ledger["n_long"].mean())
    result["average_short_positions"] = float(ledger["n_short"].mean())
    return result


def stock_reference(panel, start, config):
    indices = np.arange(start, len(panel.dates) - 2)
    actual = panel.spy_opens[indices + 2] / panel.spy_opens[indices + 1] - 1
    cash = panel.cash_opens[indices + 2] / panel.cash_opens[indices + 1] - 1
    cost = config.cost_bps / 10000
    net = actual.copy()
    net[0] = (1 + net[0]) / (1 + cost) - 1
    net[-1] = (1 + net[-1]) * (1 - cost) - 1
    return pd.DataFrame({"entry_date": panel.dates[indices + 1], "exit_date": panel.dates[indices + 2],
                         "net_return": net, "cash_return": cash, "excess_return": net - cash,
                         "equity_net": np.cumprod(1 + net), "gross_exposure": 1.0,
                         "turnover": np.r_[1 / (1 + cost), np.zeros(len(indices) - 1)],
                         "liquidation_turnover": np.r_[np.zeros(len(indices) - 1), 1.0],
                         "n_long": 1, "n_short": 0}, index=panel.dates[indices]).rename_axis("signal_date")


def stock_bootstrap(ledgers, seed=42, replications=2000, block_size=21):
    n = len(next(iter(ledgers.values())))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, (replications, int(np.ceil(n / block_size))))
    indices = ((starts[:, :, None] + np.arange(block_size)) % n).reshape(replications, -1)[:, :n]
    observed, boot = {}, {}
    for name, ledger in ledgers.items():
        values = ledger["excess_return"].to_numpy()
        sample = values[indices]
        std = sample.std(axis=1, ddof=1)
        boot[name] = np.divide(sample.mean(axis=1) * np.sqrt(252), std,
                               out=np.full(replications, np.nan), where=std > 1e-10)
        observed[name] = stock_metrics(ledger)["sharpe_excess"]
    rows = []
    for name in STOCK_METHODS:
        finite = boot[name][np.isfinite(boot[name])]
        bounds = np.quantile(finite, [0.025, 0.975]) if len(finite) >= 0.95 * replications else [None, None]
        rows.append({"method": name, "sharpe_excess": observed[name], "ci_lower": bounds[0], "ci_upper": bounds[1],
                     "replications": replications, "block_size": block_size,
                     "selection_adjusted": False})
    return pd.DataFrame(rows)


def forecast_scores(panel, features, forecasts, start, config):
    rows = []
    for name, lower, upper in (("full", start, len(panel.dates)),
                               ("2024_plus", max(start, int(panel.dates.searchsorted("2024-01-01"))), len(panel.dates))):
        for m, method in enumerate(STOCK_METHODS):
            errors, zero_errors, ics = [], [], []
            for t in range(lower, upper):
                valid = features.eligible[t] & np.isfinite(features.y[t]) & np.isfinite(forecasts[t, :, m])
                if valid.sum() < 10:
                    continue
                y, forecast = features.y[t, valid], forecasts[t, valid, m] * config.target_horizon
                errors.append(float(np.square(y - forecast).mean()))
                zero_errors.append(float(np.square(y).mean()))
                if np.std(forecast) > 1e-10 and np.std(y) > 1e-10:
                    ics.append(float(np.corrcoef(y, forecast)[0, 1]))
            if errors:
                rows.append({"method": method, "period": name, "n_signal_dates": len(errors),
                             "residual_return_rmse": float(np.sqrt(np.mean(errors))),
                             "r2_vs_zero_residual": float(1 - np.sum(errors) / np.sum(zero_errors)),
                             "mean_daily_cross_sectional_ic": float(np.mean(ics)) if ics else None})
    return pd.DataFrame(rows)


def run_stock_study(data_path: Path, universe_path: Path, output: Path, config: StockConfig,
                     reuse_cache=False, progress=print, synthetic=False):
    universe = pd.read_csv(universe_path, dtype={"issuer_id": str})
    prices = pd.read_csv(data_path)
    panel = make_stock_panel(prices, universe)
    output.mkdir(parents=True, exist_ok=True)
    source_dir = Path(__file__).parent
    key = {"data_sha256": file_digest(data_path), "universe_sha256": file_digest(universe_path),
           "config": json.loads(json.dumps(config.to_dict())),
           "source_hashes": {name: file_digest(source_dir / name) for name in
                             ("stock_features.py", "stock_models.py")},
           "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn")}}
    key_path = output / "forecast_cache.json"
    if reuse_cache:
        if not key_path.exists() or json.loads(key_path.read_text()) != key:
            raise ValueError("Stock feature/forecast cache is missing or stale; run without --reuse-cache")
        features = load_stock_features(output)
        with np.load(output / "forecasts.npz", allow_pickle=False) as archive:
            forecasts = archive["forecasts"]
        decisions = json.loads((output / "model_decisions.json").read_text())
    else:
        features = build_stock_features(panel, config, progress)
        save_stock_features(output, features)
        forecasts, decisions = stock_forecasts(panel, features, config, progress)
        np.savez_compressed(output / "forecasts.npz", forecasts=forecasts)
        (output / "model_decisions.json").write_text(json.dumps(decisions, indent=2, allow_nan=False), encoding="utf-8")
        key_path.write_text(json.dumps(key, indent=2), encoding="utf-8")
    weights, start, checks = stock_weights(panel, features, forecasts, config, progress)
    ledgers, metrics = {}, []
    scenarios = ((config.cost_bps, config.borrow_rate), (10.0, config.borrow_rate), (20.0, config.borrow_rate),
                 (config.cost_bps, 0.05), (config.cost_bps, 0.10))
    for cost, borrow in dict.fromkeys(scenarios):
        for m, method in enumerate(STOCK_METHODS):
            ledger = simulate_stock_portfolio(panel, weights[:, :, m], start, config, cost, borrow)
            for period, mask in stock_periods(ledger).items():
                if np.count_nonzero(mask) >= 2:
                    metrics.append({"method": method, "cost_bps": cost, "borrow_rate": borrow,
                                    "period": period, **stock_metrics(ledger.loc[mask])})
            if cost == config.cost_bps and borrow == config.borrow_rate:
                ledgers[method] = ledger
        progress(f"Stock accounting: {cost:g} bps and {borrow:.0%} annual borrow")
    ledgers["buy_hold"] = stock_reference(panel, start, config)
    for period, mask in stock_periods(ledgers["buy_hold"]).items():
        if np.count_nonzero(mask) >= 2:
            metrics.append({"method": "buy_hold", "cost_bps": config.cost_bps, "borrow_rate": 0.0,
                            "period": period, **stock_metrics(ledgers["buy_hold"].loc[mask])})
    metrics = pd.DataFrame(metrics)
    intervals = stock_bootstrap(ledgers, config.seed)
    scores = forecast_scores(panel, features, forecasts, start, config)
    np.savez_compressed(output / "holdings.npz", weights=weights, dates=panel.dates.strftime("%Y-%m-%d").to_numpy(dtype=str),
                        symbols=np.asarray(panel.symbols, dtype=str), methods=np.asarray(STOCK_METHODS, dtype=str))
    proposals = []
    latest = len(panel.dates) - 1
    for m, method in enumerate(STOCK_METHODS):
        for i, symbol in enumerate(panel.symbols):
            proposals.append({"as_of_close": str(panel.dates[latest].date()), "method": method,
                              "symbol": symbol, "sector": panel.sectors[i], "target_weight": weights[latest, i, m],
                              "expected_daily_return": forecasts[latest, i, m],
                              "execution_rule": "next regular-session open after as_of_close",
                              "borrow_available_verified": False})
    pd.DataFrame(proposals).to_csv(output / "latest_research_proposal.csv", index=False, float_format="%.12g")
    checks.to_csv(output / "constraint_checks.csv", index=False, float_format="%.12g")
    metrics.to_csv(output / "metrics.csv", index=False, float_format="%.12g")
    intervals.to_csv(output / "sharpe_intervals.csv", index=False, float_format="%.12g")
    scores.to_csv(output / "forecast_scores.csv", index=False, float_format="%.12g")
    pd.concat(ledgers, names=["method"]).to_csv(output / "trades.csv", float_format="%.12g")
    coverage = []
    for i, symbol in enumerate(panel.symbols):
        coverage.append({"symbol": symbol, "sector": panel.sectors[i],
                         "observed_sessions": int(np.isfinite(panel.closes[:, i]).sum()),
                         "eligible_signal_dates": int(features.eligible[:, i].sum()),
                         "first_eligible_signal": str(panel.dates[np.flatnonzero(features.eligible[:, i])[0]].date())
                         if features.eligible[:, i].any() else None})
    pd.DataFrame(coverage).to_csv(output / "eligibility_audit.csv", index=False)
    manifest = {"study_type": "stock_statarb", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "synthetic": synthetic, "data_sha256": key["data_sha256"], "universe_sha256": key["universe_sha256"],
                "data_path": str(data_path.resolve()), "universe_path": str(universe_path.resolve()),
                "config": config.to_dict(), "methods": list(STOCK_METHODS), "n_requested_stocks": len(panel.symbols),
                "factor_snapshots": len(features.snapshots), "pooled_model_refits": len(decisions),
                "forecast_start": str(panel.dates[start].date()), "latest_signal_close": str(panel.dates[-1].date()),
                "last_backtest_exit": str(ledgers[STOCK_METHODS[0]]["exit_date"].iloc[-1].date()),
                "point_in_time_universe": False, "historical_short_availability_known": False,
                "selection_caveat": "Current survivors and current sectors; historical scores are provisional",
                "evaluation_caveat": "2024+ reserved for this stock specification; broad market context already seen in ETF research",
                "cash_policy": "Only unused original equity after funding long holdings earns BIL; short proceeds earn zero",
                "max_constraint_error": float(checks["constraint_error"].max()),
                "max_absolute_estimated_beta": float(checks["estimated_beta"].abs().max()),
                "source_hashes": {path.name: file_digest(path) for path in source_dir.glob("stock_*.py")},
                "versions": {name: importlib.metadata.version(name) for name in
                             ("numpy", "pandas", "scipy", "scikit-learn", "osqp", "yfinance")}}
    protocol = source_dir.parents[1] / "docs" / "STOCK_PROTOCOL.md"
    if protocol.exists():
        shutil.copyfile(protocol, output / "protocol_snapshot.md")
        manifest["protocol_sha256"] = file_digest(protocol)
    source_metadata = data_path.parent / "metadata.json"
    if source_metadata.exists():
        shutil.copyfile(source_metadata, output / "data_metadata.json")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    from .stock_report import create_stock_report
    create_stock_report(output, metrics, ledgers, checks, intervals, scores, manifest)
    progress(metrics.loc[(metrics["period"] == "full") & (metrics["cost_bps"] == config.cost_bps)
                          & (metrics["borrow_rate"].isin([0, config.borrow_rate])),
                          ["method", "sharpe_excess", "cagr_net", "max_drawdown", "average_gross_exposure"]].to_string(index=False))
    progress(f"Report: {(output / 'report.html').resolve()}")
    return metrics, manifest

