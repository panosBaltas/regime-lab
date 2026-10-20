import numpy as np
import pandas as pd

from .backtest import MODEL_NAMES
from .config import Config


def forecast_metrics(predictions: pd.DataFrame, horizon: int) -> pd.DataFrame:
    rows = []
    for sample, frame in (("daily_overlapping", predictions),
                          ("nonoverlapping", predictions.iloc[::horizon])):
        actual = frame["target_return"].to_numpy()
        benchmark_error = np.square(actual - frame["historical_mean"].to_numpy()).sum()
        for name in MODEL_NAMES:
            forecast = frame[name].to_numpy()
            error = actual - forecast
            correlation = (float(np.corrcoef(actual, forecast)[0, 1])
                           if np.std(actual) > 0 and np.std(forecast) > 1e-12 else None)
            rows.append({
                "sample": sample, "model": name, "n": len(frame),
                "rmse": float(np.sqrt(np.square(error).mean())),
                "mae": float(np.abs(error).mean()),
                "oos_r2_vs_mean": float(1 - np.square(error).sum() / benchmark_error)
                if benchmark_error > 0 else None,
                "directional_accuracy": float(((forecast > 0) == (actual > 0)).mean()),
                "return_correlation": correlation,
            })
    return pd.DataFrame(rows)


def strategy_results(predictions: pd.DataFrame, config: Config):
    """Single nonoverlapping long/cash sleeve, next-open execution, round-trip costs."""
    frame = predictions.iloc[::config.horizon]
    if len(frame) > 1 and (frame["entry_date"].iloc[1:].to_numpy()
                           < frame["exit_date"].iloc[:-1].to_numpy()).any():
        raise ValueError("Trading intervals overlap")
    actual = frame["target_return"].to_numpy()
    cost = config.cost_bps / 10_000
    ledgers, metrics = [], []
    for name in (*MODEL_NAMES, "buy_and_hold"):
        position = np.ones(len(frame)) if name == "buy_and_hold" else (frame[name].to_numpy() > 0).astype(float)
        turnover = np.abs(position - np.r_[0.0, position[:-1]])
        gross = position * actual
        net = (1 - cost * turnover) * (1 + gross) - 1
        net[-1] = (1 + net[-1]) * (1 - cost * position[-1]) - 1
        equity = np.cumprod(1 + net)
        running_peak = np.maximum.accumulate(np.r_[1.0, equity])[1:]
        periods_per_year = 252 / config.horizon
        std = np.std(net, ddof=1) if len(net) > 1 else 0
        metrics.append({
            "model": name, "n_trades": len(frame), "total_return_net": float(equity[-1] - 1),
            "cagr_net": float(equity[-1] ** (periods_per_year / len(net)) - 1),
            "annualized_volatility_net": float(std * np.sqrt(periods_per_year)),
            "sharpe_net_zero_cash_rate": float(np.mean(net) / std * np.sqrt(periods_per_year))
            if std > 0 else None,
            "max_drawdown_net": float(np.min(equity / running_peak - 1)),
            "fraction_invested": float(position.mean()),
            "one_way_turnover": float(turnover.sum() + position[-1]),
        })
        ledger = frame[["entry_date", "exit_date", "target_return"]].copy()
        ledger["model"] = name
        ledger["forecast"] = np.nan if name == "buy_and_hold" else frame[name]
        ledger["position"] = position
        ledger["entry_turnover"] = turnover
        ledger["exit_liquidation_turnover"] = np.r_[np.zeros(len(frame) - 1), position[-1]]
        ledger["gross_return"] = gross
        ledger["net_return"] = net
        ledger["equity_net"] = equity
        ledgers.append(ledger)
    return pd.DataFrame(metrics), pd.concat(ledgers)

