"""Unlevered multi-asset accounting, excess Sharpe, and paired block uncertainty."""

import numpy as np
import pandas as pd


def simulate_portfolio(weights, asset_returns, cash_returns, signal_dates, entry_dates,
                       exit_dates, cost_bps=5.0, horizon=5):
    weights = np.asarray(weights, dtype=float)
    asset_returns = np.asarray(asset_returns, dtype=float)
    cash_returns = np.asarray(cash_returns, dtype=float)
    if (weights.shape != asset_returns.shape or weights.ndim != 2 or not len(weights)
            or not np.isfinite(weights).all() or not np.isfinite(asset_returns).all()):
        raise ValueError("Weights and asset returns must be matching finite nonempty matrices")
    if np.any(weights < -1e-12) or not np.allclose(weights.sum(axis=1), 1, atol=1e-10):
        raise ValueError("Long-only portfolio weights, including cash proxy, must sum to one")
    if np.any(asset_returns <= -1) or not 0 <= cost_bps < 1000:
        raise ValueError("Invalid asset returns or transaction cost")
    if cash_returns.shape != (len(weights),) or not np.isfinite(cash_returns).all():
        raise ValueError("Cash return vector must match the trading periods")
    entry_dates, exit_dates = pd.DatetimeIndex(entry_dates), pd.DatetimeIndex(exit_dates)
    if len(weights) > 1 and np.any(entry_dates[1:] < exit_dates[:-1]):
        raise ValueError("Portfolio holding intervals overlap")
    cost = cost_bps / 10_000
    previous = np.zeros(weights.shape[1])
    equity = 1.0
    rows = []
    for i, target in enumerate(weights):
        # After-fee NAV q solves q = 1 - cost * sum(abs(q * target - pretrade_weights)).
        # This fixed point accounts for fees and drift without borrowing or outside cash.
        q = 1.0
        for _ in range(20):
            q = 1 - cost * np.abs(q * target - previous).sum()
        turnover = float(np.abs(q * target - previous).sum())
        gross = float(target @ asset_returns[i])
        net_factor = q * (1 + gross)
        liquidation = 1.0 if i == len(weights) - 1 else 0.0
        if liquidation:
            net_factor *= 1 - cost
        equity *= net_factor
        rows.append({"signal_date": signal_dates[i], "entry_date": entry_dates[i],
                     "exit_date": exit_dates[i], "gross_return": gross,
                     "net_return": net_factor - 1, "cash_return": cash_returns[i],
                     "excess_return": net_factor - 1 - cash_returns[i],
                     "equity_net": equity, "turnover": turnover,
                     "liquidation_turnover": liquidation, "fee_fraction_at_entry": 1 - q,
                     "risky_weight": 1 - target[-1]})
        previous = target * (1 + asset_returns[i]) / (1 + gross)
    return pd.DataFrame(rows).set_index("signal_date")


def portfolio_metrics(ledger: pd.DataFrame, horizon=5):
    if len(ledger) < 2:
        raise ValueError("Need at least two periods for portfolio metrics")
    net = ledger["net_return"].to_numpy()
    excess = ledger["excess_return"].to_numpy()
    equity = np.cumprod(1 + net)
    peak = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    periods = 252 / horizon
    excess_std, total_std = np.std(excess, ddof=1), np.std(net, ddof=1)
    return {"n_periods": len(ledger),
            "sharpe_excess": float(np.mean(excess) / excess_std * np.sqrt(periods)) if excess_std > 1e-12 else None,
            "sharpe_zero_cash": float(np.mean(net) / total_std * np.sqrt(periods)) if total_std > 1e-12 else None,
            "cagr_net": float(equity[-1] ** (periods / len(net)) - 1),
            "volatility_net": float(total_std * np.sqrt(periods)),
            "max_drawdown": float(np.min(equity / peak - 1)),
            "average_risky_weight": float(ledger["risky_weight"].mean()),
            "one_way_turnover": float((ledger["turnover"] + ledger["liquidation_turnover"]).sum())}


def paired_sharpe_intervals(ledgers: dict, references=("buy_hold", "original_hmm"),
                            horizon=5, replications=2000, block_size=13, seed=42):
    """Pointwise paired circular moving-block intervals, not selection-adjusted tests."""
    names = list(ledgers)
    lengths = {len(ledger) for ledger in ledgers.values()}
    if len(lengths) != 1:
        raise ValueError("Bootstrap ledgers must cover identical periods")
    first = ledgers[names[0]]
    if any(not item.index.equals(first.index) for item in ledgers.values()):
        raise ValueError("Bootstrap dates must align")
    n = len(first)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(replications, int(np.ceil(n / block_size))))
    indices = ((starts[:, :, None] + np.arange(block_size)) % n).reshape(replications, -1)[:, :n]
    bootstraps, observed = {}, {}
    for name, ledger in ledgers.items():
        excess = ledger["excess_return"].to_numpy()
        samples = excess[indices]
        std = samples.std(axis=1, ddof=1)
        bootstraps[name] = np.divide(samples.mean(axis=1) * np.sqrt(252 / horizon), std,
                                    out=np.full(replications, np.nan), where=std > 1e-12)
        observed[name] = portfolio_metrics(ledger, horizon)["sharpe_excess"]
    rows = []
    for name in names:
        for reference in references:
            if name == reference:
                continue
            delta = bootstraps[name] - bootstraps[reference]
            finite = delta[np.isfinite(delta)]
            if len(finite) < replications * 0.95:
                raise ValueError("Too many degenerate bootstrap replications")
            lower, upper = np.quantile(finite, [0.025, 0.975])
            rows.append({"strategy": name, "reference": reference,
                         "sharpe_difference": observed[name] - observed[reference],
                         "ci_95_lower": float(lower), "ci_95_upper": float(upper),
                         "replications": replications, "block_size_periods": block_size})
    return pd.DataFrame(rows)

