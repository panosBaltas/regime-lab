"""Constrained stock holdings and actual long/short accounting."""

import numpy as np
import pandas as pd
from scipy import sparse

from .stock_features import StockConfig, StockFeatures, StockPanel
from .stock_models import STOCK_METHODS


class StockOptimizer:
    def __init__(self, constraints, variance, config: StockConfig):
        import osqp
        self.config, self.constraints = config, np.asarray(constraints)
        n = len(variance)
        self.n = n
        eye, zero = sparse.eye(n, format="csc"), sparse.csc_matrix((n, n))
        # Variables: signed stock weights w, absolute trades t, absolute holdings g.
        rows = [sparse.hstack([sparse.csc_matrix(self.constraints), sparse.csc_matrix((len(self.constraints), 2 * n))]),
                sparse.hstack([eye, zero, zero]),
                sparse.hstack([eye, -eye, zero]), sparse.hstack([-eye, -eye, zero]),
                sparse.hstack([eye, zero, -eye]), sparse.hstack([-eye, zero, -eye]),
                sparse.hstack([zero, eye, zero]), sparse.hstack([zero, zero, eye]),
                sparse.csc_matrix(np.r_[np.zeros(2 * n), np.ones(n)][None, :])]
        self.a = sparse.vstack(rows, format="csc")
        k = len(constraints)
        self.box = slice(k, k + n)
        self.trade_positive = slice(k + n, k + 2 * n)
        self.trade_negative = slice(k + 2 * n, k + 3 * n)
        self.lower = np.r_[np.zeros(k), np.full(n, -config.position_limit), np.full(4 * n, -np.inf),
                           np.zeros(2 * n), -np.inf]
        self.upper = np.r_[np.zeros(k), np.full(n, config.position_limit), np.zeros(4 * n),
                           np.full(2 * n, np.inf), config.gross_limit]
        scaled_variance = config.risk_aversion * np.maximum(variance, 0.005**2) * 10000
        self.p = sparse.diags(np.r_[scaled_variance, np.full(2 * n, 1e-8)], format="csc")
        self.solver = osqp.OSQP()
        self.solver.setup(P=self.p, q=np.zeros(3 * n), A=self.a, l=self.lower, u=self.upper,
                          verbose=False, eps_abs=1e-7, eps_rel=1e-7, max_iter=50000,
                          polishing=True, adaptive_rho_interval=25)

    def solve(self, expected_returns, previous, caps):
        n, config = self.n, self.config
        expected_returns = np.asarray(expected_returns)
        previous, caps = np.asarray(previous), np.asarray(caps)
        if (expected_returns.shape != (n,) or previous.shape != (n,) or caps.shape != (n,)
                or not np.isfinite(np.r_[expected_returns, previous, caps]).all() or np.any(caps < 0)):
            raise ValueError("Optimizer inputs must be finite matching vectors with nonnegative caps")
        transaction_penalty = config.cost_bps / 10000 / config.target_horizon
        q = np.r_[-expected_returns, np.full(n, transaction_penalty), np.full(n, config.borrow_rate / (2 * 252))] * 10000
        lower, upper = self.lower.copy(), self.upper.copy()
        lower[self.box], upper[self.box] = -caps, caps
        upper[self.trade_positive], upper[self.trade_negative] = previous, -previous
        self.solver.update(q=q, l=lower, u=upper)
        result = self.solver.solve(raise_error=False)
        retried = False
        if result.info.status_val not in (1, 2):
            # A deterministic cold retry changes only numerical settings, not the objective.
            self.solver.update_settings(adaptive_rho=False, rho=1.0, max_iter=100000)
            self.solver.warm_start(x=np.zeros(3 * n), y=np.zeros(self.a.shape[0]))
            result = self.solver.solve(raise_error=False)
            retried = True
            self.solver.update_settings(adaptive_rho=True, rho=0.1, max_iter=50000)
        if result.info.status_val not in (1, 2) or result.x is None:
            raise ValueError(f"Stock optimization failed: {result.info.status}; "
                             f"primal residual {result.info.prim_res:g}, dual residual {result.info.dual_res:g}")
        weights = result.x[:n].copy()
        raw_gross = float(np.abs(weights).sum())
        # Summed absolute-value slack can slightly exceed the aggregate bound.
        # A tiny uniform shrink preserves every neutrality constraint. Larger errors fail.
        gross_scale = 1.0
        if config.gross_limit < raw_gross <= config.gross_limit + 1e-4:
            gross_scale = config.gross_limit / raw_gross
            weights *= gross_scale
        error = float(np.max(np.abs(self.constraints @ weights)))
        if error > 2e-6 or np.any(np.abs(weights) > caps + 2e-6) or np.abs(weights).sum() > config.gross_limit + 2e-6:
            raise ValueError(f"Stock optimizer constraint violation: equality {error:g}, "
                             f"maximum cap excess {np.max(np.abs(weights) - caps):g}, "
                             f"gross {np.abs(weights).sum():g}; polish status {result.info.status_polish}")
        return weights, {"status": result.info.status, "iterations": int(result.info.iter),
                         "constraint_error": error, "gross": float(np.abs(weights).sum()),
                         "solver_retried": retried, "numerical_gross_scale": gross_scale,
                         "max_cap_violation": float(max(0, np.max(np.abs(weights) - caps)))}


def known_close_weights(weights, open_prices, close_prices):
    held = np.abs(weights) > 1e-8
    if not np.isfinite(open_prices[held]).all() or not np.isfinite(close_prices[held]).all():
        raise ValueError("A held stock cannot be marked at the known close")
    moves = np.zeros(len(weights))
    moves[held] = close_prices[held] / open_prices[held] - 1
    nav = 1 + weights @ moves
    if nav <= 0:
        raise ValueError("Portfolio exhausted before decision time")
    return weights * (1 + moves) / nav


def stock_weights(panel: StockPanel, features: StockFeatures, forecasts, config: StockConfig, progress=None):
    weights = np.zeros_like(forecasts)
    ready = np.isfinite(forecasts[:, :, 1]).sum(axis=1) >= max(10, config.min_stocks // 2)
    starts = np.flatnonzero(ready)
    if not len(starts):
        raise ValueError("No common stock-model evaluation start")
    start = int(starts[0])
    solvers, current_snapshot = {}, -1
    checks = []
    for t in range(start, len(panel.dates)):
        snapshot_id = features.snapshot_id[t]
        if snapshot_id < 0:
            continue
        snapshot = features.snapshots[snapshot_id]
        indices = np.asarray(snapshot["indices"])
        if snapshot_id != current_snapshot:
            solvers = {name: StockOptimizer(np.asarray(snapshot["constraints"]), np.asarray(snapshot["variance"]), config)
                       for name in STOCK_METHODS}
            current_snapshot = snapshot_id
        cap = np.minimum(config.position_limit, config.adv_participation * panel.adv[t, indices] / config.account_notional)
        cap = np.where(features.eligible[t, indices], cap, 0)
        for m, name in enumerate(STOCK_METHODS):
            previous = weights[t - 1, :, m] if t > start else np.zeros(len(panel.symbols))
            close_weights = known_close_weights(previous, panel.opens[t], panel.closes[t])
            prediction = forecasts[t, indices, m]
            valid = np.isfinite(prediction)
            current_caps = np.where(valid, cap, 0)
            try:
                chosen, detail = solvers[name].solve(np.where(valid, prediction, 0), close_weights[indices], current_caps)
            except ValueError as exc:
                raise ValueError(f"{panel.dates[t].date()} {name}: {exc}") from exc
            weights[t, indices, m] = chosen
            expected_turnover = float(np.abs(weights[t, :, m] - close_weights).sum())
            raw_factor_error = float(np.max(np.abs(np.asarray(snapshot["loadings"]).T @ chosen)))
            beta = float(np.asarray(snapshot["beta"]) @ chosen)
            detail.update(signal_date=str(panel.dates[t].date()), method=name,
                          estimated_turnover_at_close=expected_turnover, estimated_beta=beta,
                          pca_exposure_error=raw_factor_error, factor_snapshot=snapshot_id)
            checks.append(detail)
        if progress and (t - start) % 252 == 0:
            progress(f"Stock portfolios: {panel.dates[t].date()} | {len(indices)} stocks | {t - start + 1}/{len(panel.dates) - start} dates")
    return weights, start, pd.DataFrame(checks)


def simulate_stock_portfolio(panel: StockPanel, weights, start, config: StockConfig,
                             cost_bps=None, borrow_rate=None):
    cost = (config.cost_bps if cost_bps is None else cost_bps) / 10000
    borrow = config.borrow_rate if borrow_rate is None else borrow_rate
    previous = np.zeros(len(panel.symbols))
    equity = 1.0
    rows = []
    for t in range(start, len(panel.dates) - 2):
        target = weights[t].copy()
        held = (np.abs(target) > 1e-7) | (np.abs(previous) > 1e-7)
        if not np.isfinite(panel.opens[t + 1, held]).all() or not np.isfinite(panel.opens[t + 2, held]).all():
            raise ValueError(f"Held-position prices unavailable at {panel.dates[t].date()}; cannot assign a zero return")
        if np.abs(target).sum() > config.gross_limit + 3e-6 or abs(target.sum()) > 3e-6:
            raise ValueError("Target violates gross or dollar-neutrality constraints")
        q = 1.0
        for _ in range(20):
            q = 1 - cost * np.abs(q * target - previous).sum()
        turnover = float(np.abs(q * target - previous).sum())
        returns = np.zeros(len(panel.symbols))
        returns[held] = panel.opens[t + 2, held] / panel.opens[t + 1, held] - 1
        cash_return = float(panel.cash_opens[t + 2] / panel.cash_opens[t + 1] - 1)
        long = float(np.maximum(target, 0).sum())
        short = float(-np.minimum(target, 0).sum())
        year_fraction = (panel.dates[t + 2] - panel.dates[t + 1]).days / 365.25
        cash_income = max(0, 1 - long) * cash_return
        borrow_fee = short * borrow * year_fraction
        financing_fee = max(0, long - 1) * (cash_return + config.financing_spread * year_fraction)
        stock_pnl = float(target @ returns)
        gross_factor = 1 + stock_pnl + cash_income - borrow_fee - financing_fee
        if gross_factor <= 0:
            raise ValueError("Stock portfolio exhausted")
        liquidation_turnover = float(np.abs(target * (1 + returns)).sum()) if t == len(panel.dates) - 3 else 0.0
        factor = q * (gross_factor - cost * liquidation_turnover)
        equity *= factor
        rows.append({"signal_date": panel.dates[t], "entry_date": panel.dates[t + 1], "exit_date": panel.dates[t + 2],
                     "stock_pnl": stock_pnl, "cash_income": cash_income, "borrow_fee": borrow_fee,
                     "financing_fee": financing_fee, "net_return": factor - 1, "cash_return": cash_return,
                     "excess_return": factor - 1 - cash_return, "equity_net": equity,
                     "gross_exposure": long + short, "net_exposure": long - short,
                     "turnover": turnover, "liquidation_turnover": liquidation_turnover,
                     "n_long": int(np.count_nonzero(target > 1e-6)), "n_short": int(np.count_nonzero(target < -1e-6))})
        previous = target * (1 + returns) / gross_factor
    if len(rows) < 2:
        raise ValueError("Not enough daily holding periods")
    return pd.DataFrame(rows).set_index("signal_date")

