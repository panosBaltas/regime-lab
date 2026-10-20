"""Causal, quantity-based spot/perpetual funding carry."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .carry_data import SYMBOLS
from .data import file_digest


@dataclass(frozen=True)
class CarrySpec:
    window: int = 7
    rebalance: int = 30
    gate: str = "positive"
    allocation: float = 0.7
    spot_cost: float = 0.0012
    future_cost: float = 0.0007
    delay_hours: int = 1
    maintenance: float = 0.1
    guard: float = 0.3

    @property
    def name(self):
        return f"{self.gate}_funding{self.window}_hold{self.rebalance}"


def carry_catalog():
    family = [CarrySpec(window, rebalance, gate) for window in (3, 7, 30)
              for rebalance in (7, 30) for gate in ("positive", "net_cost")]
    family += [CarrySpec(30, rebalance, "passive") for rebalance in (7, 30)]
    family += [CarrySpec(window, rebalance, "ou") for window in (7, 30) for rebalance in (7, 30)]
    return family


@dataclass
class CarryMarket:
    times: pd.DatetimeIndex
    spot: np.ndarray
    future: np.ndarray
    mark: np.ndarray
    mark_close: np.ndarray
    high: np.ndarray
    basis_close: np.ndarray
    rate: np.ndarray
    funding_cash: np.ndarray
    tradable: np.ndarray
    cash_index: np.ndarray
    cash_yield: np.ndarray
    usd: np.ndarray
    audit: dict

    def before(self, timestamp):
        keep = self.times < pd.Timestamp(timestamp)
        return CarryMarket(self.times[keep], *(getattr(self, name)[keep] for name in
                           ("spot", "future", "mark", "mark_close", "high", "basis_close", "rate", "funding_cash", "tradable",
                            "cash_index", "cash_yield", "usd")), self.audit)


def read_frame(path):
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def load_carry_market(directory, cash_path):
    directory = Path(directory)
    frames = {(symbol, kind): read_frame(directory / f"{symbol}_{kind}.csv")
              for symbol in SYMBOLS for kind in ("spot", "perpetual", "mark", "funding", "settlements")}
    first = max(frames[symbol, kind].index.min() for symbol in SYMBOLS for kind in ("spot", "perpetual", "mark"))
    last = min(frames[symbol, kind].index.max() for symbol in SYMBOLS for kind in ("spot", "perpetual", "mark"))
    times = pd.date_range(first, last, freq="h", name="time")
    values = {name: [] for name in ("spot", "future", "mark", "mark_close", "high", "basis_close", "rate", "funding_cash", "tradable")}
    audit = {"assets": {}, "price_gaps_are_halts_not_fills": True}
    for symbol in SYMBOLS:
        sources = {kind: frames[symbol, kind].reindex(times) for kind in ("spot", "perpetual", "mark")}
        observed = {kind: sources[kind].open.notna().to_numpy() for kind in sources}
        # A venue halt has no executable open. Carry the last completed close for
        # intermediate valuation only, flag it, and prohibit trades at that hour.
        marks = {}
        for kind, frame in sources.items():
            stale = frame.close.ffill().shift(1)
            marks[kind] = frame.open.fillna(stale).to_numpy()
            if not np.isfinite(marks[kind]).all():
                raise ValueError(f"Unmarkable boundary for {symbol} {kind}")
        archive, actual = frames[symbol, "funding"], frames[symbol, "settlements"]
        merged = archive.join(actual, how="left", lsuffix="_archive", rsuffix="_api")
        if merged.rate_api.isna().any() or not np.allclose(merged.rate_archive, merged.rate_api, atol=1e-12):
            raise ValueError(f"API and archive funding mismatch: {symbol}")
        hours = merged.index.floor("h")
        if hours.has_duplicates or np.max(np.abs((merged.index - hours).total_seconds())) > 60:
            raise ValueError("Unexpected funding timestamp schedule")
        positions = times.get_indexer(hours)
        good = positions >= 0
        mark_api = merged["mark"].to_numpy()[good]
        fallback = ~np.isfinite(mark_api) | (mark_api <= 0)
        event_marks = np.where(fallback, marks["mark"][positions[good]], mark_api)
        rate = np.zeros(len(times))
        funding_cash = np.zeros(len(times))
        rate[positions[good]] = merged.rate_archive.to_numpy()[good]
        funding_cash[positions[good]] = rate[positions[good]] * event_marks
        values["spot"].append(marks["spot"])
        values["future"].append(marks["perpetual"])
        values["mark"].append(marks["mark"])
        values["mark_close"].append(sources["mark"].close.to_numpy())
        values["high"].append(sources["mark"].high.fillna(sources["perpetual"].high).fillna(pd.Series(marks["mark"], index=times)).to_numpy())
        values["basis_close"].append(np.log(sources["mark"].close / sources["spot"].close).to_numpy())
        values["rate"].append(rate)
        values["funding_cash"].append(funding_cash)
        values["tradable"].append(observed["spot"] & observed["perpetual"] & observed["mark"])
        audit["assets"][symbol] = {"missing_hours": {kind: int((~observed[kind]).sum()) for kind in observed},
                                  "funding_events": int(good.sum()), "api_mark_fallback_events": int(fallback.sum())}
    cash = pd.read_csv(cash_path, parse_dates=["date"])
    if "symbol" in cash:
        cash = cash[cash.symbol == "BIL"]
    index = pd.DatetimeIndex(cash.date) + pd.Timedelta(hours=14, minutes=30)
    cash_series = pd.Series(cash.open.to_numpy(), index=index).sort_index()
    grid = cash_series.index.union(times).sort_values()
    # Ex-post interpolation is used only as the matched benchmark, never as a signal.
    cash_index = np.exp(np.log(cash_series).reindex(grid).interpolate(method="time")).reindex(times).to_numpy()
    yearly = np.log(cash_series / cash_series.shift(21)) / ((cash_series.index.to_series() - cash_series.index.to_series().shift(21)).dt.total_seconds() / (365.25 * 86400))
    cash_yield = yearly.reindex(grid).ffill().reindex(times).to_numpy()
    if not np.isfinite(np.r_[cash_index, cash_yield]).all():
        raise ValueError("Cash history does not cover the crypto study and lookback")
    stable = read_frame(directory / "usdt_usd.csv").open
    usd = stable.reindex(stable.index.union(times)).ffill().reindex(times).to_numpy()
    if not np.isfinite(usd).all() or (usd <= 0).any():
        raise ValueError("USDT/USD observations do not cover the study")
    audit["cash_proxy"] = "BIL adjusted opens, calendar log interpolation for benchmark; past 21-session yield for decisions"
    return CarryMarket(times, *(np.column_stack(values[name]) for name in
                       ("spot", "future", "mark", "mark_close", "high", "basis_close", "rate", "funding_cash", "tradable")),
                       cash_index, cash_yield, usd, audit)


def carry_signals(market, spec):
    rates = pd.DataFrame(market.rate, index=market.times)
    apr = rates.rolling(spec.window * 24, min_periods=spec.window * 24).sum().shift(1) * 365 / spec.window
    expected = apr.to_numpy() * spec.rebalance / 365
    hurdle = np.zeros_like(expected)
    if spec.gate in ("net_cost", "ou"):
        hurdle += 2 * (spec.spot_cost + spec.future_cost)
        hurdle += market.cash_yield[:, None] * spec.rebalance / 365 / spec.allocation
    if spec.gate == "ou":
        basis = pd.DataFrame(market.basis_close)
        lag = basis.shift(1)
        window = 30 * 24
        covariance = basis.rolling(window, min_periods=window).cov(lag, pairwise=False)
        variance = lag.rolling(window, min_periods=window).var()
        rho = covariance / variance
        theta = (basis.rolling(window).mean() - rho * lag.rolling(window).mean()) / (1 - rho)
        valid = (rho > 0) & (rho < 1)
        forecast = ((basis - theta) * (1 - rho.pow(spec.rebalance * 24))).where(valid, 0).shift(1)
        expected += forecast.to_numpy()
    active = expected > hurdle
    if spec.gate == "passive":
        active[:] = True
    return active & np.isfinite(apr.to_numpy()), apr.to_numpy()


def simulate_carry(market: CarryMarket, spec: CarrySpec, start="2020-02-02", stop=None):
    if spec.gate not in ("positive", "net_cost", "ou", "passive") or not 0 < spec.allocation <= 0.7:
        raise ValueError("Invalid carry specification")
    active, apr = carry_signals(market, spec)
    count = len(market.times)
    first = int(market.times.searchsorted(pd.Timestamp(start)))
    last = count - 1 if stop is None else min(count - 1, int(market.times.searchsorted(pd.Timestamp(stop))) - 1)
    last -= market.times[last].hour
    quantity = np.zeros(market.spot.shape[1])
    cash, previous_nav = 1.0, 1.0
    previous_day_index = first
    pending, decisions, daily, trades = {}, [], [], []
    previous_mark = market.mark[first].copy()
    minimum_margin = np.inf
    margin_breaches, stale_valuation_hours = 0, 0
    day_funding = day_fees = day_turnover = 0.0
    day_risk_trades = 0
    previous_close_margin = np.inf
    for t in range(first, last + 1):
        clock = market.times[t]
        cash -= float(quantity @ (market.mark[t] - previous_mark))
        funding = float(quantity @ market.funding_cash[t])
        cash += funding
        day_funding += funding
        nav_before = float(cash + quantity @ market.spot[t])
        if nav_before <= 0:
            raise ValueError("Carry equity exhausted")
        # Funding is booked to OLD quantity before any fill at this timestamp.
        if clock.hour == 0 and (clock.normalize() - pd.Timestamp(start).normalize()).days % spec.rebalance == 0:
            target = active[t].astype(float)
            target *= spec.allocation / max(1, target.sum())
            pending[t + spec.delay_hours] = (target, False, t)
            decisions.append({"signal_time": str(clock), "latest_funding_allowed": str(clock - pd.Timedelta(nanoseconds=1)),
                              "expected_apr": apr[t].tolist(), "target_weights": target.tolist()})
        if previous_close_margin < spec.guard and np.any(quantity):
            # Known risk at the previous completed hour triggers a next-open reduction.
            invested = float(quantity @ market.spot[t])
            scale = min(1.0, 0.65 * nav_before / max(invested, 1e-12))
            risk_quantity = quantity * scale
        else:
            risk_quantity = None
        if t in pending or risk_quantity is not None:
            if t in pending:
                weights, risk_trade, signal_t = pending.pop(t)
                desired = nav_before * weights / market.spot[t]
            else:
                desired, risk_trade, signal_t = risk_quantity, True, t - 1
            if risk_quantity is not None:
                desired = np.minimum(desired, risk_quantity)
                risk_trade = True
            available = market.tradable[t]
            desired = np.where(available, desired, quantity)
            if not available.all() and not risk_trade:
                pending[t + 1] = (weights, risk_trade, signal_t)
            # Self-financing targets solve for post-fee NAV, including the execution/mark spread.
            if not risk_trade:
                for _ in range(12):
                    delta = desired - quantity
                    fee = float(np.abs(delta) @ (spec.spot_cost * market.spot[t] + spec.future_cost * market.future[t]))
                    new_nav = nav_before - fee + float(delta @ (market.future[t] - market.mark[t]))
                    desired = np.where(available, new_nav * weights / market.spot[t], quantity)
            delta = desired - quantity
            fee = float(np.abs(delta) @ (spec.spot_cost * market.spot[t] + spec.future_cost * market.future[t]))
            cash += float(delta @ (market.future[t] - market.mark[t])) - float(delta @ market.spot[t]) - fee
            quantity = desired
            day_fees += fee
            day_turnover += float(np.abs(delta) @ (market.spot[t] + market.future[t]))
            day_risk_trades += int(risk_trade)
            trades.append({"fill_time": str(clock), "signal_time": str(market.times[signal_t]), "risk_trade": risk_trade,
                           "fee": fee, "cash": cash, **{f"quantity_{symbol}": quantity[i] for i, symbol in enumerate(SYMBOLS)}})
        # Last mark is observable; intrahour highs are risk audits, never trading signals.
        notional = float(quantity @ market.mark[t])
        worst_cash = cash - float(quantity @ (market.high[t] - market.mark[t]))
        worst_notional = float(quantity @ market.high[t])
        margin = worst_cash / worst_notional if worst_notional > 1e-12 else np.inf
        minimum_margin = min(minimum_margin, margin)
        margin_breaches += int(margin < spec.maintenance)
        stale_valuation_hours += int(np.any((quantity > 0) & ~market.tradable[t]))
        close_cash = cash - float(quantity @ (market.mark_close[t] - market.mark[t]))
        close_notional = float(quantity @ market.mark_close[t])
        previous_close_margin = close_cash / close_notional if close_notional > 1e-12 else np.inf
        nav = float(cash + quantity @ market.spot[t])
        if clock.hour == 0 and t > first:
            if np.any((quantity > 0) & ~market.tradable[t]):
                raise ValueError(f"Daily NAV has stale held prices at {clock}")
            net = nav / previous_nav - 1
            rf = market.cash_index[t] / market.cash_index[previous_day_index] - 1
            usd_return = (1 + net) * market.usd[t] / market.usd[previous_day_index] - 1
            daily.append({"time": clock, "net_return": net, "usd_return": usd_return, "cash_return": rf,
                          "excess_return": usd_return - rf, "equity_quote": nav, "spot_fraction": float(quantity @ market.spot[t]) / nav,
                          "funding": day_funding / previous_nav, "fees": day_fees / previous_nav,
                          "turnover": day_turnover / previous_nav, "cash": cash, "margin_ratio": margin,
                          "risk_trades": day_risk_trades})
            previous_nav, previous_day_index = nav, t
            day_funding = day_fees = day_turnover = 0.0
            day_risk_trades = 0
        previous_mark = market.mark[t]
    ledger = pd.DataFrame(daily).set_index("time")
    if len(ledger) < 2:
        raise ValueError("Insufficient carry evaluation days")
    # Liquidate at the last REPORTED timestamp, not at an unreported later mark.
    end_t = int(market.times.get_loc(ledger.index[-1]))
    if end_t != last:
        raise ValueError("Final valuation must match final daily timestamp")
    closing_fee = float(quantity @ (spec.spot_cost * market.spot[end_t] + spec.future_cost * market.future[end_t]))
    closing_spread = float(quantity @ (market.future[end_t] - market.mark[end_t]))
    final_nav = nav - closing_fee - closing_spread
    beginning_nav = nav / (1 + ledger.net_return.iloc[-1])
    fx_ratio = (1 + ledger.usd_return.iloc[-1]) / (1 + ledger.net_return.iloc[-1])
    ledger.loc[ledger.index[-1], "net_return"] = final_nav / beginning_nav - 1
    ledger.loc[ledger.index[-1], "usd_return"] = (1 + ledger.net_return.iloc[-1]) * fx_ratio - 1
    ledger.loc[ledger.index[-1], "excess_return"] = ledger.usd_return.iloc[-1] - ledger.cash_return.iloc[-1]
    ledger.loc[ledger.index[-1], "equity_quote"] = final_nav
    ledger.loc[ledger.index[-1], "fees"] += closing_fee / beginning_nav
    info = {"spec": asdict(spec), "minimum_intrahour_margin_ratio": minimum_margin,
            "maintenance_breaches": margin_breaches, "stale_intermediate_valuation_hours": stale_valuation_hours,
            "n_rebalances": len(trades), "decisions": decisions}
    return ledger, pd.DataFrame(trades), info


def carry_metrics(ledger):
    net, excess = ledger.usd_return.to_numpy(), ledger.excess_return.to_numpy()
    wealth = np.cumprod(1 + net)
    peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    return {"days": len(net), "sharpe_excess": float(excess.mean() / excess.std(ddof=1) * np.sqrt(365)) if excess.std(ddof=1) > 1e-10 else None,
            "sharpe_zero": float(net.mean() / net.std(ddof=1) * np.sqrt(365)) if net.std(ddof=1) > 1e-10 else None,
            "cagr": float(wealth[-1] ** (365 / len(net)) - 1), "volatility": float(net.std(ddof=1) * np.sqrt(365)),
            "drawdown": float(np.min(wealth / peak - 1)), "average_spot_fraction": float(ledger.spot_fraction.mean()),
            "annual_fees_mean": float(ledger.fees.mean() * 365), "annual_funding_mean": float(ledger.funding.mean() * 365)}


def run_carry_study(directory, cash_path, output, progress=print):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    market = load_carry_market(directory, cash_path)
    (output / "data_audit.json").write_text(json.dumps(market.audit, indent=2), encoding="utf-8")
    development_market = market.before("2024-01-01")
    catalog = carry_catalog()
    rows, eligible = [], []
    for spec in catalog:
        ledger, _, info = simulate_carry(development_market, spec)
        metrics = {}
        for period, mask in (("development", ledger.index < "2023-01-01"),
                             ("validation", ledger.index >= "2023-01-01")):
            metrics[period] = carry_metrics(ledger.loc[mask])
            rows.append({"method": spec.name, "period": period, **metrics[period]})
        valid = info["maintenance_breaches"] == 0 and all(metrics[p]["average_spot_fraction"] >= 0.1 for p in metrics)
        score = min(metrics[p]["sharpe_excess"] if metrics[p]["sharpe_excess"] is not None else -1e6 for p in metrics)
        if valid:
            eligible.append((score, spec))
        progress(f"Carry development: {spec.name} | development {metrics['development']['sharpe_excess']:.2f}, validation {metrics['validation']['sharpe_excess']:.2f}, breaches {info['maintenance_breaches']}")
    if not eligible:
        raise ValueError("No active carry specification passed margin validation")
    selected = max(eligible, key=lambda item: item[0])[1]
    selection = {"selected": selected.name, "spec": asdict(selected), "criterion": "maximum worst development/validation excess Sharpe; active >=10%; zero maintenance breaches",
                 "latest_selection_data": "2023-12-31", "test_begin": "2024-01-01",
                 "source_sha256": file_digest(__file__), "data_hashes": {p.name: file_digest(p) for p in Path(directory).glob("*.csv")}}
    # This artifact is written BEFORE any test-period strategy is simulated.
    (output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output / "development_catalog.csv", index=False)
    progress(f"FROZEN SELECTION: {selected.name}; evaluation scores have not been loaded")
    ledgers, diagnostics = {}, []
    for spec in catalog:
        ledger, trades, info = simulate_carry(market, spec)
        ledgers[spec.name] = ledger
        for period, mask in (("full", np.ones(len(ledger), bool)), ("evaluation", ledger.index >= "2024-01-01")):
            rows.append({"method": spec.name, "period": period, **carry_metrics(ledger.loc[mask])})
        diagnostics.append({"method": spec.name, **{k: v for k, v in info.items() if k not in ("decisions", "spec")}})
        if spec.name == selected.name:
            trades.to_csv(output / "selected_trades.csv", index=False)
            (output / "selected_decisions.json").write_text(json.dumps(info["decisions"], indent=2), encoding="utf-8")
            for label, alternative in (("double_cost", replace(spec, spot_cost=2*spec.spot_cost, future_cost=2*spec.future_cost)),
                                       ("four_hour_delay", replace(spec, delay_hours=4))):
                stressed, _, stressed_info = simulate_carry(market, alternative)
                ledgers[label] = stressed
                rows.append({"method": label, "period": "evaluation", **carry_metrics(stressed.loc[stressed.index >= "2024-01-01"])})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "margin_audit.csv", index=False)
    pd.concat(ledgers, names=["method"]).to_csv(output / "daily_returns.csv")
    progress(metrics[(metrics.period == "evaluation") & metrics.method.isin([selected.name, "double_cost", "four_hour_delay"])].to_string(index=False))
    return metrics, selection
