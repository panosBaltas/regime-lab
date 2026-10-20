"""A reproducible direct-account study and a small, fully offline demonstration."""
from dataclasses import asdict
from html import escape
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import file_digest
from .ensemble import allocation, load_sleeves, metrics, selected_allocator
from .execution import ExecutionEngine, ExecutionSpec, timestamp
from .macro import MacroSpec, macro_panel, macro_targets
from .provenance import now_utc, source_hashes, write_new


def selected_configuration(inverse_root, macro_root, ensemble_root):
    selections = {name: json.loads((Path(root)/"selection.json").read_text(encoding="utf-8"))
                  for name, root in (("inverse", inverse_root), ("macro", macro_root), ("ensemble", ensemble_root))}
    rule = selected_allocator(selections["ensemble"])
    if selections["inverse"]["spec"]["gate"] != "passive":
        raise ValueError("Execution v1 implements the selected passive carry control only")
    return {"allocator": rule, "carry": selections["inverse"]["spec"],
            "macro": selections["macro"]["spec"], "selections": selections}


def risk_history(inverse_root, macro_root):
    r, rf, _ = load_sleeves(inverse_root, macro_root)
    monthly = ((1+r).resample("ME").prod()-1).subtract((1+rf).resample("ME").prod()-1, axis=0)
    # Only month-end NAVs actually present in the input can be complete.
    return monthly[monthly.index <= r.index.max().normalize()]


def risk_weights(history, when, rule):
    when = timestamp(when).tz_localize(None)
    earlier = history[history.index < when.normalize().replace(day=1)]
    if rule == "static_risk":
        earlier = earlier[(earlier.index >= "2021-01-01") & (earlier.index < "2023-01-01")]
    else:
        earlier = earlier.tail(12)
    if len(earlier) < 12 or not np.isfinite(earlier.to_numpy()).all():
        raise ValueError("Twelve completed monthly observations are required")
    return allocation(earlier, rule), earlier


def exchange_time(day, hour, minute=0):
    return (pd.Timestamp(day).normalize()+pd.Timedelta(hours=hour, minutes=minute)).tz_localize(
        "America/New_York").tz_convert("UTC")


def historical_events(market, low, panel, macro_spec, history, rule, start, end):
    """Build timestamped prices/decisions, never precomputed strategy returns."""
    start, end = timestamp(start), timestamp(end)
    dates, names, opens, closes, cash_open, cash_close = panel
    targets = macro_targets(*[dates, names, closes, cash_close], macro_spec)
    initial_anchor = max(252, int(dates.searchsorted("2010-01-01")))
    events = {}
    def event(when):
        key = timestamp(when)
        return events.setdefault(key, {"time": key.isoformat(), "prices": {}, "decisions": []})
    retained = np.flatnonzero((market.times >= start.tz_localize(None)) & (market.times <= end.tz_localize(None)))
    if not len(retained):
        raise ValueError("No crypto prices in the requested period")
    for i in retained:
        when = timestamp(market.times[i])
        row = event(when)
        row["prices"].update(spot=market.spot[i].tolist(), future=market.future[i].tolist(),
                             mark=market.mark[i].tolist(), quote_usd=float(market.usd[i]))
        row["crypto_trade"] = bool(market.tradable[i].all())
        row["funding_coin_per_face"] = market.funding_cash[i].tolist()
        row["margin_bounds"] = [market.high[i].tolist(), low[i].tolist()]
        if when.hour == 0:
            if when.day == 1 or i == retained[0]:
                weights, past = risk_weights(history, when, rule)
                row["decisions"].append({"time": when.isoformat(), "latest_input_time": timestamp(past.index[-1]).isoformat(),
                                         "allocation": weights.tolist()})
            if (when.tz_localize(None).normalize()-pd.Timestamp("2020-11-02")).days % 30 == 0:
                row["decisions"].append({"time": when.isoformat(), "latest_input_time": (when-pd.Timedelta(hours=1)).isoformat(), "hedge": True})
    first = event(start)
    prior_sessions = [i for i, day in enumerate(dates) if exchange_time(day, 16) < start]
    if not prior_sessions:
        raise ValueError("Missing pre-start ETF history")
    last = prior_sessions[-1]
    first["prices"]["etf"] = {**dict(zip(names, closes[last].tolist())), "BIL": float(cash_close[last])}
    scheduled = [i for i in prior_sessions if i >= initial_anchor and (i-initial_anchor) % macro_spec.rebalance == 0]
    if not scheduled:
        raise ValueError("Missing pre-start ETF target")
    j = scheduled[-1]
    first["decisions"].append({"time": start.isoformat(), "latest_input_time": exchange_time(dates[j], 16).isoformat(),
                               "macro": dict(zip(names, targets[j].tolist()))})
    for i, day in enumerate(dates):
        op, cl = exchange_time(day, 9, 30), exchange_time(day, 16)
        if start <= op <= end:
            row = event(op)
            row["prices"]["etf"] = {**dict(zip(names, opens[i].tolist())), "BIL": float(cash_open[i])}
            row["etf_trade"] = True
        if start <= cl <= end:
            row = event(cl)
            row["prices"]["etf"] = {**dict(zip(names, closes[i].tolist())), "BIL": float(cash_close[i])}
            if i >= initial_anchor and (i-initial_anchor) % macro_spec.rebalance == 0:
                row["decisions"].append({"time": cl.isoformat(), "latest_input_time": cl.isoformat(),
                                         "macro": dict(zip(names, targets[i].tolist()))})
    return [events[key] for key in sorted(events)]


def simulate_events(events, spec):
    engine = ExecutionEngine(spec)
    decisions, holdings = [], []
    benchmark_initial = None
    for event in events:
        engine.advance(event)
        for decision in event.get("decisions", []):
            engine.plan(decision)
            decisions.append(decision)
        if timestamp(event["time"]).hour == 0 and timestamp(event["time"]).minute == 0:
            bil = engine.prices["etf"].get("BIL")
            if bil is None:
                raise ValueError("BIL must be marked before the first daily valuation")
            if benchmark_initial is None:
                benchmark_initial = bil
            holdings.append({"time": event["time"], "nav": engine.nav(), "benchmark": bil/benchmark_initial,
                             "state": engine.state()})
    daily = pd.DataFrame([{k: v for k, v in row.items() if k != "state"} for row in holdings]).set_index("time")
    daily.index = pd.DatetimeIndex(daily.index).tz_convert(None)
    daily["net_return"] = daily.nav.pct_change(fill_method=None)
    daily["cash_return"] = daily.benchmark.pct_change(fill_method=None)
    daily["excess_return"] = daily.net_return-daily.cash_return
    daily.iloc[0, daily.columns.get_indexer(["net_return", "cash_return", "excess_return"])] = 0.
    return engine, daily, decisions, holdings


def complete_monthly_excess(ledger):
    """Midnight NAV at consecutive month starts spans one COMPLETE calendar month."""
    boundaries = ledger[(ledger.index.day == 1) & (ledger.index.hour == 0)]
    if len(boundaries) < 2:
        return pd.Series(dtype=float)
    grid = pd.date_range(boundaries.index.min(), boundaries.index.max(), freq="MS")
    anchored = boundaries[["nav", "benchmark"]].reindex(grid)
    changes = anchored.pct_change(fill_method=None)
    return (changes.nav-changes.benchmark).dropna()


def account_metrics(ledger):
    # First NAV is the starting boundary, not an additional observed return.
    returns = ledger.copy()
    returns["net_return"] = returns.nav.pct_change(fill_method=None)
    returns["cash_return"] = returns.benchmark.pct_change(fill_method=None)
    returns["excess_return"] = returns.net_return-returns.cash_return
    result = metrics(returns.iloc[1:])
    monthly = complete_monthly_excess(ledger)
    sigma = monthly.std(ddof=1)
    result["monthly_excess_sharpe"] = float(monthly.mean()/sigma*np.sqrt(12)) if len(monthly) > 1 and sigma > 0 else float("nan")
    result["complete_months"] = len(monthly)
    return result


def monthly_interval(ledger, replications=5000):
    values = complete_monthly_excess(ledger).to_numpy()
    if len(values) < 12:
        return {"months": len(values), "interval": None, "reason": "Fewer than 12 complete months"}
    rng = np.random.default_rng(42)
    starts = rng.integers(0, len(values), (replications, int(np.ceil(len(values)/3))))
    indices = ((starts[:, :, None]+np.arange(3)) % len(values)).reshape(replications, -1)[:, :len(values)]
    samples = values[indices]
    score = samples.mean(axis=1)/np.maximum(samples.std(axis=1, ddof=1), 1e-15)*np.sqrt(12)
    return {"months": len(values), "interval": np.quantile(score, [.025, .975]).tolist(),
            "block_months": 3, "replications": replications, "selection_adjusted": False}


def _write_study(output, studies, settings, synthetic):
    output = Path(output)
    # Never destroy an earlier experiment by reusing its directory.
    write_new(output/"run.json", {"created_at": now_utc(), "synthetic": synthetic,
                                "status": "synthetic accounting demonstration" if synthetic else "exploratory historical replay",
                                "settings": settings, "sources": source_hashes()})
    rows, ledgers, attribution, audits, intervals = [], {}, [], {}, {}
    for name, (engine, ledger, decisions, holdings) in studies.items():
        scoring = ledger if synthetic else ledger[ledger.index >= "2024-01-01"]
        rows.append({"method": name, **account_metrics(scoring)})
        intervals[name] = monthly_interval(scoring)
        ledgers[name] = ledger
        events = pd.DataFrame(engine.daily).set_index("time")
        events.to_csv(output/f"{name}_events.csv")
        pd.DataFrame(engine.blotter).to_csv(output/f"{name}_orders.csv", index=False)
        write_new(output/f"{name}_decisions.json", decisions)
        write_new(output/f"{name}_holdings.json", holdings)
        components = [c for c in events if c.startswith("pnl_")]+["fees_usd"]
        attribution.append({"method": name, **{c: float(events[c].sum()) for c in components}})
        audits[name] = {"max_event_reconciliation_error_usd": float(events.reconciliation_error.abs().max()),
                        "nav_from_attribution": float(engine.spec.capital+sum(events[c].sum() for c in components if c != "fees_usd")-events.fees_usd.sum()),
                        "ending_nav": engine.nav(), "minimum_margin": engine.minimum_margin,
                        "open_positions_at_end": True, "events": len(events), "fills_and_transfers": len(engine.blotter)}
    pd.concat(ledgers, names=["method"]).to_csv(output/"daily_nav.csv")
    pd.DataFrame(attribution).to_csv(output/"pnl_attribution.csv", index=False)
    summary = pd.DataFrame(rows).set_index("method")
    summary.to_csv(output/"metrics.csv")
    yearly = []
    for name, ledger in ledgers.items():
        for year in sorted(set(ledger.index.year)):
            group = ledger[(ledger.index >= f"{year}-01-01") & (ledger.index <= f"{year+1}-01-01")]
            if len(group) > 2:
                yearly.append({"method": name, "year": int(year), **account_metrics(group)})
    pd.DataFrame(yearly).to_csv(output/"yearly_metrics.csv", index=False)
    write_new(output/"reconciliation.json", audits)
    write_new(output/"uncertainty.json", intervals)
    _report(output, summary, ledgers, attribution, synthetic, settings)
    return summary, audits


def _report(output, summary, ledgers, attribution, synthetic, settings):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 3.3))
    for name, ledger in ledgers.items():
        ax.plot(ledger.index, ledger.nav/ledger.nav.iloc[0], label=name)
    first = next(iter(ledgers.values()))
    ax.plot(first.index, first.benchmark/first.benchmark.iloc[0], label="BIL total-return benchmark", color="#94a3b8", ls="--")
    ax.legend(frameon=False, fontsize=8); ax.set_ylabel("Marked wealth")
    fig.tight_layout(); fig.savefig(output/"wealth.png", dpi=140); plt.close(fig)
    picture = base64.b64encode((output/"wealth.png").read_bytes()).decode("ascii")
    table = summary[["monthly_excess_sharpe", "daily_excess_sharpe", "hac28_sharpe", "cagr", "drawdown"]].copy()
    for col in table.columns[:3]:
        table[col] = table[col].map(lambda v: f"{v:.2f}" if np.isfinite(v) else "undefined")
    for col in table.columns[3:]:
        table[col] = table[col].map(lambda v: f"{v:.1%}")
    status = "Synthetic demonstration — no evidence of profitability" if synthetic else "Exploratory historical replay — inspected research data; table scores 2024 onward, plot includes the full replay"
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Regime Lab | Execution research</title><style>body{{max-width:1050px;margin:40px auto;padding:0 22px;font:15px/1.6 system-ui;color:#203047}}h1{{font-size:32px}}.notice{{background:#eef3fc;border-left:4px solid #3564be;padding:16px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:9px;border-bottom:1px solid #d9e0ec;text-align:right}}.table{{overflow:auto}}img{{width:100%}}code{{overflow-wrap:anywhere}}</style>
<h1>Funding carry: from returns to an account</h1><p class="notice">{escape(status)}. No Sharpe target is an acceptance test. The new execution variant has its own results.</p>
<p>The reference policy is <strong>{escape(settings.get('selected_allocator', 'synthetic_policy'))}</strong>. Both controls use the same account engine. Funding accrues to old whole-contract holdings; transfer cash earns zero; positions fill only at subsequent executable observations.</p>
<p>Monthly Sharpe uses complete calendar months between consecutive midnight month-start NAVs; boundary partial months are excluded. The pointwise three-month block-bootstrap intervals in <code>uncertainty.json</code> do not correct research selection.</p>
<div class="table">{table.to_html(border=0)}</div><img src="data:image/png;base64,{picture}" alt="Marked account wealth and BIL benchmark">
<h2>Where the dollars came from</h2><p>Funding, collateral price movements, inverse-contract PnL, ETF PnL, quote-currency changes, execution basis/spread and fees reconcile to total account PnL at every event. A small execution component can reflect rounding and realizing a trade/mark basis.</p>
<div class="table">{pd.DataFrame(attribution).round(2).to_html(index=False,border=0)}</div>
<h2>Interpretation and remaining limits</h2><p>Daily and monthly Sharpe use matched BIL excess returns and different aggregation. The 28-day HAC value is a dependence diagnostic. ETF marks and crypto marks remain asynchronous. Historical ETF positions are fractional adjusted total-return units; raw shares, corporate actions and actual broker fills require different data. Ending NAV includes open positions and is not a liquidation guarantee.</p>
<p>Two business days excludes weekends but does not reconstruct bank holidays, withdrawal queues or outages. The maintenance buffer is a research constraint, not an historical exchange-tier simulator. Fees are stated assumptions. Venue default, ADL, custody loss, historical order-book depth and executable capacity are not estimated by this backtest.</p>
<h2>Audit files</h2><p><code>run.json</code>, <code>daily_nav.csv</code>, per-policy event ledgers, orders, decisions and daily holdings, <code>pnl_attribution.csv</code>, <code>yearly_metrics.csv</code> and <code>reconciliation.json</code> make the accounting inspectable. The forward journal is a separate, versioned record; synthetic/history records must never be presented as live evidence.</p></html>'''
    (output/"report.html").write_text(html, encoding="utf-8")


def run_execution_study(output, inverse_root="outputs/inverse_v1", macro_root="outputs/macro_v1",
                        ensemble_root="outputs/ensemble_v1", market_path="data/market.csv",
                        cash_path="data/cash_bil.csv", inverse_data="data/inverse", carry_data="data/carry",
                        start="2023-01-01", end=None, spec=None):
    from .inverse import load_inverse_market
    if (Path(output)/"run.json").exists():
        raise ValueError("Output already contains a run; use a new version directory")
    configuration = selected_configuration(inverse_root, macro_root, ensemble_root)
    carry = configuration["carry"]
    spec = spec or ExecutionSpec(spot_cost=carry["spot_cost"], future_cost=carry["future_cost"],
                                etf_cost=configuration["macro"]["cost"], collateral_fraction=carry["allocation"])
    market, low = load_inverse_market(inverse_data, carry_data, cash_path)
    panel = macro_panel(market_path, cash_path)
    history = risk_history(inverse_root, macro_root)
    end = end or str(market.times[-1].normalize())
    studies = {}
    for rule in dict.fromkeys((configuration["allocator"], "static_risk")):
        events = historical_events(market, low, panel, MacroSpec(**configuration["macro"]), history, rule, start, end)
        studies[rule] = simulate_events(events, spec)
    settings = {"selected_allocator": configuration["allocator"], "selection_override": False,
                "configuration": configuration, "execution": asdict(spec), "start": start, "end": end,
                "data": {str(p): file_digest(p) for p in [Path(market_path), Path(cash_path),
                          *Path(inverse_data).glob("*.csv"), *Path(carry_data).glob("*.csv")]}}
    return _write_study(output, studies, settings, synthetic=False)


def project_demo(output):
    """Synthetic prices test contracts, timing, settlement and journal mechanics."""
    from .carry import CarryMarket
    from .macro import GROUPS
    if (Path(output)/"run.json").exists():
        raise ValueError("Output already contains a run; use a new version directory")
    rng = np.random.default_rng(721)
    times = pd.date_range("2023-01-01", "2024-07-01", freq="h")
    spot = np.array([20_000., 1500.])*np.exp(np.cumsum(rng.normal(0, .0018, (len(times), 2)), axis=0))
    basis = rng.normal(0, .0004, (len(times), 2))
    future = spot*np.exp(basis)
    rates = np.zeros_like(spot)
    rates[times.hour % 8 == 0] = rng.normal(.00004, .00015, ((times.hour % 8 == 0).sum(), 2))
    m = CarryMarket(times, spot, future, future, future, future*1.01, basis, rates, rates/future,
                    np.ones_like(spot, dtype=bool), np.ones(len(times)), np.zeros(len(times)), np.ones(len(times)), {})
    dates = pd.bdate_range("2019-01-01", "2024-07-01")
    names = [n for group in GROUPS.values() for n in group]
    closes = 100*np.exp(np.cumsum(rng.normal(.0001, .008, (len(dates), len(names))), axis=0))
    opens = closes*np.exp(rng.normal(0, .001, closes.shape))
    cash = 100*np.exp(np.arange(len(dates))*.0001)
    panel = (dates, names, opens, closes, cash, cash)
    months = pd.date_range("2021-01-31", "2024-06-30", freq="ME")
    history = pd.DataFrame(rng.normal([.002, .003], [.006, .025], (len(months), 2)), index=months)
    studies = {}
    for rule in ("rolling_risk", "static_risk"):
        events = historical_events(m, future*.99, panel, MacroSpec("inverse_variance"), history, rule,
                                  "2023-01-01", "2024-07-01")
        studies[rule] = simulate_events(events, ExecutionSpec())
    result = _write_study(output, studies, {"selected_allocator": "rolling_risk", "seed": 721,
                                           "execution": asdict(ExecutionSpec())}, synthetic=True)
    from .paper import synthetic_paper_demo
    synthetic_paper_demo(Path(output)/"paper_demo")
    return result
