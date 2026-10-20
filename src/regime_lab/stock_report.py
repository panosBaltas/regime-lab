"""Portable report for the free-data long/short stock pilot."""

import base64
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .stock_models import STOCK_METHODS

NAMES = {"residual_reversion": "Residual mean reversion", "pooled_ridge": "Pooled ridge",
         "forecast_blend": "Equal forecast blend", "buy_hold": "SPY reference"}
COLORS = {"residual_reversion": "#2563eb", "pooled_ridge": "#7c3aed", "forecast_blend": "#14b8a6", "buy_hold": "#64748b"}


def create_stock_report(output, metrics, ledgers, checks, intervals, scores, manifest):
    config = manifest["config"]
    primary = metrics.loc[(metrics["cost_bps"] == config["cost_bps"])
                           & ((metrics["borrow_rate"] == config["borrow_rate"]) | (metrics["method"] == "buy_hold"))]
    full = primary.loc[primary["period"] == "full"].set_index("method")
    table = full.loc[[*STOCK_METHODS, "buy_hold"],
                      ["sharpe_excess", "cagr_net", "max_drawdown", "average_gross_exposure", "average_daily_turnover"]].copy()
    table.index = [NAMES[name] for name in table.index]
    table.columns = ["Net excess Sharpe", "Net CAGR", "Drawdown", "Mean gross exposure", "Daily turnover"]
    table["Net excess Sharpe"] = table["Net excess Sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    for column in table.columns[1:]:
        table[column] = table[column].map(lambda x: f"{x:.1%}")
    periods = primary.pivot(index="method", columns="period", values="sharpe_excess").reindex(
        index=[*STOCK_METHODS, "buy_hold"], columns=["before_2021", "2021_2023", "2024_plus"])
    periods.index = [NAMES[name] for name in periods.index]
    periods.columns = ["Before 2021", "2021–2023", "2024 onward"]
    stress = metrics.loc[(metrics["period"] == "full") & (metrics["method"] != "buy_hold")].copy()
    stress["Scenario"] = stress.apply(lambda r: f"{r.cost_bps:g} bps / {r.borrow_rate:.0%} borrow", axis=1)
    stress_table = stress.pivot(index="method", columns="Scenario", values="sharpe_excess").reindex(STOCK_METHODS)
    stress_table.index = [NAMES[name] for name in stress_table.index]
    ci_table = intervals[["method", "sharpe_excess", "ci_lower", "ci_upper"]].copy()
    ci_table["method"] = ci_table["method"].map(NAMES)
    ci_table.columns = ["Method", "Sharpe", "95% lower", "95% upper"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "text.color": "#64748b", "axes.labelcolor": "#64748b", "xtick.color": "#64748b",
                         "ytick.color": "#64748b", "axes.spines.top": False, "axes.spines.right": False})
    figure, axis = plt.subplots(figsize=(9, 3))
    for name in STOCK_METHODS:
        ledger = ledgers[name]
        axis.plot(ledger["exit_date"], ledger["equity_net"], label=NAMES[name], color=COLORS[name], linewidth=1.6)
    cash = next(iter(ledgers.values()))
    axis.plot(cash["exit_date"], np.cumprod(1 + cash["cash_return"].to_numpy()), label="BIL cash proxy", color="#64748b", linestyle="--")
    axis.set_ylabel("Net wealth (initial = 1)")
    axis.legend(frameon=False, ncols=2, fontsize=9)
    figure.savefig(output / "stock_equity.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    figure, axes = plt.subplots(2, 1, figsize=(9, 4), sharex=True)
    for name in STOCK_METHODS:
        item = ledgers[name]
        axes[0].plot(item.index, item["gross_exposure"], color=COLORS[name], label=NAMES[name], linewidth=0.9, alpha=0.8)
        axes[1].plot(item.index, item["turnover"], color=COLORS[name], linewidth=0.7, alpha=0.65)
    axes[0].set_ylabel("Gross exposure")
    axes[1].set_ylabel("One-way turnover")
    axes[0].legend(frameon=False, ncols=3, fontsize=8)
    figure.savefig(output / "stock_exposure.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    model_scores = scores.copy()
    model_scores["method"] = model_scores["method"].map(NAMES)
    model_scores = model_scores.rename(columns={"method": "Method", "period": "Period",
                                                "r2_vs_zero_residual": "R² vs zero residual",
                                                "mean_daily_cross_sectional_ic": "Mean cross-sectional IC"})
    model_scores = model_scores[["Method", "Period", "R² vs zero residual", "Mean cross-sectional IC"]]
    reached = bool((full.loc[list(STOCK_METHODS), "sharpe_excess"] >= 2).any())
    result_text = ("At least one provisional historical estimate reaches Sharpe 2; it requires independent validation."
                   if reached else "None of the three fixed candidates establishes the target Sharpe of 2 in this pilot.")
    notice = ("Synthetic fixture: verifies the pipeline only, with no evidence about real-market performance."
              if manifest["synthetic"] else
              "Free-data prototype: current surviving stocks and current sectors introduce selection bias. Historical borrow availability is unknown.")
    content = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Free-data stock stat-arb pilot</title><style>
body {{margin:0;color:var(--foreground,#25324a);font:15px/1.55 system-ui,sans-serif;}}main {{max-width:1040px;margin:auto;}}
h1 {{font-size:25px;margin:16px 0 6px;}}h2 {{font-size:18px;margin:25px 0 8px;}}p {{margin:8px 0;}}.muted {{color:var(--muted-foreground,#64748b);}}
.notice {{border-left:3px solid #7c3aed;background:color-mix(in srgb,#7c3aed 7%,transparent);padding:12px 16px;border-radius:10px;}}
.table {{overflow:auto;}}table {{width:100%;border-collapse:collapse;font-size:13px;text-align:right;}}
th,td {{padding:8px 6px;border-bottom:1px solid var(--border,#cbd5e1);white-space:nowrap;}}th:first-child,td:first-child {{text-align:left;}}
img {{display:block;width:100%;height:auto;}}li {{margin:6px 0;}}
</style></head><body><main><h1>Free-data stock statistical arbitrage</h1>
<p class="muted">{manifest['n_requested_stocks']} requested stocks · {manifest['forecast_start']} to {manifest['last_backtest_exit']} · three fixed methods</p>
<p class="notice">{notice}</p><p>{result_text}</p>
<h2>Actual long/short portfolio results</h2><div class="table">{table.to_html(border=0)}</div>
<p class="muted">Daily next-open execution, gross limit {config['gross_limit']:.0%}, position cap {config['position_limit']:.0%}, market/sector/PCA neutrality.
Primary fees: {config['cost_bps']:g} bps one-way stock turnover; {config['borrow_rate']:.0%} annual borrow.
Only unused original capital after funding longs earns BIL; short proceeds earn no interest. Sharpe subtracts matched BIL returns.
SPY is a directional reference with a different exposure profile.</p>
<img src="stock_equity.png" alt="Actual-stock net wealth for three fixed long/short methods against the cash proxy">
<h2>Chronological comparisons</h2><div class="table">{periods.to_html(float_format=lambda x: f'{x:.2f}', border=0)}</div>
<p class="muted">2024 onward is reserved for this first stock specification, but broad market context was previously seen in ETF studies.
Current-universe selection affects all periods; this is not a clean survivorship-free holdout.</p>
<h2>Trading and borrowing cost sensitivity</h2><div class="table">{stress_table.to_html(float_format=lambda x: f'{x:.2f}', border=0)}</div>
<p class="muted">Sensitivity replays the same dated target weights with different costs; it does not retune or reoptimize a strategy for each scenario.</p>
<h2>Exposure and turnover</h2><img src="stock_exposure.png" alt="Gross stock exposure and actual daily turnover across the study">
<h2>Forecast diagnostics</h2><div class="table">{model_scores.to_html(index=False, float_format=lambda x: f'{x:.4f}', border=0)}</div>
<p class="muted">Five-session residual-return labels overlap. IC is the daily correlation across eligible stocks; these descriptive numbers are not independent-sample significance tests.</p>
<h2>Uncertainty</h2><div class="table">{ci_table.to_html(index=False, float_format=lambda x: f'{x:.2f}', border=0)}</div>
<p class="muted">2,000 circular 21-session block-bootstrap replications. Intervals are pointwise and do not correct current-universe selection or all prior project trials.</p>
<details><summary>Implementation, data quality, and reproducibility</summary><ul>
<li>{manifest['factor_snapshots']} dated PCA fits; {manifest['pooled_model_refits']} pooled ridge refits. Old supervised features retain the factor model known on their own signal dates.</li>
<li>Maximum optimizer equality error: {manifest['max_constraint_error']:.2e}; maximum absolute estimated market beta: {manifest['max_absolute_estimated_beta']:.2e}.
Solver retries and any tiny uniform correction to the gross limit are recorded in constraint_checks.csv.</li>
<li>Every ridge alpha is chosen using earlier purged chronological MSE. Exact label endpoints and all inner decisions are in model_decisions.json.</li>
<li>Optimization uses holdings marked at the known close. Fees use actual next-open drifted holdings, with self-financing stock trades. Missing held-position prices fail explicitly.</li>
<li>Total-return-adjusted prices include dividends; the short-side PnL does not deduct those distributions a second time. Borrow fees and collateral interest assumptions remain approximations without a broker's historical records.</li>
<li>Download failures and gaps are audited; IPOs require their own trailing history before eligibility. Current sector classifications can differ from historical ones.</li>
<li>Latest research proposal: {manifest['latest_signal_close']}. It specifies target weights for the next regular-session open; no orders are sent and borrow availability is unverified.</li>
<li>Source and data hashes, configuration, binary feature/holding archives, universe snapshot, and protocol are saved with the report.</li>
<li>Research: <a href="https://math.nyu.edu/~avellane/AvellanedaLeeStatArb071008.pdf">Avellaneda–Lee</a>;
<a href="https://pubsonline.informs.org/doi/10.1287/mnsc.2022.03132">Guijarro-Ordonez–Pelger–Zanotti</a>.
These methods are adaptations, not claimed paper replications.</li>
</ul></details></main></body></html>'''
    for filename in ("stock_equity.png", "stock_exposure.png"):
        data = base64.b64encode((output / filename).read_bytes()).decode("ascii")
        content = content.replace(f'src="{filename}"', f'src="data:image/png;base64,{data}"')
    (output / "report.html").write_text(content, encoding="utf-8")
    return content


def render_saved_stock_report(output: Path):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    trades = pd.read_csv(output / "trades.csv", parse_dates=["signal_date", "entry_date", "exit_date"])
    ledgers = {name: frame.drop(columns="method").set_index("signal_date") for name, frame in trades.groupby("method", sort=False)}
    return create_stock_report(output, pd.read_csv(output / "metrics.csv"), ledgers,
                                pd.read_csv(output / "constraint_checks.csv"), pd.read_csv(output / "sharpe_intervals.csv"),
                                pd.read_csv(output / "forecast_scores.csv"), manifest)
