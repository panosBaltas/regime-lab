"""Static research report for the fixed Sharpe experiment catalog."""

from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .portfolio import portfolio_metrics, simulate_portfolio
from .sharpe_study import PRIMARY, DIVERSIFIED, period_masks

NAMES = {
    "buy_hold": "Buy and hold", "original_hmm": "Original HMM", "original_hmm_vol": "Original HMM + vol sizing",
    "vol_target": "Volatility target", "inverse_variance": "Inverse variance", "trend_12m": "12-month trend",
    "trend_multiscale": "Multiscale trend", "trend_vol": "Trend + vol sizing", "hmm_risk_target": "HMM risk target",
    "shrunk_hmm": "Shrunk HMM forecast", "shrunk_hmm_vol": "Shrunk HMM + vol sizing",
    "small_ridge": "Small ridge", "small_hmm_ridge": "Small HMM ridge", "fixed_ensemble": "Fixed ensemble",
    "div_equal": "Diversified equal weight", "div_inverse_vol": "Diversified inverse vol",
    "div_min_variance": "Diversified minimum variance", "div_trend": "Diversified trend",
}


def create_sharpe_report(output: Path, metrics, intervals, ledgers, manifest):
    selected = manifest["selected_on_development_only"]
    base = metrics.loc[(metrics["cost_bps"] == 5) & (metrics["phase"] == 0)]
    later = base.loc[base["period"] == "later"].set_index("strategy")
    full = base.loc[base["period"] == "full"].set_index("strategy")
    development = base.loc[base["period"] == "development"].set_index("strategy")
    selected_row = later.loc[selected]
    original = later.loc["original_hmm"]
    benchmark = later.loc["buy_hold"]
    ci = intervals.loc[(intervals["strategy"] == selected) & (intervals["reference"] == "buy_hold")].iloc[0]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "text.color": "#64748b", "axes.labelcolor": "#64748b",
                         "xtick.color": "#64748b", "ytick.color": "#64748b",
                         "axes.edgecolor": "#94a3b8", "axes.spines.top": False, "axes.spines.right": False})
    figure, axis = plt.subplots(figsize=(9, 3.2))
    chosen = list(dict.fromkeys(["buy_hold", "original_hmm", selected, "hmm_risk_target", "vol_target"]))
    colors = ["#64748b", "#d97706", "#7c3aed", "#2563eb", "#14b8a6"]
    for name, color in zip(chosen, colors, strict=False):
        ledger = ledgers[name]
        item = ledger.loc[period_masks(ledger)["later"]]
        wealth = np.cumprod(1 + item["net_return"].to_numpy())
        axis.plot(item["exit_date"], wealth, label=NAMES[name], color=color, linewidth=1.8)
    axis.set_ylabel("Net wealth from 2019 (initial = 1)")
    axis.legend(frameon=False, ncols=2, fontsize=9)
    figure.savefig(output / "later_equity.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    phase = metrics.loc[(metrics["period"] == "later") & (metrics["cost_bps"] == 5)
                         & metrics["strategy"].isin(PRIMARY)].groupby("strategy")["sharpe_excess"].agg(["min", "max"])
    figure, axis = plt.subplots(figsize=(9, 5))
    for i, name in enumerate(reversed(PRIMARY)):
        color = "#7c3aed" if name == selected else "#2563eb"
        axis.plot([phase.loc[name, "min"], phase.loc[name, "max"]], [i, i], color=color, alpha=0.5, linewidth=3)
        axis.scatter(later.loc[name, "sharpe_excess"], i, color=color, s=28)
    axis.set_yticks(range(len(PRIMARY)), [NAMES[name] for name in reversed(PRIMARY)])
    axis.axvline(benchmark["sharpe_excess"], color="#94a3b8", linestyle="--", linewidth=1)
    axis.set_xlabel("Net excess Sharpe: dot = phase 0; range = all five starting phases")
    figure.savefig(output / "phase_sensitivity.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    def table(names):
        frame = later.loc[list(names), ["sharpe_excess", "cagr_net", "volatility_net", "max_drawdown", "average_risky_weight"]].copy()
        frame.insert(0, "development_sharpe", development.loc[list(names), "sharpe_excess"])
        frame.insert(0, "Strategy", [NAMES[name] + (" *" if name == selected else "") for name in names])
        frame.columns = ["Strategy", "2013–18 Sharpe", "2019+ Sharpe", "Net CAGR", "Volatility", "Drawdown", "Risky exposure"]
        for column in ("2013–18 Sharpe", "2019+ Sharpe"):
            frame[column] = frame[column].map(lambda x: f"{x:.2f}")
        for column in ("Net CAGR", "Volatility", "Drawdown", "Risky exposure"):
            frame[column] = frame[column].map(lambda x: f"{x:.1%}")
        return frame.to_html(index=False, border=0)

    stress = metrics.loc[(metrics["period"] == "later") & (metrics["phase"] == 0)
                         & metrics["strategy"].isin(["buy_hold", "original_hmm", selected, "vol_target", "hmm_risk_target"])]
    stress = stress.pivot(index="strategy", columns="cost_bps", values="sharpe_excess")
    stress.index = [NAMES[name] for name in stress.index]
    stress.columns = [f"{cost:g} bps" for cost in stress.columns]
    stress_html = stress.to_html(float_format=lambda x: f"{x:.2f}", border=0)
    periods = base.loc[base["strategy"].isin(list(dict.fromkeys(["buy_hold", "original_hmm", selected, "hmm_risk_target"])))]
    periods = periods.pivot(index="strategy", columns="period", values="sharpe_excess")
    periods = periods.reindex(columns=["development", "2019_2022", "2023_plus", "full"])
    periods.index = [NAMES[name] for name in periods.index]
    periods.columns = ["2013–2018", "2019–2022", "2023+", "Full period"]
    matched = manifest["ex_post_exposure_matched_later"]
    improvement = selected_row["sharpe_excess"] - original["sharpe_excess"]
    versus_benchmark = selected_row["sharpe_excess"] - benchmark["sharpe_excess"]
    statement = (f"The development-selected method changes later-period Sharpe by {improvement:+.2f} versus the original HMM "
                 f"and {versus_benchmark:+.2f} versus buy and hold.")
    risk = full.loc["hmm_risk_target"]
    risk_ci = intervals.loc[(intervals["strategy"] == "hmm_risk_target") &
                            (intervals["reference"] == "original_hmm")].iloc[0]
    summary = full.loc[["original_hmm", "hmm_risk_target", "buy_hold"],
                        ["sharpe_excess", "cagr_net", "max_drawdown", "average_risky_weight"]].copy()
    summary.index = [NAMES[name] for name in summary.index]
    summary.columns = ["Net excess Sharpe", "Net CAGR", "Max drawdown", "Risky exposure"]
    summary["Net excess Sharpe"] = summary["Net excess Sharpe"].map(lambda x: f"{x:.2f}")
    for column in summary.columns[1:]:
        summary[column] = summary[column].map(lambda x: f"{x:.1%}")
    exposure_rows = []
    reference_ledger = ledgers["buy_hold"]
    for target in dict.fromkeys([selected, "hmm_risk_target"]):
        target_ledger = ledgers[target].loc[period_masks(ledgers[target])["later"]]
        average = float(target_ledger["risky_weight"].mean())
        weights = np.tile([average, 1 - average], (len(reference_ledger), 1))
        diagnostic = simulate_portfolio(weights, reference_ledger[["gross_return", "cash_return"]].to_numpy(),
                                         reference_ledger["cash_return"].to_numpy(), reference_ledger.index,
                                         reference_ledger["entry_date"], reference_ledger["exit_date"])
        exposure_rows.append({"target": target, "ex_post_risky_exposure": average,
                              **portfolio_metrics(diagnostic.loc[period_masks(diagnostic)["later"]])})
    exposure_table = pd.DataFrame(exposure_rows)
    exposure_table.to_csv(output / "exposure_diagnostics.csv", index=False, float_format="%.12g")
    risk_static = exposure_table.set_index("target").loc["hmm_risk_target", "sharpe_excess"]
    content = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Sharpe research study</title>
<style>
body {{margin:0; color:var(--foreground,#25324a); font:15px/1.5 system-ui,sans-serif;}}
main {{max-width:1040px; margin:auto;}} h1 {{font-size:25px; margin:16px 0 6px;}}
h2 {{font-size:18px; margin:24px 0 8px;}} p {{margin:8px 0;}} .muted {{color:var(--muted-foreground,#64748b);}}
.notice {{padding:12px 16px; border-left:3px solid #7c3aed; background:color-mix(in srgb,#7c3aed 7%,transparent); border-radius:10px;}}
.table {{overflow:auto;}} table {{width:100%; border-collapse:collapse; font-size:12px; text-align:right;}}
th,td {{padding:8px 6px; border-bottom:1px solid var(--border,#cbd5e1); white-space:nowrap;}}
th:first-child,td:first-child {{text-align:left;}} img {{display:block; width:100%; height:auto;}} li {{margin:6px 0;}}
</style></head><body><main>
<h1>Sharpe research study</h1><p class="muted">18 methods · long-only, unlevered · next-open execution · interest-bearing cash proxy</p>
<p class="notice">Later-period history was previously inspected in aggregate. These are research results, with no pristine final holdout.</p>
<h2>What improved</h2>
<p>Using HMM states for risk sizing raises full-period net excess Sharpe from
<strong>{full.loc['original_hmm', 'sharpe_excess']:.2f} to {risk['sharpe_excess']:.2f}</strong>.
Buy and hold remains at {full.loc['buy_hold', 'sharpe_excess']:.2f}. The risk-sizing method has lower growth and a smaller drawdown.</p>
<div class="table">{summary.to_html(border=0)}</div>
<p class="muted">Full period: 2013–2026, phase 0, 5 bps. Risk sizing blends recent EWMA variance with transition-weighted HMM state variances;
the target is 10% annual volatility with a 100% equity cap. This candidate is highlighted after inspecting the family;
it was not the method selected by pre-2019 results.</p>
<p>Its later-period Sharpe difference versus the original HMM is {risk_ci['sharpe_difference']:+.2f}, with a paired 95% block-bootstrap interval
[{risk_ci['ci_95_lower']:+.2f}, {risk_ci['ci_95_upper']:+.2f}]. The interval includes zero, so the gain is not established as reliable.</p>
<p>A static SPY/BIL portfolio matched after the fact to the risk model's later exposure has Sharpe {risk_static:.2f},
versus {later.loc['hmm_risk_target', 'sharpe_excess']:.2f} for the dynamic risk model. The dynamic sizing has not demonstrated a Sharpe gain over passive exposure.</p>
<h2>What failed to generalize</h2>
<p><strong>{escape(NAMES[selected])}</strong> was selected using only completed trades before 2019. {statement}</p>
<p>Later-period net excess Sharpe: <strong>{selected_row['sharpe_excess']:.2f}</strong> selected method,
{original['sharpe_excess']:.2f} original HMM, {benchmark['sharpe_excess']:.2f} buy and hold.</p>
<h2>SPY and Treasury-bill ETF opportunity set</h2><div class="table">{table(PRIMARY)}</div>
<p class="muted">* Selected by development Sharpe, not later-period results. Primary results use starting phase 0 and 5 bps one-way costs.
Sharpe subtracts the same holding-period BIL return; both risky-asset and BIL trades incur costs.
These values use a different cash and accounting convention from the original zero-cash report.</p>
<h2>Later-period wealth</h2><img src="later_equity.png" alt="Cost-adjusted wealth from 2019 for selected and reference strategies">
<h2>Rebalancing-date sensitivity</h2><img src="phase_sensitivity.png" alt="Sharpe ranges across all five weekly starting phases">
<p class="muted">Ranges describe start-date sensitivity, not confidence intervals. No phase was chosen for its performance.</p>
<h2>Trading-cost sensitivity</h2><div class="table">{stress_html}</div>
<h2>Time-period sensitivity</h2><div class="table">{periods.to_html(float_format=lambda x: f'{x:.2f}', border=0)}</div>
<h2>Uncertainty and exposure</h2>
<p>The selected method's later-period Sharpe difference versus buy and hold is {ci['sharpe_difference']:+.2f}.
Its paired 95% block-bootstrap interval is [{ci['ci_95_lower']:+.2f}, {ci['ci_95_upper']:+.2f}].
This is a pointwise diagnostic interval, without adjustment for selecting among strategies or prior project trials.</p>
<p>The selected method's mean risky exposure is {selected_row['average_risky_weight']:.1%}.
A static SPY/BIL portfolio matched to that exposure after the fact has Sharpe {matched['sharpe_excess']:.2f}.
That exposure match is an ex post diagnostic, not a separately selected strategy.</p>
<h2>Separate diversified opportunity set</h2><div class="table">{table(DIVERSIFIED)}</div>
<p class="muted">These four portfolios can hold SPY, TLT, GLD, and DBC, with BIL as residual cash.
Any gain also reflects a broader investment universe; it does not establish better SPY return prediction.</p>
<details><summary>Methodology and research</summary><ul>
<li>Protocol fixed before new results: docs/SHARPE_PROTOCOL.md. All 18 candidates, five starting phases, and 5/10/25 bps costs are retained.</li>
<li>Scaling, PCA, and HMM use earlier calibration data. Smaller excess-return models choose ridge alpha by purged inner chronological MSE;
all inner boundaries and errors are in enhanced_folds.json.</li>
<li>Exposure changes use drifted holdings and self-financing transaction fees. Positions are unlevered and long-only. BIL is a traded cash proxy with expenses and small price risk.</li>
<li>Bootstrap: 2,000 paired circular block replications, blocks of 13 holding periods. Serial dependence, selection, regime drift,
and the reused sample limit confidence in the estimated gains.</li>
<li>Research: <a href="https://www.nber.org/papers/w22208">Moreira–Muir</a>;
<a href="https://www.lehigh.edu/~xuy219/research/COWY.pdf">Cederburg et al.</a>;
<a href="https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum">Moskowitz–Ooi–Pedersen</a>;
<a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1544161">Rapach–Strauss–Zhou</a>;
<a href="https://ledoit.net/honey.pdf">Ledoit–Wolf</a>;
<a href="https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf">Bailey–López de Prado</a>.</li>
</ul></details></main></body></html>'''
    # Embed image bytes in the portable export; preview helpers can instead use local PNG paths.
    import base64
    for filename in ("later_equity.png", "phase_sensitivity.png"):
        encoded = base64.b64encode((output / filename).read_bytes()).decode("ascii")
        content = content.replace(f'src="{filename}"', f'src="data:image/png;base64,{encoded}"')
    (output / "report.html").write_text(content, encoding="utf-8")
    return content


def render_saved_sharpe_report(output: Path):
    import json
    from datetime import datetime, timezone
    from .data import file_digest

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    trades = pd.read_csv(output / "trades.csv", parse_dates=["signal_date", "entry_date", "exit_date"])
    ledgers = {name: frame.drop(columns="strategy").set_index("signal_date")
               for name, frame in trades.groupby("strategy", sort=False)}
    manifest["report_regenerated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["report_source_sha256"] = file_digest(Path(__file__))
    content = create_sharpe_report(output, pd.read_csv(output / "metrics.csv"),
                                   pd.read_csv(output / "sharpe_intervals.csv"), ledgers, manifest)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return content

