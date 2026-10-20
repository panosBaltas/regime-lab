"""Portable report: inline SVG plots, no scripts, services, or external assets."""

from html import escape
from io import StringIO
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {"historical_mean": "Historical mean", "ridge": "Ridge", "pca_ridge": "PCA + ridge",
          "regime_probabilities": "Regime probabilities", "pca_hmm_ridge": "PCA + HMM + ridge",
          "buy_and_hold": "Buy and hold"}
COLORS = {"historical_mean": "#94a3b8", "ridge": "#3b82f6", "pca_ridge": "#14b8a6",
          "regime_probabilities": "#d97706", "pca_hmm_ridge": "#a855f7", "buy_and_hold": "#64748b"}


def _svg(figure):
    buffer = StringIO()
    figure.savefig(buffer, format="svg", bbox_inches="tight", transparent=True)
    plt.close(figure)
    svg = buffer.getvalue()
    return svg[svg.index("<svg"):]


def create_report(output: Path, experiment, metrics, strategies, ledger, manifest):
    predictions = experiment.predictions
    config = manifest["config"]
    synthetic = manifest["data"]["synthetic"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "text.color": "#64748b", "axes.labelcolor": "#64748b",
                         "xtick.color": "#64748b", "ytick.color": "#64748b",
                         "axes.edgecolor": "#94a3b8", "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    figure, axis = plt.subplots(figsize=(9, 2.8))
    columns = [f"p_state_{k}" for k in range(config["n_states"])]
    axis.stackplot(predictions.index, *[predictions[column] for column in columns],
                   labels=[f"Risk rank {k}" for k in range(config["n_states"])],
                   colors=["#3b82f6", "#d97706", "#a855f7", "#14b8a6", "#64748b"][:config["n_states"]], alpha=0.85)
    axis.set(ylim=(0, 1), ylabel="Filtered probability")
    axis.legend(loc="upper left", ncols=min(3, config["n_states"]), frameon=False)
    figure.savefig(output / "regime_probabilities.png", dpi=160, bbox_inches="tight")
    probabilities_chart = _svg(figure)
    figure, axis = plt.subplots(figsize=(9, 2.8))
    for name in ("buy_and_hold", "historical_mean", "ridge", "pca_ridge", "pca_hmm_ridge"):
        item = ledger.loc[ledger["model"] == name]
        axis.plot(item["exit_date"], item["equity_net"], label=LABELS[name], color=COLORS[name], linewidth=1.6)
    axis.set(ylabel="Net wealth (initial = 1)")
    axis.legend(loc="upper left", ncols=3, frameon=False, fontsize=8)
    figure.savefig(output / "strategy_equity.png", dpi=160, bbox_inches="tight")
    equity_chart = _svg(figure)
    table = metrics.loc[metrics["sample"] == "daily_overlapping",
                        ["model", "rmse", "oos_r2_vs_mean", "directional_accuracy"]].copy()
    table["model"] = table["model"].map(LABELS)
    table.columns = ["Model", "RMSE", "OOS R² vs mean", "Direction accuracy"]
    for column in table.columns[1:]:
        table[column] = table[column].map(lambda x: f"{x:.2%}")
    strategy_table = strategies[["model", "cagr_net", "sharpe_net_zero_cash_rate", "max_drawdown_net"]].copy()
    strategy_table["model"] = strategy_table["model"].map(LABELS)
    strategy_table.columns = ["Model", "Net CAGR", "Net Sharpe", "Max drawdown"]
    for column in ("Net CAGR", "Max drawdown"):
        strategy_table[column] = strategy_table[column].map(lambda x: f"{x:.2%}")
    strategy_table["Net Sharpe"] = strategy_table["Net Sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    states = experiment.states.groupby("risk_rank").agg(
        calibration_vol=("calibration_target_volatility", "mean"),
        test_mass=("test_probability_mass", "mean"),
        duration=("expected_duration_sessions", "median"))
    states["calibration_vol"] = states["calibration_vol"].map(lambda x: f"{x:.1%}")
    states["test_mass"] = states["test_mass"].map(lambda x: f"{x:.1%}")
    states["duration"] = states["duration"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "∞")
    states.columns = ["Mean calibration volatility", "Mean fold test mass", "Median duration (sessions)"]
    latest_fold = max(fold["fold"] for fold in experiment.folds)
    latest_loadings = experiment.loadings.loc[experiment.loadings["fold"] == latest_fold].copy()
    latest_loadings["magnitude"] = latest_loadings["loading_on_standardized_feature"].abs()
    largest_loadings = latest_loadings.sort_values("magnitude", ascending=False).groupby("component").head(4)
    loading_rows = []
    for component, group in largest_loadings.groupby("component"):
        leaders = ", ".join(f"{row.feature} ({row.loading_on_standardized_feature:+.2f})" for row in group.itertuples())
        loading_rows.append({"Latest component": f"PC{component}", "Largest absolute loadings": leaders})
    loading_table = pd.DataFrame(loading_rows).to_html(index=False, border=0)
    selected_not_converged = sum(
        not next(c for c in fold["hmm_candidates"] if c["seed"] == fold["selected_hmm_seed"])["tolerance_reached"]
        for fold in experiment.folds)
    banner = ("Synthetic demonstration • validates the pipeline only; does not establish market predictability."
              if synthetic else "Historical research • a first experiment, with no tuned or untouched final holdout.")
    summary = metrics.loc[(metrics["sample"] == "daily_overlapping") &
                          (metrics["model"] == "pca_hmm_ridge")].iloc[0]
    conclusion = ("The regime model reduced forecast squared error against the historical mean in this run."
                  if summary["oos_r2_vs_mean"] > 0 else
                  "The regime model did not beat the historical-mean forecast on squared error in this run.")
    content = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Regime Lab research report</title><style>
:root {{color-scheme:light dark; --foreground:#25324a; --muted-foreground:#64748b; --border:#cbd5e1; --radius:10px;}}
@media(prefers-color-scheme:dark) {{:root {{--foreground:#e2e8f0; --muted-foreground:#94a3b8; --border:#475569;}}}}
body {{margin:0; color:var(--foreground); font:15px/1.55 system-ui,sans-serif;}}
main {{max-width:980px; margin:auto; padding:0;}} h1 {{font-size:25px; margin:16px 0 6px;}}
h2 {{font-size:18px; margin:26px 0 8px;}} p {{margin:8px 0;}} .muted {{color:var(--muted-foreground);}}
.notice {{border-left:3px solid #a855f7; padding:12px 16px; background:color-mix(in srgb,#a855f7 7%,transparent); border-radius:var(--radius);}}
.table {{overflow-x:auto;}} table {{width:100%; border-collapse:collapse; font-size:13px; text-align:right;}}
th,td {{padding:9px 8px; border-bottom:1px solid var(--border); white-space:nowrap;}} th:first-child,td:first-child {{text-align:left;}}
.loadings td:last-child {{white-space:normal; text-align:left;}}
.chart svg {{display:block; width:100%; height:auto;}} code {{font-size:12px;}} details {{margin:16px 0;}} li {{margin:5px 0;}}
</style></head><body><main>
<h1>Regime Lab</h1><p class="muted">{escape(config['target'])} · next-open {config['horizon']}-session returns ·
{predictions.index[0].date()} to {predictions.index[-1].date()} · {len(experiment.folds)} chronological folds</p>
<p class="notice">{banner}</p>
<p>{conclusion} Its daily OOS R² is <strong>{summary['oos_r2_vs_mean']:.2%}</strong>.</p>
<h2>Return prediction</h2><div class="table">{table.to_html(index=False, border=0)}</div>
<p class="muted">Daily targets overlap; rows are dependent. These are descriptive scores, without significance claims.
The separate nonoverlapping scores are in metrics.csv. All models use the same mature return-training labels.</p>
<h2>Market regimes</h2><div class="chart">{probabilities_chart}</div>
<p class="muted">States are ranked by target volatility in each fold's calibration period. Rank 0 is lowest.
The ranks can change economic meaning after a refit; they are not permanent bull/bear labels.</p>
<div class="table">{states.to_html(border=0)}</div>
<h2>What the factors represent</h2><div class="table loadings">{loading_table}</div>
<p class="muted">Largest standardized-feature loadings from the latest calibration fit.
Component signs are arbitrary; importance here is variance representation, not predictive importance.</p>
<h2>One nonoverlapping long/cash sleeve</h2><div class="chart">{equity_chart}</div>
<div class="table">{strategy_table.to_html(index=False, border=0)}</div>
<p class="muted">Trade every {config['horizon']} sessions; long if predicted return is positive, otherwise cash.
Costs: {config['cost_bps']:g} bps per one-way change, including initial entry and final liquidation.
Cash earns zero. Sharpe uses zero cash return; annualization assumes 252 sessions/year.
This rule is illustrative and fixed before evaluation.</p>
<details><summary>Design, diagnostics, and reproducibility</summary><ul>
<li>Each window: {config['calibration_size']} earlier sessions calibrate scaling, PCA, and HMM;
later mature labels fit the return regressions; the next {config['test_size']} sessions evaluate frozen parameters.</li>
<li>HMM uses {config['n_components']} PCs, {config['n_states']} full-covariance Gaussian states,
{config['hmm_restarts']} initializations, and a weak persistence prior. Selection uses calibration likelihood only.</li>
<li>Filtered probabilities use a forward recursion. Purged sessions still update the filter.
No smoothed or Viterbi state sequence enters the forecasts.</li>
<li>Selected HMM fits not meeting the likelihood tolerance: {selected_not_converged}/{len(experiment.folds)}.
Inspect folds.json for all initialization scores, transition matrices, and convergence diagnostics.</li>
<li>PCA + HMM + ridge adds penalized regime intercepts and signal interactions to the same standardized raw features as ridge.
Ridge alpha is fixed at {config['ridge_alpha']:g}; it has not been tuned on this test period.</li>
<li>Current vendor-adjusted prices are not point-in-time vintages. A fixed present-day ETF universe can introduce selection bias.
Gaussian emissions, feature overlap, regime drift, cash-rate assumptions, and start-date sensitivity remain research limitations.</li>
<li>Data source: {escape(str(manifest['data'].get('source', 'user supplied')))}.
Input SHA-256: <code>{escape(manifest['data']['sha256'])}</code>.</li>
</ul></details>
</main></body></html>'''
    (output / "report.html").write_text(content, encoding="utf-8")
    return content


def render_saved_report(output: Path):
    """Regenerate plots and HTML without refitting any research models."""
    from .backtest import Experiment

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(output / "predictions.csv", index_col="signal_date",
                              parse_dates=["signal_date", "entry_date", "exit_date"])
    experiment = Experiment(
        predictions, json.loads((output / "folds.json").read_text(encoding="utf-8")),
        pd.read_csv(output / "state_diagnostics.csv"), pd.read_csv(output / "pca_loadings.csv"))
    return create_report(output, experiment, pd.read_csv(output / "metrics.csv"),
                         pd.read_csv(output / "strategy_metrics.csv"),
                         pd.read_csv(output / "trades.csv", index_col="signal_date",
                                     parse_dates=["signal_date", "entry_date", "exit_date"]), manifest)

