import argparse
from datetime import date, datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

from .backtest import walk_forward
from .config import Config
from .data import DEFAULT_SYMBOLS, fetch_prices, file_digest, load_prices, synthetic_prices
from .evaluation import forecast_metrics, strategy_results
from .features import build_dataset
from .report import create_report, render_saved_report


def run_experiment(prices, config: Config, output: Path, data_metadata: dict):
    dataset = build_dataset(prices, config.target, config.horizon)
    print(f"{len(dataset.features)} labeled sessions; {dataset.features.shape[1]} trailing features", flush=True)
    experiment = walk_forward(dataset, config, lambda text: print(text, flush=True))
    metrics = forecast_metrics(experiment.predictions, config.horizon)
    strategies, ledger = strategy_results(experiment.predictions, config)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "config": config.to_dict(),
        "data": data_metadata, "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("numpy", "pandas", "scipy", "scikit-learn", "hmmlearn", "matplotlib", "threadpoolctl")},
        "symbols": sorted(prices["symbol"].unique().tolist()),
        "feature_columns": dataset.features.columns.tolist(),
        "forecast_start": str(experiment.predictions.index[0].date()),
        "forecast_end": str(experiment.predictions.index[-1].date()),
        "n_forecasts": len(experiment.predictions), "n_folds": len(experiment.folds),
        "information_timing": "Signal after close t; enter adjusted open t+1; exit adjusted open t+h+1",
        "holdout_status": "Chronological research evaluation; no untouched final holdout reserved",
    }
    output.mkdir(parents=True, exist_ok=True)
    experiment.predictions.to_csv(output / "predictions.csv", float_format="%.12g")
    experiment.states.to_csv(output / "state_diagnostics.csv", index=False, float_format="%.12g")
    experiment.loadings.to_csv(output / "pca_loadings.csv", index=False, float_format="%.12g")
    metrics.to_csv(output / "metrics.csv", index=False, float_format="%.12g")
    strategies.to_csv(output / "strategy_metrics.csv", index=False, float_format="%.12g")
    ledger.to_csv(output / "trades.csv", float_format="%.12g")
    (output / "folds.json").write_text(json.dumps(experiment.folds, indent=2, allow_nan=False), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    create_report(output, experiment, metrics, strategies, ledger, manifest)
    print("\nDaily forecast metrics (overlapping targets):")
    print(metrics.loc[metrics["sample"] == "daily_overlapping",
                      ["model", "rmse", "oos_r2_vs_mean", "directional_accuracy"]].to_string(index=False))
    print(f"\nReport: {(output / 'report.html').resolve()}", flush=True)
    return experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description="Chronological PCA/HMM and long/short stock research")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="Run a deterministic synthetic end-to-end experiment")
    demo.add_argument("--sessions", type=int, default=2200)
    demo.add_argument("--config", type=Path)
    demo.add_argument("--output", type=Path, default=Path("outputs/demo"))
    fetch = subparsers.add_parser("fetch", help="Download adjusted ETF OHLC from Yahoo Finance")
    fetch.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    fetch.add_argument("--start", default="2008-01-01")
    fetch.add_argument("--end", default=date.today().isoformat(), help="Exclusive end date")
    fetch.add_argument("--output", type=Path, default=Path("data/market.csv"))
    run = subparsers.add_parser("run", help="Evaluate a saved adjusted OHLC CSV")
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--config", type=Path)
    run.add_argument("--output", type=Path, default=Path("outputs/market"))
    report = subparsers.add_parser("report", help="Regenerate a report from saved results without refitting")
    report.add_argument("--output", type=Path, required=True)
    sharpe = subparsers.add_parser("sharpe", help="Run the fixed-catalog Sharpe improvement study")
    sharpe.add_argument("--data", type=Path, required=True)
    sharpe.add_argument("--cash-data", type=Path, required=True)
    sharpe.add_argument("--baseline", type=Path, default=Path("outputs/market"))
    sharpe.add_argument("--output", type=Path, default=Path("outputs/sharpe_v1"))
    sharpe.add_argument("--reuse-forecasts", action="store_true")
    stock_universe = subparsers.add_parser("stock-universe", help="Freeze a sector-balanced current-stock universe")
    stock_universe.add_argument("--output", type=Path, required=True)
    stock_universe.add_argument("--count", type=int, default=200)
    stock_universe.add_argument("--seed", type=int, default=42)
    stock_fetch = subparsers.add_parser("stock-fetch", help="Download audited free stock histories")
    stock_fetch.add_argument("--universe", type=Path, required=True)
    stock_fetch.add_argument("--output", type=Path, default=Path("data/stocks"))
    stock_fetch.add_argument("--start", default="2014-01-01")
    stock_fetch.add_argument("--end", default=date.today().isoformat())
    stock_run = subparsers.add_parser("stock-run", help="Run causal PCA-residual long/short research")
    stock_run.add_argument("--data", type=Path, required=True)
    stock_run.add_argument("--universe", type=Path, required=True)
    stock_run.add_argument("--output", type=Path, default=Path("outputs/stocks_v1"))
    stock_run.add_argument("--config", type=Path, default=Path("configs/stocks.toml"))
    stock_run.add_argument("--reuse-cache", action="store_true")
    stock_demo = subparsers.add_parser("stock-demo", help="Run a synthetic stock-study fixture")
    stock_demo.add_argument("--output", type=Path, default=Path("outputs/stock_demo"))
    alpha_fetch=subparsers.add_parser("alpha-fetch",help="Cache free hourly crypto carry and inverse-contract data")
    alpha_fetch.add_argument("--data-root",type=Path,default=Path("data"))
    alpha_fetch.add_argument("--end-month",default="2026-09")
    alpha_run=subparsers.add_parser("alpha-run",help="Run the broad carry, macro, inverse and allocation research")
    alpha_run.add_argument("--data",type=Path,default=Path("data/market.csv"))
    alpha_run.add_argument("--cash-data",type=Path,default=Path("data/cash_bil.csv"))
    alpha_run.add_argument("--data-root",type=Path,default=Path("data"))
    alpha_run.add_argument("--output",type=Path,default=Path("outputs/alpha_v2"))
    alpha_report=subparsers.add_parser("alpha-report",help="Build the broad research report from completed studies")
    alpha_report.add_argument("--carry-results",type=Path,default=Path("outputs/carry_v1"))
    alpha_report.add_argument("--macro-results",type=Path,default=Path("outputs/macro_v1"))
    alpha_report.add_argument("--inverse-results",type=Path,default=Path("outputs/inverse_v1"))
    alpha_report.add_argument("--ensemble-results",type=Path,default=Path("outputs/ensemble_v1"))
    alpha_report.add_argument("--output",type=Path,default=Path("outputs/alpha_v1"))
    project_demo = subparsers.add_parser("project-demo", help="Offline direct-account demonstration; synthetic results only")
    project_demo.add_argument("--output", type=Path, default=Path("outputs/project_demo"))
    execution_run = subparsers.add_parser("execution-run", help="Replay underlying positions with explicit cash and settlement")
    execution_run.add_argument("--output", type=Path, required=True)
    execution_run.add_argument("--start", default="2023-01-01")
    execution_run.add_argument("--end")
    for command in (execution_run,):
        command.add_argument("--inverse-results", type=Path, default=Path("outputs/inverse_v1"))
        command.add_argument("--macro-results", type=Path, default=Path("outputs/macro_v1"))
        command.add_argument("--ensemble-results", type=Path, default=Path("outputs/ensemble_v1"))
        command.add_argument("--data", type=Path, default=Path("data/market.csv"))
        command.add_argument("--cash-data", type=Path, default=Path("data/cash_bil.csv"))
        command.add_argument("--inverse-data", type=Path, default=Path("data/inverse"))
        command.add_argument("--carry-data", type=Path, default=Path("data/carry"))
    freeze = subparsers.add_parser("paper-freeze", help="Create an exclusive, source-pinned prospective paper model")
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--inverse-results", type=Path, default=Path("outputs/inverse_v1"))
    freeze.add_argument("--macro-results", type=Path, default=Path("outputs/macro_v1"))
    freeze.add_argument("--ensemble-results", type=Path, default=Path("outputs/ensemble_v1"))
    freeze.add_argument("--data", type=Path, default=Path("data/market.csv"))
    freeze.add_argument("--cash-data", type=Path, default=Path("data/cash_bil.csv"))
    paper_plan = subparsers.add_parser("paper-plan", help="Record the next scheduled model decision before its outcome")
    paper_plan.add_argument("--model", type=Path, required=True)
    paper_plan.add_argument("--inverse-results", type=Path, default=Path("outputs/inverse_v1"))
    paper_plan.add_argument("--macro-results", type=Path, default=Path("outputs/macro_v1"))
    paper_plan.add_argument("--data", type=Path, default=Path("data/market.csv"))
    paper_plan.add_argument("--cash-data", type=Path, default=Path("data/cash_bil.csv"))
    capture = subparsers.add_parser("paper-capture", help="Capture a single public-data snapshot; no orders or background worker")
    capture.add_argument("--model", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--include-etfs", action="store_true")
    capture.add_argument("--quote-usd", type=float, help="Explicit USDT/USD assumption; otherwise fetch the Yahoo proxy")
    paper_step = subparsers.add_parser("paper-step", help="Mark and simulate eligible fills from a fresh captured snapshot")
    paper_step.add_argument("--model", type=Path, required=True)
    paper_step.add_argument("--snapshot", type=Path, required=True)
    verify = subparsers.add_parser("paper-verify", help="Replay the paper journal and verify its model, chain and account")
    verify.add_argument("--model", type=Path, required=True)
    verify.add_argument("--expected-head", help="Previously saved trusted chain head; detects truncation")
    refresh = subparsers.add_parser("paper-refresh", help="Rebuild the frozen reference specs on updated histories without selecting new models")
    refresh.add_argument("--model", type=Path, required=True)
    refresh.add_argument("--output", type=Path, required=True)
    refresh.add_argument("--data", type=Path, default=Path("data/market.csv"))
    refresh.add_argument("--cash-data", type=Path, default=Path("data/cash_bil.csv"))
    refresh.add_argument("--inverse-data", type=Path, default=Path("data/inverse"))
    refresh.add_argument("--carry-data", type=Path, default=Path("data/carry"))
    stats = subparsers.add_parser("selection-audit", help="Allocator-only block-bootstrap and IID trial-count diagnostics")
    stats.add_argument("--ensemble-results", type=Path, default=Path("outputs/ensemble_v1"))
    stats.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "project-demo":
            from .project import project_demo
            summary, _ = project_demo(args.output)
            print(summary.to_string())
            print(f"Synthetic demonstration: {(args.output/'report.html').resolve()}")
            return
        if args.command == "execution-run":
            from .project import run_execution_study
            summary, _ = run_execution_study(args.output, args.inverse_results, args.macro_results,
                args.ensemble_results, args.data, args.cash_data, args.inverse_data, args.carry_data, args.start, args.end)
            print(summary.to_string())
            print(f"Exploratory execution report: {(args.output/'report.html').resolve()}")
            return
        if args.command == "paper-freeze":
            from .paper import freeze_model
            result = freeze_model(args.output, args.inverse_results, args.macro_results,
                                  args.ensemble_results, args.data, args.cash_data)
            print(json.dumps({"model_id": result["model_id"], "created_at": result["created_at"], "mode": result["mode"]}, indent=2))
            return
        if args.command == "paper-plan":
            from .paper import plan_paper
            result = plan_paper(args.model, args.inverse_results, args.macro_results, args.data, args.cash_data)
            print(json.dumps({"sha256": result["sha256"], "decision": result["payload"]["decision"]}, indent=2))
            return
        if args.command == "paper-capture":
            from .capture import capture_public
            try:
                result = capture_public(args.model, args.output, args.include_etfs, args.quote_usd)
            except Exception as exc:
                raise ValueError(f"Public capture failed; no fills recorded: {exc}") from exc
            print(json.dumps(result, indent=2))
            return
        if args.command == "paper-step":
            from .paper import observe_paper
            result = observe_paper(args.model, args.snapshot)
            print(json.dumps({"sha256": result["sha256"], "valuation": result["payload"]["valuation"], "fills": result["payload"]["fills"]}, indent=2))
            return
        if args.command == "paper-verify":
            from .paper import verify_paper
            print(json.dumps(verify_paper(args.model, args.expected_head), indent=2))
            return
        if args.command == "paper-refresh":
            from .paper import refresh_references
            print(json.dumps(refresh_references(args.model, args.output, args.data, args.cash_data,
                                                args.inverse_data, args.carry_data), indent=2))
            return
        if args.command == "selection-audit":
            from .statistics import allocator_statistics
            args.output.parent.mkdir(parents=True, exist_ok=True)
            print(json.dumps(allocator_statistics(args.ensemble_results, args.output), indent=2))
            return
        if args.command=="alpha-fetch":
            import pandas as pd
            import yfinance as yf
            from .carry_data import download_carry_data,download_settlements,repair_mark_archive_gaps
            from .inverse_data import download_inverse_data
            carry_directory=args.data_root/"carry"
            download_carry_data(carry_directory,end=args.end_month)
            repair_mark_archive_gaps(carry_directory)
            end=(pd.Period(args.end_month,freq="M")+1).start_time.date().isoformat()
            download_settlements(carry_directory,end=end)
            stable=yf.Ticker("USDT-USD").history(start="2019-01-01",end=end,auto_adjust=True)
            if stable.empty:raise ValueError("No USDT/USD data")
            stable.index=stable.index.tz_localize(None)
            stable[["Open","Close"]].rename(columns={"Open":"open","Close":"close"}).to_csv(carry_directory/"usdt_usd.csv")
            download_inverse_data(args.data_root/"inverse",end=args.end_month)
            return
        if args.command=="alpha-run":
            from .carry import run_carry_study
            from .macro import run_macro_study
            from .inverse import run_inverse_study
            from .ensemble import run_ensemble_study
            from .alpha_report import build_alpha_report
            carry_output=args.output/"carry";macro_output=args.output/"macro"
            inverse_output=args.output/"inverse";ensemble_output=args.output/"ensemble"
            run_carry_study(args.data_root/"carry",args.cash_data,carry_output)
            run_macro_study(args.data,args.cash_data,macro_output)
            run_inverse_study(args.data_root/"inverse",args.data_root/"carry",args.cash_data,inverse_output)
            run_ensemble_study(inverse_output,macro_output,ensemble_output)
            build_alpha_report(carry_output,macro_output,inverse_output,ensemble_output,args.output,
                               args.data,args.cash_data,args.data_root/"carry",args.data_root/"inverse")
            return
        if args.command=="alpha-report":
            from .alpha_report import build_alpha_report
            build_alpha_report(args.carry_results,args.macro_results,args.inverse_results,args.ensemble_results,args.output)
            return
        if args.command == "stock-universe":
            from .stock_data import snapshot_universe
            universe = snapshot_universe(args.output, args.count, args.seed)
            print(f"Saved {len(universe)} current-stock members to {args.output}")
            return
        if args.command == "stock-fetch":
            from .stock_data import fetch_stock_panel
            fetch_stock_panel(args.universe, args.output, args.start, args.end, lambda s: print(s, flush=True))
            return
        if args.command in ("stock-run", "stock-demo"):
            from .stock_features import StockConfig
            from .stock_study import run_stock_study
            if args.command == "stock-demo":
                from .stock_data import synthetic_stock_panel
                prices, universe = synthetic_stock_panel()
                args.output.mkdir(parents=True, exist_ok=True)
                data_path, universe_path = args.output / "synthetic_prices.csv", args.output / "synthetic_universe.csv"
                prices.to_csv(data_path, index=False, float_format="%.12g")
                universe.to_csv(universe_path, index=False)
                config = StockConfig(factor_lookback=126, n_components=3, min_stocks=16,
                                     minimum_training_dates=126, training_window=252,
                                     minimum_inner_dates=63, validation_block=42, position_limit=0.05)
                run_stock_study(data_path, universe_path, args.output, config, progress=lambda s: print(s, flush=True), synthetic=True)
            else:
                run_stock_study(args.data, args.universe, args.output, StockConfig.load(args.config),
                                  args.reuse_cache, lambda s: print(s, flush=True))
            return
        if args.command == "sharpe":
            from .sharpe_study import run_sharpe_study
            run_sharpe_study(args.data, args.cash_data, args.baseline, args.output,
                              args.reuse_forecasts, lambda text: print(text, flush=True))
            return
        if args.command == "report":
            if (args.output / "report_inputs.json").exists():
                from .alpha_report import render_saved_alpha_report
                render_saved_alpha_report(args.output)
            elif (args.output / "factor_snapshots.json").exists():
                from .stock_report import render_saved_stock_report
                render_saved_stock_report(args.output)
            elif (args.output / "enhanced_folds.json").exists():
                from .sharpe_report import render_saved_sharpe_report
                render_saved_sharpe_report(args.output)
            else:
                render_saved_report(args.output)
            print(f"Report: {(args.output / 'report.html').resolve()}")
            return
        if args.command == "fetch":
            prices = fetch_prices(args.symbols, args.start, args.end, args.output)
            print(f"Saved {prices['date'].nunique()} sessions × {prices['symbol'].nunique()} symbols to {args.output}")
            return
        config = Config.load(args.config) if args.config else (
            Config(train_size=1008, calibration_size=504, n_states=2, hmm_restarts=2, hmm_iterations=150)
            if args.command == "demo" else Config())
        if args.command == "demo":
            prices = synthetic_prices(args.sessions, config.seed)
            args.output.mkdir(parents=True, exist_ok=True)
            input_path = args.output / "synthetic_prices.csv"
            prices.to_csv(input_path, index=False, float_format="%.12g")
            # Use the persisted bytes, so the saved input exactly reproduces the experiment.
            prices = load_prices(input_path, config.target)
            metadata = {"source": "Regime Lab synthetic generator", "synthetic": True,
                        "adjusted_ohlc": True, "seed": config.seed, "sha256": file_digest(input_path)}
            input_path.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        else:
            prices = load_prices(args.data, config.target)
            metadata_path = args.data.with_suffix(".meta.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {
                "source": "User-supplied OHLC; adjusted-price contract not independently verified", "synthetic": False}
            digest = file_digest(args.data)
            if metadata.get("sha256", digest) != digest:
                raise ValueError("CSV digest differs from its metadata; refresh the data and metadata together")
            metadata["sha256"] = digest
        run_experiment(prices, config, args.output, metadata)
    except (ValueError, OSError, ImportError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

