# Regime Lab

This document preserves the earlier research and its original output paths. For
a fresh checkout, use the current [README](README.md) and
[reproduction guide](docs/REPRODUCIBILITY.md), which provide one coherent layout.

A Python research project for **return prediction and market-regime interpretation** using PCA, a Gaussian HMM, and regularized regression. The first target is SPY's next-open five-session return. The asset, horizon, and window sizes are configurable.

The main question is whether filtered regime probabilities and regime-dependent signal relationships improve return forecasts beyond a regression without regimes. A well-separated volatility regime does not automatically imply predictable expected returns.

## Quick start (PowerShell)

Python 3.11 or newer is required. From this directory:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[market,dev]"
.\.venv\Scripts\python.exe -m regime_lab demo
.\.venv\Scripts\python.exe -m pytest -q
```

The existing local environment was created with `--system-site-packages` to reuse installed scientific libraries. The install command above also works in a fresh isolated environment. `requirements-lock.txt` records the resolved direct dependency versions used for verification.

Open `outputs/demo/report.html` for a self-contained report. **The demo uses synthetic prices. It checks the experiment machinery; its results are not evidence of market predictability.** Demo dates are business-day placeholders, not an exchange holiday calendar.

For market data:

```powershell
.\.venv\Scripts\python.exe -m regime_lab fetch --start 2008-01-01 --end 2026-10-17
.\.venv\Scripts\python.exe -m regime_lab run --data data/market.csv --config configs/default.toml --output outputs/market
```

Regenerate a saved report after editing its presentation, without fitting the models again:

```powershell
.\.venv\Scripts\python.exe -m regime_lab report --output outputs/market
```

`fetch` uses Yahoo Finance through the optional `yfinance` dependency, with `auto_adjust=True`. Its end date is exclusive. The default universe is SPY, QQQ, IWM, EFA, EEM, TLT, IEF, LQD, HYG, GLD, DBC, and UUP. These are US-listed instruments on a shared daily clock. Provider access and data availability may vary. The downloader requires every selected symbol to have all sessions: it will not silently drop an asset or fill a missing price.

You can supply an alternative adjusted-price CSV:

```csv
date,symbol,open,close
2020-01-02,SPY,299.01,301.25
2020-01-02,TLT,125.13,125.56
2020-01-03,SPY,298.42,299.31
2020-01-03,TLT,126.15,126.72
```

This short example demonstrates the schema only; the experiment needs several years of history. Both prices must use the **same split/dividend adjustment convention**. The program cannot verify that convention for custom data. It rejects duplicates, nonpositive or nonfinite prices, intraday dates, and unbalanced panels. It does not forward-fill data. Hashes and download metadata are saved beside fetched files; edited files must have their metadata refreshed too.

## Information and execution timing

- Observe all features after the close of session `t`.
- Predict `open[t + horizon + 1] / open[t + 1] - 1` on adjusted prices.
- Enter at the next session's open; hold exactly `horizon` open-to-open intervals.
- A training target is eligible only when its exit date is **strictly before** the first evaluation signal date. This deliberately conservative purge includes the next-open delay.
- The strategy illustration trades one sleeve every `horizon` sessions. Positions never overlap. Daily forecast scores are also provided, but their targets overlap and the observations are dependent.

Date offsets use actual supplied sessions, not calendar days. Features use close-to-close log returns over 1, 5, 21, and 63 sessions; 21- and 63-session annualized trailing volatility; and drawdown from the trailing 63-session high. There are seven features per asset, with a 63-session warmup. The annualization convention is 252 trading sessions/year.

## Chronological experiment

Default settings in `configs/default.toml` use a rolling 1,260-session history and refit every 126 sessions:

1. **Earlier 756 sessions:** fit the feature scaler, unwhitened three-component PCA, and three-state full-covariance Gaussian HMM. Try three deterministic seeds and select the highest calibration likelihood among finite fits where each state has at least 1% filtered probability mass. A weak diagonal transition prior encourages persistence. Initialization selection never uses evaluation returns.
2. **Later mature labels:** freeze the encoder and filter forward through this later period. Train return models on those observations and their completed targets. The purged tail still updates the state filter, even though its return labels cannot be used.
3. **Next 126 sessions:** freeze all parameters, update filtered probabilities one observation at a time, and issue forecasts. Then roll the window forward and repeat.

The three chronological segments avoid estimating the encoder from observations later than the supervised training signals it produces. This costs sample size and deliberately leaves the encoder older than the latest return-training observations. It is a clear first design, not a claim that this window structure is optimal.

The state probability calculation is an explicit log-space forward recursion using the HMM's public Gaussian parameters. It does not call `predict_proba`, smoothing, or Viterbi for forecast features. The filter continues through training, the purged gap, and evaluation.

Compared models:

| Model | Inputs and purpose |
| --- | --- |
| `historical_mean` | Mean of the same mature return-training labels used by all other models |
| `ridge` | All standardized trailing features, without regimes |
| `pca_ridge` | The PCA factors, without regime probabilities |
| `regime_probabilities` | Filtered probabilities alone, with a ridge penalty |
| `pca_hmm_ridge` | The same raw features as ridge, plus regime intercept and signal-interaction terms |

The last model has shared feature slopes plus deviations multiplied by centered state probabilities, with the final state as reference. Ridge penalizes every slope and regime deviation; the shared intercept is unpenalized. It is a jointly fitted linear interaction model, not a collection of separately trained regressions. The fixed `ridge_alpha=100` is a research starting value. No hyperparameters are selected on reported evaluation returns.

PCA's unconditional orthogonality does not imply within-state independence, which is why the HMM uses full covariance matrices. Gaussian emissions, static transition probabilities within a fold, and the HMM's conditional-independence assumption are explicit simplifications. Overlapping trailing features can cause the model to discover persistence introduced by feature construction.

## Evaluation and interpretation

Forecast outputs include RMSE, MAE, directional accuracy, return correlation, and:

```text
OOS R² = 1 - sum((actual - forecast)²) / sum((actual - historical_mean)²)
```

The mean benchmark is refitted in every fold. A negative OOS R² means worse squared error than that benchmark. Daily and fixed-phase nonoverlapping samples are reported separately. These are descriptive statistics: there are no naive independent-observation p-values or confidence intervals.

The illustrative strategy is long when the forecast is positive and otherwise holds cash. It includes configurable one-way transaction costs on allocation changes, initial entry, and final liquidation. Entry costs reduce capital before the holding-period return; final exit costs reduce liquidation wealth. Cash earns zero; there is no leverage or shorting. Sharpe is a descriptive annualization on nonoverlapping returns at a zero cash rate. Buy and hold uses the same entry/exit range and cost convention. Changing the trading phase may change results; the initial version evaluates one fixed phase.

Regime IDs are sorted by the target's **calibration-period daily volatility**, from lowest to highest. This provides a consistent risk ranking for plots but does not establish identical economic states across folds. Inspect probability mass, duration, transition matrices, PCA loadings, and forward-return distributions. Calibration volatility and later forward-return means describe different quantities; neither establishes a profitable strategy. Forward-return summaries use soft weights and dependent labels; their effective counts are weight diagnostics, not counts of independent observations.

## Outputs

Every successful run writes:

| File | Contents |
| --- | --- |
| `report.html` | Portable summary with embedded SVG regime and net-equity plots |
| `regime_probabilities.png`, `strategy_equity.png` | Standalone plots for inspection and sharing |
| `predictions.csv` | Signal dates, target entry/exit dates, actual returns, forecasts, probabilities, fold IDs |
| `metrics.csv` | Daily and nonoverlapping forecast scores |
| `trades.csv` | Nonoverlapping position, turnover, cost-adjusted return, and wealth ledger |
| `strategy_metrics.csv` | Net CAGR, volatility, Sharpe, drawdown, exposure, turnover |
| `state_diagnostics.csv` | Fold-level state occupancy, volatility, forward-return summaries, and duration |
| `pca_loadings.csv` | Per-fold loadings on standardized input features |
| `folds.json` | Exact segment dates, mature label endpoints, transition matrices, explained variance, all initialization diagnostics |
| `manifest.json` | Config, input hash, source, package versions, and timing conventions |

Running into an existing output directory replaces files with the same names. Use a new output directory for each experiment you want to preserve. Downloaded prices and generated outputs are excluded from Git.

## Validation and next research steps

The test suite checks forward probabilities against exhaustive hidden-state enumeration, batch continuation and terminal-posterior agreement, causal features and forecasts under future-price changes, next-open target timing, purged label boundaries, calibration-only transformations, input rejection, nonoverlapping trades, entry/exit costs, and drawdown from initial capital.

Before claiming predictive value:

1. Reserve a final time period that remains untouched while designing the model. The initial report is exploratory chronological evaluation, not that final holdout.
2. Tune a small, preregistered set of component/state counts, regularization values, and window lengths using nested chronological validation. Record every trial.
3. Compare with a small-feature HMM without PCA and alternative regime specifications. Check dependence on feature windows, random seeds, market episodes, and rebalancing phases.
4. Add block-bootstrap uncertainty and benchmark-relative tests suitable for dependent forecast losses. Review turnover and more realistic slippage/cash returns.
5. Validate adjusted-price assumptions against a point-in-time data source. A current, fixed ETF universe and retrospectively adjusted history can introduce selection and vintage effects. Confirm whether apparent gains reflect expected-return information or risk scaling.

## Sharpe improvement research

The fixed experiment family in [docs/SHARPE_PROTOCOL.md](docs/SHARPE_PROTOCOL.md) evaluates 14 SPY/BIL strategies and four separate diversified ETF strategies. It tests volatility targeting, inverse-variance exposure, trend filters, HMM risk estimates, forecast shrinkage, smaller inner-validated return models, and covariance shrinkage. The protocol was written before viewing the new family results.

The initial historical sample had already been inspected in aggregate. A pre-2019 development selection and later-period comparison provide a chronological research exercise, **not a pristine final holdout**. Every method, five rebalancing phases, and three cost assumptions are retained.

```powershell
.\.venv\Scripts\python.exe -m regime_lab fetch --symbols BIL --start 2008-01-01 --end 2026-10-17 --output data/cash_bil.csv
.\.venv\Scripts\python.exe -m regime_lab sharpe --data data/market.csv --cash-data data/cash_bil.csv --baseline outputs/market --output outputs/sharpe_v1
```

The original market CSV and baseline outputs must exist and match the saved input hash. The fixed study currently requires SPY, a five-session horizon, the original six-feature small-model set, and SPY/TLT/GLD/DBC/BIL for portfolios. It preserves `outputs/market` and verifies that the recomputed baseline forecasts agree with the original run. It needs no additional dependencies beyond the base scientific stack and optional market downloader.

The smaller ridge models select alpha by purged chronological inner MSE. Their scaling is fitted within each inner training fold. They predict excess returns, shrink toward the mature-label mean, and add only probability intercepts in the HMM version. Their forecasts are cached with source/config/data hashes. To reuse a valid forecast cache while regenerating strategy comparisons:

```powershell
.\.venv\Scripts\python.exe -m regime_lab sharpe --data data/market.csv --cash-data data/cash_bil.csv --baseline outputs/market --output outputs/sharpe_v1 --reuse-forecasts
```

For presentation-only changes, `python -m regime_lab report --output outputs/sharpe_v1` rebuilds the report directly from saved results.

The new portfolio simulator accounts for drifted holdings, self-financing fees, BIL sleeve trades, initial purchases, and final liquidation. Weights are long-only and fully funded, with no leverage. Primary Sharpe is calculated from net portfolio returns **minus matched-period gross BIL returns**. BIL is an investable Treasury-bill ETF proxy with expenses and small price risk. These figures are not directly comparable to the original report's zero-cash Sharpe; the new report recomputes original-method benchmarks under the same convention.

Outputs under `outputs/sharpe_v1/` include:

- `report.html`, `later_equity.png`, `phase_sensitivity.png`: portable report and plots.
- `metrics.csv`: every strategy, cost, rebalancing phase, and chronological subperiod.
- `trades.csv`, `weights.csv`: phase-zero primary-cost trade ledgers and daily proposed weights.
- `enhanced_predictions.csv`, `enhanced_folds.json`: new excess-return forecasts and all inner-validation decisions alongside reproduced baseline forecasts.
- `sharpe_intervals.csv`: paired later-period Sharpe-difference block-bootstrap intervals. These are pointwise diagnostics, not selection-adjusted significance tests.
- `exposure_diagnostics.csv`: explicitly ex post static exposure matches for the development-selected method and the highlighted HMM risk model.
- `risk_diagnostics.csv`, `state_diagnostics.csv`: causal volatility and state estimates.
- `manifest.json`, `protocol_snapshot.md`, `forecast_cache.json`: provenance, candidate catalog, frozen protocol, and cache validity.

The selected strategy is chosen only from SPY/BIL methods using completed development trades. Diversified portfolios are shown separately because they change the opportunity set. A static exposure-matched diagnostic uses later-period average exposure **after the fact** and is explicitly excluded from candidate selection. Positive backtest changes still require independent future validation.

## Free-data long/short stock pilot

The stock study adds daily market-neutral statistical arbitrage alongside the ETF research. It uses rolling PCA to estimate common exposures, then compares residual mean reversion, a pooled ridge return forecast, and their equal forecast blend. Dollar, sector, estimated market-beta, and PCA neutrality constrain actual stock holdings. The frozen candidate family and accounting conventions are in [docs/STOCK_PROTOCOL.md](docs/STOCK_PROTOCOL.md); settings are in [configs/stocks.toml](configs/stocks.toml).

Install the additional downloader, HTML parser, and quadratic optimizer:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[statarb,dev]"
```

Freeze the current membership snapshot once, then download daily prices. Use a new snapshot path for a new date; existing universe snapshots are never overwritten. The completed pilot uses the 19 October 2026 snapshot and an exclusive download end of 17 October 2026.

```powershell
.\.venv\Scripts\python.exe -m regime_lab stock-universe --count 200 --seed 42 --output data/stocks/universe_2026-10-19.csv
.\.venv\Scripts\python.exe -m regime_lab stock-fetch --universe data/stocks/universe_2026-10-19.csv --start 2014-01-01 --end 2026-10-17 --output data/stocks
.\.venv\Scripts\python.exe -m regime_lab stock-run --data data/stocks/prices.csv --universe data/stocks/universe_2026-10-19.csv --config configs/stocks.toml --output outputs/stocks_v1
```

Yahoo Finance supplies free adjusted and raw OHLCV histories. The loader caches individual symbols, audits failures and missing sessions, and keeps shorter listing histories. Each stock needs its own trailing price/liquidity history before entering PCA or a portfolio. It does not replace failed names or forward-fill missing stock returns. Missing prices for a held position stop the backtest.

**The current membership and current sector labels introduce historical selection bias. Historical borrow availability is also unknown.** These free-data results are provisional. Retrospectively adjusted vendor prices are not historical data vintages. No historical score in this study establishes a survivorship-free, executable Sharpe of 2.

The primary backtest charges 5 bps per one-way stock notional and 2% annual short borrow, including weekends. It reserves short proceeds without interest; only original capital left after funding longs earns the matched BIL return. Sharpe uses net portfolio returns minus those BIL returns. Costs include drifted holdings, initial purchases, and final liquidation. Gross exposure is capped at 100% and individual positions at 2%. Target position notional is also screened against 1% of trailing ADV at a fixed illustrative $1 million account size; this is not a trade-impact model. Next-open fills at target weights are idealized. The same weights are replayed at higher trading and borrowing costs.

For offline validation without downloads, run:

```powershell
.\.venv\Scripts\python.exe -m regime_lab stock-demo --output outputs/stock_demo
.\.venv\Scripts\python.exe -m pytest -q
```

Synthetic results test the plumbing only. To rebuild portfolios from unchanged cached features and forecasts, add `--reuse-cache` to `stock-run`; the command rejects mismatched data, universe, model source, versions, or configuration. To change only the presentation, run `python -m regime_lab report --output outputs/stocks_v1`.

Stock-study artifacts include:

| File | Contents |
| --- | --- |
| `report.html`, `stock_equity.png`, `stock_exposure.png` | Portable report, net wealth, exposures, and turnover |
| `metrics.csv`, `forecast_scores.csv`, `sharpe_intervals.csv` | Every fixed candidate, cost/borrow stress, chronological slice, forecast score, and pointwise block-bootstrap interval |
| `trades.csv`, `holdings.npz` | Daily actual-stock portfolio accounting and all dated target weights |
| `latest_research_proposal.csv` | Latest forecasts and proposed weights for the next regular-session open, with unverified borrow explicitly flagged |
| `factor_snapshots.json`, `features.npz`, `forecasts.npz` | Dated PCA exposures/hedges, causally stored supervised features, labels, and forecasts |
| `model_decisions.json`, `constraint_checks.csv` | Purged inner validation, alpha decisions, solver diagnostics, and neutrality checks |
| `eligibility_audit.csv`, `data_metadata.json`, `manifest.json`, `protocol_snapshot.md` | Coverage, provenance, research limitations, source/data hashes, package versions, and protocol |

All names and all three candidates are retained in the outputs. No broker connection or order submission is implemented.

## Broader carry and multi-asset model

The next study broadens the opportunity set beyond stock return prediction. It evaluates 18 cash-collateralized funding strategies, 29 multi-asset ETF methods, 18 native inverse-contract carry strategies, and three capital-allocation methods. Protocols are in [ALPHA_PROTOCOL.md](docs/ALPHA_PROTOCOL.md), [INVERSE_PROTOCOL.md](docs/INVERSE_PROTOCOL.md), and [ENSEMBLE_PROTOCOL.md](docs/ENSEMBLE_PROTOCOL.md). All failed candidates remain available.

The preferred **research candidate** combines coin-collateralized BTC/ETH funding carry with the selected inverse-variance ETF portfolio. Fixed sleeve weights are estimated from 2021–2022 monthly risk. They are restored monthly with execution/transfer fees and a two-day idle-cash allowance. This candidate was preferred after inspecting the research comparisons. The separate 2023 validation selector chose the rolling-risk allocator; both are reported. The broader market context has been reused, so the results are not a pristine final holdout.

For 2024 through September 2026 the fixed-risk candidate has monthly excess Sharpe **2.10**, daily excess Sharpe **1.87**, and a 28-day serial-dependence diagnostic of **2.03**. Net annualized growth is **10.34%**, with **-3.51%** maximum drawdown in this sample. Its pointwise monthly Sharpe interval is approximately **[1.08, 3.44]**, without a multiple-selection correction. These historical estimates do not establish a future Sharpe of two.

The native carry sleeve uses the coin as its own derivative collateral, reciprocal-price futures PnL, actual funding payments, coin-denominated fees, and whole contracts. It does not invent interest on reserve cash or ignore the hedge's price changes. Its 3.81 daily Sharpe becomes 1.31 on monthly returns; serial dependence matters. A 10% hypothetical loss of crypto venue capital cuts the combined monthly Sharpe to about 1.01. That tail exposure is not captured by the ordinary mark-to-market Sharpe.

Download the free hourly archives and currency conversion data, then run all four studies into a new output directory:

```powershell
.\.venv\Scripts\python.exe -m regime_lab alpha-fetch --end-month 2026-09
.\.venv\Scripts\python.exe -m regime_lab alpha-run --data data/market.csv --cash-data data/cash_bil.csv --output outputs/alpha_v2
```

These commands need the existing 12-ETF and BIL inputs described earlier, plus the `statarb` optional dependencies. No exchange account, API key, or paid historical feed is used. Public Binance ZIPs and official funding/mark APIs are cached, including recovery of missing archive days. Spot exchange halts remain untradable, with any delayed fills and stale intermediate marks audited. Spot timestamps from 2025 onward are parsed as microseconds.

The completed research outputs are under `outputs/carry_v1`, `outputs/macro_v1`, `outputs/inverse_v1`, and `outputs/ensemble_v1`. To rebuild the consolidated report and its selected-model audits:

```powershell
.\.venv\Scripts\python.exe -m regime_lab alpha-report --output outputs/alpha_v1
.\.venv\Scripts\python.exe -m pytest -q
```

The report rebuild replays selected stress cases and wallet checks; it does not reselect model parameters. `python -m regime_lab report --output outputs/alpha_v1` also uses saved input locations. The consolidated directory contains:

- `report.html`, `equity.png`, `risk.png`: portable results, monthly returns, and drawdown.
- `model.json`: estimated sleeve weights, actual selected strategy settings, capital movement rules, and selection limitations.
- `summary_metrics.csv`, `yearly_metrics.csv`, `stress_metrics.csv`: results at multiple frequencies and costs, funding loss, delay, and venue-loss scenarios.
- `bootstrap_intervals.json`, `replay_audit.json`: dependent-month uncertainty and an independent reconstruction of native coin-wallet NAV.
- `manifest.json`, `report_inputs.json`, protocol snapshots: data/source hashes, package versions, exact selection records, and reproducible input locations.

The capital allocator is a return-sleeve research model with asynchronous valuation across markets. Direct multi-venue execution, bank/crypto settlement timing, custody, liquidation procedures, and live availability still require implementation and forward validation. No live orders or broker integration are included.

## References

- [Ang and Timmermann, Regime Changes and Financial Markets](https://www.nber.org/papers/w17182)
- [Campbell and Thompson, Predicting the Equity Premium Out of Sample](https://www.nber.org/papers/w11468)
- [scikit-learn: preprocessing and data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
- [hmmlearn API and inference algorithms](https://hmmlearn.readthedocs.io/en/stable/api.html)
- [yfinance download API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)
- [Avellaneda and Lee: Statistical Arbitrage in the U.S. Equities Market](https://math.nyu.edu/~avellane/AvellanedaLeeStatArb071008.pdf)
- [Guijarro-Ordonez, Pelger, and Zanotti: Deep Learning Statistical Arbitrage](https://pubsonline.informs.org/doi/10.1287/mnsc.2022.03132)
- [OSQP Python interface](https://osqp.org/docs/interfaces/python.html)
- [BIS: Crypto carry](https://www.bis.org/publ/work1087.htm)
- [Binance: COIN-M collateral and contract multipliers](https://www.binance.com/en-IN/support/faq/detail/85eac2bba0b342819122dc9bd4745e9b)
- [Andrew Lo: The Statistics of Sharpe Ratios](https://alo.mit.edu/research-page/the-statistics-of-sharpe-ratios/)
