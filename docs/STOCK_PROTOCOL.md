# Free-data stock statistical-arbitrage pilot

Protocol written before inspecting this study's strategy performance, 19 October 2026. Sharpe 2 is a research ambition, not a guaranteed result or a threshold for stopping a parameter search.

## Data and scope

- Preserve the ETF experiments and add a separate stock experiment.
- Freeze a dated current S&P 500 membership/sector snapshot from Wikipedia. Choose 200 securities by deterministic, sector-balanced round-robin sampling, ordered by SHA-256 of seed plus ticker. No selection by historical strategy performance. Keep duplicate share classes out using issuer CIK where available.
- Download free Yahoo Finance daily OHLCV and adjustment data for 2014 onward, plus SPY and BIL references. Cache each symbol with hashes and an audit of first/last dates, missing sessions, and failures. Retry bounded transient failures. Do not silently replace failed tickers.
- Current membership and current sector labels introduce hindsight selection. This is **not** a survivorship-free historical universe. A historical score is provisional; all reports must say so.
- Keep variable listing histories. Eligibility uses only trailing information: 252 complete daily returns, price at least $5, and mean 21-session dollar volume at least $20 million. A later IPO cannot appear in earlier factor models. Missing prices for an actual held position cause an error, not a zero return or a retrospectively convenient exit.
- Use raw vendor close times volume for the liquidity proxy; use consistently dividend/split-adjusted open and close for return calculations. Do not add dividends again to adjusted-return PnL. Today's vendor adjustments are not point-in-time vintages.
- No paid service, broker credentials, live orders, or recurring jobs are required. The latest dated prediction/weight proposal is saved for manual forward research; it is not evidence of past short availability.

## Factor and return models

- Every 21 sessions (or when the observation set breaks), fit five PCs to standardized trailing 252-session close returns for then-eligible stocks. Fit all transformations at that time. Specific volatility has a 0.5% daily floor.
- Use PCA only to estimate common exposures and factor-hedged residuals. The linear return residual is `r - D U' U D^-1 r`; PCA's centering is used to estimate components, but no arbitrary sample mean is subtracted from tradable PnL.
- Build each date's features using that date's available factor model. Never recompute old supervised features using a later PCA fit.
- Features: standardized residual moves over 1/5/20 sessions, risk-normalized 21-session stock momentum, log relative volume against the preceding 21-session median, and the 21/63-session volatility ratio. Clip features to [-8, 8].
- Labels: the next-open five-session return, residualized with the **signal-date** factor hedge. Label endpoints must be strictly before a model refit date to enter training. Missing labels remain unavailable.
- Pooled ridge uses the latest 504 feature dates and requires 252 mature dates. Each date gets equal total sample weight, irrespective of its number of eligible stocks. Alpha choices [1, 10, 100] are selected by purged, expanding chronological validation on earlier dates (initial 126 dates, subsequent blocks of 63). Fit scaling separately in each inner training fold. Refit the chosen model every 21 sessions.
- Three fixed candidates: (1) residual mean reversion with daily expected return `-0.1 * five-session residual sum / 5`, reconstructing that sum from the clipped five-session residual feature; (2) pooled ridge's predicted five-session residual return divided by 5; (3) their equal forecast blend. Clip expected daily returns to +/-0.5%, not realized returns. No HMM or tree-model search in this first pilot.

## Portfolio and execution

- Optimize actual stock holdings, not a synthetic residual equity curve. Dollar, current-sector, estimated market-beta, and PCA-factor exposures are neutral within numerical tolerances.
- Convex objective: expected daily return minus specific-variance risk, expected amortized turnover costs, and short borrowing costs. Use OSQP with explicit solver and constraint checks. Numerical failures cannot silently become flat trades. Implementation validation added a deterministic cold retry with a different numerical penalty when convergence fails; the objective is unchanged. If summed absolute-value slack exceeds the gross bound by at most 0.0001 of NAV, uniformly shrink the solution to the bound, preserving neutrality; larger errors fail. Log every retry and correction. These numerical fixes were made before viewing strategy performance.
- Risk aversion 20, gross exposure at most 100%, per-stock exposure at most 2%, and each target position's notional bounded by 1% of trailing ADV at a fixed research account size of $1 million. The position-size bound is an illustrative capacity screen, not a guarantee about each rebalance trade's ADV participation or an impact model.
- Signals are computed after close t; target weights execute at open t+1; the daily portfolio is valued at open t+2. Rebalance daily. The forecasting/planning horizon is five sessions, so optimizer turnover cost is amortized over five sessions; the ledger charges actual costs immediately.
- Decisions use previous holdings marked at the **known current close**, not tomorrow's opening prices. Actual fees use drifted holdings at execution, with self-financing post-fee target weights. Record the difference between estimated and realized turnover. Opening fills are an explicit idealized assumption to stress, not guaranteed fills.
- Charge 5 bps per one-way stock notional, including entry and liquidation; annual short borrow 2% on short notional, using actual calendar days. Fully reserve short proceeds without interest. Only unused original capital after funding long holdings earns the matched BIL return. Long financing above original equity incurs the BIL rate plus a 1% annual spread; with gross capped at one and dollar neutrality this loan is normally zero.
- Backtest assumptions do not assert historical borrow availability. Test the fixed weights at 10/20 bps and 5%/10% borrowing costs as separate sensitivity diagnostics, not optimized new candidates.

## Evaluation and evidence

- Preserve every candidate. Report daily net excess Sharpe versus BIL, CAGR, volatility, drawdown, turnover, gross/net/sector/factor exposure, and forecast diagnostics.
- Show the predeclared periods before 2021, 2021–2023, and 2024 onward. The last is the first stock-study evaluation slice, but its broad market context has already appeared in the project's ETF studies. It is not a pristine project-wide holdout.
- Use 2,000 paired circular 21-session block-bootstrap replications for strategy Sharpe diagnostics; intervals are pointwise, not corrected for all prior experiments or current-universe selection.
- Save per-symbol download metadata, failed downloads, universe snapshots, configuration and source hashes, dated factor models, inner-validation decisions, actual-stock trade/weight ledgers, and latest research proposals. No winner is promoted because it happens to reach Sharpe 2.

## Research references

- [Avellaneda and Lee: Statistical Arbitrage in the U.S. Equities Market](https://math.nyu.edu/~avellane/AvellanedaLeeStatArb071008.pdf): PCA residuals and relative-value signals; historical results varied substantially across periods.
- [Guijarro-Ordonez, Pelger, and Zanotti: Deep Learning Statistical Arbitrage](https://pubsonline.informs.org/doi/10.1287/mnsc.2022.03132): factor-residual portfolios, learned signals, and constrained trading policies. This pilot deliberately starts with a smaller estimator.
- [OSQP formulation and Python API](https://osqp.org/docs/interfaces/python.html).
- [yfinance download API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html).

