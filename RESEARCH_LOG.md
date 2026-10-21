# Initial experiment — 19 October 2026

Question: do PCA-derived market-regime probabilities improve forecasts of SPY's next-open five-session return?

This is the first exploratory run. Its configuration was fixed before inspecting these results. Future experiments on the same period are model development, not an untouched final test.

## Data and design

- Yahoo Finance adjusted daily OHLC downloaded through yfinance, from 2 January 2008 to 16 October 2026: 4,728 shared sessions across 12 ETFs.
- Input SHA-256: `15cf4533d18058fa7fe032cba542f28aec417048f0049a5a7558a807db5aa685`.
- Configuration: `configs/default.toml`; three PCs, three HMM states, three initializations, ridge alpha 100.
- Rolling five-year histories: earlier three years calibrate the encoder, later mature targets fit the return regressions, following six months evaluate frozen parameters.
- 27 chronological folds, with 3,399 daily forecasts from 5 April 2013 through 8 October 2026. The final forecast's return requires later opens, explaining the gap to the last downloaded session.
- All 27 selected HMM fits met the likelihood tolerance. Some rejected initializations reported small likelihood decreases; all initialization diagnostics are preserved in `outputs/market/folds.json`.

## Forecast results

RMSE is measured in five-session return units. The mean forecast is estimated on the same mature labels as the other return models.

| Model | Daily RMSE | Daily OOS R² vs mean | Nonoverlapping OOS R² vs mean |
| --- | ---: | ---: | ---: |
| Historical mean | 2.1600% | 0.00% | 0.00% |
| Ridge | 2.6460% | -50.06% | -44.00% |
| PCA + ridge | 2.2084% | -4.53% | -4.25% |
| Regime probabilities only | 2.1674% | -0.69% | -0.62% |
| PCA + HMM + ridge interactions | 2.7744% | -64.98% | -57.51% |

The initial regime-interaction model does **not** establish improved expected-return prediction. Its squared forecast error is about 65% greater than the mean benchmark on daily overlapping targets. Negative OOS R² describes forecast error; it is not a trading return. The regime-probability-only model is close to the mean forecast but does not beat it on this metric either. Daily observations overlap, so no independent-sample significance claim follows from these scores.

## Regime interpretation and strategy illustration

Across calibration folds, the risk-ranked states have average target volatility of roughly 11%, 16%, and 32%. These rankings provide descriptive market-risk context. They do not establish persistent economic state identities or profitable timing signals.

On the fixed nonoverlapping long/cash illustration, using 5 bps per one-way allocation change and a zero cash rate, the PCA/HMM model has approximately 10.17% net annualized growth and -22.78% maximum drawdown. Buy and hold on the same dates has approximately 14.60% net annualized growth and -29.16% maximum drawdown. Exposure differs substantially, so the lower drawdown alone does not demonstrate predictive value.

These numbers use vendor-adjusted history, a fixed ETF universe, a single starting phase, and simplified costs. They describe this implementation and dataset, not a validated trading strategy. Full results, trades, state diagnostics, and factor loadings are under `outputs/market/`.

## Next experiments

Preserve this run. Before further model selection, define a new validation protocol and reserve future or otherwise untouched data. Investigate a smaller set of economically motivated return predictors, stronger regularization selected using nested chronological validation, and reduced interaction complexity. Compare HMM inputs and state counts using a small prespecified search. Evaluate whether regime information improves risk estimates even when expected-return forecasts remain weak.

# Sharpe study — 19 October 2026

The fixed [Sharpe protocol](docs/SHARPE_PROTOCOL.md) tested 14 SPY/BIL strategies and four diversified ETF strategies, across all five weekly starting phases and 5/10/25 bps one-way costs. It was written before inspecting this new family. All of the historical period was previously seen in aggregate, so the later-period comparison is reused research history, not a pristine holdout.

## What improved

The useful initial direction is **HMM risk sizing**, combining transition-weighted state variances with recent EWMA variance and targeting 10% annual volatility with a 100% equity cap. It uses the HMM to size an equity position, without relying on the original return forecast's sign.

Primary full-period results, phase zero and 5 bps:

| Strategy | Net excess Sharpe | Net CAGR | Maximum drawdown | Mean risky exposure |
| --- | ---: | ---: | ---: | ---: |
| Original HMM | 0.724 | 10.53% | -22.55% | 54.56% |
| HMM risk target | 0.825 | 9.38% | -17.63% | 70.13% |
| Buy and hold | 0.848 | 14.60% | -29.16% | 100.00% |

Here Sharpe is calculated from net portfolio returns minus matched-period **BIL** returns. It is not directly comparable to the initial zero-cash Sharpe. BIL is traded as the residual sleeve; its transactions are charged too. Fees use drifted holdings and self-financing allocations, including initial purchases and final liquidation.

In 2019 onward, HMM risk sizing has excess Sharpe 0.777 versus 0.598 for the original HMM and 0.846 for buy and hold. It improves on the original HMM in four of five starting phases, but not all five. Median later-period phase Sharpe is approximately 0.786 versus 0.391 for the original. At 25 bps, phase-zero later-period Sharpe is 0.691 versus 0.279 for the original.

**These gains are not established as statistically reliable.** The paired later-period 95% block-bootstrap interval for the Sharpe change versus the original HMM is approximately [-0.383, +0.786]. It includes zero and is not corrected for selecting among candidates. The risk method is highlighted after examining the family, rather than being presented as a prospectively selected winner. It still underperforms buy and hold on primary excess Sharpe.

An ex post static SPY/BIL allocation matched to the risk model's later-period average exposure has Sharpe approximately 0.844, versus 0.777 for the dynamic risk model. This control reinforces that improved performance relative to the original return-sign strategy does not establish a timing advantage over passive exposure.

## What did not generalize

The actual pre-2019 development selector chose `shrunk_hmm` (75% historical mean / 25% original forecast, with bounded mean-variance sizing). Its development Sharpe was 0.888, but its later-period Sharpe fell to 0.496. Preserve this failure instead of relabeling the best later performer as the selected method.

Smaller, inner-validated return models did not solve the forecasting problem. Inner MSE selected ridge alpha 10,000 in 19/27 raw-model folds and 20/27 probability-augmented folds, favoring strong regularization. Their later-period strategy Sharpes were approximately 0.401 and 0.404. Their near-identical performance provides little evidence that the probability intercepts add useful return information in this specification.

Simple volatility targeting and 12-month trend improved on the original HMM's later-period Sharpe but did not beat buy and hold. Inverse-variance and multiscale trend methods were weaker in phase-zero later-period results. The fixed strategy ensemble did not rescue the forecast-heavy members.

Diversified trend achieved later-period Sharpe 0.855 with approximately -9.36% drawdown, but it changes the investment universe, has full-period Sharpe only 0.680, and is cost-sensitive. Its later-period Sharpe difference from buy and hold is approximately +0.009, with a wide bootstrap interval [-0.640, +0.607]. It does not establish an edge over passive SPY.

## Reproducibility and implication

The original price CSV and results remain preserved under `outputs/market`. Recomputed original forecasts differ from the saved baseline by at most `4.9e-13`. New cash input SHA-256 is `1b94b30f8286e252bdb6200c1566f2cbd2e3bc15c0877ec7428f9a09ff0bc6bd`. Every inner split passed the target-maturity audit; all 23 focused tests passed before the study.

All candidate/phase/cost/subperiod scores, trade and weight ledgers, nested-validation decisions, bootstrap intervals, and source hashes are under `outputs/sharpe_v1/`. The report includes the development-selection failure, all candidates, and the exposure-matched diagnostic.

Next, freeze the risk-sizing hypothesis for genuinely new observations, improve the match between risk forecasts and next-open holding-period risk, and gather additional independent evidence for expected returns. This experiment supports further research on regime-aware risk management; it does not justify promoting a return-prediction strategy to live use.

# Free-data stock pilot — 19 October 2026

The new implementation preserves both ETF studies and adds daily long/short stock research. The [stock protocol](docs/STOCK_PROTOCOL.md) fixed three candidates before viewing their real-market strategy performance: PCA-residual mean reversion, pooled ridge forecasts of five-session residual returns, and their equal forecast blend. There is no search for a parameter combination that happens to reach Sharpe 2.

The dated universe contains 200 current S&P 500 securities, selected with deterministic sector-balanced hash ordering and duplicate-issuer removal. Yahoo Finance supplied 2014-onward adjusted and raw OHLCV for all 200 names plus SPY and BIL. There were no failed symbols or internal session gaps. IPOs retain their shorter histories and become eligible only after their own trailing history is sufficient. Input price SHA-256 is `40981e8e4e8c1cd5498ba054d65fd86766e3a68c901cfa6b9b15e48a23fbeed7`; universe SHA-256 is `a178bfbe387ca9a1aefa2e5923d13e342624f7ad28fb70f266d19bcfe24e43c6`.

**Current survivors, current sector labels, retrospectively adjusted prices, and unknown historical borrow availability limit this evidence.** It is not a survivorship-free historical strategy test. The 2024-onward slice was reserved for this stock specification, but broad market context had already appeared in the ETF research. It is not a pristine project-wide holdout.

Each date retains its then-known PCA exposures and features. The pooled model uses equal total weight per date and purged chronological inner MSE to select ridge regularization. All 129 real-data model refits and all inner splits passed the strict label-maturity audit. Portfolio optimization uses known-close holdings estimates and imposes dollar, sector, estimated market-beta, and PCA neutrality on actual stock holdings. The backtest charges next-open drifted turnover, self-financing stock fees, calendar-day borrowing, and final liquidation. Short proceeds earn no interest; unused original capital after funding longs earns BIL. The account size used for the target-position liquidity screen is a fixed illustrative $1 million, not a requirement for using the model.

Numerical validation found sparse-matrix assembly and optimizer convergence issues before performance was evaluated. These were fixed without changing the economic objective or searching strategy results. Deterministic cold retries and small uniform corrections to the gross bound are logged. Equality, gross, and position checks still fail explicitly on larger violations.

All 34 tests passed, including future-price invariance of forecasts and holdings, dated IPO eligibility, purged labels, factor-neutral actual-stock PnL, optimizer constraints, weekend short borrowing, cash opportunity costs, self-financing fees, missing held-price failures, cache validation, and full offline report generation. The documented `stock-demo` and saved-report commands also completed. Synthetic performance is plumbing evidence only.

## Stock results and interpretation

The full real-data simulation covers 2,705 daily next-open intervals from signals beginning 12 January 2016 through the final exit on 16 October 2026. It uses 142 dated PCA snapshots and 129 pooled-model refits. Primary costs are 5 bps per one-way stock notional and 2% annual short borrow. Sharpe subtracts matched BIL returns.

| Fixed method | Net excess Sharpe | Net CAGR | Maximum drawdown | Mean daily turnover |
| --- | ---: | ---: | ---: | ---: |
| Residual mean reversion | -0.972 | -4.04% | -39.09% | 60.06% |
| Pooled ridge | -1.097 | -2.57% | -27.11% | 23.31% |
| Equal forecast blend | -0.788 | -2.78% | -30.62% | 49.95% |
| SPY directional reference | 0.812 | 15.56% | -32.05% | 0.04% |

All three methods also have negative excess Sharpe in every predeclared chronological slice. In 2024 onward the values are -1.577, -1.754, and -1.418 respectively. Pointwise full-period 95% circular-block bootstrap intervals are [-1.675, -0.391], [-1.681, -0.504], and [-1.441, -0.221]. They do not correct survivorship bias, model selection, or prior project trials. The target Sharpe of 2 is not established.

The forecasting problem remains weak. Full-period residual-return R-squared versus a zero forecast is negative for all three methods; ridge is approximately -0.10%. Ridge's average cross-sectional forecast/return correlation is 0.0068 over the full period and -0.0111 in 2024 onward. These overlapping-label diagnostics are descriptive, not independent-observation significance tests.

After viewing the primary results, a fixed-weight zero-transaction-fee replay was added as a **diagnostic**, without changing signals, weights, or the 2% annual short-borrow assumption. Its excess Sharpes are 0.240, -0.404, and 0.251. Even eliminating transaction fees does not produce evidence near Sharpe 2. At the primary 5 bps, approximate annualized mean entry/rebalance fee drag is 7.57%, 2.94%, and 6.29% of NAV; borrowing adds about 1.00%, 0.90%, and 1.00%. These arithmetic fee summaries are not CAGR decompositions. The zero-fee diagnosis and primary replay checks are saved in `cost_diagnostics.csv` and `replay_audit.json`.

Replaying saved binary holdings reproduces primary daily net returns to within `5.0e-14`. Maximum optimizer neutrality error is `6.93e-12`, estimated market-beta exposure is at most `1.42e-11`, and position-cap excess is at most `4.49e-8`. No cold retry was needed in the completed run. Uniform gross corrections were below 0.000307% of NAV. These checks support implementation consistency; they do not validate the economic model or the missing borrow-history assumptions.

The next hypothesis should address both weak expected returns and trading frequency: test a small, separately frozen family of slower decisions and explicit no-trade bands, then require stronger forecasts to justify any additional complexity. Keep PCA as an exposure/risk tool and retain the original HMM research for market-regime interpretation. Do not promote this stock strategy or infer that a larger neural model will solve its signal problem. A dated forward universe and new observations can improve the evidence without purchasing historical data.

All three failed candidates, all prespecified cost stresses, model decisions, and source/data hashes remain under `outputs/stocks_v1/`. The portable report and interactive `summary.html` are suitable for reviewing this result without rerunning the study.

# Broader alpha research — 20 October 2026

The user requested a much broader attempt at Sharpe two, allowing any model or market while retaining the preference for free historical data. This research evaluated 68 candidates: 18 linear funding-carry methods, 29 macro/ETF methods, 18 native inverse-contract carry methods, and three cross-sleeve allocators. Protocols were written before each family's scores, and later extensions explicitly identify already-inspected context. All unsuccessful methods remain in their catalogs.

The economic change is collecting a structural funding premium while hedging coin price risk, combined with diversification across a separate ETF risk allocation. Brownian motion alone supplies no expected-return edge; reciprocal-price contract accounting and funding economics determine the hedge's payoff. OU spread forecasts were also tested, but the selector preferred the simpler native carry control.

## Data, accounting and first results

Public Binance hourly spot/perpetual/mark archives and actual funding settlements were downloaded without credentials. Archive omissions were recovered from official APIs with provenance. The linear and native products use different actual funding series and different payoff equations. Spot exchange halts remain untradable; stale intermediate marks and deferred fills are explicit. USDT/USD conversion and BIL are retained. No future interest is used in signals and no exchange cash interest is invented.

Cash-collateralized linear carry was economically capital-inefficient: its selected 2024-onward daily excess Sharpe was -0.126 despite positive absolute returns. Reserving cash collateral reduced its return below the Treasury opportunity cost. The separate 29-method ETF study selected the inverse-variance risk model with 21-session rebalancing. Its 2024-onward trading-day excess Sharpe was 1.459, net CAGR 13.77%, and drawdown -7.29%. Its market history was already inspected in earlier project research.

Native COIN-M contracts can use the purchased BTC/ETH as collateral. The short's coin PnL is USD face times the change in reciprocal mark price; actual funding is paid in coins. The wallet, unrealized coin PnL, basis, fees and whole-contract counts are all valued. No short sale proceeds or fictitious dollar loan are added to capital. Monthly hedge adjustments neutralize accumulated funding coin delta. The two coin sleeves begin with equal capital and retain their own drifting wealth.

Development/2023 validation selected native passive carry with 30-day hedge adjustments. The selection artifact was written before evaluating its 2024+ strategy scores. In 2024–September 2026 it delivered daily excess Sharpe 3.813, CAGR 7.23%, and mark-to-market drawdown -0.371%. Double trading fees gave 3.587 daily Sharpe; a 10% haircut to positive funding gave 2.819. These daily scores are insufficient by themselves: monthly Sharpe was only 1.313 and the 28-day serial-dependence diagnostic about 1.370. Carry was especially strong in 2024, weaker in 2025, and negative versus BIL in 2026. Venue-default risk is not priced by the historical quote series.

## Preferred research candidate and selection caveat

The fixed training-risk mix uses **50.5947% native carry / 49.4053% ETF sleeve**, estimated from completed 2021–2022 monthly volatility. These are fitted risk weights, not numbers chosen to maximize the 2024+ Sharpe. Restore capital weights monthly, charge 29 bps on capital entering/leaving the carry sleeve and 15 bps on ETF capital movement, and keep moved funds idle for two calendar days. New allocations earn only subsequent returns. Native underlying terminal liquidation is retained; ETF exit and fiat/venue allowances are charged as well.

This fixed-risk mix is the **preferred research candidate after comparing the completed results**. It is not the allocator picked by the 2023 validation selector. That selector chose rolling risk, which subsequently scored daily excess Sharpe 1.994 and monthly Sharpe 1.809. Do not relabel the fixed-risk model as validation-selected. The calendar evaluation context was reused across these extensions, so none of this is a pristine final holdout.

2024–September 2026, fixed training-risk mix:

| Metric | Result |
| --- | ---: |
| Monthly net excess Sharpe, annualized by sqrt(12) | 2.095 |
| Daily net excess Sharpe, annualized by sqrt(365) | 1.870 |
| Bartlett/Newey–West 28-day risk diagnostic | 2.035 |
| Net CAGR | 10.34% |
| Daily annualized volatility | 3.00% |
| Maximum drawdown | -3.51% |
| Monthly Sharpe pointwise 95% block-bootstrap interval | [1.085, 3.440] |

The interval uses 33 monthly observations and 5,000 circular three-month blocks, without a multiple-selection correction. This is a historical model near the requested target, not proof that its future Sharpe is two. The capital allocator is an asynchronous return-sleeve model; direct cross-venue execution and settlement still need forward validation.

Stress tests keep the fixed capital weights. Double trading fees give monthly Sharpe 2.057; four-hour carry fills 2.100; a 10% haircut to positive funding, with all negative payments retained, 1.977. A hypothetical 10% loss of capital allocated to the crypto venue on 2 January 2025 lowers monthly Sharpe to 1.008 and drawdown to -6.41%. This is a loss scenario, not an estimate of venue-default probability. A separate $10,000 whole-contract carry account retains daily Sharpe 3.663, but monthly Sharpe remains 1.313.

The independent native-wallet audit rebuilds all 2,158 daily marks from saved contract snapshots and intervening funding, including reciprocal-price PnL and terminal liquidation. Maximum equity error is `1.14e-14`. All native candidates passed the conservative 10% maintenance test on hourly high/low marks. The preferred native model's minimum marked margin ratio was 0.979. Accumulated funding creates residual coin exposure between hedge adjustments; the maximum over the full native history was about 5.94% of native NAV, explicitly reported rather than called perfectly delta-neutral.

The consolidated report, fitted model, complete trial catalog, source/data hashes, yearly metrics, bootstrap results and stresses are under `outputs/alpha_v1`. Original family outputs remain under `outputs/carry_v1`, `outputs/macro_v1`, `outputs/inverse_v1`, and `outputs/ensemble_v1`. The next evidence should be a frozen forward paper record and a direct execution simulation, not further tuning on this evaluation period.

All 51 tests passed after the new implementations, including the existing ETF/stock suites, funding and basis PnL, native reciprocal payoffs, collateral margin, whole contracts, future-data invariance, allocation timing, transfer costs and idle cash, and the effect of serial dependence on risk estimates. The completed CLI report, model/source/data hashes and embedded figures passed the final artifact audit.

# Project maturity and reporting correction — 20 October 2026

Preserved the prior exploratory project in baseline commit `3bb5de1`, then removed
the report-level preference that promoted fixed risk after evaluation. The headline
now follows the saved 2023 rolling-risk selector. The old 2.10 monthly figure remains
historical research context; the corrected reference is 1.81 monthly / 1.99 daily.
Stress replays retain the primary policy's original allocation decisions.

Added a direct position/cash account engine with native coin wallets, whole inverse
contracts, ETF units, USDT conversion, funding priority, delayed business-day
transfers, bid/ask support, fees, collateral checks, corporate actions and dollar
PnL attribution. It compares the recorded selector and a fixed-risk control under
the same accounting, without selecting a new historical winner. The selected new
variant scores 1.888 monthly / 1.953 daily / 1.834 HAC-28, 8.389% CAGR and -2.281%
drawdown in the 2024 onward daily sample. There are 32 complete calendar months;
partial boundary months are excluded. Its pointwise monthly interval is [0.902,
3.341], not corrected for project-wide research selection. The control's 2.315
monthly score is reported as a control. Every event's NAV reconciles within
1.34e-10 USD. Total dollar attribution also reproduces final NAV.

Added exclusive source/library-pinned model freezes, scheduled pre-outcome plans,
public quote/funding capture, append-only local hash chains, trusted-head checks,
and replay of paper account states. Reference-refresh rebuilds only frozen specs
and avoids artificial terminal liquidation. A prospective model and its first plan
were recorded on 20 October; public capture timed out at Binance. No live market
observations, fills or forward Sharpe are claimed at the initial checkpoint.

The separate Python 3.13 environment has system packages disabled. Package-server
and DNS failures prevented the first online installation; cached original wheels
supplied a clean offline installation instead. The complete 47-package version
lock installs, `pip check` passes, and all 70 tests pass in that isolated environment.
The offline CLI demonstration also exercises a labeled synthetic paper journal.
Windows/Linux CI is configured but no remote CI run or repository publication is
claimed. Curated portable reports, the research note, paper runbook, protocol and
interview guide are included. New HMM controls, broker integration, verified
dividend payment settlement and a substantial forward record remain future work.

## Final audit revision (0.2.1)

Final review replaced residual execution attribution with an independent
calculation from each spot, ETF and inverse trade's fill price versus its mark.
Injected unexplained cash now fails reconciliation instead of entering a residual
bucket. All 71 tests pass in the isolated environment, `pip check` passes and both
direct/synthetic CLI studies reproduce their prior returns. Version 0.2.0 and the
initial paper-v1 source are preserved in commit `d296200`; the revised active paper
model is `paper/v2`. Model parameters and historical performance were not optimized
in this correction. The remaining public-data capture limitation is connectivity,
not a fabricated replacement series.

The retry did capture public data. One snapshot was rejected as stale at recording;
the next capture and observation were executed in one process and accepted. Active
paper-v2 now has one actual market observation, no fills and no inferred Sharpe.
The plan's initial one-hour eligibility delay is respected. The provider did not
qualify all indicative ETF quotes as fresh/executable. A checkpoint of the verified
active chain is retained with the curated artifacts.

# Publication preparation — 21 October 2026

Added the standard MIT license with copyright held by Panagiotis Baltas, matching
package author/SPDX license metadata, a GitHub citation file, and contributor
guidance. Retaining copyright/license notices is the binding MIT requirement;
explicit citation in research and presentations is requested as a courtesy.
Dependency and public-data terms remain separate.

Replaced mixed legacy-path instructions with a single repository runner for free
data download, exploratory family studies, direct account replay, and selection
diagnostics. Commands and the runner hash are retained in an exclusive new run
directory. Dry runs have no side effects. The original research and its output
paths remain documented rather than erased.

Added a static report gallery built only from an explicit list of curated files,
the original account equity figure, and legal/citation files. The gallery labels
the selected allocator, retained control, synthetic demo, and dated forward
checkpoint separately. A manually triggered GitHub Pages workflow is prepared;
no remote repository, hosted CI success, or public deployment is claimed.

Added a user-run public capture-and-record operation. It includes ETF capture,
serializes overlapping ticks, checks an optional trusted prior head, immediately
records a fresh snapshot, independently replays the resulting account, and saves
a separate checkpoint with its runner hash. It never retries automatically after
an error because a record can commit before a later receipt write fails. No
background schedule, live orders, or new real-market observation was created in
this publication pass.

All 79 tests pass in the isolated locked environment, including eight operational
tests for successful replay, stale data, provider errors, overlap, checkpoint
mismatch, post-commit receipt failure, dry-run behavior, and workflow failure.
`pip check` passes. A clean staged-source archive with no market cache built a
wheel, and its installed wheel ran the full offline demo and synthetic paper replay.
The source distribution contains citation, scripts, documentation, curated reports,
and the equity figure; wheel metadata contains the author and MIT license notice.
The gallery also builds from the clean archive. Desktop/phone browser checks found
no document overflow, missing images, or broken gallery links. A tracked-file
credential-pattern scan found no matches.

No strategy-package source, selection, parameter, historical return, or active
paper record changed. The existing paper-v2 model still verifies against its
original trusted head. Continued forward observation, verified dividend payment
settlement, broker reconciliation, and predeclared new model comparisons remain
research work, not publication blockers for an honestly labeled portfolio project.

