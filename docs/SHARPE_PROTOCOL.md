# Sharpe research protocol — 19 October 2026

Written before examining results from this experiment family. The previous 2013–2026 aggregate results have already been inspected, so **none of this history is a pristine final holdout**. This is a research improvement study, with an explicit development/later-period comparison and a full trial ledger.

## Hypotheses and research

1. **Risk management:** inverse-volatility or inverse-variance sizing might improve risk-adjusted performance even without improved return forecasts. [Moreira and Muir](https://www.nber.org/papers/w22208) motivate testing volatility management. [Cederburg et al.](https://www.lehigh.edu/~xuy219/research/COWY.pdf) find that impressive historical results do not systematically translate into implementable out-of-sample improvements. Our sizing constants are fixed in advance, with no full-sample volatility normalization.
2. **Trend:** test 12-month and combined 3/6/12-month trend signals. [Moskowitz, Ooi, and Pedersen](https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum) study diversified futures; these ETF long/cash rules are adaptations, not paper replications.
3. **Estimation risk:** shrink noisy forecasts toward the historical mean and reduce the return model's feature count. [Rapach, Strauss, and Zhou](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1544161) motivate forecast combinations as a response to model instability. Our particular blend weights are assumptions to test, not values established by that paper.
4. **Diversification:** separately test a fixed four-asset opportunity set using equal weights, inverse volatility, and constrained minimum variance. [Ledoit and Wolf](https://ledoit.net/honey.pdf) motivate covariance shrinkage; the implementation uses scikit-learn's LedoitWolf estimator with its scaled-identity target, not the constant-correlation estimator from that particular paper.
5. **Multiple testing:** keep all candidates, use chronological selection, inspect alternate rebalancing phases and costs, and report paired block-bootstrap uncertainty. [Bailey and López de Prado](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) explain selection inflation of Sharpe estimates. We do not interpret a winning backtest or a pointwise bootstrap interval as proof of a significant discovery.

## Frozen design

- Preserve the original prices, forecasts, config, and results under `outputs/market`.
- Same after-close signal and next-open five-session execution convention.
- Long-only, no leverage or shorting; risky weights sum to at most 100%.
- Default cash proxy: BIL adjusted open-to-open returns. Treat BIL as an actual sleeve, charge its trades, and report net excess Sharpe relative to gross BIL returns. BIL has small price risk and expenses, so it is a cash proxy rather than a literal risk-free account.
- Rebalance every five sessions. Test all five starting phases without selecting the best phase.
- Primary cost: 5 bps per one-way dollar traded, including initial purchases, cash-sleeve changes, and final liquidation. Also test 10 and 25 bps.
- Portfolio accounting uses drifted pretrade holdings; transaction fees are solved with self-financing post-fee target weights. Sharpe uses net returns minus matched-period BIL returns, annualized by sqrt(252/5).
- Volatility: exponentially weighted close-return standard deviation, half-life 21 sessions, at least 63 observations; floor 5% annualized. Volatility target and inverse-variance anchor: 10% annualized.
- HMM risk estimate: state-weighted calibration variances propagated through the saved transition matrix over the forecast horizon, blended equally with recent EWMA variance.
- Mean-variance allocation: `clip(expected_excess_return / (5 * expected_horizon_variance), 0, 1)`. This is a risk-aversion-five rule, not exact Kelly sizing. Hard exposure limits matter.
- Original HMM return forecast blend: 75% historical mean and 25% HMM forecast. Smaller return models use the same 75/25 shrinkage. No blend-weight search.
- Small models: SPY 21/63-session returns, SPY 21-session volatility, SPY trailing 63-session drawdown, TLT 21-session return, HYG 21-session return. Inputs are fixed, not selected by evaluation performance.
- Small ridge models compare alphas `[100, 1000, 10000]` using purged, expanding chronological inner validation on mature labels. Minimum initial inner history: 252 rows; validation blocks: 63 rows. Select by squared forecast error, not realized Sharpe. Fit scaling separately in each inner training fold. The HMM encoder is calibrated earlier than all these training/validation observations.
- Smaller HMM model adds only state-probability intercept terms to the six raw predictors; no feature-by-state interactions.

## Candidate catalog

SPY/BIL opportunity set:

1. `buy_hold`: 100% SPY reference.
2. `original_hmm`: original HMM forecast sign, long/cash.
3. `original_hmm_vol`: original sign multiplied by volatility target weight.
4. `vol_target`: inverse-volatility exposure.
5. `inverse_variance`: exposure proportional to inverse variance.
6. `trend_12m`: SPY 252-session total return above BIL's matching trailing return.
7. `trend_multiscale`: fraction of 63/126/252-session excess trends above zero.
8. `trend_vol`: multiscale trend multiplied by volatility target weight.
9. `hmm_risk_target`: target weight based on blended EWMA/HMM risk.
10. `shrunk_hmm`: shrunk original forecast with mean-variance allocation.
11. `shrunk_hmm_vol`: volatility target multiplied by shrunk forecast's allocation.
12. `small_ridge`: inner-validated six-feature excess-return model, shrunk and allocated by mean-variance rule.
13. `small_hmm_ridge`: the same model plus probability intercepts.
14. `fixed_ensemble`: equal allocation across `vol_target`, `trend_vol`, `shrunk_hmm_vol`, and `small_hmm_ridge` sleeves.

Separate diversification opportunity set: SPY, TLT, GLD, DBC, plus BIL cash.

15. `div_equal`: fixed 25% per risky asset.
16. `div_inverse_vol`: inverse EWMA volatility, normalized across the four assets.
17. `div_min_variance`: long-only LedoitWolf minimum variance over the latest 252 closes, each asset capped at 60%.
18. `div_trend`: inverse-volatility allocation multiplied assetwise by multiscale excess-trend scores; leftover goes to BIL.

## Evaluation and selection

- **Development selection:** use only signal dates before 1 January 2019 and phase zero, then lock the highest net-excess-Sharpe SPY/BIL candidate for the later-period comparison. Ties follow catalog order. Show all later-period candidate results as descriptive comparisons, not independently selected live recommendations.
- **Later period:** signal dates on/after 1 January 2019. Refit forecasting models chronologically as originally designed. Clearly label this reused history, not a pristine holdout.
- For the development selector, include only trades whose exits occur before the 2019 cutoff, avoiding a target that crosses the boundary.
- Report full-period and later-period Sharpe, CAGR, volatility, drawdown, exposure, turnover, five-phase sensitivity, cost sensitivity, and fixed subperiods 2013–2018 / 2019–2022 / 2023 onward.
- Estimate paired circular moving-block bootstrap intervals for Sharpe differences against buy-and-hold and the original HMM, with 13 five-session periods per block and 2,000 replications. These are pointwise diagnostic intervals; they are not corrected for strategy selection or every prior project trial.
- Report an evaluation-period exposure-matched static SPY/BIL diagnostic, explicitly ex post, to distinguish dynamic timing from average exposure. It is not a deployable new strategy or part of candidate selection.
- Record code, input and baseline-forecast hashes, package versions, exact settings, every inner alpha score, and every candidate's metrics. Do not tune these settings after seeing the results without appending a new protocol and preserving this run.

