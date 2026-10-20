# Funding carry: economics, execution and evidence

## Research question

How much of crypto funding carry survives explicit trading and capital movement,
and how does a diversified ETF allocation change the risk? The project studies an
economic premium and its implementation. A future Sharpe of two is not established.

The original PCA/HMM return-forecasting study and a stock residual-reversion pilot
remain in the repository. Their weak forecasting and adverse cost-adjusted stock
results are findings. Their full explanations are in `LEGACY_RESEARCH.md` and
`RESEARCH_LOG.md`. The current execution variant does not include a new HMM overlay.

## Economic mechanism

A spot-long/perpetual-short position receives funding when the funding rate is
positive and pays it when negative. Leveraged directional demand can sustain that
premium. Arbitrage is limited by collateral, venue/custody exposure, financing,
execution, operational constraints and changes in the spot/perpetual basis.
The [BIS crypto-carry research](https://www.bis.org/publ/work1087.htm) provides
economic context, not proof of this particular perpetual strategy's performance.

For an inverse short with USD face `F`, entry price `P0`, and exit/mark price `P`,
coin PnL is `F * (1/P - 1/P0)`. For multiple fills retain the entry coin term
`E = sum(face_at_fill / fill_price)`. Marked coin equity is `wallet + F/P - E`.
Its USD value is that coin equity times the USD collateral-coin price. Funding
adds `F * rate / event_mark` coins to the old wallet; it is not separate cash
income in addition to those coins. Retained coin income creates residual exposure
until another hedge adjustment. Whole contract faces are $100 BTC and $10 ETH
in this specified research model; current contract metadata is also captured.

Brownian motion specifies uncertainty; it does not supply an expected-return
edge. OU forecasts of basis convergence were investigated in the earlier candidate
family. The original selector favored simpler passive carry. More mathematical
complexity must justify itself against that control.

## Data and timing

Historical data use public Binance hourly spot, inverse perpetual and mark bars,
actual funding settlements, Yahoo adjusted ETF opens/closes, a dated USDT/USD
conversion and BIL. Repairs and halts in the original downloads remain audited.
Free ETF histories are retrospectively adjusted, not a point-in-time vintage
archive. The stock pilot additionally has current-universe and unknown-borrow bias.

The direct replay marks crypto hourly and ETFs at US opens/closes with New York
daylight-saving conversion. Decisions use completed prices; fills require a later
eligible observation. Funding at an event belongs to positions held before its
fills. Transfers debit source cash immediately and remain idle in transit for two
business days. Destination positions are funded only after arrival. An ETF signal
after a completed close cannot fill that close. Macro target weights come from the
selected strategy's past-only features; allocator risk inputs use earlier completed
monthly reference-sleeve returns.

Historical ETF holdings are fractional **adjusted total-return units**. This is
an explicit execution proxy, not raw shares with reconstructed corporate actions.
The prospective paper engine can instead process raw-share splits and accrue
dividend receivables; verified payment-date settlement remains unfinished.

## Selection and the reporting correction

The original families explored funding controls, OU gates, diversified ETF rules,
native inverse carry and three allocators. Development used earlier observations,
2023 supplied validation, and 2024–September 2026 supplied an already-inspected
comparison. There is no pristine project-wide historical holdout.

An earlier report highlighted fixed training-risk weights after inspecting the
results, despite the 2023 selector choosing rolling risk. The resulting 2.10 monthly
annualized excess Sharpe was a research-selected headline. Its daily equivalent
was 1.87. That choice was selection bias, not a fabricated return series.

The corrected original report follows rolling risk: about **1.81 monthly**, **1.99
daily**, and **1.75 HAC-28** excess Sharpe in 2024–September 2026. It labels fixed
risk as a control. Its pointwise monthly bootstrap interval is approximately
[0.83, 3.04], before accounting for research selection. These numbers belong to the
old return-sleeve model. The direct account variant has separate outputs and can
have different costs, timing and returns. It is evaluated on reused history too.

The `selection-audit` command adds a centered paired circular-block maximum-Sharpe
diagnostic for the three allocators and an IID deflated-Sharpe sensitivity analysis.
Its catalog excludes earlier adaptive research. Dispersion estimated from three
related allocators is not a project-wide trial correction. IID probabilities are
not reliable significance estimates for persistent funding returns. See
[Bailey and Lopez de Prado](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
for the selection problem and DSR assumptions. None of these diagnostics restores
an untouched test set.

## New direct-account results

The selected rolling-risk rule scores **1.89 monthly / 1.95 daily / 1.83 HAC-28**,
about **8.39% net CAGR** and **-2.28% drawdown**. Daily scores span 1,003 intervals
from 1 January 2024 NAV through 30 September 2026 NAV. Monthly scores use 32
complete calendar months between month-start NAVs, excluding incomplete September.
Its pointwise three-month-block monthly Sharpe interval is approximately
**[0.90, 3.34]**, without selection adjustment. Fixed risk scores 2.31 monthly as a
separate control; it is not promoted over the validation winner. Exact values and
attribution are in `examples/historical-metrics.csv` and the accompanying report.

## Account attribution and validation

At every event, NAV must reconcile to bank cash, broker cash/receivables/ETF units,
venue quote cash, marked native coin equity and transfer cash. Changes reconcile
to collateral price PnL, inverse PnL, funding, ETFs, dividends where applicable,
quote-currency changes, execution basis/spread and fees. Saved decisions, order
blotters and daily holdings permit inspection; an independent reconstruction of
the earlier native wallet also remains available.

Tests cover reciprocal payoffs, funding signs and same-time priority, whole lots,
fees, stale fills, delayed/weekend settlement, collateral breaches, DST, corporate
actions, future-data invariance, immutable manifests, journal tampering and replay.
These establish implementation consistency; they do not establish alpha.

## What would strengthen or invalidate the hypothesis?

Evidence would strengthen with source-pinned pre-outcome decisions, repeated
forward quote observations, reconciled broker/exchange accounting, and performance
that survives reasonable execution assumptions across funding environments. The
current forward workflow provides infrastructure; its length starts at deployment.
It cannot manufacture a prospective history today.

The thesis weakens if net funding fails to compensate basis/financing costs, the
premium is concentrated in one episode, residual exposures explain returns, or
venue losses dominate ordinary mark-to-market risk. Stress tests are hypothetical
scenarios, not estimates of default probability. The 10% venue-capital-loss stress
already substantially reduces the original model's Sharpe.

Further research should predeclare a small comparison of passive carry, simple
volatility/funding controls and an HMM risk overlay. The HMM must use filtered
probabilities, past-only transformations and chronological model selection.
Assess downside-risk reduction and net incremental returns at comparable risk,
including the cost of reduced funding exposure. PCA should earn inclusion through
an ablation. This is a subsequent experiment, not a feature silently added to the
frozen baseline.
