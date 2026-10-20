# Execution and forward-validation project protocol

Implementation started 20 October 2026. All existing historical results have been
inspected. New replays of that history are exploratory, including the new direct
execution model. They are not a new independent test and have no Sharpe target
used as an acceptance criterion.

## Question and scope

How much of native BTC/ETH funding carry survives explicit trading, financing,
and capital movement, and what does adding a diversified ETF allocation change?
The historical validation-selected carry, ETF and allocator specifications remain
the reference. The report must use the recorded selector, never a report-level
override. The fixed training-risk allocation remains a separately labeled control.
No new HMM/PCA model is promoted without a separate prospective experiment.

## New execution variant (v1)

Maintain actual coin wallets, whole USD-face inverse contracts, ETF total-return
units, broker cash, USDT venue cash and transfers in transit. Funding settles to
the position that existed before a same-time fill. Orders use subsequent available
prices. Both legs of carry trades, ETF turnover, and capital transfers incur fees.
Transferred cash earns zero; two business days is an explicit configurable model
of settlement, excluding weekends but not a bank-holiday calendar. Exchange cash
earns no interest; uninvested broker cash can be allocated to BIL.

The historical ETF data are retrospectively adjusted OHLC. Positions therefore
represent fractional total-return units, not a claim of a raw-share corporate-
action ledger. Crypto valuation uses the dated USDT/USD conversion. Intrahour
high/low inverse marks test the specified 10% maintenance buffer. This buffer is
a conservative research rule, not a reconstruction of historical exchange tiers.
Missing executable prices prohibit fills; stale marks are recorded. Staleness,
transfer delays, whole-contract rounding, and unfilled orders must remain visible.

Compare the recorded rolling-risk selector, fixed training-risk control and BIL
benchmark with identical accounting. Never select the new historical best as a
forward model. Report daily and monthly excess Sharpe, a dependence-adjusted
diagnostic, drawdown, funding/basis/price/currency/fee attribution, and yearly
results. Preserve a self-financing NAV reconciliation at every event.

## Evidence and paper record

Freeze source hashes, selections, costs, instruments, data provenance and model
parameters in an exclusive-create manifest. A paper journal records timestamped
plans before fills, their input hashes, subsequent observations, actual simulated
fills and account states. Historical demonstrations must be marked synthetic or
historical replay. A live journal cannot be backdated or accept old quotes as new
executable observations. No live broker orders are part of this project.

Each change starts a new version. Hash chaining detects edits relative to a
trusted saved chain head; local files alone are not a trusted timestamp service or
tamper-proof archive. Unscheduled monitoring is user-run; no background service is
implicitly installed. Short forward records establish operational behavior, not a
precise long-run Sharpe.

## Statistical and engineering acceptance

Pointwise dependent-month intervals and any multiple-trial diagnostics are labeled
with their assumptions and catalog scope. IID deflated Sharpe is a sensitivity
diagnostic, not valid evidence by itself for serially correlated carry returns.
Selection-adjusted inference cannot undo previously inspected history.

Acceptance requires causal decision tests, independent account reconciliation,
funding signs and inverse payoffs, fees, whole lots, collateral and transfer timing,
immutable freeze/journal validation, a clean-install offline test run and portable
demo artifacts. Failure to achieve Sharpe two is not a software test failure.
