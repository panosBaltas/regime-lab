# Next research and engineering work

The next milestone is a useful prospective record, not another historical Sharpe
target. The current one-observation checkpoint establishes an operational start,
not trading performance.

## Forward evidence

Run the user-controlled paper tick regularly, retain checkpoints independently,
and document provider outages, missed sessions, stale quotes, and rejected fills.
Never reconstruct missed fills as though they were observed. Freeze any new model
before its observations begin. Compare daily, calendar-aligned NAV only after
enough observations exist; an irregular journal is not a daily return series.

## Execution and settlement

Add verified ETF dividend payment dates and reconcile receivables against a broker
statement or authoritative event feed. Extend the business-day approximation to
venue/broker calendars. Test actual settlement, funding coverage, quote spread,
depth, and contract changes before claiming execution realism or capacity. Keep
custody/default and liquidity losses distinct from ordinary return volatility.

## A controlled model comparison

Before new outcomes, write and freeze a comparison protocol. Use the same data
availability, decision clock, account engine, turnover, and costs for every arm:

| Arm | Question |
| --- | --- |
| Passive carry + ETF policy | Does the current economic baseline survive forward operation? |
| Volatility-only risk adjustment | Does simple risk scaling explain any improvement? |
| HMM on directly observed features | Does state conditioning help beyond volatility? |
| PCA followed by HMM | Does dimensionality reduction improve stable state estimation? |

For PCA/HMM, fit transforms and state models only on the past; use filtered rather
than full-sample smoothed state probabilities for decisions. Account for arbitrary
state-label permutations between refits. Keep state interpretation separate from
return prediction: a state can explain volatility without forecasting excess PnL.

Predeclare refit schedules, components/states, signal-to-position rules, controls,
the primary excess-return metric, block-based uncertainty, minimum observation
horizon, and stopping rule. Retain all attempts. The already-inspected historical
sample can debug the pipeline but cannot become an untouched holdout again.

## Portfolio presentation

Keep the README concise, retain a runnable offline demo, publish only curated
reports, and explain the selection correction candidly. Strong interview evidence
is a defensible economic hypothesis, correct account arithmetic, causal timing,
reproducibility, and understanding of why a promising model may fail.
