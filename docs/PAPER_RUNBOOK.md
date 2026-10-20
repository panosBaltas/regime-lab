# Frozen forward paper workflow

This is a user-run research workflow, with no exchange credentials, live orders,
or background scheduler. Use Python 3.13 and the locked environment in the README.

## Freeze once

From the repository root, after the existing free-data studies have been run:

```powershell
python -m regime_lab paper-freeze --output paper/v1
python -m regime_lab paper-refresh --model paper/v1 --output outputs/reference_v1
python -m regime_lab paper-plan --model paper/v1 --inverse-results outputs/reference_v1/inverse --macro-results outputs/reference_v1/macro
python -m regime_lab paper-verify --model paper/v1
```

The freeze records source hashes, library versions, costs, whole-contract faces,
the selected strategy specifications, original input hashes and training control
weights. It refuses to replace an existing manifest. Editing source after a freeze
requires a new version. Keep the original commit/environment to replay the old one.

`paper-plan` computes decisions from completed sessions and completed monthly
reference-sleeve returns. Inputs and their file hashes are retained in the journal.
It uses the original macro rebalance calendar, monthly allocator, and 30-day carry
schedule. Repeated calls without a newly scheduled decision fail explicitly. If a
scheduled decision was missed, it is recorded when actually run, never backdated.

Refresh ETF/cash histories with `fetch` and crypto histories with `alpha-fetch`,
then rebuild the **same frozen specs** without a new parameter search:

```powershell
python -m regime_lab paper-refresh --model paper/v1 --output outputs/reference_v2
python -m regime_lab paper-plan --model paper/v1 --inverse-results outputs/reference_v2/inverse --macro-results outputs/reference_v2/macro
```

`paper-plan` needs the latest completed month for rolling risk;
it refuses an outdated month rather than silently fitting a stale risk window.
Reference risk inputs currently come from the independently simulated original
sleeves, not the combined account's endogenous allocation returns.
`paper-refresh` leaves those reference positions open at the end of the input,
avoiding an artificial terminal-liquidation fee in the latest risk observation.
The earlier finite-horizon study outputs remain unchanged. Supply the refreshed
reference paths when planning; the default old paths remain a legacy-input option.

## Capture, then mark

```powershell
python -m regime_lab paper-capture --model paper/v1 --include-etfs --output data/captures/observation-001.json
python -m regime_lab paper-step --model paper/v1 --snapshot data/captures/observation-001.json
python -m regime_lab paper-verify --model paper/v1
```

Use a new capture filename every time. Each plan waits at least one hour before
eligible fills. Transfers then take two business days, excluding weekends. Crypto
can fill at a subsequent fresh observation; ETF orders also need the regular market
and fresh valid quotes for all required assets. Missing/stale quotes leave orders
unfilled. Running a command infrequently produces a correspondingly coarse paper
record; this does not reconstruct unobserved intraday trading or margin breaches.

Capture queries public Binance spot/native inverse books, native marks, contract
metadata and all funding events since the previous mark. Funding-event marks must
come from the event or its historical mark bar, never today's mark. It also reads
an indicative USDT/USD proxy and optionally Yahoo ETF bid/ask and last-trade data.
An explicit `--quote-usd VALUE` is recorded as an operator FX assumption; there is
no silent USDT/USD parity fallback. Provider failures stop capture without fills.

Public quote freshness has limits: spot uses HTTP receipt time, and Yahoo's last
trade timestamp does not establish the bid/ask's age. Fees plus any observed spread
are charged. These indicative quotes do not establish executable depth, capacity,
queue priority or broker fills. Delayed providers may prevent ETF fills entirely.
The workflow accepts a normalized snapshot from another source using the same
schema; retain its raw response, origin and observation times.

ETF capture checks dividends and splits across every observation interval once
shares are held. Splits adjust units; ex-dividend income becomes an **unspendable
receivable**, rather than immediately reusable cash. Verified dividend payment
dates and automatic receivable settlement are not yet implemented. Do not use the
paper log as a reconciled broker statement. Historical replay instead uses adjusted
ETF total-return units and does not apply raw-share actions a second time.

## Verify and retain a checkpoint

Every record includes its predecessor's SHA-256 and the resulting account-state
hash. Verification replays plans and observations from initial cash and compares
the computed states. Save the printed chain head independently, then use:

```powershell
python -m regime_lab paper-verify --model paper/v1 --expected-head YOUR_SAVED_HEAD
```

An expected head detects deletion of the tail as well as ordinary content changes.
If legitimate new records have been appended, use the new saved head. The local
chain alone cannot stop an owner from rewriting the whole history, nor establish
independently trusted timestamps. Source-pinned commits and independently retained
checkpoints improve the evidence. The CLI does not report Sharpe from a short or
irregular journal.

An interrupted writer can leave `.writer.lock`. Inspect the incomplete record and
establish that no other writer is running before manually recovering it. Concurrent
writers fail instead of silently interleaving records.

## Snapshot contract

`time`, `funding_from` and `funding_through` are UTC timestamps. Coverage must start
at the preceding account mark (or model freeze) and end at this snapshot. Live paper
observations must be captured within two minutes of recording and after the freeze.

`prices` contains two-element `spot`, `future`, `mark` arrays, `quote_usd`, optional
two-element executable bid/ask arrays, and ETF symbol-to-price/bid/ask dictionaries.
`funding_coin_per_face` contains the sum of each settled rate divided by its event
mark. `quote_times` and trade-availability flags distinguish valuations from fills.
`corporate_actions` retains effective times, split ratios and per-unit dividends;
`corporate_actions_checked` must be true when ETF positions are held. The raw
responses and normalized data are retained in the capture/journal.

For a complete offline example, run `project-demo` and inspect its `paper_demo/`
subdirectory. Its manifest says `synthetic`; those observations are not forward data.
