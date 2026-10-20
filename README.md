# Regime Lab

**Auditable funding carry, explicit account simulation and forward paper validation.**

The project asks how much native BTC/ETH funding carry survives trading costs,
collateral and capital movement, and what diversified ETFs change. It preserves
PCA/HMM forecasting and the failed stock-arbitrage pilot as research history.
No future Sharpe of two is established.

## Offline quick start

Python 3.13 is the locked reference environment; the package supports 3.11+.
From the repository root in PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m regime_lab project-demo --output outputs/demo_v1
```

On Linux/macOS use python3.13 and .venv/bin/python. Open the resulting report.html.
The demo exercises positions, funding, fees, transfers and a replayable paper
journal. **Its observations are synthetic; its Sharpe is not trading evidence.**
Use a new output directory each time. Windows/Linux CI generates the same demo.
Remote CI requires pushing this repository. The original local .venv used system
packages; the separate .venv-repro environment is isolated.

## What is implemented

- Coin wallets, whole inverse contracts, reciprocal PnL and positive/negative funding.
- ETF holding drift, cash, costs and past-only allocation features.
- Combined venue/broker cash, idle transfers and subsequent eligible fills.
- Event-level PnL attribution, order blotters, daily holdings and reconciliation.
- Reports that follow the saved validation selector and retain controls/failures.
- Source/library-pinned paper models, pre-outcome plans, public snapshots,
  quote-age checks, linked journal records and account-state replay.

The direct variant implements passive carry plus a long-only ETF strategy. A new
HMM/PCA overlay is a subsequent experiment; the original HMM research is preserved.

## Market reproduction

The original free-data download and study commands are in
[LEGACY_RESEARCH.md](LEGACY_RESEARCH.md). With those input data and family outputs:

```powershell
python -m regime_lab alpha-report --output outputs/selected_report_v2
python -m regime_lab selection-audit --output outputs/selection_audit_v1.json
python -m regime_lab execution-run --output outputs/execution_v1
```

The new execution model trades underlying holdings, rather than historical return
series. It compares the validation-selected allocator and a fixed training-risk
control with the same accounting. Table scores use 2024 onward; the full replay
starts in 2023. This history is already inspected and all results are exploratory.
See --help for input paths and date bounds. Raw downloads stay outside Git.

## Selection and evidence

An earlier report highlighted fixed risk after inspecting evaluation results.
Its 2.10 monthly annualized excess Sharpe was research-selected. The corrected
original report follows rolling risk: approximately **1.81 monthly / 1.99 daily /
1.75 HAC-28** for 2024-September 2026. These belong to the original return-sleeve
model; the new account variant has separate performance and assumptions.

Sharpe frequency, benchmark, costs and dependence matter. Strategy Sharpes subtract
matched BIL returns. Pointwise intervals and allocator-only selection diagnostics
do not undo earlier adaptive research or create an untouched holdout.

## Forward paper record

```powershell
python -m regime_lab paper-freeze --output paper/v1
python -m regime_lab paper-refresh --model paper/v1 --output outputs/reference_v1
python -m regime_lab paper-plan --model paper/v1 --inverse-results outputs/reference_v1/inverse --macro-results outputs/reference_v1/macro
python -m regime_lab paper-capture --model paper/v1 --include-etfs --output data/captures/observation-001.json
python -m regime_lab paper-step --model paper/v1 --snapshot data/captures/observation-001.json
python -m regime_lab paper-verify --model paper/v1
```

Plans wait at least an hour before eligible trading; transfers take two business
days. Capture/mark again later with a new filename. Stale ETF quotes prohibit fills.
There are no live broker orders, credentials or background jobs. A short paper log
is operational evidence, not a Sharpe estimate. Read the runbook before using it.

## Artifacts and explanation

Each run contains a portable report, metrics, yearly scores, daily NAV, event
ledgers, orders, decisions, daily holdings, PnL attribution and reconciliation.
Ending NAV marks open positions; it is not guaranteed liquidation proceeds.

- [Curated reports](examples/README.md)
- [Research note](docs/RESEARCH_NOTE.md): economics, equations, selection and limitations.
- [Protocol](docs/PROJECT_PROTOCOL.md): scope and acceptance criteria.
- [Paper runbook](docs/PAPER_RUNBOOK.md): operation and checkpointing.
- [Interview guide](docs/INTERVIEW_GUIDE.md): demonstration and technical questions.
- [Research log](RESEARCH_LOG.md) and [earlier research](LEGACY_RESEARCH.md).

Remaining limits include asynchronous marks, adjusted-price ETF proxies, indicative
quotes without depth, exchange/custody/default risk, simplified settlement calendars
and maintenance rules. Paper splits and dividend receivables are supported; verified
payment-date settlement and broker reconciliation remain future work.
