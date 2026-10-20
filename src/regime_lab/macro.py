"""Slow multi-asset strategies, with actual drift and no final-period selection."""

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .data import file_digest

GROUPS = {"equities": ("SPY", "QQQ", "IWM", "EFA", "EEM"), "bonds": ("TLT", "IEF", "LQD", "HYG"),
          "real_assets": ("GLD", "DBC"), "currency": ("UUP",)}


@dataclass(frozen=True)
class MacroSpec:
    family: str
    lookback: int = 126
    rebalance: int = 21
    top: int = 3
    cost: float = 0.0005
    borrow: float = 0.02

    @property
    def name(self):
        return f"{self.family}_{self.lookback}_{self.rebalance}_top{self.top}"


def macro_catalog():
    specs = [MacroSpec(family, lookback, frequency) for family in ("trend_long", "trend_both")
             for lookback in (63, 126, 252) for frequency in (5, 21)]
    specs += [MacroSpec(family, 126, frequency) for family in ("multispeed_long", "multispeed_both", "risk_parity", "inverse_variance")
              for frequency in (5, 21)]
    specs += [MacroSpec(family, 252, 21, top) for family in ("dual_momentum", "fast_momentum", "breadth_defensive") for top in (1, 3, 5)]
    return specs


def macro_panel(data_path, cash_path):
    prices = pd.read_csv(data_path, parse_dates=["date"])
    cash = pd.read_csv(cash_path, parse_dates=["date"])
    all_prices = pd.concat([prices, cash])
    if all_prices.duplicated(["date", "symbol"]).any():
        raise ValueError("Duplicate macro prices")
    opens = all_prices.pivot(index="date", columns="symbol", values="open").sort_index()
    closes = all_prices.pivot(index="date", columns="symbol", values="close").reindex(opens.index)
    # Cash was downloaded with a longer history than the risky-asset panel.
    dates = pd.DatetimeIndex(prices.date.unique()).sort_values()
    opens, closes = opens.reindex(dates), closes.reindex(dates)
    names = [name for group in GROUPS.values() for name in group]
    if opens[names+["BIL"]].isna().any().any() or closes[names+["BIL"]].isna().any().any():
        raise ValueError("Unbalanced macro panel")
    return dates, names, opens[names].to_numpy(), closes[names].to_numpy(), opens.BIL.to_numpy(), closes.BIL.to_numpy()


def macro_targets(dates, names, closes, cash_closes, spec):
    frame = pd.DataFrame(closes)
    returns = frame.pct_change(fill_method=None)
    sigma = returns.rolling(63, min_periods=63).std().clip(lower=0.002)
    momentum = frame / frame.shift(spec.lookback) - 1
    if spec.family.startswith("multispeed"):
        signal = sum(np.tanh((frame/frame.shift(h)-1)/(sigma*np.sqrt(h))) for h in (21,63,126,252))/4
    else:
        signal = np.sign(momentum)
    if spec.family in ("risk_parity", "inverse_variance"):
        signal[:] = 1.0
    if spec.family.endswith("_long"):
        signal = signal.clip(lower=0)
    if spec.family == "inverse_variance":
        signal = signal / sigma
    weights = np.zeros_like(closes)
    for group in GROUPS.values():
        indices = [names.index(name) for name in group]
        raw = (signal.iloc[:, indices] / sigma.iloc[:, indices]).to_numpy()
        denom = np.abs(raw).sum(axis=1)
        weights[:, indices] = np.divide(raw, denom[:,None], out=np.zeros_like(raw), where=denom[:,None]>0)/len(GROUPS)
    if spec.family in ("dual_momentum", "fast_momentum", "breadth_defensive"):
        slow = momentum.to_numpy()
        fast = sum(weight*(frame/frame.shift(h)-1).to_numpy() for h,weight in ((21,12),(63,4),(126,2),(252,1)))/19
        score = slow if spec.family == "dual_momentum" else fast
        cash_score = cash_closes / np.r_[np.full(252,np.nan),cash_closes[:-252]] - 1
        weights[:] = 0
        canaries = [names.index(name) for name in ("SPY","EFA","EEM","HYG")]
        defensive = [names.index(name) for name in ("IEF","TLT","GLD")]
        for t in range(252,len(dates)):
            order = np.argsort(score[t])[::-1]
            chosen = order[:spec.top]
            chosen = chosen[score[t,chosen]>cash_score[t]]
            if spec.family == "breadth_defensive" and np.any(fast[t,canaries]<0):
                ranked = np.asarray(defensive)[np.argsort(fast[t,defensive])[::-1]]
                chosen = ranked[:1]
                chosen = chosen[fast[t,chosen]>max(0,cash_score[t])]
            if len(chosen):
                raw = 1/sigma.iloc[t,chosen].to_numpy()
                weights[t,chosen] = raw/raw.sum()
    # Risk uses the signal date's past covariance, with 50% diagonal shrinkage.
    r = returns.to_numpy()
    for t in range(252,len(dates)):
        covariance = np.cov(r[t-62:t+1],rowvar=False)
        covariance = .5*covariance + .5*np.diag(np.diag(covariance))
        annual_vol = np.sqrt(max(0,weights[t]@covariance@weights[t])*252)
        weights[t] *= min(1, .10/max(annual_vol,1e-8))
    return weights


def simulate_macro(panel, spec):
    dates,names,opens,closes,cash_open,cash_close = panel
    targets = macro_targets(dates,names,closes,cash_close,spec)
    start = max(252,int(dates.searchsorted("2010-01-01")))
    previous = np.zeros(len(names))
    equity, rows = 1.0, []
    for t in range(start,len(dates)-2):
        rebalance = (t-start)%spec.rebalance == 0
        target = targets[t].copy() if rebalance else previous.copy()
        fee = spec.cost
        q = 1.0
        if rebalance:
            for _ in range(15):
                q = 1-fee*np.abs(q*target-previous).sum()
        turnover = float(np.abs(q*target-previous).sum())
        changes = opens[t+2]/opens[t+1]-1
        rf = cash_open[t+2]/cash_open[t+1]-1
        long,short = np.maximum(target,0).sum(),-np.minimum(target,0).sum()
        days = (dates[t+2]-dates[t+1]).days/365.25
        carry = max(0,1-long)*rf -short*spec.borrow*days-max(0,long-1)*(rf+.01*days)
        before_exit = 1+float(target@changes)+carry
        liquidation = float(np.abs(target*(1+changes)).sum()) if t == len(dates)-3 else 0
        factor = q*(before_exit-fee*liquidation)
        equity *= factor
        rows.append({"time":dates[t+2],"net_return":factor-1,"cash_return":rf,"excess_return":factor-1-rf,
                     "equity":equity,"gross":float(np.abs(target).sum()),"turnover":turnover,"risky_fraction":long})
        previous = target*(1+changes)/before_exit
    return pd.DataFrame(rows).set_index("time")


def macro_metrics(ledger):
    r,e = ledger.net_return.to_numpy(),ledger.excess_return.to_numpy()
    wealth = np.cumprod(1+r)
    return {"sharpe_excess":float(e.mean()/e.std(ddof=1)*np.sqrt(252)),
            "cagr":float(wealth[-1]**(252/len(r))-1),"drawdown":float(np.min(wealth/np.maximum.accumulate(np.r_[1.,wealth])[1:]-1)),
            "volatility":float(r.std(ddof=1)*np.sqrt(252)),"average_gross":float(ledger.gross.mean())}


def run_macro_study(data_path,cash_path,output,progress=print):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    panel=macro_panel(data_path,cash_path)
    ledgers,rows,scores={},[],[]
    for spec in macro_catalog():
        ledger=simulate_macro(panel,spec)
        ledgers[spec.name]=ledger
        values={}
        for period,mask in (("development",ledger.index<"2023-01-01"),("validation",(ledger.index>="2023-01-01")&(ledger.index<"2024-01-01"))):
            values[period]=macro_metrics(ledger.loc[mask])
            rows.append({"method":spec.name,"period":period,**values[period]})
        score=min(values[p]["sharpe_excess"] for p in values)
        scores.append((score,spec))
        progress(f"Macro development {spec.name}: {values['development']['sharpe_excess']:.2f} / {values['validation']['sharpe_excess']:.2f}")
    selected=max(scores,key=lambda pair:pair[0])[1]
    selection={"selected":selected.name,"spec":asdict(selected),"criterion":"maximum worst development/2023 validation excess Sharpe",
               "latest_selection_data":"2023-12-31","historical_context_previously_seen":True,"data_hash":file_digest(data_path),"source_hash":file_digest(__file__)}
    (output/"selection.json").write_text(json.dumps(selection,indent=2),encoding="utf-8")
    for name,ledger in ledgers.items():
        for period,mask in (("full",np.ones(len(ledger),bool)),("evaluation",ledger.index>="2024-01-01")):
            rows.append({"method":name,"period":period,**macro_metrics(ledger.loc[mask])})
    metrics=pd.DataFrame(rows);metrics.to_csv(output/"metrics.csv",index=False)
    pd.concat(ledgers,names=["method"]).to_csv(output/"daily_returns.csv")
    progress(metrics[(metrics.method==selected.name)&(metrics.period=="evaluation")].to_string(index=False))
    return metrics,selection
