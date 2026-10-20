"""Past-only capital allocation across the selected carry and ETF sleeves."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .data import file_digest

RULES=("static_risk","rolling_risk","shrunk_mean_variance")


def selected_allocator(selection):
    """The saved validation decision is the sole source of the report headline."""
    name = selection["selected"]
    if name not in RULES:
        raise ValueError(f"Unknown saved allocator: {name}")
    return name


def allocation(history,rule):
    covariance=history.cov().to_numpy()
    sigma=np.sqrt(np.maximum(np.diag(covariance),1e-10))
    raw=1/sigma
    if rule=="shrunk_mean_variance":
        covariance=.5*covariance+.5*np.diag(np.diag(covariance))
        mean=history.mean().to_numpy()
        mean=.5*mean+.5*mean.mean()
        candidate=np.maximum(np.linalg.solve(covariance+np.eye(2)*1e-8,mean),0)
        if candidate.sum()>0:raw=candidate
    w=float(np.clip(raw[0]/raw.sum(),.1,.9))
    return np.array([w,1-w])


def load_sleeves(inverse_root,macro_root,inverse_method=None):
    inverse_root,macro_root=Path(inverse_root),Path(macro_root)
    selections=[json.loads((p/"selection.json").read_text()) for p in (inverse_root,macro_root)]
    native=pd.read_csv(inverse_root/"daily_returns.csv",parse_dates=["time"])
    native=native[native.method==(inverse_method or selections[0]["selected"])].set_index("time")
    macro=pd.read_csv(macro_root/"daily_returns.csv",parse_dates=["time"])
    macro=macro[macro.method==selections[1]["selected"]].set_index("time")
    # The US opening NAV becomes known before the FOLLOWING UTC midnight.
    macro_wealth=(1+macro.net_return).cumprod()
    macro_wealth.index+=pd.Timedelta(days=1)
    grid=pd.date_range(native.index.min(),native.index.max(),freq="D",name="time")
    macro_daily=macro_wealth.reindex(macro_wealth.index.union(grid)).ffill().reindex(grid)
    risky=pd.DataFrame({"inverse_carry":native.equity_quote.reindex(grid),"macro":macro_daily})
    r=risky.pct_change(fill_method=None).iloc[1:]
    rf=native.cash_return.reindex(r.index)
    if not np.isfinite(r.to_numpy()).all() or not np.isfinite(rf).all():raise ValueError("Unaligned portfolio sleeves")
    return r,rf,selections


def monthly_returns(ledger):
    values=ledger[["net_return","cash_return"]].resample("ME").apply(lambda r:np.prod(1+r)-1)
    values["excess_return"]=values.net_return-values.cash_return
    return values


def metrics(ledger):
    r,e=ledger.net_return.to_numpy(),ledger.excess_return.to_numpy()
    wealth=np.cumprod(1+r)
    monthly=monthly_returns(ledger)
    me=monthly.excess_return.to_numpy()
    centered=e-e.mean();gamma=np.mean(centered**2)
    lag=min(28,len(e)//4)
    long_var=gamma+2*sum((1-k/(lag+1))*np.mean(centered[k:]*centered[:-k]) for k in range(1,lag+1))
    return {"days":len(e),"daily_excess_sharpe":float(e.mean()/e.std(ddof=1)*np.sqrt(365)),
            "monthly_excess_sharpe":float(me.mean()/me.std(ddof=1)*np.sqrt(12)),
            "hac28_sharpe":float(e.mean()/np.sqrt(max(long_var,1e-16))*np.sqrt(365)),
            "cagr":float(wealth[-1]**(365/len(r))-1),"volatility":float(r.std(ddof=1)*np.sqrt(365)),
            "drawdown":float(np.min(wealth/np.maximum.accumulate(np.r_[1.,wealth])[1:]-1))}


def simulate_allocation(returns,rf,rule,transfer_cost=.001,static_weights=None,weight_schedule=None):
    monthly=(1+returns).resample("ME").prod()-1
    monthly_cash=(1+rf).resample("ME").prod()-1
    excess=monthly.subtract(monthly_cash,axis=0)
    training=excess[(excess.index>="2021-01-01")&(excess.index<"2023-01-01")]
    static=allocation(training,"static_risk") if static_weights is None else np.asarray(static_weights)
    values=np.zeros(2);previous_equity=1.
    # Capital transfers include underlying spot/future or ETF execution as well
    # as a fiat/quote/venue allowance. Cash in transit earns zero for two days.
    costs=np.array([transfer_cost+.0019,transfer_cost+.0005])
    pending=[];rows=[];decisions=[]
    for day,r in returns.loc[returns.index>="2023-01-01"].iterrows():
        values*=1+r.to_numpy()
        arrived=[item for item in pending if item[0]<=day]
        for _,amount in arrived:values+=amount
        pending=[item for item in pending if item[0]>day]
        fees=0.
        if (day.day==2 and not pending) or not rows:
            # Month-end observations are complete before this next-day allocation.
            history=excess[excess.index<day.normalize().replace(day=1)].tail(12)
            if len(history)<12:raise ValueError("Insufficient past ensemble history")
            weights=static if rule=="static_risk" else allocation(history,rule)
            if weight_schedule is not None:
                weights=np.asarray(weight_schedule[str(day)],dtype=float)
                if weights.shape!=(2,) or not np.isfinite(weights).all() or np.any(weights<0) or not np.isclose(weights.sum(),1):
                    raise ValueError("Invalid frozen allocation weights")
            equity=float(values.sum()) if rows else 1.
            previous=values/equity;q=1.
            for _ in range(12):q=1-costs@np.abs(q*weights-previous)
            desired=equity*q*weights
            delta=desired-values
            fees=equity*(1-q)
            values+=np.minimum(delta,0)
            incoming=np.maximum(delta,0)
            if incoming.sum()>0:pending.append((day+pd.Timedelta(days=2),incoming))
            decisions.append({"fill_time":str(day),"latest_return_observation":str(history.index.max()),
                              "inverse_weight":float(weights[0]),"macro_weight":float(weights[1]),"fee_fraction":1-q})
        transit=sum(float(amount.sum()) for _,amount in pending)
        equity=float(values.sum())+transit;ret=equity/previous_equity-1
        rows.append({"time":day,"net_return":ret,"cash_return":rf.loc[day],"excess_return":ret-rf.loc[day],
                     "equity":equity,"inverse_weight":values[0]/equity,"transfer_fee":fees/previous_equity,"transit_cash":transit/equity})
        previous_equity=equity
    ledger=pd.DataFrame(rows).set_index("time")
    # Carry already pays its final spot/future liquidation in the input ledger.
    exit_fee=float(values@np.array([transfer_cost,transfer_cost+.0005]))
    beginning=equity/(1+ledger.net_return.iloc[-1])
    ledger.loc[ledger.index[-1],"net_return"]=(equity-exit_fee)/beginning-1
    ledger.loc[ledger.index[-1],"excess_return"]=ledger.net_return.iloc[-1]-ledger.cash_return.iloc[-1]
    ledger.loc[ledger.index[-1],"equity"]=equity-exit_fee
    ledger.loc[ledger.index[-1],"transfer_fee"]+=exit_fee/beginning
    return ledger,decisions


def run_ensemble_study(inverse_root,macro_root,output,progress=print):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    returns,rf,selections=load_sleeves(inverse_root,macro_root)
    ledgers={};decisions={};rows=[]
    for rule in RULES:
        ledger,records=simulate_allocation(returns,rf,rule)
        ledgers[rule],decisions[rule]=ledger,records
        validation=metrics(ledger[ledger.index<"2024-01-01"])
        rows.append({"method":rule,"period":"validation",**validation})
    chosen=max(rows,key=lambda r:r["monthly_excess_sharpe"])["method"]
    selection={"selected":chosen,"criterion":"2023 monthly excess Sharpe","latest_selection_data":"2023-12-31",
               "evaluation_context_reused":True,"sleeves":[s["selected"] for s in selections],"source_hash":file_digest(__file__)}
    (output/"selection.json").write_text(json.dumps(selection,indent=2),encoding="utf-8")
    for rule,ledger in ledgers.items():
        evaluation=ledger[ledger.index>="2024-01-01"]
        rows.append({"method":rule,"period":"evaluation",**metrics(evaluation)})
    result=pd.DataFrame(rows);result.to_csv(output/"metrics.csv",index=False)
    pd.concat(ledgers,names=["method"]).to_csv(output/"daily_returns.csv")
    (output/"selected_decisions.json").write_text(json.dumps(decisions[chosen],indent=2),encoding="utf-8")
    progress(result.to_string(index=False))
    return result,selection
