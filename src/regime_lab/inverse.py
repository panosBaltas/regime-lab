"""Coin-collateralized inverse contracts: reciprocal-price PnL and coin wallets."""

from dataclasses import asdict, replace
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .carry import CarryMarket, carry_catalog, carry_metrics, carry_signals, load_carry_market, read_frame
from .data import file_digest
from .inverse_data import PAIRS


def load_inverse_market(directory, linear_directory, cash_path):
    root=Path(directory)
    linear=load_carry_market(linear_directory,cash_path)
    native={(s,k):read_frame(root/f"{s}_{k}.csv") for s in PAIRS for k in ("perpetual","mark","funding")}
    start=max(native[s,"mark"].index.min() for s in PAIRS)
    stop=min(native[s,"mark"].index.max() for s in PAIRS)
    mask=(linear.times>=start)&(linear.times<=stop)
    times=linear.times[mask]
    spot=linear.spot[mask]*linear.usd[mask,None]
    future=np.column_stack([native[s,"perpetual"].open.reindex(times) for s in PAIRS])
    mark=np.column_stack([native[s,"mark"].open.reindex(times) for s in PAIRS])
    mark_close=np.column_stack([native[s,"mark"].close.reindex(times) for s in PAIRS])
    high=np.column_stack([native[s,"mark"].high.reindex(times) for s in PAIRS])
    low=np.column_stack([native[s,"mark"].low.reindex(times) for s in PAIRS])
    rate=np.zeros_like(mark)
    audit={"mark_source":"official native hourly mark at funding timestamp", "funding_events":{},"spot_halts":linear.audit}
    for i,symbol in enumerate(PAIRS):
        frame=native[symbol,"funding"]
        hour=frame.index.floor("h")
        positions=times.get_indexer(hour)
        if np.max(np.abs((frame.index-hour).total_seconds()))>60 or hour.has_duplicates:raise ValueError("Unexpected inverse funding timestamps")
        valid=positions>=0
        rate[positions[valid],i]=frame.rate.to_numpy()[valid]
        audit["funding_events"][symbol]=int(valid.sum())
    if not np.isfinite(np.r_[future.ravel(),mark.ravel(),mark_close.ravel(),high.ravel(),low.ravel()]).all():
        raise ValueError("Incomplete native inverse prices")
    basis=np.log(mark_close/(linear.spot[mask]*linear.usd[mask,None]))
    market=CarryMarket(times,spot,future,mark,mark_close,high,basis,rate,rate/mark,
                       linear.tradable[mask],linear.cash_index[mask],linear.cash_yield[mask],linear.usd[mask],audit)
    return market,low


def change_short(wallet,face,entry_sum,delta,price,fee_rate):
    """USD short face >0; wallet and entry_sum are coin amounts."""
    if delta>=0:
        fee=delta/price*fee_rate
        return wallet-fee,face+delta,entry_sum+delta/price,fee
    closed=min(face,-delta)
    realized=closed/price-(closed/face)*entry_sum if face>0 else 0
    fee=closed/price*fee_rate
    new_face=face-closed
    new_entry=entry_sum*(new_face/face) if face>0 else 0
    return wallet+realized-fee,new_face,new_entry,fee


def simulate_inverse(market,spec,low=None,start="2020-11-02",capital=100_000.,funding_haircut=1.,contract_scale=1.,liquidate=True):
    if not 0<spec.allocation<1 or not 0<funding_haircut<=1:raise ValueError("Invalid inverse allocation/haircut")
    active,apr=carry_signals(market,spec)
    first=int(market.times.searchsorted(pd.Timestamp(start)))
    last=len(market.times)-1;last-=market.times[last].hour
    lot=np.array([100.,10.])*contract_scale
    cash=np.full(2,capital/2/market.usd[first])
    wallet=np.zeros(2);face=np.zeros(2);entry=np.zeros(2)
    previous_nav=capital;previous_t=first
    pending={};rows=[];trades=[];decisions=[]
    fees_day=funding_day=turnover_day=0.
    breaches=0;min_margin=np.inf;max_abs_delta=0.
    for t in range(first,last+1):
        clock=market.times[t]
        received=face*market.funding_cash[t]
        received=np.where(received>0,received*funding_haircut,received)
        wallet+=received
        funding_day+=float(received@market.spot[t])
        if clock.hour==0 and (clock.normalize()-pd.Timestamp(start).normalize()).days%spec.rebalance==0:
            pending[t+spec.delay_hours]=active[t].copy()
            decisions.append({"signal_time":str(clock),"estimated_apr":apr[t].tolist(),"active":active[t].tolist()})
        if t in pending:
            target=pending.pop(t)
            for i in range(2):
                if not market.tradable[t,i]:
                    pending[t+1]=target
                    continue
                before_face=face[i]
                spot_quote=market.spot[t,i]/market.usd[t]
                if not target[i]:
                    wallet[i],face[i],entry[i],coin_fee=change_short(wallet[i],face[i],entry[i],-face[i],market.future[t,i],spec.future_cost)
                    fees_day+=coin_fee*market.spot[t,i]
                    turnover_day+=before_face
                    if wallet[i]>0:
                        gross_quote=wallet[i]*spot_quote
                        cash[i]+=gross_quote*(1-spec.spot_cost)
                        fees_day+=gross_quote*market.usd[t]*spec.spot_cost
                        turnover_day+=gross_quote*market.usd[t]
                        wallet[i]=0
                else:
                    if face[i]==0 and wallet[i]==0:
                        budget=spec.allocation*cash[i]
                        wallet[i]=budget/(spot_quote*(1+spec.spot_cost))
                        cash[i]-=budget
                        fees_day+=wallet[i]*market.spot[t,i]*spec.spot_cost
                        turnover_day+=wallet[i]*market.spot[t,i]
                    # Wallet minus entry coin amount is the directional delta.
                    delta_coin=wallet[i]-entry[i]
                    denominator=1+spec.future_cost if delta_coin>=0 else 1-spec.future_cost
                    raw=delta_coin*market.future[t,i]/denominator
                    delta=np.sign(raw)*np.floor(abs(raw)/lot[i])*lot[i]
                    delta=max(-face[i],delta)
                    wallet[i],face[i],entry[i],coin_fee=change_short(wallet[i],face[i],entry[i],delta,market.future[t,i],spec.future_cost)
                    fees_day+=coin_fee*market.spot[t,i]
                    turnover_day+=abs(delta)
                trades.append({"time":str(clock),"asset":PAIRS[i],"cash_quote":cash[i],"coin_wallet":wallet[i],
                               "short_face_usd":face[i],"entry_coin_term":entry[i],"future_trade_price":market.future[t,i],
                               "spot_usd":market.spot[t,i],"actual_contracts":face[i]/lot[i]})
        coin_equity=wallet+face/market.mark[t]-entry
        nav=float(cash.sum()*market.usd[t]+coin_equity@market.spot[t])
        if nav<=0:raise ValueError("Inverse equity exhausted")
        coin_delta=wallet-entry
        max_abs_delta=max(max_abs_delta,float(np.abs(coin_delta*market.spot[t]).sum()/nav))
        for bound in (market.high[t],low[t] if low is not None else market.mark[t]):
            equity_bound=wallet+face/bound-entry
            denominator=face/bound
            margin=np.divide(equity_bound,denominator,out=np.full(2,np.inf),where=denominator>1e-12)
            min_margin=min(min_margin,float(margin.min()))
            breaches+=int(np.any(margin<spec.maintenance))
        if clock.hour==0 and t>first:
            if np.any((face>0)&~market.tradable[t]):raise ValueError("Stale daily inverse valuation")
            ret=nav/previous_nav-1
            rf=market.cash_index[t]/market.cash_index[previous_t]-1
            rows.append({"time":clock,"net_return":ret,"usd_return":ret,"cash_return":rf,"excess_return":ret-rf,
                         "equity_quote":nav/capital,"spot_fraction":float(coin_equity@market.spot[t])/nav,
                         "funding":funding_day/previous_nav,"fees":fees_day/previous_nav,"turnover":turnover_day/previous_nav,
                         "margin_ratio":min_margin,"cash":float(cash.sum()*market.usd[t]),"risk_trades":0,
                         "residual_coin_delta":float(coin_delta@market.spot[t])/nav})
            previous_nav,previous_t=nav,t;fees_day=funding_day=turnover_day=0.
    ledger=pd.DataFrame(rows).set_index("time")
    # Realize the inverse contract at a trade price, then sell collateral coins.
    ending_fee=0.
    for i in range(2) if liquidate else ():
        wallet[i],face[i],entry[i],fee=change_short(wallet[i],face[i],entry[i],-face[i],market.future[last,i],spec.future_cost)
        ending_fee+=fee*market.spot[last,i]
        sale=wallet[i]*market.spot[last,i]
        ending_fee+=sale*spec.spot_cost
        cash[i]+=sale/market.usd[last]*(1-spec.spot_cost);wallet[i]=0
    final=float(cash.sum()*market.usd[last]) if liquidate else nav
    beginning=nav/(1+ledger.net_return.iloc[-1])
    ledger.loc[ledger.index[-1],"net_return"]=final/beginning-1
    ledger.loc[ledger.index[-1],"usd_return"]=ledger.net_return.iloc[-1]
    ledger.loc[ledger.index[-1],"excess_return"]=ledger.net_return.iloc[-1]-ledger.cash_return.iloc[-1]
    ledger.loc[ledger.index[-1],"equity_quote"]=final/capital
    ledger.loc[ledger.index[-1],"fees"]+=ending_fee/beginning
    info={"minimum_margin_ratio":min_margin,"maintenance_breaches":breaches,"max_absolute_delta_fraction":max_abs_delta,
          "capital":capital,"contract_face_usd":lot.tolist(),"decisions":decisions,"spec":asdict(spec),"terminal_liquidation":liquidate}
    return ledger,pd.DataFrame(trades),info


def run_inverse_study(directory,linear_directory,cash_path,output,progress=print):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    market,low=load_inverse_market(directory,linear_directory,cash_path)
    catalog=[replace(s,allocation=.98) for s in carry_catalog()]
    development=market.before("2024-01-01")
    dev_low=low[:len(development.times)]
    rows=[];eligible=[]
    for spec in catalog:
        ledger,_,info=simulate_inverse(development,spec,dev_low)
        values={}
        for period,mask in (("development",ledger.index<"2023-01-01"),("validation",ledger.index>="2023-01-01")):
            values[period]=carry_metrics(ledger.loc[mask]);rows.append({"method":spec.name,"period":period,**values[period]})
        if not info["maintenance_breaches"] and all(values[p]["average_spot_fraction"]>.1 for p in values):
            score=min(values[p]["sharpe_excess"] if values[p]["sharpe_excess"] is not None else -1e6 for p in values)
            eligible.append((score,spec))
        progress(f"Inverse development {spec.name}: {values['development']['sharpe_excess']:.2f} / {values['validation']['sharpe_excess']:.2f}, breaches {info['maintenance_breaches']}")
    if not eligible:raise ValueError("No active margin-valid inverse candidate")
    selected=max(eligible,key=lambda v:v[0])[1]
    selection={"selected":selected.name,"spec":asdict(selected),"criterion":"maximum worst development/2023 validation excess Sharpe; active >10%; no margin breach",
               "latest_selection_data":"2023-12-31","evaluation_context_reused":True,"source_hash":file_digest(__file__),
               "data_hashes":{p.name:file_digest(p) for p in Path(directory).glob("*.csv")}}
    (output/"selection.json").write_text(json.dumps(selection,indent=2),encoding="utf-8")
    progress(f"FROZEN native inverse selection: {selected.name}")
    pd.DataFrame(rows).to_csv(output/"development_catalog.csv",index=False)
    ledgers={};diagnostics=[]
    for spec in catalog:
        ledger,trades,info=simulate_inverse(market,spec,low)
        ledgers[spec.name]=ledger
        diagnostics.append({"method":spec.name,**{k:v for k,v in info.items() if k not in ("decisions","spec")}})
        for period,mask in (("full",np.ones(len(ledger),bool)),("evaluation",ledger.index>="2024-01-01")):
            rows.append({"method":spec.name,"period":period,**carry_metrics(ledger.loc[mask])})
        if spec.name==selected.name:
            trades.to_csv(output/"selected_trades.csv",index=False)
            (output/"selected_decisions.json").write_text(json.dumps(info["decisions"],indent=2),encoding="utf-8")
            for label,alternate,haircut in (("double_cost",replace(spec,spot_cost=2*spec.spot_cost,future_cost=2*spec.future_cost),1.),
                                            ("four_hour_delay",replace(spec,delay_hours=4),1.),("funding_haircut_10pct",spec,.9)):
                stressed,_,stressed_info=simulate_inverse(market,alternate,low,funding_haircut=haircut)
                ledgers[label]=stressed
                rows.append({"method":label,"period":"evaluation",**carry_metrics(stressed.loc[stressed.index>="2024-01-01"])})
    metrics=pd.DataFrame(rows);metrics.to_csv(output/"metrics.csv",index=False)
    pd.concat(ledgers,names=["method"]).to_csv(output/"daily_returns.csv")
    pd.DataFrame(diagnostics).to_csv(output/"margin_audit.csv",index=False)
    (output/"data_audit.json").write_text(json.dumps(market.audit,indent=2),encoding="utf-8")
    progress(metrics[(metrics.period=="evaluation")&metrics.method.isin([selected.name,"double_cost","four_hour_delay","funding_haircut_10pct"])].to_string(index=False))
    return metrics,selection
