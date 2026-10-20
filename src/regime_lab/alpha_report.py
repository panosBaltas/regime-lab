"""Research summary, robustness diagnostics, and an explicit selection audit."""
from dataclasses import replace
from html import escape
from pathlib import Path
import base64
import importlib.metadata
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .carry import CarrySpec
from .data import file_digest
from .ensemble import allocation,load_sleeves,metrics,monthly_returns,simulate_allocation,selected_allocator
from .inverse import load_inverse_market,simulate_inverse
from .macro import MacroSpec,macro_panel,simulate_macro


def read_ledger(root,name):
    frame=pd.read_csv(Path(root)/"daily_returns.csv",parse_dates=["time"])
    return frame[frame.method==name].drop(columns="method").set_index("time")


def bootstrap_months(ledger,seed=42,replications=5000):
    values=monthly_returns(ledger).excess_return.to_numpy()
    rng=np.random.default_rng(seed);n=len(values);block=3
    starts=rng.integers(0,n,(replications,int(np.ceil(n/block))))
    indices=((starts[:,:,None]+np.arange(block))%n).reshape(replications,-1)[:,:n]
    samples=values[indices];scores=samples.mean(axis=1)/samples.std(axis=1,ddof=1)*np.sqrt(12)
    return {"lower":float(np.quantile(scores,.025)),"upper":float(np.quantile(scores,.975)),
            "replications":replications,"block_months":block,"selection_adjusted":False,"months":n}


def audit_native_replay(inverse_root,market,spec,capital=100_000.):
    root=Path(inverse_root)
    selected=json.loads((root/"selection.json").read_text())["selected"]
    original=read_ledger(root,selected)
    trades=pd.read_csv(root/"selected_trades.csv",parse_dates=["time"])
    cumulative=np.cumsum(market.funding_cash,axis=0)
    snapshots=[trades[trades.asset==symbol].set_index("time") for symbol in ("BTCUSD_PERP","ETHUSD_PERP")]
    errors=[]
    for timestamp,row in original.iterrows():
        t=int(market.times.get_loc(timestamp));nav=0.
        terminal=timestamp==original.index[-1]
        for i,frame in enumerate(snapshots):
            loc=int(frame.index.searchsorted(timestamp,side="right"))-1
            if loc<0:raise ValueError("Replay predates a native holding snapshot")
            snapshot=frame.iloc[loc];s=int(market.times.get_loc(frame.index[loc]))
            face=snapshot.short_face_usd
            wallet=snapshot.coin_wallet+face*(cumulative[t,i]-cumulative[s,i])
            cash=snapshot.cash_quote*market.usd[t]
            if terminal:
                wallet+=face/market.future[t,i]-snapshot.entry_coin_term-face/market.future[t,i]*spec.future_cost
                nav+=cash+wallet*market.spot[t,i]*(1-spec.spot_cost)
            else:
                nav+=cash+(wallet+face/market.mark[t,i]-snapshot.entry_coin_term)*market.spot[t,i]
        errors.append(abs(nav/capital-row.equity_quote))
    result={"maximum_equity_replay_error":float(max(errors)),"n_daily_marks":len(errors),
            "method":"Rebuild native coin wallets from saved trades and intervening funding; reciprocal-price mark and final liquidation"}
    if result["maximum_equity_replay_error"]>1e-9:raise ValueError(f"Native replay failed: {result}")
    return result


def build_alpha_report(carry_root,macro_root,inverse_root,ensemble_root,output,
                       market_path="data/market.csv",cash_path="data/cash_bil.csv",
                       carry_data="data/carry",inverse_data="data/inverse"):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    roots={"linear":Path(carry_root),"macro":Path(macro_root),"inverse":Path(inverse_root),"ensemble":Path(ensemble_root)}
    trial_tables=[]
    for family,root in roots.items():
        table=pd.read_csv(root/"metrics.csv")
        table.insert(0,"family",family)
        table["daily_annualization"]=252 if family=="macro" else 365
        trial_tables.append(table)
    pd.concat(trial_tables,ignore_index=True).to_csv(output/"trial_catalog.csv",index=False)
    selections={name:json.loads((p/"selection.json").read_text()) for name,p in roots.items()}
    primary=selected_allocator(selections["ensemble"])
    names={"static_risk":"Fixed training-risk control", "rolling_risk":"Rolling risk allocation",
           "shrunk_mean_variance":"Shrunk mean–variance mix", "inverse":"Inverse funding carry", "macro":"Diversified ETFs"}
    returns,rf,_=load_sleeves(inverse_root,macro_root)
    frames={name:read_ledger(ensemble_root,name) for name in ("static_risk","rolling_risk","shrunk_mean_variance")}
    frames["inverse"]=read_ledger(inverse_root,selections["inverse"]["selected"]).rename(columns={"usd_return":"usd_value_return"})
    frames["macro"]=pd.DataFrame({"net_return":returns.macro,"cash_return":rf,"excess_return":returns.macro-rf})
    evaluations={name:frame[frame.index>="2024-01-01"].copy() for name,frame in frames.items()}
    summary={name:metrics(frame) for name,frame in evaluations.items()}
    interval={name:bootstrap_months(evaluations[name]) for name in evaluations}
    monthly=(1+returns).resample("ME").prod()-1
    monthly_cash=(1+rf).resample("ME").prod()-1
    excess=monthly.subtract(monthly_cash,axis=0)
    training=excess[(excess.index>="2021-01-01")&(excess.index<"2023-01-01")]
    weights=allocation(training,"static_risk")
    _,base_decisions=simulate_allocation(returns,rf,primary)
    frozen_schedule={d["fill_time"]:[d["inverse_weight"],d["macro_weight"]] for d in base_decisions}
    # Replay the primary strategy's original decisions under stresses.
    stress=[]
    macro_spec=MacroSpec(**selections["macro"]["spec"])
    doubled_macro=simulate_macro(macro_panel(market_path,cash_path),replace(macro_spec,cost=2*macro_spec.cost))
    for label,native in (("Double trading fees","double_cost"),("Four-hour carry fills","four_hour_delay"),
                         ("10% haircut to positive funding","funding_haircut_10pct")):
        r,benchmark,_=load_sleeves(inverse_root,macro_root,inverse_method=native)
        if native=="double_cost":
            wealth=(1+doubled_macro.net_return).cumprod();wealth.index+=pd.Timedelta(days=1)
            aligned=wealth.reindex(wealth.index.union(r.index)).ffill().reindex(r.index)
            # Include the previous calendar wealth point to obtain the first return.
            source=wealth.reindex(wealth.index.union(r.index).union(pd.DatetimeIndex([r.index[0]-pd.Timedelta(days=1)]))).ffill()
            r["macro"]=source.reindex(pd.DatetimeIndex([r.index[0]-pd.Timedelta(days=1)]).append(r.index)).pct_change().iloc[1:].to_numpy()
        ledger,_=simulate_allocation(r,benchmark,primary,static_weights=weights,weight_schedule=frozen_schedule)
        stress.append({"scenario":label,**metrics(ledger[ledger.index>="2024-01-01"])})
    venue=evaluations[primary].copy();shock_date=pd.Timestamp("2025-01-02")
    loss=.1*float(venue.loc[shock_date,"inverse_weight"])
    venue.loc[shock_date,"net_return"]=(1+venue.loc[shock_date,"net_return"])*(1-loss)-1
    venue["excess_return"]=venue.net_return-venue.cash_return
    stress.append({"scenario":"Hypothetical 10% loss of venue capital",**metrics(venue)})
    native_market,native_low=load_inverse_market(inverse_data,carry_data,cash_path)
    native_spec=CarrySpec(**selections["inverse"]["spec"])
    replay=audit_native_replay(inverse_root,native_market,native_spec)
    small,_,small_info=simulate_inverse(native_market,native_spec,native_low,capital=10_000.)
    small_evaluation=small[small.index>="2024-01-01"]
    stress.append({"scenario":"Carry-only $10k whole-contract account",**metrics(small_evaluation)})
    pd.DataFrame(stress).to_csv(output/"stress_metrics.csv",index=False)
    yearly=[]
    for name,ledger in evaluations.items():
        for year,g in ledger.groupby(ledger.index.year):
            yearly.append({"method":name,"year":int(year),"total_return":float(np.prod(1+g.net_return)-1),**metrics(g)})
    pd.DataFrame(yearly).to_csv(output/"yearly_metrics.csv",index=False)
    pd.DataFrame(summary).T.to_csv(output/"summary_metrics.csv")
    (output/"bootstrap_intervals.json").write_text(json.dumps(interval,indent=2),encoding="utf-8")
    (output/"replay_audit.json").write_text(json.dumps(replay,indent=2),encoding="utf-8")
    model={"status":"exploratory_historical_reference","primary_allocator":primary,
           "validation_selected_allocator":selections["ensemble"]["selected"],
           "weights_fitted_on":"2021–2022 completed monthly excess returns" if primary=="static_risk" else "previous 12 completed monthly excess returns",
           "training_control_weights":{"inverse":float(weights[0]),"macro":float(weights[1])},
           "primary_weight_policy":primary,"selection_override":False,
           "inverse_spec":selections["inverse"]["spec"],"macro_spec":selections["macro"]["spec"],
           "allocation_timing":"monthly capital allocation; subsequent returns only; two-day idle transfer allowance",
           "capital_transfer_costs":{"native_carry":.0029,"macro":.0015},
           "data_end":"2026-09-30","evaluation":"2024–September 2026; reused research context",
           "scope":"asynchronous return-sleeve allocation research; direct multi-venue implementation needs forward validation"}
    (output/"model.json").write_text(json.dumps(model,indent=2,ensure_ascii=False),encoding="utf-8")
    preferred=evaluations[primary]
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    figure,axis=plt.subplots(figsize=(9,3.5))
    colors={"static_risk":"#2563eb","rolling_risk":"#0f9b8e","shrunk_mean_variance":"#ca8a04","inverse":"#a855f7","macro":"#64748b"}
    for name in colors:
        item=evaluations[name]
        axis.plot(item.index,np.cumprod(1+item.net_return),label=names[name],color=colors[name],lw=1.6)
    axis.plot(preferred.index,np.cumprod(1+preferred.cash_return),label="BIL benchmark",color="#a1a1aa",ls="--")
    axis.set_ylabel("Net wealth, rebased to 1");axis.legend(frameon=False,fontsize=8,ncols=2)
    figure.savefig(output/"equity.png",dpi=160,bbox_inches="tight");plt.close(figure)
    figure,axes=plt.subplots(1,2,figsize=(9,2.7))
    m=monthly_returns(preferred)
    axes[0].bar(m.index,100*m.excess_return,width=22,color=np.where(m.excess_return>=0,"#2563eb","#e05b61"))
    axes[0].axhline(0,color="#94a3b8",lw=.7);axes[0].set_ylabel("Monthly excess return (%)")
    drawdown=np.cumprod(1+preferred.net_return);peak=np.maximum.accumulate(np.r_[1,drawdown])[1:]
    axes[1].fill_between(preferred.index,100*(drawdown/peak-1),0,color="#2563eb",alpha=.22)
    axes[1].set_ylabel("Drawdown (%)");figure.autofmt_xdate()
    figure.savefig(output/"risk.png",dpi=160,bbox_inches="tight");plt.close(figure)
    table=pd.DataFrame(summary).T.loc[["static_risk","rolling_risk","shrunk_mean_variance","inverse","macro"],
                     ["monthly_excess_sharpe","daily_excess_sharpe","hac28_sharpe","cagr","drawdown"]]
    table.index=table.index.map(names);table.columns=["Monthly Sharpe","Daily Sharpe","HAC 28-day","Net CAGR","Drawdown"]
    for c in table.columns[:3]:table[c]=table[c].map(lambda x:f"{x:.2f}")
    for c in table.columns[3:]:table[c]=table[c].map(lambda x:f"{x:.1%}")
    stress_table=pd.DataFrame(stress)[["scenario","monthly_excess_sharpe","daily_excess_sharpe","cagr","drawdown"]].copy()
    for c in ("monthly_excess_sharpe","daily_excess_sharpe"):stress_table[c]=stress_table[c].map(lambda x:f"{x:.2f}")
    for c in ("cagr","drawdown"):stress_table[c]=stress_table[c].map(lambda x:f"{x:.1%}")
    ci=interval[primary];best=summary[primary]
    data={"summary":summary,"intervals":interval,"model":model,"stress":stress,"names":names,"replay":replay}
    (output/"results.json").write_text(json.dumps(data,indent=2,allow_nan=False),encoding="utf-8")
    content=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Carry and multi-asset research</title><style>
body{{font:15px/1.55 system-ui,sans-serif;color:#25324a;margin:20px auto;max-width:1040px;padding:0 16px}}h1{{font-size:28px}}h2{{font-size:19px;margin-top:26px}}p{{margin:9px 0}}.notice{{background:#f2efff;border-left:3px solid #8b5cf6;padding:14px 16px;border-radius:8px}}.score{{font-size:38px;font-weight:700}}.muted{{color:#64748b}}table{{width:100%;border-collapse:collapse;font-size:13px}}td,th{{border-bottom:1px solid #dbe2eb;padding:9px 7px;text-align:right;white-space:nowrap}}td:first-child,th:first-child{{text-align:left}}.table{{overflow:auto}}img{{width:100%;height:auto}}li{{margin:6px 0}}
</style></head><body><h1>Funding carry + diversified ETFs</h1><p class="muted">Free-data research · 68 candidates across four families · evaluation valuation dates 2024–September 2026</p>
<p><span class="score">{best['monthly_excess_sharpe']:.2f}</span> monthly net excess Sharpe · {best['daily_excess_sharpe']:.2f} from daily returns · {best['hac28_sharpe']:.2f} serial-dependence diagnostic</p>
<p>{best['cagr']:.1%} annualized net growth; {best['drawdown']:.1%} maximum drawdown. Sharpe subtracts matched BIL performance from net returns; growth is the portfolio's net growth. Trading and capital-transfer costs are included.</p>
<p class="notice"><strong>Exploratory historical reference.</strong> The headline follows the recorded 2023 validation selector: {escape(primary)}. The fixed training-risk mix is a separate control. No report-level preference overrides the selector. Evaluation context has been reused; this is not a pristine final holdout or a live trading record.</p>
<h2>What the model trades</h2><p>The selected {escape(primary)} policy allocates between coin-collateralized BTC/ETH funding carry and diversified ETFs. Rolling rules use the previous 12 completed monthly observations. The fixed control uses 2021–2022 volatility, giving {weights[0]:.1%} carry and {weights[1]:.1%} ETF weights. Moved cash spends two calendar days idle in this original return-sleeve approximation.</p>
<p>The carry sleeve buys BTC/ETH and uses those coins as collateral for matching native inverse perpetual shorts. It earns or pays actual funding and marks both the collateral and reciprocal-price futures payoff. Contract counts are whole numbers. It adjusts the hedge for accumulated coin funding every 30 days, rather than paying to repurchase the whole position daily. The ETF sleeve uses inverse-volatility asset-class risk allocation with an additional inverse-volatility tilt, a past-covariance risk ceiling, and 21-session rebalancing. It spans equities, bonds, real assets and the dollar.</p>
<h2>Frequency and model-selection checks</h2><div class="table">{table.to_html(border=0)}</div>
<p class="muted">Carry's {summary['inverse']['daily_excess_sharpe']:.2f} daily Sharpe falls to {summary['inverse']['monthly_excess_sharpe']:.2f} on monthly returns because funding is persistent. The combined candidate is evaluated primarily on monthly excess returns. Daily combined marks are asynchronous: US opening NAV is recognized at the following UTC midnight, while crypto is marked at midnight. The HAC diagnostic uses Bartlett-weighted covariances through 28 days; it is not a guarantee of an annual realized Sharpe.</p>
<img src="equity.png" alt="Net evaluation wealth for the research candidate, validation-selected alternative, individual sleeves and BIL">
<h2>Risk and uncertainty</h2><img src="risk.png" alt="Monthly excess returns and drawdown of the validation-selected allocator">
<p>The pointwise 95% three-month circular-block bootstrap interval for monthly Sharpe is [{ci['lower']:.2f}, {ci['upper']:.2f}], using {ci['months']} monthly observations and {ci['replications']:,} resamples. It is not corrected for selecting among these candidates or earlier project experiments. The true future Sharpe is not established as two.</p>
<h2>Costs, delays and failure scenarios</h2><div class="table">{stress_table.to_html(index=False,border=0)}</div>
<p class="muted">Doubled trading fees cover both the native carry and ETF sleeves; financing/capital-transfer rules remain fixed. Funding haircut removes 10% of positive receipts while retaining every negative payment. The venue-loss scenario is an invented 10% loss of capital allocated to the crypto venue on 2 January 2025, not an observed event or an estimate of default probability. The $10k row concerns whole-contract carry alone.</p>
<h2>Implementation and limitations</h2><ul>
<li>Primary carry costs: 12 bps one-way spot and 7 bps futures, including slippage assumptions; ETF turnover 5 bps, ETF shorts 2% annual borrow where used. The selected ETF sleeve is long-only. Capital transfer/execution allowances are 29 bps carry and 15 bps ETF per transferred notional, plus the two-day idle allowance.</li>
<li>Actual native funding rates, hourly spot/perpetual/mark prices, a dated USDT/USD conversion series and BIL are used. Early missing native funding ZIPs are recovered from the official funding API, with their provenance recorded. Missing mark days are recovered from the official mark API. Spot halts remain untradable.</li>
<li>Funding is booked to the old position before any same-timestamp fill. Future data cannot change old signals, contracts, ETF targets, or allocation decisions. An independent wallet reconstruction reproduces all {replay['n_daily_marks']:,} native daily equity marks within {replay['maximum_equity_replay_error']:.2e}.</li>
<li>Coin funding retained between hedge adjustments creates residual coin exposure; the margin and delta audits disclose it. Per-coin margin tests include hourly high and low marks. No candidate is selected after a maintenance breach.</li>
<li>Return-sleeve capital allocation is a research abstraction. Direct multi-venue order execution, bank/crypto transfers, settlement calendars, changing margin rules, auto-deleveraging, withdrawal freezes, custody loss and venue failure require additional implementation and forward evidence. Historical price Sharpe omits unobserved venue-default jumps.</li>
<li>2024 was especially favorable for carry; 2026 carry-only excess Sharpe was negative. The mixture reduces reliance on that one premium but has directional ETF risk and a short evaluation record. All failed stock, linear-carry and macro trials remain preserved.</li>
</ul><h2>Research basis</h2><p><a href="https://www.bis.org/publ/work1087.htm">BIS: crypto carry</a>; <a href="https://www.binance.com/en-AE/support/faq/detail/3a55a23768cb416fb404f06ffedde4b2">native inverse payoff mechanics</a>; <a href="https://www.aqr.com/insights/research/journal-article/a-century-of-evidence-on-trend-following-investing">diversified trend evidence</a>; <a href="https://www.nber.org/papers/w22208">volatility-managed portfolios</a>; <a href="https://alo.mit.edu/research-page/the-statistics-of-sharpe-ratios/">Lo: serial dependence and Sharpe</a>.</p>
</body></html>'''
    for image_name in ("equity.png","risk.png"):
        encoded=base64.b64encode((output/image_name).read_bytes()).decode("ascii")
        content=content.replace(f'src="{image_name}"',f'src="data:image/png;base64,{encoded}"')
    (output/"report.html").write_text(content,encoding="utf-8")
    source=Path(__file__).parent
    manifest={"sources":{p.name:file_digest(p) for pattern in ("carry*.py","inverse*.py","macro.py","ensemble.py","alpha_report.py") for p in source.glob(pattern)},
              "data":{str(p):file_digest(p) for folder in (Path(carry_data),Path(inverse_data)) for p in folder.glob("*.csv")},
              "selections":selections,"primary_allocator":primary,"selection_override":False,"test_context_reused":True,
              "versions":{name:importlib.metadata.version(name) for name in ("numpy","pandas","scipy","scikit-learn","yfinance")}}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    inputs={"carry_root":str(Path(carry_root).resolve()),"macro_root":str(Path(macro_root).resolve()),
            "inverse_root":str(Path(inverse_root).resolve()),"ensemble_root":str(Path(ensemble_root).resolve()),
            "output":str(output.resolve()),"market_path":str(Path(market_path).resolve()),"cash_path":str(Path(cash_path).resolve()),
            "carry_data":str(Path(carry_data).resolve()),"inverse_data":str(Path(inverse_data).resolve())}
    (output/"report_inputs.json").write_text(json.dumps(inputs,indent=2),encoding="utf-8")
    for name in ("ALPHA_PROTOCOL.md","INVERSE_PROTOCOL.md","ENSEMBLE_PROTOCOL.md"):
        shutil.copyfile(source.parents[1]/"docs"/name,output/name)
    print(json.dumps({"model":model,"metrics":summary[primary],"bootstrap":interval[primary],"stress":stress,"replay":replay},indent=2))
    print(f"Report: {(output/'report.html').resolve()}")
    return data


def render_saved_alpha_report(output):
    inputs=json.loads((Path(output)/"report_inputs.json").read_text())
    return build_alpha_report(**inputs)
