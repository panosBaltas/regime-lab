from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from regime_lab.carry import CarrySpec
from regime_lab.inverse import change_short, simulate_inverse
from regime_lab.macro import GROUPS, MacroSpec, macro_targets, simulate_macro
from test_carry import market


def inverse_market(**kwargs):
    result=market(**kwargs)
    result.funding_cash=result.rate/result.mark
    return result


def inverse_spec(**kwargs):
    return replace(CarrySpec(window=3,rebalance=30,gate="passive",allocation=.98,spot_cost=0,future_cost=0),**kwargs)


def test_inverse_short_realizes_reciprocal_price_pnl_in_coins():
    wallet,face,entry,fee=change_short(1.,0.,0.,10_000.,10_000.,0.)
    assert wallet==1 and face==10_000 and entry==1
    wallet,face,entry,fee=change_short(wallet,face,entry,-10_000.,20_000.,0.)
    assert wallet==pytest.approx(.5)
    assert wallet*20_000==pytest.approx(10_000)
    assert face==entry==0


def test_native_collateral_hedge_cancels_large_coin_price_moves_without_cash_loan():
    m=inverse_market(days=80,daily_move=.03,rate=0)
    ledger,trades,info=simulate_inverse(m,inverse_spec(),start="2020-02-02")
    np.testing.assert_allclose(ledger.net_return,0,atol=1e-12)
    assert info["maintenance_breaches"]==0
    assert info["minimum_margin_ratio"]==pytest.approx(1,abs=1e-10)
    assert (trades.actual_contracts%1==0).all()
    assert (ledger.cash>=0).all()


def test_coin_funding_is_not_double_counted_as_both_spot_and_cash_income():
    m=inverse_market(days=40)
    ledger,_,_=simulate_inverse(m,inverse_spec(),start="2020-02-02")
    assert ledger.net_return.iloc[0]==pytest.approx(.98*.0001*3)
    assert ledger.funding.iloc[0]==pytest.approx(.98*.0001*3)
    assert np.prod(1+ledger.net_return)==pytest.approx(ledger.equity_quote.iloc[-1],abs=1e-12)


def test_inverse_entry_and_exit_pay_both_legs_and_respect_contract_granularity():
    m=inverse_market(days=40,rate=0)
    free,_,_=simulate_inverse(m,inverse_spec(),start="2020-02-02")
    paid,trades,info=simulate_inverse(m,inverse_spec(spot_cost=.0012,future_cost=.0007),start="2020-02-02")
    assert paid.equity_quote.iloc[-1]<free.equity_quote.iloc[-1]
    assert paid.fees.sum()>0
    assert (trades.actual_contracts%1==0).all()
    assert info["max_absolute_delta_fraction"]<.002
    assert np.prod(1+paid.net_return)==pytest.approx(paid.equity_quote.iloc[-1],abs=1e-12)


def test_inverse_future_data_cannot_change_old_contracts():
    original=inverse_market(days=80)
    changed=inverse_market(days=80)
    cutoff=original.times.get_loc(pd.Timestamp("2020-02-25"))
    changed.rate[cutoff+1:]=-.001
    changed.funding_cash[cutoff+1:]=changed.rate[cutoff+1:]/changed.mark[cutoff+1:]
    s=inverse_spec(gate="positive",rebalance=7)
    _,a,_=simulate_inverse(original,s,start="2020-02-02")
    _,b,_=simulate_inverse(changed,s,start="2020-02-02")
    pd.testing.assert_frame_equal(a[a.time<="2020-02-25"].reset_index(drop=True),b[b.time<="2020-02-25"].reset_index(drop=True))


def test_continuous_reference_does_not_move_an_artificial_exit_fee_into_past_returns():
    m=inverse_market(days=80)
    s=inverse_spec(spot_cost=.0012,future_cost=.0007)
    full,_,_=simulate_inverse(m,s,start="2020-02-02",liquidate=False)
    shorter,_,info=simulate_inverse(m.before("2020-03-01"),s,start="2020-02-02",liquidate=False)
    np.testing.assert_allclose(full.loc[shorter.index,"net_return"],shorter.net_return,atol=1e-14)
    assert info["terminal_liquidation"] is False
    dates=pd.bdate_range("2015-01-01",periods=400)
    names=[name for group in GROUPS.values() for name in group]
    prices=np.full((len(dates),len(names)),100.)
    cash=np.full(len(dates),100.)
    panel=(dates,names,prices,prices,cash,cash)
    closed=simulate_macro(panel,MacroSpec("risk_parity"))
    ongoing=simulate_macro(panel,MacroSpec("risk_parity"),liquidate=False)
    assert np.prod(1+closed.net_return)==pytest.approx(np.prod(1+ongoing.net_return)*(1-.0005))


def test_macro_flat_portfolio_only_pays_initial_and_final_trades():
    dates=pd.bdate_range("2015-01-01",periods=400)
    names=[name for group in GROUPS.values() for name in group]
    prices=np.full((len(dates),len(names)),100.)
    cash=np.full(len(dates),100.)
    s=MacroSpec("risk_parity")
    ledger=simulate_macro((dates,names,prices,prices,cash,cash),s)
    assert np.prod(1+ledger.net_return)==pytest.approx((1-s.cost)/(1+s.cost),abs=1e-12)
    assert ledger.turnover.iloc[1:].sum()==pytest.approx(0,abs=1e-12)


def test_macro_signals_and_covariance_ignore_future_prices():
    rng=np.random.default_rng(7)
    dates=pd.bdate_range("2015-01-01",periods=450)
    names=[name for group in GROUPS.values() for name in group]
    prices=100*np.exp(np.cumsum(rng.normal(.0002,.01,(len(dates),len(names))),axis=0))
    altered=prices.copy();altered[351:]*=2
    cash=np.linspace(100,102,len(dates))
    for family in ("multispeed_long","fast_momentum","breadth_defensive"):
        s=MacroSpec(family)
        a=macro_targets(dates,names,prices,cash,s)
        b=macro_targets(dates,names,altered,cash,s)
        np.testing.assert_allclose(a[:351],b[:351],equal_nan=True)
