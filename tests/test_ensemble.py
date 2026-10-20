import numpy as np
import pandas as pd

from regime_lab.ensemble import allocation,metrics,simulate_allocation


def fixture():
    rng=np.random.default_rng(8)
    dates=pd.date_range("2020-12-01","2024-06-30",freq="D")
    returns=pd.DataFrame(rng.normal([.0002,.0003],[.0003,.005],(len(dates),2)),index=dates,columns=["carry","macro"])
    rf=pd.Series(.00005,index=dates)
    return returns,rf


def test_allocator_respects_bounds_and_uses_only_earlier_observations():
    returns,rf=fixture()
    changed=returns.copy();changed.loc[changed.index>"2024-02-20"]*=-10
    for rule in ("static_risk","rolling_risk","shrunk_mean_variance"):
        original,decisions=simulate_allocation(returns,rf,rule)
        altered,new_decisions=simulate_allocation(changed,rf,rule)
        pd.testing.assert_frame_equal(original.loc[:"2024-02-20"],altered.loc[:"2024-02-20"])
        assert all(pd.Timestamp(d["latest_return_observation"])<pd.Timestamp(d["fill_time"]) for d in decisions)
        assert all(.1<=d["inverse_weight"]<=.9 for d in decisions)


def test_new_allocations_cannot_earn_a_return_that_ended_before_the_fill():
    returns,rf=fixture()
    returns[:]=0
    # The first deposit is filled Jan 1 and spends two days in transit. A Jan 1
    # or Jan 2 return cannot be earned by those newly deposited dollars.
    returns.loc["2023-01-01":"2023-01-02"]=1
    ledger,_=simulate_allocation(returns,rf,"static_risk")
    assert ledger.loc["2023-01-01":"2023-01-03","net_return"].sum()<=0
    assert ledger.transit_cash.iloc[0]>.99
    assert ledger.transit_cash.iloc[1]>.99
    assert ledger.transit_cash.iloc[2]==0


def test_transfers_and_final_exit_reduce_wealth_with_flat_underlying_prices():
    returns,rf=fixture();returns[:]=0;rf[:]=0
    ledger,_=simulate_allocation(returns,rf,"static_risk")
    assert 0<ledger.equity.iloc[-1]<1
    assert np.prod(1+ledger.net_return)==ledger.equity.iloc[-1]
    assert ledger.transfer_fee.sum()>0


def test_serially_persistent_returns_have_lower_hac_than_naive_daily_sharpe():
    rng=np.random.default_rng(12)
    values=np.zeros(1500)
    for t in range(1,len(values)):
        values[t]=.8*values[t-1]+rng.normal(0,.001)
    values+=.0005
    dates=pd.date_range("2015-01-01",periods=len(values),freq="D")
    ledger=pd.DataFrame({"net_return":values,"excess_return":values,"cash_return":0.},index=dates)
    result=metrics(ledger)
    assert result["hac28_sharpe"]<.6*result["daily_excess_sharpe"]
