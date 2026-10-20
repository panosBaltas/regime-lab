"""Native COIN-M data, kept separate from the linear-futures experiment."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import zipfile
from io import BytesIO

import pandas as pd
import requests

from .carry_data import BASE, read_archive
from .data import file_digest

PAIRS=("BTCUSD_PERP","ETHUSD_PERP")


def fetch_one(root,symbol,kind,month):
    family="fundingRate" if kind=="funding" else ("markPriceKlines" if kind=="mark" else "klines")
    suffix=f"{symbol}-fundingRate-{month}.zip" if kind=="funding" else f"1h/{symbol}-1h-{month}.zip"
    relative=f"futures/cm/monthly/{family}/{symbol}/{suffix}"
    path=root/"archives"/kind/symbol/f"{month}.zip"
    path.parent.mkdir(parents=True,exist_ok=True)
    source_url=BASE+relative
    if path.with_suffix(".api.csv").exists():
        path=path.with_suffix(".api.csv")
        source_url=json.loads(path.with_suffix(".json").read_text())["source"]
    if not path.exists():
        r=requests.get(BASE+relative,timeout=40)
        if r.status_code==404 and kind=="funding":
            begin=pd.Timestamp(month+"-01");finish=begin+pd.offsets.MonthBegin()
            response=requests.get("https://dapi.binance.com/dapi/v1/fundingRate",params={"symbol":symbol,"startTime":int(begin.timestamp()*1000),"endTime":int(finish.timestamp()*1000)-1,"limit":1000},timeout=30)
            response.raise_for_status();frame=pd.DataFrame(response.json())
            if frame.empty:raise ValueError(f"No official funding history for {symbol} {month}")
            normalized=pd.DataFrame({"calc_time":frame.fundingTime,"funding_interval_hours":8,"last_funding_rate":frame.fundingRate})
            path=path.with_suffix(".api.csv");normalized.to_csv(path,index=False)
            source_url=response.url
            path.with_suffix(".json").write_text(json.dumps({"source":source_url,"reason":"ZIP not published; normalized official funding API"}),encoding="utf-8")
        else:
            r.raise_for_status();zipfile.ZipFile(BytesIO(r.content)).testzip();path.write_bytes(r.content)
    return {"symbol":symbol,"kind":kind,"month":month,"path":str(path.resolve()),"sha256":file_digest(path),"url":source_url}


def download_inverse_data(root,start="2020-09",end="2026-09",progress=print):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    jobs=[(s,k,m) for s in PAIRS for k in ("perpetual","mark","funding") for m in pd.period_range(start,end,freq="M").astype(str)]
    records=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        pending={pool.submit(fetch_one,root,*j):j for j in jobs}
        for future in as_completed(pending):
            records.append(future.result())
            if len(records)%60==0:progress(f"COIN-M archives {len(records)}/{len(jobs)}")
    audit=pd.DataFrame(records).sort_values(["symbol","kind","month"])
    audit.to_csv(root/"download_audit.csv",index=False)
    repairs=[]
    for (symbol,kind),g in audit.groupby(["symbol","kind"]):
        frame=pd.concat([read_archive(p,kind) for p in g.path]).sort_index()
        if kind!="funding":
            expected=pd.date_range(frame.index.min(),frame.index.max(),freq="h")
            missing=expected.difference(frame.index)
            for day in missing.normalize().unique():
                endpoint="markPriceKlines" if kind=="mark" else "klines"
                r=requests.get(f"https://dapi.binance.com/dapi/v1/{endpoint}",params={"symbol":symbol,"interval":"1h","startTime":int(day.timestamp()*1000),"endTime":int((day+pd.Timedelta(days=1)).timestamp()*1000)-1,"limit":1000},timeout=30)
                r.raise_for_status();raw=pd.DataFrame(r.json()).iloc[:,:6]
                raw.columns=["time","open","high","low","close","volume"]
                raw.index=pd.to_datetime(raw.pop("time"),unit="ms");raw=raw.apply(pd.to_numeric)
                frame=frame.combine_first(raw)
                repairs.append({"symbol":symbol,"kind":kind,"date":str(day.date()),"rows":len(raw),"url":r.url})
            if not expected.equals(frame.index):raise ValueError(f"COIN-M price gap: {symbol} {kind}")
        if frame.index.has_duplicates:raise ValueError("Duplicate inverse observations")
        frame.index.name="time";frame.to_csv(root/f"{symbol}_{kind}.csv",float_format="%.14g")
    (root/"repair_audit.json").write_text(json.dumps(repairs,indent=2),encoding="utf-8")
    return True
