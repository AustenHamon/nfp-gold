
# Builds one table: each past NFP release + pre-release clues + gold's reaction
# pip install fredapi yfinance pandas
import os
import pandas as pd
import yfinance as yf
from fredapi import Fred

fred = Fred(api_key="b3935149b33724831582541f8f482663")

# 1) NFP first-print history (with real release dates)
vint = fred.get_series_all_releases("PAYEMS")
vint["realtime_start"] = pd.to_datetime(vint["realtime_start"])
vint["date"] = pd.to_datetime(vint["date"])

rows = []
for month, grp in vint.groupby("date"):
    first = grp.sort_values("realtime_start").iloc[0]
    rel = first["realtime_start"]
    prev_month = month - pd.DateOffset(months=1)
    prev = vint[(vint["date"] == prev_month) & (vint["realtime_start"] <= rel)]
    if prev.empty:
        continue
    prev_val = prev.sort_values("realtime_start").iloc[-1]["value"]
    rows.append({"release_date": rel, "ref_month": month,
                 "nfp_change": first["value"] - prev_val})
nfp = pd.DataFrame(rows).query("release_date >= '2005-01-01'")

# 2) Pre-release clues (only data known BEFORE each release is used)
clues = {
    "jobless_claims": fred.get_series("ICSA"),      # weekly
    "adp": fred.get_series("ADPMNUSNERSA"),         # monthly ADP employment
    "us10y": fred.get_series("DGS10"),              # 10-year yield
    "usd_index": fred.get_series("DTWEXBGS"),       # broad dollar index
}

def last_before(series, day):
    s = series.dropna()
    s = s[s.index < day]
    return s.iloc[-1] if len(s) else None

def change_before(series, day, lookback=5):
    s = series.dropna()
    s = s[s.index < day]
    return s.iloc[-1] - s.iloc[-1 - lookback] if len(s) > lookback else None

for name, s in clues.items():
    nfp[name] = nfp["release_date"].apply(lambda d: last_before(s, d))
nfp["us10y_5d_chg"] = nfp["release_date"].apply(lambda d: change_before(clues["us10y"], d))
nfp["usd_5d_chg"] = nfp["release_date"].apply(lambda d: change_before(clues["usd_index"], d))

# 3) Gold prices (daily) and reaction on release day
gold = yf.download("GC=F", start="2004-12-01", progress=False)["Close"].squeeze()

def gold_stats(day):
    before = gold[gold.index < day]
    on = gold[gold.index >= day]
    if len(before) < 6 or on.empty:
        return pd.Series([None] * 3)
    prev_close, day_close = before.iloc[-1], on.iloc[0]
    run_up = (prev_close / before.iloc[-6] - 1) * 100   # gold's move into the event
    move = (day_close / prev_close - 1) * 100
    return pd.Series([run_up, move, "BUY" if move > 0 else "SELL"])

nfp[["gold_5d_runup_pct", "gold_move_pct", "direction"]] = \
    nfp["release_date"].apply(gold_stats)

nfp = nfp.dropna().reset_index(drop=True)
nfp.to_csv("nfp_gold_dataset.csv", index=False)
print(nfp.tail(10))
print(f"\nSaved {len(nfp)} releases. BUY share: {(nfp.direction == 'BUY').mean():.0%}")
