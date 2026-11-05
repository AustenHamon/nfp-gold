# update_pipeline.py -- runs on GitHub after each NFP
import os, subprocess, sys
import pandas as pd
from fredapi import Fred

fred = Fred(api_key=os.environ["FRED_API_KEY"])
cal = pd.read_csv("nfp_consensus.csv", parse_dates=["release_date"])
today = pd.Timestamp.today().normalize()

# Fill in the actual for any past release that's still blank, straight from FRED
for i, r in cal[cal["actual"].isna() & (cal["release_date"] <= today)].iterrows():
    s = fred.get_series_as_of_date("PAYEMS", r.release_date.strftime("%Y-%m-%d"))
    s = s.sort_values("date").drop_duplicates("date", keep="last")
    if len(s) >= 2:
        cal.loc[i, "actual"] = float(s["value"].iloc[-1] - s["value"].iloc[-2])
        print(f"{r.release_date.date()}: actual = {cal.loc[i, 'actual']:.0f}K")

cal["release_date"] = cal["release_date"].dt.date
cal.to_csv("nfp_consensus.csv", index=False)

for script in ["nfp_gold_dukascopy.py", "export_stats.py"]:
    subprocess.run([sys.executable, script], check=True)