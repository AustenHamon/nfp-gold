# update_pipeline.py -- runs on GitHub every weekday evening
# 1) fills in actuals from FRED for any release that has happened
# 2) measures gold's reaction (Dukascopy)  3) rebuilds the app's JSON files
import os, subprocess, sys
import pandas as pd
from fredapi import Fred

fred = Fred(api_key=os.environ["FRED_API_KEY"])
today = pd.Timestamp.today().normalize()

# event: (calendar file, FRED series, how to turn the last two index values into the headline number)
EVENTS = {
    "nfp": ("nfp_consensus.csv", "PAYEMS", lambda new, old: new - old),                          # jobs added, thousands
    "cpi": ("cpi_consensus.csv", "CPILFESL", lambda new, old: round((new / old - 1) * 100, 1)),  # core CPI % m/m
}


def first_print(series, day, calc):
    """The number as it was published on the release day (None if FRED doesn't have it yet)."""
    s = fred.get_series_as_of_date(series, day.strftime("%Y-%m-%d"))
    s["realtime_start"] = pd.to_datetime(s["realtime_start"])
    s = s.sort_values(["date", "realtime_start"]).drop_duplicates("date", keep="last")
    if len(s) < 2 or s["realtime_start"].iloc[-1] < day - pd.Timedelta(days=3):
        return None  # newest month wasn't published around this release yet
    return float(calc(float(s["value"].iloc[-1]), float(s["value"].iloc[-2])))


for event, (path, series, calc) in EVENTS.items():
    if not os.path.exists(path):
        continue
    cal = pd.read_csv(path, parse_dates=["release_date"])
    changed = False
    for i, r in cal[cal["actual"].isna() & (cal["release_date"] <= today)].iterrows():
        val = first_print(series, r.release_date, calc)
        if val is None:
            print(f"{event} {r.release_date.date()}: not on FRED yet, will retry next run")
            continue
        cal.loc[i, "actual"] = val
        changed = True
        print(f"{event} {r.release_date.date()}: actual = {val:g} (forecast {r.forecast:g})")
    if changed:
        cal["release_date"] = cal["release_date"].dt.date
        cal.to_csv(path, index=False)
    if os.path.exists(path):
        subprocess.run([sys.executable, "gold_reaction.py", event], check=True)

subprocess.run([sys.executable, "export_stats.py"], check=True)