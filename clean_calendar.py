# clean_calendar.py -- turns an Investing.com history copy-paste into a clean CSV
# Usage:  python clean_calendar.py raw_cpi.txt cpi_consensus.csv
# Works for any event (NFP, CPI, ...). Handles values like 0.3%, 165K, 1,371, and blanks.
import re, sys
import pandas as pd

raw_path, out_path = sys.argv[1], sys.argv[2]
raw = open(raw_path, encoding="utf-8").read()
raw = raw.replace("−", "-")                        # unicode minus -> normal minus
if "Previous" in raw:
    raw = raw.split("Previous", 1)[1]                    # drop the header row

months = "Jan|eb|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
val = r"-?[\d,]*\.?\d*[%KMB]?"
pat = re.compile(rf"({months}) (\d{{1,2}}), (\d{{4}})(?: \([A-Za-z]+\))?[;\t]\s*(\d{{1,2}}:\d{{2}})[;\t]\s*"
                 rf"({val})[;\t]\s*({val})[;\t]\s*({val})")


def num(s):
    s = re.sub(r"[%KMB,\s]", "", s)
    return float(s) if s not in ("", "-", ".") else None


rows = []
for m, d, y, t, a, f, p in pat.findall(raw):
    m = "Feb" if m == "eb" else m
    rows.append({"release_date": pd.to_datetime(f"{m} {d} {y}", format="%b %d %Y").date(),
                 "actual": num(a), "forecast": num(f), "previous": num(p)})

df = pd.DataFrame(rows).drop_duplicates("release_date").sort_values("release_date")
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} releases to {out_path}")
print(f"With a forecast: {df.forecast.notna().sum()} | first: {df.release_date.min()} | last: {df.release_date.max()}")
print(df.tail(5).to_string(index=False))