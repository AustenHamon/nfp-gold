# cpi_clues.py -- do seasonality or last month's surprise predict hot/cool core CPI?
import os
import pandas as pd

cal = pd.read_csv("cpi_consensus.csv", parse_dates=["release_date"])
cal = cal.dropna(subset=["actual", "forecast"]).sort_values("release_date")
cal = cal[cal.release_date >= "2015-01-01"].reset_index(drop=True)
cal["surprise"] = (cal.actual - cal.forecast).round(2)
cal["side"] = cal.surprise.apply(lambda s: "HOT" if s > 0 else ("COOL" if s < 0 else "INLINE"))
cal["ref_month"] = (cal.release_date - pd.DateOffset(months=1)).dt.month   # month the data is for
cal["last_side"] = cal.side.shift(1)

base_hot, base_cool = (cal.side == "HOT").mean(), (cal.side == "COOL").mean()
print(f"{len(cal)} releases | HOT {base_hot:.0%} | INLINE {(cal.side=='INLINE').mean():.0%} | COOL {base_cool:.0%}\n")

def table(col, label):
    print(f"--- by {label} ---")
    g = cal.groupby(col).side.value_counts(normalize=True).unstack(fill_value=0)
    g["n"] = cal.groupby(col).size()
    print((g[["HOT", "INLINE", "COOL", "n"]] * [100, 100, 100, 1]).round(0).astype(int).to_string(), "\n")

table("ref_month", "data month (1 = January)")
table("last_side", "last month's result")

# Honest test: for each release, use ONLY earlier releases to pick HOT or COOL
def walk_forward(key, min_hist=24):
    hits, calls = 0, 0
    for i in range(min_hist, len(cal)):
        past, row = cal.iloc[:i], cal.iloc[i]
        grp = past[past[key] == row[key]]
        if len(grp) < 5 or row.side == "INLINE":
            continue
        pick = "HOT" if (grp.side == "HOT").mean() >= (grp.side == "COOL").mean() else "COOL"
        calls += 1
        hits += pick == row.side
    return hits, calls

print("--- walk-forward, releases that weren't in line ---")
for key, label in [("ref_month", "seasonal"), ("last_side", "last surprise")]:
    h, c = walk_forward(key)
    print(f"{label:14s}: right {h}/{c} = {h / c:.0%}")
nonflat = cal[cal.side != "INLINE"]
print(f"always COOL   : right {(nonflat.side == 'COOL').mean():.0%}  (the number to beat)")

# Optional: Manheim used vehicle index, two months before the data month
if os.path.exists("manheim.csv"):
    m = pd.read_csv("manheim.csv", parse_dates=["date"]).set_index("date")["index"].resample("MS").last()
    chg = m.pct_change() * 100
    cal["manheim_lag2"] = cal.release_date.apply(
        lambda d: chg.get(pd.Timestamp(d.year, d.month, 1) - pd.DateOffset(months=3)))
    x = cal.dropna(subset=["manheim_lag2"])
    x = x[x.side != "INLINE"]
    up = x[x.manheim_lag2 > 1]
    down = x[x.manheim_lag2 < -1]
    print(f"\nManheim up >1% two months earlier  ({len(up)}): HOT {(up.side == 'HOT').mean():.0%}")
    print(f"Manheim down >1% two months earlier ({len(down)}): COOL {(down.side == 'COOL').mean():.0%}")