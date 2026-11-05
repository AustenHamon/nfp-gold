# export_stats.py  -- runs on your PC, turns the CSVs into nfp_stats.json for the phone app
# Run after every NFP (once you've updated nfp_consensus.csv and rerun nfp_gold_dukascopy.py),
# then copy nfp_stats.json into the Flutter project's assets folder and rebuild.
import json, math
from datetime import date
from nfp_logic import load_reactions, load_calendar, pre_release_lean, BUCKETS

rx, cal = load_reactions(), load_calendar()
lean = pre_release_lean(cal, rx)
nxt = cal[cal["actual"].isna() & cal["forecast"].notna()].tail(1)


def clean(x, default=0.0):
    """JSON can't hold NaN, so swap it for a default."""
    x = float(x)
    return default if math.isnan(x) else round(x, 4)


buckets = []
for lo, hi, label in BUCKETS:
    g = rx[(rx.surprise > lo) & (rx.surprise <= hi)]
    buckets.append({
        "label": label,
        "lo": None if lo <= -1e9 else lo,          # null = no lower limit
        "hi": None if hi >= 1e9 else hi,           # null = no upper limit
        "n": int(len(g)),
        "p_buy_1m": clean((g.dir_1m == "BUY").mean(), 0.5),
        "p_buy_5m": clean((g.dir_5m == "BUY").mean(), 0.5),
        "avg_1m": clean(g.move_1m_pct.mean()),
        "avg_5m": clean(g.move_5m_pct.mean()),
    })

recent = rx.sort_values("release_date").tail(12).iloc[::-1]
history = [{"date": str(r.release_date.date()), "actual": clean(r.actual),
            "forecast": clean(r.forecast), "surprise": clean(r.surprise),
            "move_1m": clean(r.move_1m_pct)} for r in recent.itertuples()]

stats = {
    "updated": str(date.today()),
    "next_release": str(nxt["release_date"].iloc[0].date()) if len(nxt) else None,
    "next_forecast": clean(nxt["forecast"].iloc[0]) if len(nxt) else None,
    "releases": int(len(rx)),
    "p_beat": clean(lean["p_beat"]),
    "p_buy": clean(lean["p_buy"]),
    "buckets": buckets,
    "history": history,
}

with open("nfp_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(f"Saved nfp_stats.json | next release {stats['next_release']} | "
      f"beat {stats['p_beat']:.0%} | gold BUY {stats['p_buy']:.0%} / SELL {1 - stats['p_buy']:.0%}")