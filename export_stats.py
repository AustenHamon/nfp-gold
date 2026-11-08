# export_stats.py  -- runs on your PC, turns the CSVs into nfp_stats.json for the phone app
# Run after every NFP (once you've updated nfp_consensus.csv and rerun nfp_gold_dukascopy.py),
# then copy nfp_stats.json into the Flutter project's assets folder and rebuild.
import os
import pandas as pd
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

# ---- Live scoreboard: predictions saved BEFORE each release, scored after ----
LIVE = "predictions.csv"
preds = (pd.read_csv(LIVE, parse_dates=["release_date"]) if os.path.exists(LIVE)
         else pd.DataFrame(columns=["release_date", "forecast", "p_beat", "call", "made_on"]))
today = pd.Timestamp.today().normalize()

# Save the prediction for the next release, only if it's at least a day away
if len(nxt):
    d = nxt["release_date"].iloc[0]
    if d > today and d not in set(preds["release_date"]):
        preds = pd.concat([preds, pd.DataFrame([{
            "release_date": d, "forecast": float(nxt["forecast"].iloc[0]),
            "p_beat": round(float(lean["p_beat"]), 4),
            "call": "SELL" if lean["p_buy"] < 0.5 else "BUY",
            "made_on": today.date()}])], ignore_index=True)
        preds.to_csv(LIVE, index=False, date_format="%Y-%m-%d")

res = preds.merge(rx[["release_date", "surprise", "dir_1m", "move_1m_pct"]],
                  on="release_date", how="left").sort_values("release_date")
rows, wins, losses = [], 0, 0
for r in res.itertuples():
    done = pd.notna(r.dir_1m)
    won = bool(r.call == r.dir_1m) if done else None
    wins += won is True
    losses += won is False
    rows.append({"date": str(r.release_date.date()), "call": r.call,
                 "p_beat": clean(r.p_beat), "forecast": clean(r.forecast),
                 "beat_called": bool(r.p_beat > 0.5),
                 "beat_actual": bool(r.surprise > 0) if done else None,
                 "move_1m": clean(r.move_1m_pct) if done else None, "won": won})
streak = 0
for row in reversed([x for x in rows if x["won"] is not None]):
    if not row["won"]:
        break
    streak += 1
live = {"wins": wins, "losses": losses, "streak": streak, "rows": rows[::-1]}

stats = {
    "updated": str(date.today()),
    "next_release": str(nxt["release_date"].iloc[0].date()) if len(nxt) else None,
    "next_forecast": clean(nxt["forecast"].iloc[0]) if len(nxt) else None,
    "releases": int(len(rx)),
    "p_beat": clean(lean["p_beat"]),
    "p_buy": clean(lean["p_buy"]),
    "buckets": buckets,
    "history": history,
    "live": live,
}

with open("nfp_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(f"Saved nfp_stats.json | next release {stats['next_release']} | "
      f"beat {stats['p_beat']:.0%} | gold BUY {stats['p_buy']:.0%} / SELL {1 - stats['p_buy']:.0%}")