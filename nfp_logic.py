# nfp_logic.py  -- the middle layer: loads data, computes odds
import pandas as pd

START = "2015-01-01"

def load_reactions(path="nfp_gold_m1.csv"):
    df = pd.read_csv(path, parse_dates=["release_date"])
    return df[df["release_date"] >= START]

def load_calendar(path="nfp_consensus.csv"):
    cal = pd.read_csv(path, parse_dates=["release_date"]).sort_values("release_date")
    cal["surprise"] = cal["actual"] - cal["forecast"]
    return cal

def base_rates(cal, rx):
    hist = cal[(cal["release_date"] >= START) & cal["surprise"].notna() & (cal["surprise"] != 0)]
    p_beat = (hist["surprise"] > 0).mean()
    buy_beat = (rx.loc[rx.surprise > 0, "dir_1m"] == "BUY").mean()
    buy_miss = (rx.loc[rx.surprise < 0, "dir_1m"] == "BUY").mean()
    return p_beat, buy_beat, buy_miss

def pre_release_lean(cal, rx):
    p_beat, buy_beat, buy_miss = base_rates(cal, rx)
    p_buy = p_beat * buy_beat + (1 - p_beat) * buy_miss
    return {"p_beat": p_beat, "p_buy": p_buy, "p_sell": 1 - p_buy}

BUCKETS = [(-1e9, -75, "Big miss"), (-75, 0, "Small miss"),
           (0, 75, "Small beat"), (75, 1e9, "Big beat")]

def reaction_odds(rx, surprise):
    for lo, hi, label in BUCKETS:
        if lo < surprise <= hi:
            g = rx[(rx.surprise > lo) & (rx.surprise <= hi)]
            return {"bucket": label, "n": len(g),
                    "p_buy_1m": (g.dir_1m == "BUY").mean(),
                    "p_buy_5m": (g.dir_5m == "BUY").mean(),
                    "avg_1m": g.move_1m_pct.mean(), "avg_5m": g.move_5m_pct.mean()}
    return None

def bucket_table(rx):
    rows = []
    for lo, hi, label in BUCKETS:
        g = rx[(rx.surprise > lo) & (rx.surprise <= hi)]
        rows.append({"Surprise": label, "Releases": len(g),
                     "Gold BUY 1m": f"{(g.dir_1m == 'BUY').mean():.0%}",
                     "Avg 1m move": f"{g.move_1m_pct.mean():+.2f}%"})
    return pd.DataFrame(rows)