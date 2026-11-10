# export_stats.py -- builds <event>_stats.json for the phone app
# Usage:  python export_stats.py        (all events)
#         python export_stats.py cpi    (one event)
import json, math, os, sys
from datetime import date
import pandas as pd

START = "2015-01-01"

EVENTS = {
    "nfp": dict(
        title="NFP", unit="K", decimals=0, up="Beat", down="Miss",
        up_phrase="beat the forecast", down_phrase="missed it",
        call_rule="The call is sell on a beat and buy on a miss",
        calendar="nfp_consensus.csv", reactions="nfp_gold_m1.csv",
        predictions="predictions.csv", out="nfp_stats.json",
        # (lo, hi, label, description) -- a surprise s belongs where lo < s <= hi
        buckets=[(-1e9, -75, "Big miss", "Missed by 75K or more"),
                 (-75, 0, "Small miss", "Missed by up to 75K"),
                 (0, 75, "Small beat", "Beat by up to 75K"),
                 (75, 1e9, "Big beat", "Beat by more than 75K")],
        # (hi, key, label, when) -- by size of surprise; first tier with |s| <= hi wins
        tiers=[(20, "skip", "Too close to call", "With a surprise under 20K"),
               (40, "weak", "Weak signal", "With a surprise of 20K to 40K"),
               (None, "strong", "Strong signal", "With a surprise over 40K")],
    ),
    "cpi": dict(
        title="Core CPI", unit="%", decimals=1, up="Hotter", down="Cooler",
        up_phrase="came in hotter than forecast", down_phrase="cooler",
        call_rule="The call is sell when core CPI comes in hotter than forecast and buy when it comes in cooler",
        calendar="cpi_consensus.csv", reactions="cpi_gold_m1.csv",
        predictions="cpi_predictions.csv", out="cpi_stats.json",
        lean_rule="last_surprise",   # hot months tend to follow hot months
        buckets=[(-1e9, -0.15, "Much cooler", "Cooler by 0.2 or more"),
                 (-0.15, -0.05, "Cooler", "Cooler by 0.1"),
                 (-0.05, 0.05, "In line", "Matched the forecast"),
                 (0.05, 0.15, "Hotter", "Hotter by 0.1"),
                 (0.15, 1e9, "Much hotter", "Hotter by 0.2 or more")],
        tiers=[(0.05, "skip", "In line", "When core CPI matches the forecast"),
               (None, "strong", "Clear surprise", "When core CPI misses the forecast by 0.1 or more")],
    ),
}


def clean(x, default=0.0):
    """JSON can't hold NaN, so swap it for a default."""
    x = float(x)
    return default if math.isnan(x) else round(x, 4)


def side_of(x):
    return "up" if x > 0 else ("down" if x < 0 else "flat")


def lean_from(past_cal, past_rx, last):
    """Chance of each outcome after a month like `last`, and the gold buy chance that follows."""
    g = past_cal[past_cal["last"] == last]
    if len(g) < 5:
        g = past_cal
    p = {k: (g["side"] == k).mean() for k in ("up", "flat", "down")}
    sides = past_rx["surprise"].apply(side_of)
    rate = {}
    for k in p:
        r = past_rx[sides == k]
        rate[k] = (r["dir_1m"] == "BUY").mean() if len(r) else 0.5
    return p, sum(p[k] * rate[k] for k in p)


def export(key):
    cfg = EVENTS[key]
    if not (os.path.exists(cfg["calendar"]) and os.path.exists(cfg["reactions"])):
        print(f"{key}: skipped (needs {cfg['calendar']} and {cfg['reactions']})")
        return

    cal = pd.read_csv(cfg["calendar"], parse_dates=["release_date"]).sort_values("release_date")
    rx = pd.read_csv(cfg["reactions"], parse_dates=["release_date"])
    rx = rx[rx["release_date"] >= START].copy()
    rx["surprise"] = rx["surprise"].round(3)
    nxt = cal[cal["actual"].isna() & cal["forecast"].notna()].tail(1)

    # ---- Lean before the number: how often each outcome happened, and how gold reacted ----
    n = len(rx)
    p_up = (rx.surprise > 0).mean()
    p_flat = (rx.surprise == 0).mean()
    p_down = (rx.surprise < 0).mean()
    p_buy = (rx.dir_1m == "BUY").mean()
    lean_side = "buy" if p_buy >= 0.5 else "sell"
    note = f"Since 2015, {cfg['title']} {cfg['up_phrase']} in {p_up:.0%} of releases and {cfg['down_phrase']} in {p_down:.0%}"
    if p_flat >= 0.05:
        note += f", with {p_flat:.0%} in line"
    note += f". The lean lands {'slightly ' if abs(p_buy - 0.5) < 0.1 else ''}on {lean_side}."

    lean_desc, lean_win_rule = None, None
    if cfg.get("lean_rule") == "last_surprise":
        hist = cal.dropna(subset=["actual", "forecast"]).copy()
        hist = hist[hist["release_date"] >= START]
        hist["side"] = (hist["actual"] - hist["forecast"]).round(3).apply(side_of)
        hist["last"] = hist["side"].shift(1)
        hist = hist.dropna(subset=["last"])
        word = {"up": cfg["up"].lower(), "flat": "in line", "down": cfg["down"].lower()}

        # today's lean, from the most recent release
        last = hist["side"].iloc[-1]
        p, p_buy = lean_from(hist, rx, last)
        p_up, p_flat, p_down = p["up"], p["flat"], p["down"]
        lean_side = "buy" if p_buy >= 0.5 else "sell"
        if_surprise = p_up / (p_up + p_down) if (p_up + p_down) else 0.5
        lean_dir = cfg["up"].lower() if if_surprise >= 0.5 else cfg["down"].lower()
        note = (f"Last month {cfg['title']} came in {word[last]}. Since 2015, after a month like that the next one "
                f"came in {word['up']} {p_up:.0%}, in line {p_flat:.0%} and {word['down']} {p_down:.0%}. "
                f"If it surprises, it's {max(if_surprise, 1 - if_surprise):.0%} likely to be {lean_dir}, "
                f"so the lean lands on {lean_side}.")

        # honest test: score each past release using only what was known before it
        hits = calls = 0
        for r in rx.itertuples():
            past_cal = hist[hist["release_date"] < r.release_date]
            past_rx = rx[rx["release_date"] < r.release_date]
            if len(past_cal) < 24:
                continue
            _, pb = lean_from(past_cal, past_rx, past_cal["side"].iloc[-1])
            calls += 1
            hits += ("BUY" if pb >= 0.5 else "SELL") == r.dir_1m
        lean_win_rule = hits / calls if calls else None
        lean_desc = (f"right using last month's surprise, tested on {calls} releases "
                     f"with only the data known before each one")

    # ---- Reaction by surprise bucket ----
    buckets = []
    for lo, hi, label, desc in cfg["buckets"]:
        g = rx[(rx.surprise > lo) & (rx.surprise <= hi)]
        buckets.append({
            "label": label, "desc": desc,
            "lo": None if lo <= -1e9 else lo, "hi": None if hi >= 1e9 else hi,
            "n": int(len(g)),
            "p_buy_1m": clean((g.dir_1m == "BUY").mean(), 0.5),
            "p_buy_5m": clean((g.dir_5m == "BUY").mean(), 0.5),
            "avg_1m": clean(g.move_1m_pct.mean()),
            "avg_5m": clean(g.move_5m_pct.mean()),
        })

    # ---- Track record: sell when above forecast, buy when below ----
    rx["call"] = rx["surprise"].apply(lambda s: None if s == 0 else ("SELL" if s > 0 else "BUY"))
    called = rx[rx["call"].notna()]
    won_1m = called["call"] == called["dir_1m"]
    won_5m = called["call"] == called["dir_5m"]
    track = {
        "n": int(len(called)), "wins_1m": int(won_1m.sum()),
        "win_1m": clean(won_1m.mean()), "win_5m": clean(won_5m.mean()),
        "lean_side": lean_side.upper(),
        "lean_win": clean(lean_win_rule if lean_win_rule is not None
                          else (rx["dir_1m"] == lean_side.upper()).mean()),
        "lean_desc": lean_desc,
        "since": str(rx["release_date"].min().date()),
    }

    # ---- Signal strength by size of surprise ----
    def tier_of(s):
        for hi, k, _, _ in cfg["tiers"]:
            if hi is None or abs(s) <= hi:
                return k
    rx["tier"] = rx["surprise"].apply(tier_of)
    tiers = []
    lo = 0
    for hi, k, label, when in cfg["tiers"]:
        g = rx[rx.tier == k]
        nz, z = g[g.call.notna()], g[g.call.isna()]
        wins = int((nz.call == nz.dir_1m).sum())
        if len(z):  # in-line releases: score the side gold took most often
            wins += int(max((z.dir_1m == "BUY").sum(), (z.dir_1m == "SELL").sum()))
        tiers.append({"key": k, "label": label, "when": when, "lo": lo, "hi": hi,
                      "n": int(len(g)), "win": clean(wins / len(g) if len(g) else float("nan"), 0.5),
                      "avg_move": clean(g.move_1m_pct.abs().mean())})
        lo = hi

    # ---- History (newest first) ----
    history = []
    for r in rx.sort_values("release_date", ascending=False).itertuples():
        call = r.call if isinstance(r.call, str) else None   # pandas turns None into NaN
        history.append({"date": str(r.release_date.date()), "actual": clean(r.actual),
                        "forecast": clean(r.forecast), "surprise": clean(r.surprise),
                        "move_1m": clean(r.move_1m_pct), "call": call,
                        "won": None if call is None else bool(call == r.dir_1m)})

    # ---- Live scoreboard: predictions saved BEFORE each release, scored after ----
    path = cfg["predictions"]
    preds = (pd.read_csv(path, parse_dates=["release_date"]) if os.path.exists(path)
             else pd.DataFrame(columns=["release_date", "forecast", "p_beat", "call", "made_on"]))
    preds["release_date"] = pd.to_datetime(preds["release_date"])
    today = pd.Timestamp.today().normalize()
    if len(nxt):
        d = nxt["release_date"].iloc[0]
        if d > today and d not in set(preds["release_date"]):
            preds = pd.concat([preds, pd.DataFrame([{
                "release_date": d, "forecast": float(nxt["forecast"].iloc[0]),
                "p_beat": round(float(p_up), 4), "call": lean_side.upper(),
                "made_on": today.date()}])], ignore_index=True)
            preds["release_date"] = pd.to_datetime(preds["release_date"])
            preds.to_csv(path, index=False, date_format="%Y-%m-%d")
            print(f"{key}: saved prediction for {d.date()} -> {lean_side.upper()}")

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
                     "move_1m": clean(r.move_1m_pct) if done else None, "won": won})
    streak = 0
    for row in reversed([x for x in rows if x["won"] is not None]):
        if not row["won"]:
            break
        streak += 1

    stats = {
        "event": key, "title": cfg["title"], "unit": cfg["unit"], "decimals": cfg["decimals"],
        "up_word": cfg["up"], "down_word": cfg["down"], "call_rule": cfg["call_rule"],
        "lean_note": note,
        "updated": str(date.today()),
        "next_release": str(nxt["release_date"].iloc[0].date()) if len(nxt) else None,
        "next_forecast": clean(nxt["forecast"].iloc[0]) if len(nxt) else None,
        "releases": int(n),
        "p_beat": clean(p_up), "p_buy": clean(p_buy),
        "buckets": buckets, "tiers": tiers, "track": track, "history": history,
        "live": {"wins": wins, "losses": losses, "streak": streak, "rows": rows[::-1]},
    }

    # Only rewrite the file when something besides the date changed (keeps git quiet)
    new = json.loads(json.dumps(stats, allow_nan=False))  # fails loudly instead of writing NaN
    if os.path.exists(cfg["out"]):
        old = json.load(open(cfg["out"]))
        if {k: v for k, v in old.items() if k != "updated"} == {k: v for k, v in new.items() if k != "updated"}:
            print(f"{key}: no change")
            return
    with open(cfg["out"], "w") as f:
        json.dump(stats, f, indent=2, allow_nan=False)
    print(f"{key}: saved {cfg['out']} | next {stats['next_release']} | {cfg['up'].lower()} {p_up:.0%} | "
          f"gold BUY {p_buy:.0%} / SELL {1 - p_buy:.0%} | live {wins}W-{losses}L")


if __name__ == "__main__":
    for k in (sys.argv[1:] or EVENTS.keys()):
        export(k)