
# Free 1-minute gold reaction to every NFP since 2008, built from Dukascopy tick data.
#
# FIXES vs v1
#  1) Each hour's ticks were time-stamped from HH:30 instead of HH:00, so the
#     script was measuring gold 30 minutes BEFORE the release (tiny 0.02% moves).
#  2) Downloads are cached in ./duka_cache and retried properly, so failed hours
#     ("short data" / "no data") are fetched again on the next run instead of lost.
#  3) Sanity check against your MT5 numbers for the last 3 releases.

import lzma, os, struct, time, urllib.request, urllib.error
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd

NY, UTC = ZoneInfo("America/New_York"), ZoneInfo("UTC")
URL = "https://datafeed.dukascopy.com/datafeed/XAUUSD/{y}/{m:02d}/{d:02d}/{h:02d}h_ticks.bi5"
CACHE = "duka_cache"
os.makedirs(CACHE, exist_ok=True)


def download(hour_utc):
    """Return raw bi5 bytes for one UTC hour (cached). b'' = no ticks that hour."""
    path = os.path.join(CACHE, hour_utc.strftime("%Y%m%d_%H") + ".bi5")
    if os.path.exists(path):
        return open(path, "rb").read()
    # Dukascopy months are 0-based
    url = URL.format(y=hour_utc.year, m=hour_utc.month - 1, d=hour_utc.day, h=hour_utc.hour)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(5):
        try:
            raw = urllib.request.urlopen(req, timeout=30).read()
            open(path, "wb").write(raw)          # cache only successful downloads
            return raw
        except urllib.error.HTTPError as e:
            if e.code == 404:
                open(path, "wb").write(b"")
                return b""
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))            # back off: 2s, 4s, 6s, 8s, 10s
    return None                                  # failed; not cached, retried next run


def hour_ticks(hour_utc):
    hour_utc = hour_utc.replace(minute=0, second=0, microsecond=0)   # <-- THE FIX
    raw = download(hour_utc)
    if not raw:
        return pd.DataFrame()
    data = lzma.decompress(raw)
    rows = [struct.unpack(">IIIff", data[i:i + 20]) for i in range(0, len(data) - 19, 20)]
    df = pd.DataFrame(rows, columns=["ms", "ask", "bid", "ask_vol", "bid_vol"])
    df["time"] = pd.Timestamp(hour_utc.replace(tzinfo=None)) + pd.to_timedelta(df["ms"], unit="ms")
    df["mid"] = (df["ask"] + df["bid"]) / 2
    return df[["time", "mid"]]


def fix_scale(s):
    """Dukascopy stores prices as integers; scale gold into a sensible range."""
    for div in (1000, 100, 10, 1):
        if 200 < s.median() / div < 10000:
            return s / div
    return s


cal = pd.read_csv("nfp_consensus.csv", parse_dates=["release_date"])
cal = cal.dropna(subset=["actual", "forecast"])
cal = cal[cal["release_date"] >= "2015-01-01"]

rows, failed = [], []
for _, r in cal.iterrows():
    ny = datetime.combine(r.release_date.date(), datetime.min.time()).replace(hour=8, minute=30, tzinfo=NY)
    rel = ny.astimezone(UTC)                                      # 12:30 or 13:30 UTC
    ticks = pd.concat([hour_ticks(rel - timedelta(hours=1)), hour_ticks(rel)], ignore_index=True)
    if ticks.empty:
        failed.append(r.release_date.date()); print("no data:", r.release_date.date()); continue
    ticks["mid"] = fix_scale(ticks["mid"])
    m1 = ticks.set_index("time")["mid"].resample("1min").ohlc().dropna()
    rel_naive = pd.Timestamp(rel.replace(tzinfo=None))
    before, after = m1[m1.index < rel_naive], m1[m1.index >= rel_naive]
    if before.empty or len(after) < 15:
        failed.append(r.release_date.date()); print("short data:", r.release_date.date()); continue
    p0 = before["close"].iloc[-1]                                 # last price before 8:30 NY
    mv = lambda n: round((after["close"].iloc[n - 1] / p0 - 1) * 100, 3)
    rows.append({
        "release_date": r.release_date.date(), "actual": r.actual, "forecast": r.forecast,
        "surprise": r.actual - r.forecast,
        "move_1m_pct": mv(1), "move_5m_pct": mv(5), "move_15m_pct": mv(15),
        "spike_high_pct": round((after["high"].iloc[:5].max() / p0 - 1) * 100, 3),
        "spike_low_pct": round((after["low"].iloc[:5].min() / p0 - 1) * 100, 3),
    })
    print(f"{r.release_date.date()}  surprise {r.actual - r.forecast:+.0f}  1m {mv(1):+.2f}%")
    time.sleep(0.3)

df = pd.DataFrame(rows)
df["dir_1m"] = df["move_1m_pct"].apply(lambda x: "BUY" if x > 0 else "SELL")
df["dir_5m"] = df["move_5m_pct"].apply(lambda x: "BUY" if x > 0 else "SELL")
df.to_csv("nfp_gold_m1.csv", index=False)

# ---- Sanity check against your Goat Funded MT5 numbers ----
mt5_check = {"2026-08-07": 1.057, "2026-09-04": -1.505, "2026-10-02": 0.667}
print("\nSanity check vs MT5 (1-minute move %):")
for d, mt5_val in mt5_check.items():
    hit = df[df.release_date.astype(str) == d]
    duka = f"{hit.move_1m_pct.iloc[0]:+.3f}" if len(hit) else "missing"
    print(f"  {d}  MT5 {mt5_val:+.3f}  |  Dukascopy {duka}")

beats, misses = df[df.surprise > 0], df[df.surprise < 0]
print(f"\nSaved {len(df)} releases ({df.release_date.min()} to {df.release_date.max()})")
print(f"Still missing: {len(failed)}  (run the script again to retry them; cached hours are skipped)")
print(f"NFP BEAT   ({len(beats)}): gold SOLD 1st min {(beats.dir_1m == 'SELL').mean():.0%} | by 5 min {(beats.dir_5m == 'SELL').mean():.0%}")
print(f"NFP MISSED ({len(misses)}): gold BOUGHT 1st min {(misses.dir_1m == 'BUY').mean():.0%} | by 5 min {(misses.dir_5m == 'BUY').mean():.0%}")
for lo, hi, label in [(-1e9, -75, "big miss"), (-75, 0, "small miss"), (0, 75, "small beat"), (75, 1e9, "big beat")]:
    g = df[(df.surprise > lo) & (df.surprise <= hi)]
    if len(g):
        print(f"  {label:10s} ({len(g):3d}): gold BUY {(g.dir_1m == 'BUY').mean():.0%}, avg 1m move {g.move_1m_pct.mean():+.2f}%")