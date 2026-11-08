# Pulls 1-minute XAUUSD from your MT5 terminal around every NFP release,
# measures gold's move after 1, 5 and 15 minutes, and links it to the surprise.
# pip install MetaTrader5 pandas tzdata
# MT5 must be open and logged in to your broker.

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pandas as pd
import MetaTrader5 as mt5

NY = ZoneInfo("America/New_York")

if not mt5.initialize():
    raise SystemExit(f"MT5 connection failed: {mt5.last_error()}")

# ---- Find the gold symbol your broker uses (XAUUSD, XAUUSD.m, GOLD...) ----
names = [s.name for s in mt5.symbols_get() if "XAU" in s.name.upper() and "USD" in s.name.upper()]
if not names:
    names = [s.name for s in mt5.symbols_get() if "GOLD" in s.name.upper()]
SYMBOL = names[0]
mt5.symbol_select(SYMBOL, True)
print("Using symbol:", SYMBOL)

# ---- Broker server time offset vs New York (most brokers stay NY+7 all year) ----
tick = mt5.symbol_info_tick(SYMBOL)
server_now = datetime.fromtimestamp(tick.time, tz=timezone.utc).replace(tzinfo=None)
ny_now = datetime.now(NY).replace(tzinfo=None)
OFFSET = timedelta(hours=round((server_now - ny_now).total_seconds() / 3600))
print("Server is NY +", OFFSET)

# ---- Releases from your cleaned calendar ----
cal = pd.read_csv("nfp_consensus.csv", parse_dates=["release_date"]).dropna(subset=["actual"])

rows = []
for _, r in cal.iterrows():
    ny_release = datetime.combine(r.release_date.date(), datetime.min.time()).replace(hour=8, minute=30)
    srv = ny_release + OFFSET                                   # release time in broker clock
    start = (srv - timedelta(minutes=5)).replace(tzinfo=timezone.utc)
    end = (srv + timedelta(minutes=16)).replace(tzinfo=timezone.utc)
    bars = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M1, start, end)
    if bars is None or len(bars) == 0:
        continue
    b = pd.DataFrame(bars)
    b["time"] = pd.to_datetime(b["time"], unit="s")
    before = b[b["time"] < srv]
    after = b[b["time"] >= srv]
    if before.empty or len(after) < 15:
        continue
    p0 = before["close"].iloc[-1]                               # price just before release
    def move(n):
        return round((after["close"].iloc[n - 1] / p0 - 1) * 100, 3)
    rows.append({
        "release_date": r.release_date.date(),
        "actual": r.actual, "forecast": r.forecast,
        "surprise": r.actual - r.forecast if pd.notna(r.forecast) else None,
        "move_1m_pct": move(1), "move_5m_pct": move(5), "move_15m_pct": move(15),
        "spike_high_pct": round((after["high"].iloc[:5].max() / p0 - 1) * 100, 3),
        "spike_low_pct": round((after["low"].iloc[:5].min() / p0 - 1) * 100, 3),
    })

mt5.shutdown()

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("No M1 data found. Your broker's M1 history may be too short.")
df["dir_1m"] = df["move_1m_pct"].apply(lambda x: "BUY" if x > 0 else "SELL")
df["dir_5m"] = df["move_5m_pct"].apply(lambda x: "BUY" if x > 0 else "SELL")
df.to_csv("nfp_gold_m1.csv", index=False)

print(f"\nSaved {len(df)} releases | from {df.release_date.min()} to {df.release_date.max()}")
s = df.dropna(subset=["surprise"])
beats, misses = s[s.surprise > 0], s[s.surprise < 0]
print(f"\nWhen NFP BEAT  ({len(beats)}): gold SOLD in 1st minute {(beats.dir_1m == 'SELL').mean():.0%}, by 5 min {(beats.dir_5m == 'SELL').mean():.0%}")
print(f"When NFP MISSED ({len(misses)}): gold BOUGHT in 1st minute {(misses.dir_1m == 'BUY').mean():.0%}, by 5 min {(misses.dir_5m == 'BUY').mean():.0%}")
print("\nLast 5:\n", df.tail(5).to_string(index=False))