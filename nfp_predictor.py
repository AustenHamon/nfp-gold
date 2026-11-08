# Step 1: probability that NFP BEATS the forecast, using only pre-release data.
# Step 2: turn that into a gold BUY/SELL probability for the first minute,
#         using how gold actually reacted to beats and misses (nfp_gold_m1.csv).
#
# Needs in this folder: nfp_consensus.csv, nfp_gold_m1.csv
# pip install fredapi scikit-learn pandas
import os
import numpy as np
import pandas as pd
from fredapi import Fred
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss

fred = Fred(api_key="a66cba3e7117cf546810d8c58082965b ")
START = "2015-01-01"


class Clipper(BaseEstimator, TransformerMixin):
    """Caps extreme values, learned from training data only."""
    def fit(self, X, y=None):
        self.lo_ = np.nanquantile(X, 0.05, axis=0)
        self.hi_ = np.nanquantile(X, 0.95, axis=0)
        return self
    def transform(self, X):
        return np.clip(X, self.lo_, self.hi_)


# ---------------- Calendar + pre-release clues ----------------
cal = pd.read_csv("nfp_consensus.csv", parse_dates=["release_date"])
cal = cal.sort_values("release_date").reset_index(drop=True)
cal["surprise"] = cal["actual"] - cal["forecast"]
cal["last_surprise"] = cal["surprise"].shift(1)                 # last month's beat/miss
cal["forecast_vs_last"] = cal["forecast"] - cal["actual"].shift(1)  # do economists expect a rebound?

adp_chg = fred.get_series("ADPMNUSNERSA").dropna().diff() / 1000   # thousands of jobs
claims = fred.get_series("ICSA").dropna() / 1000                   # thousands


def last_before(s, day):
    s = s[s.index < day]
    return s.iloc[-1] if len(s) else np.nan


def claims_4wk_chg(day):
    s = claims[claims.index < day]
    return s.iloc[-1] - s.iloc[-5] if len(s) > 5 else np.nan


cal["adp_change"] = cal["release_date"].apply(lambda d: last_before(adp_chg, d))
cal["adp_vs_forecast"] = cal["adp_change"] - cal["forecast"]       # ADP hint at a beat or miss?
cal["claims_k"] = cal["release_date"].apply(lambda d: last_before(claims, d))
cal["claims_4wk_chg"] = cal["release_date"].apply(claims_4wk_chg)

FEATURES = ["adp_vs_forecast", "claims_k", "claims_4wk_chg", "last_surprise", "forecast_vs_last"]

# ---------------- Training data ----------------
hist = cal.dropna(subset=["actual", "forecast"] + FEATURES)
hist = hist[hist["release_date"] >= START]
hist = hist[~hist["release_date"].between("2020-03-01", "2021-06-30")]   # COVID chaos
hist = hist[hist["surprise"] != 0].reset_index(drop=True)
X, y = hist[FEATURES], (hist["surprise"] > 0).astype(int)              # 1 = BEAT
print(f"Training releases: {len(hist)} | beat rate: {y.mean():.0%}")

model = make_pipeline(Clipper(), StandardScaler(), LogisticRegression(C=0.3))

# ---------------- Honest walk-forward test (beat vs miss) ----------------
p_beat_oos, actual, base = pd.Series(index=hist.index, dtype=float), [], []
for tr, te in TimeSeriesSplit(n_splits=5).split(X):
    model.fit(X.iloc[tr], y.iloc[tr])
    p_beat_oos.iloc[te] = model.predict_proba(X.iloc[te])[:, 1]
    base += [y.iloc[tr].mean()] * len(te)
tested = p_beat_oos.dropna().index
yt, pt = y.loc[tested], p_beat_oos.loc[tested]

print("\n--- STEP 1: will NFP beat? (tested on unseen releases) ---")
print(f"Model log-loss:    {log_loss(yt, pt):.3f}")
print(f"Baseline log-loss: {log_loss(yt, base):.3f}  (model must be LOWER)")
print(f"Model hit rate:    {((pt > 0.5) == yt).mean():.0%}")
print(f"Always-guess rate: {max(yt.mean(), 1 - yt.mean()):.0%}")

# ---------------- Step 2: gold's reaction to beats / misses ----------------
m1 = pd.read_csv("nfp_gold_m1.csv", parse_dates=["release_date"])
m1 = m1[m1["release_date"] >= START]
buy_after_beat = (m1.loc[m1.surprise > 0, "dir_1m"] == "BUY").mean()
buy_after_miss = (m1.loc[m1.surprise < 0, "dir_1m"] == "BUY").mean()
print(f"\n--- STEP 2: gold first-minute reaction ---")
print(f"After a BEAT: gold BUY {buy_after_beat:.0%} | After a MISS: gold BUY {buy_after_miss:.0%}")

# ---------------- Combined backtest: did the gold probability work? ----------------
def gold_p_buy(p_beat):
    return p_beat * buy_after_beat + (1 - p_beat) * buy_after_miss

bt = hist.loc[tested, ["release_date"]].assign(p_buy=gold_p_buy(pt).values)
bt = bt.merge(m1[["release_date", "dir_1m"]], on="release_date")
if len(bt):
    hit = ((bt.p_buy > 0.5) == (bt.dir_1m == "BUY")).mean()
    always = max((bt.dir_1m == "BUY").mean(), (bt.dir_1m == "SELL").mean())
    print(f"\n--- COMBINED (gold direction, first minute, {len(bt)} unseen releases) ---")
    print(f"Model hit rate: {hit:.0%} | always-same-side rate: {always:.0%}")

# ---------------- Final model + what drives it ----------------
model.fit(X, y)
coefs = pd.Series(model[-1].coef_[0], index=FEATURES).sort_values()
print("\nWhat pushes toward a BEAT (+) or MISS (-):\n", coefs.round(3).to_string())

# ---------------- Next release ----------------
nxt = cal[cal["actual"].isna() & cal["forecast"].notna()].tail(1)
if nxt.empty:
    raise SystemExit("\nNo upcoming release in nfp_consensus.csv (add a row with the forecast and no actual).")
row = nxt.iloc[0]
print(f"\n--- NEXT NFP: {row.release_date.date()} | forecast {row.forecast:.0f}K ---")
print(nxt[FEATURES].round(1).T.to_string(header=False))
if nxt[FEATURES].isna().any(axis=None):
    print("\n!! Some inputs are missing (e.g. ADP not on FRED yet). Prediction may be off.")
p_beat = model.predict_proba(nxt[FEATURES].fillna(X.median()))[0, 1]
p_buy = gold_p_buy(p_beat)
print(f"\nChance NFP BEATS forecast: {p_beat:.0%} | MISSES: {1 - p_beat:.0%}")
print(f"GOLD first minute ->  BUY {p_buy:.0%} | SELL {1 - p_buy:.0%}")