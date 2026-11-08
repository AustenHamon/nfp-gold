# Fixes: COVID outliers capped/removed, stronger regularisation,
# optional consensus forecasts + last month's surprise
# Optional file: nfp_consensus.csv with columns  release_date,actual,forecast  (in thousands)

import os
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score

fred = Fred(api_key=" a66cba3e7117cf546810d8c58082965b")
df = pd.read_csv("nfp_gold_dataset.csv", parse_dates=["release_date"])

# ---------- Outlier cap learned ONLY from training data ----------
class Clipper(BaseEstimator, TransformerMixin):
    def __init__(self, lo=0.02, hi=0.98):
        self.lo, self.hi = lo, hi
    def fit(self, X, y=None):
        self.lo_ = np.nanquantile(X, self.lo, axis=0)
        self.hi_ = np.nanquantile(X, self.hi, axis=0)
        return self
    def transform(self, X):
        return np.clip(X, self.lo_, self.hi_)

# ---------- Features ----------
adp = fred.get_series("ADPMNUSNERSA").dropna()
adp_chg = adp.diff() / 1000
claims = fred.get_series("ICSA").dropna()

def last_before(s, day):
    s = s[s.index < day]
    return s.iloc[-1] if len(s) else np.nan

def claims_4wk_chg(day):
    s = claims[claims.index < day]
    return (s.iloc[-1] - s.iloc[-5]) / 1000 if len(s) > 5 else np.nan

df["adp_change"] = df["release_date"].apply(lambda d: last_before(adp_chg, d))
df["claims_k"] = df["jobless_claims"] / 1000
df["claims_4wk_chg"] = df["release_date"].apply(claims_4wk_chg)
df["last_nfp_change"] = df["nfp_change"].shift(1)   # previous month's print

# Drop the COVID shock period entirely
df = df[~df["release_date"].between("2020-03-01", "2021-06-30")]

FEATURES = ["adp_change", "claims_k", "claims_4wk_chg", "last_nfp_change",
            "us10y_5d_chg", "usd_5d_chg", "gold_5d_runup_pct"]

# ---------- Optional consensus forecasts ----------
has_consensus = os.path.exists("nfp_consensus.csv")
if has_consensus:
    cons = pd.read_csv("nfp_consensus.csv", parse_dates=["release_date"]).sort_values("release_date")
    cons["surprise"] = cons["actual"] - cons["forecast"]
    cons["last_surprise"] = cons["surprise"].shift(1)        # known before release
    df = pd.merge_asof(df.sort_values("release_date"),
                       cons[["release_date", "forecast", "last_surprise"]],
                       on="release_date", tolerance=pd.Timedelta("3D"))
    df["adp_vs_forecast"] = df["adp_change"] - df["forecast"]  # does ADP hint at a beat/miss?
    FEATURES += ["forecast", "last_surprise", "adp_vs_forecast"]

df = df.dropna(subset=FEATURES).reset_index(drop=True)
X, y = df[FEATURES], (df["direction"] == "BUY").astype(int)
print(f"Training rows: {len(df)} | consensus data: {'YES' if has_consensus else 'NO'}")

model = make_pipeline(Clipper(), StandardScaler(), LogisticRegression(C=0.1))

# ---------- Honest walk-forward test ----------
probs, actual, base_probs = [], [], []
for tr, te in TimeSeriesSplit(n_splits=5).split(X):
    model.fit(X.iloc[tr], y.iloc[tr])
    probs += list(model.predict_proba(X.iloc[te])[:, 1])
    actual += list(y.iloc[te])
    base_probs += [y.iloc[tr].mean()] * len(te)   # baseline uses only past data too

print(f"\nModel log-loss:    {log_loss(actual, probs):.3f}")
print(f"Baseline log-loss: {log_loss(actual, base_probs):.3f}  (model must be LOWER)")
print(f"Model hit rate:    {accuracy_score(actual, [p > 0.5 for p in probs]):.0%}")
print(f"Always-BUY rate:   {np.mean(actual):.0%}")

# ---------- Final model + drivers ----------
model.fit(X, y)
coefs = pd.Series(model[-1].coef_[0], index=FEATURES).sort_values()
print("\nFeature influence (+ BUY, - SELL):\n", coefs.round(3))

# ---------- Next NFP ----------
today = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
gold = yf.download("GC=F", period="1mo", progress=False)["Close"].squeeze()
us10y, usd = fred.get_series("DGS10").dropna(), fred.get_series("DTWEXBGS").dropna()

now = {
    "adp_change": last_before(adp_chg, today),
    "claims_k": last_before(claims, today) / 1000,
    "claims_4wk_chg": claims_4wk_chg(today),
    "last_nfp_change": df["nfp_change"].iloc[-1],
    "us10y_5d_chg": us10y.iloc[-1] - us10y.iloc[-6],
    "usd_5d_chg": usd.iloc[-1] - usd.iloc[-6],
    "gold_5d_runup_pct": (gold.iloc[-1] / gold.iloc[-6] - 1) * 100,
}
if has_consensus:
    now["forecast"] = float(input("Enter this Friday's NFP forecast (thousands): "))
    now["last_surprise"] = cons["surprise"].dropna().iloc[-1]
    now["adp_vs_forecast"] = now["adp_change"] - now["forecast"]

now = pd.DataFrame([now])[FEATURES]
p_buy = model.predict_proba(now)[0, 1]
print("\nInputs right now:\n", now.round(3).T)
print(f"\nNEXT NFP -> Gold BUY: {p_buy:.0%} | SELL: {1 - p_buy:.0%}")