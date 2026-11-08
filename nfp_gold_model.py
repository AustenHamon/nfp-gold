# Trains a probability model on nfp_gold_dataset.csv and predicts the NEXT NFP
# pip install scikit-learn
import os
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score

fred = Fred(api_key=" a66cba3e7117cf546810d8c58082965b ")
df = pd.read_csv("nfp_gold_dataset.csv", parse_dates=["release_date"])

# ---- Fix ADP: use monthly CHANGE, plus claims trend ----
adp = fred.get_series("ADPMNUSNERSA").dropna()
adp_chg = adp.diff() / 1000          # thousands of jobs, like NFP
claims = fred.get_series("ICSA").dropna()

def last_before(s, day):
    s = s.dropna(); s = s[s.index < day]
    return s.iloc[-1] if len(s) else None

def claims_4wk_chg(day):
    s = claims[claims.index < day]
    return (s.iloc[-1] - s.iloc[-5]) / 1000 if len(s) > 5 else None

df["adp_change"] = df["release_date"].apply(lambda d: last_before(adp_chg, d))
df["claims_4wk_chg"] = df["release_date"].apply(claims_4wk_chg)
df["claims_k"] = df["jobless_claims"] / 1000
df = df.dropna(subset=["adp_change", "claims_4wk_chg"])

FEATURES = ["adp_change", "claims_k", "claims_4wk_chg",
            "us10y_5d_chg", "usd_5d_chg", "gold_5d_runup_pct"]
X, y = df[FEATURES], (df["direction"] == "BUY").astype(int)

model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5))

# ---- Honest test: always train on the past, test on the future ----
probs, actual = [], []
for train_idx, test_idx in TimeSeriesSplit(n_splits=5).split(X):
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    probs += list(model.predict_proba(X.iloc[test_idx])[:, 1])
    actual += list(y.iloc[test_idx])

base = sum(actual) / len(actual)
print(f"Model log-loss:    {log_loss(actual, probs):.3f}")
print(f"Baseline log-loss: {log_loss(actual, [base]*len(actual)):.3f}  (lower is better)")
print(f"Model hit rate:    {accuracy_score(actual, [p > 0.5 for p in probs]):.0%}")
print(f"Always-BUY rate:   {base:.0%}")

# ---- Train on everything, show what drives the probability ----
model.fit(X, y)
coefs = pd.Series(model[-1].coef_[0], index=FEATURES).sort_values()
print("\nFeature influence (+ pushes toward BUY, - toward SELL):")
print(coefs.round(3))

# ---- Predict the next NFP using today's data ----
today = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
gold = yf.download("GC=F", period="1mo", progress=False)["Close"].squeeze()
us10y, usd = fred.get_series("DGS10").dropna(), fred.get_series("DTWEXBGS").dropna()

now = pd.DataFrame([{
    "adp_change": last_before(adp_chg, today),
    "claims_k": last_before(claims, today) / 1000,
    "claims_4wk_chg": claims_4wk_chg(today),
    "us10y_5d_chg": us10y.iloc[-1] - us10y.iloc[-6],
    "usd_5d_chg": usd.iloc[-1] - usd.iloc[-6],
    "gold_5d_runup_pct": (gold.iloc[-1] / gold.iloc[-6] - 1) * 100,
}])
p_buy = model.predict_proba(now[FEATURES])[0, 1]
print("\nInputs right now:\n", now.round(3).T)
print(f"\nNEXT NFP -> Gold BUY: {p_buy:.0%} | SELL: {1 - p_buy:.0%}")