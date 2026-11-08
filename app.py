# app.py  -- the top layer: Streamlit page
# pip install streamlit
# run with:  python -m streamlit run app.py
import streamlit as st
from nfp_logic import (load_reactions, load_calendar, pre_release_lean,
                       reaction_odds, bucket_table)

st.set_page_config(page_title="NFP Gold Predictor", page_icon="📊")
st.title("NFP → Gold Predictor")

rx, cal = load_reactions(), load_calendar()
nxt = cal[cal["actual"].isna() & cal["forecast"].notna()].tail(1)

# ---- Before the release ----
st.header("1. Before the release")
default_fc = float(nxt["forecast"].iloc[0]) if len(nxt) else 0.0
if len(nxt):
    st.caption(f"Next release: {nxt['release_date'].iloc[0].date()}")
forecast = st.number_input("Forecast (K)", value=default_fc, step=1.0)

lean = pre_release_lean(cal, rx)
c1, c2, c3 = st.columns(3)
c1.metric("Chance NFP beats", f"{lean['p_beat']:.0%}")
c2.metric("Gold BUY", f"{lean['p_buy']:.0%}")
c3.metric("Gold SELL", f"{lean['p_sell']:.0%}")
st.caption("Based on how often NFP has beaten the forecast since 2015, "
           "combined with gold's first-minute reaction.")

# ---- After the release ----
st.header("2. The number is out")
actual = st.number_input("Actual (K)", value=None, step=1.0, placeholder="Type the actual")
if actual is not None:
    surprise = actual - forecast
    r = reaction_odds(rx, surprise)
    side = "BUY" if r["p_buy_1m"] >= 0.5 else "SELL"
    conf = max(r["p_buy_1m"], 1 - r["p_buy_1m"])
    st.subheader(f"{r['bucket']} ({surprise:+.0f}K) → Gold {side} {conf:.0%}")
    d1, d2 = st.columns(2)
    d1.metric("First minute", f"BUY {r['p_buy_1m']:.0%}", f"avg {r['avg_1m']:+.2f}%")
    d2.metric("By 5 minutes", f"BUY {r['p_buy_5m']:.0%}", f"avg {r['avg_5m']:+.2f}%")
    st.caption(f"Based on {r['n']} past releases with a similar surprise.")

# ---- History ----
with st.expander("History since 2015"):
    st.dataframe(bucket_table(rx), hide_index=True)
    st.line_chart(rx.set_index("release_date")["move_1m_pct"])

st.warning("Probabilities, not guarantees. Check your prop firm's news-trading rules.")