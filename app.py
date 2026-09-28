"""
Spxder Signals
Consistency hybrid for MNQ · NQ · ES

Rules (in order):
1. Higher-timeframe bias required
2. Killzone only (London 02:00–05:00 or NY 08:30–11:00)
3. Liquidity sweep required
4. Mean-reversion / equilibrium entry (not breakout chase)
5. Max quality filter — all steps must pass

Signals are informational only. No broker orders are placed.
Default risk: $250 on a $50k account.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Spxder Signals",
    page_icon="🕷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label { color: #a0a0c0 !important; }
    .signal-buy {
        background: linear-gradient(135deg, #0d3320, #1a5c3a);
        border: 1px solid #2ecc71;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        color: #2ecc71;
    }
    .signal-sell {
        background: linear-gradient(135deg, #3d1515, #5c1a1a);
        border: 1px solid #e74c3c;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        color: #e74c3c;
    }
    .signal-none {
        background: linear-gradient(135deg, #1a1a2e, #222240);
        border: 1px solid #444466;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        font-size: 1.15rem;
        font-weight: 600;
        color: #8899aa;
    }
    .step-ok { color: #2ecc71; }
    .step-miss { color: #e74c3c; }
    h1 { letter-spacing: -0.5px; }
</style>
""", unsafe_allow_html=True)

DEFAULTS = {
    "account_size": 50000.0,
    "risk_dollars": 250.0,
    "max_daily_loss_pct": 2.0,
    "min_rr": 1.8,
    "swing_lookback": 3,
    "sl_buffer_pts": 2.0,
    "eq_tolerance": 0.18,
    "require_killzone": True,
}

INSTRUMENTS = [
    {"symbol": "MNQ=F", "display": "MNQ", "point_value": 2.0},
    {"symbol": "NQ=F",  "display": "NQ",  "point_value": 20.0},
    {"symbol": "ES=F",  "display": "ES",  "point_value": 50.0},
]


@st.cache_data(ttl=60, show_spinner=False)
def fetch_bars(symbol: str, interval: str = "5m", limit: int = 400) -> pd.DataFrame:
    try:
        period = "5d" if interval in ("5m", "15m") else "60d"
        df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
        if (df is None or df.empty) and symbol == "MNQ=F":
            df = yf.Ticker("NQ=F").history(period=period, interval=interval, auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume"
        })
        df = df[["open", "high", "low", "close", "volume"]].dropna().tail(limit)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception as e:
        logger.error("fetch %s: %s", symbol, e)
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_daily(symbol: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(symbol).history(period="60d", interval="1d", auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df[["open", "high", "low", "close"]].dropna()
    except Exception:
        return pd.DataFrame()


def in_killzone(ts: datetime) -> bool:
    t = ts.time()
    return (time(2, 0) <= t <= time(5, 0)) or (time(8, 30) <= t <= time(11, 0))


def htf_bias(daily: pd.DataFrame) -> str:
    if daily is None or len(daily) < 4:
        return "neutral"
    c = daily["close"].values
    if c[-1] > c[-3] and c[-1] > c[-4]:
        return "bullish"
    if c[-1] < c[-3] and c[-1] < c[-4]:
        return "bearish"
    return "neutral"


def session_levels(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    d = df.copy()
    d["date"] = d.index.date
    d["t"] = d.index.time
    out = {}
    for day, g in d.groupby("date"):
        asia = g[(g["t"] >= time(0, 0)) & (g["t"] < time(6, 0))]
        lon = g[(g["t"] >= time(2, 0)) & (g["t"] < time(8, 0))]
        if not asia.empty:
            out["asia_high"] = float(asia["high"].max())
            out["asia_low"] = float(asia["low"].min())
        if not lon.empty:
            out["london_high"] = float(lon["high"].max())
            out["london_low"] = float(lon["low"].min())
    days = sorted(d["date"].unique())
    if len(days) >= 2:
        prev = d[d["date"] == days[-2]]
        if not prev.empty:
            out["pdh"] = float(prev["high"].max())
            out["pdl"] = float(prev["low"].min())
    return out


def find_swings(df: pd.DataFrame, lb: int = 3):
    h, l = df["high"].values, df["low"].values
    sh, sl = [], []
    for i in range(lb, len(df) - lb):
        if h[i] == max(h[i - lb:i + lb + 1]):
            sh.append({"i": i, "price": float(h[i])})
        if l[i] == min(l[i - lb:i + lb + 1]):
            sl.append({"i": i, "price": float(l[i])})
    return sh, sl


def analyze(df: pd.DataFrame, daily: pd.DataFrame, cfg: dict) -> dict:
    res = {
        "htf_bias": "neutral",
        "in_killzone": False,
        "sweep": False,
        "sweep_side": None,
        "mean_rev_entry": False,
        "direction": None,
        "entry": None,
        "stop": None,
        "eq": None,
        "impulse_high": None,
        "impulse_low": None,
        "reasons": [],
        "present": [],
        "missing": [],
    }

    if df.empty or len(df) < 40:
        res["reasons"].append("Insufficient data")
        res["missing"] = ["data"]
        return res

    price = float(df["close"].iloc[-1])
    ts = df.index[-1]
    res["in_killzone"] = in_killzone(ts)
    bias = htf_bias(daily)
    res["htf_bias"] = bias

    if bias == "neutral":
        res["missing"].append("htf_bias")
        res["reasons"].append("No clear higher-timeframe bias")
    else:
        res["present"].append("htf_bias")
        res["reasons"].append(f"HTF bias: {bias}")

    if res["in_killzone"]:
        res["present"].append("killzone")
        res["reasons"].append("Inside killzone")
    else:
        res["missing"].append("killzone")
        res["reasons"].append("Outside killzone")

    if cfg["require_killzone"] and not res["in_killzone"]:
        return res

    levels = session_levels(df)
    look = min(36, len(df) - 1)
    window = df.iloc[-look:]
    sh, sl = find_swings(df, cfg["swing_lookback"])

    swept_high = swept_low = False
    level_hit = None

    candidates_high = [levels[k] for k in ("asia_high", "london_high", "pdh") if levels.get(k)]
    candidates_low = [levels[k] for k in ("asia_low", "london_low", "pdl") if levels.get(k)]
    if len(sh) >= 2:
        candidates_high.append(sh[-2]["price"])
    if len(sl) >= 2:
        candidates_low.append(sl[-2]["price"])

    for lvl in candidates_high:
        if window["high"].max() > lvl and price < lvl:
            swept_high = True
            level_hit = lvl
            break
    for lvl in candidates_low:
        if window["low"].min() < lvl and price > lvl:
            swept_low = True
            level_hit = lvl
            break

    res["sweep"] = swept_high or swept_low
    if swept_low and not swept_high:
        res["sweep_side"] = "sell_side"
        res["present"].append("liquidity_sweep")
        res["reasons"].append(f"Sell-side liquidity swept ({level_hit:.2f})")
    elif swept_high and not swept_low:
        res["sweep_side"] = "buy_side"
        res["present"].append("liquidity_sweep")
        res["reasons"].append(f"Buy-side liquidity swept ({level_hit:.2f})")
    else:
        res["missing"].append("liquidity_sweep")
        res["reasons"].append("No clear liquidity sweep")
        return res

    if res["sweep_side"] == "sell_side":
        direction = "BUY"
    else:
        direction = "SELL"

    if bias == "bullish" and direction == "SELL":
        res["missing"].append("bias_aligned")
        res["reasons"].append("Sweep direction conflicts with bullish HTF bias")
        return res
    if bias == "bearish" and direction == "BUY":
        res["missing"].append("bias_aligned")
        res["reasons"].append("Sweep direction conflicts with bearish HTF bias")
        return res

    res["present"].append("bias_aligned")
    res["direction"] = direction
    res["impulse_high"] = float(window["high"].max())
    res["impulse_low"] = float(window["low"].min())
    hi, lo = res["impulse_high"], res["impulse_low"]
    if hi <= lo:
        res["reasons"].append("Invalid impulse range")
        return res

    eq = (hi + lo) / 2.0
    res["eq"] = eq
    tol = cfg["eq_tolerance"] * (hi - lo)

    if direction == "BUY":
        in_value = price <= eq + tol
        near_eq = abs(price - eq) <= tol * 2.5 or price < eq
    else:
        in_value = price >= eq - tol
        near_eq = abs(price - eq) <= tol * 2.5 or price > eq

    if in_value and near_eq:
        res["mean_rev_entry"] = True
        res["present"].append("mean_reversion_entry")
        res["reasons"].append(f"Price in value relative to equilibrium {eq:.2f}")
    else:
        res["missing"].append("mean_reversion_entry")
        res["reasons"].append(f"Price not yet in mean-reversion zone around {eq:.2f}")
        return res

    buf = cfg["sl_buffer_pts"]
    res["entry"] = price
    if direction == "BUY":
        res["stop"] = lo - buf
    else:
        res["stop"] = hi + buf

    return res


@dataclass
class Signal:
    instrument: str
    direction: str = "NO TRADE"
    entry: Optional[float] = None
    stop: Optional[float] = None
    t1: Optional[float] = None
    t2: Optional[float] = None
    t3: Optional[float] = None
    rr: Optional[float] = None
    score: float = 0.0
    confidence: str = "low"
    present: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    price: Optional[float] = None
    bias: str = "neutral"
    in_killzone: bool = False
    analysis: dict = field(default_factory=dict)


def build_signal(display: str, df: pd.DataFrame, daily: pd.DataFrame, cfg: dict, data_ok: bool) -> Signal:
    sig = Signal(instrument=display)
    if not data_ok or df.empty:
        sig.reasons = ["DATA DISCONNECTED — no signals generated"]
        sig.missing = ["data_feed"]
        return sig

    a = analyze(df, daily, cfg)
    sig.analysis = a
    sig.price = float(df["close"].iloc[-1])
    sig.bias = a["htf_bias"]
    sig.in_killzone = a["in_killzone"]
    sig.present = a["present"]
    sig.missing = a["missing"]
    sig.reasons = a["reasons"]

    score = len(a["present"]) * 20
    if a.get("mean_rev_entry"):
        score = min(100, score + 10)
    sig.score = float(score)
    sig.confidence = "high" if score >= 80 else "medium" if score >= 60 else "low"

    needed = {"htf_bias", "killzone", "liquidity_sweep", "bias_aligned", "mean_reversion_entry"}
    if cfg["require_killzone"] is False:
        needed.discard("killzone")

    if needed.issubset(set(a["present"])) and a["direction"] and a["entry"] and a["stop"]:
        sig.direction = a["direction"]
        sig.entry = a["entry"]
        sig.stop = a["stop"]
        risk = abs(sig.entry - sig.stop)
        if risk > 0:
            rr = cfg["min_rr"]
            if sig.direction == "BUY":
                sig.t1 = sig.entry + risk * rr
                sig.t2 = sig.entry + risk * rr * 1.5
                sig.t3 = sig.entry + risk * rr * 2.0
            else:
                sig.t1 = sig.entry - risk * rr
                sig.t2 = sig.entry - risk * rr * 1.5
                sig.t3 = sig.entry - risk * rr * 2.0
            sig.rr = rr
    else:
        sig.direction = "NO TRADE"

    return sig


def calc_risk(sig: Signal, point_value: float, cfg: dict) -> dict:
    risk_amt = cfg["risk_dollars"]
    max_daily = cfg["account_size"] * (cfg["max_daily_loss_pct"] / 100)
    contracts = 0
    notes = []
    if sig.direction in ("BUY", "SELL") and sig.entry and sig.stop and point_value > 0:
        pts = abs(sig.entry - sig.stop)
        if pts > 0:
            contracts = max(1, int(risk_amt / (pts * point_value)))
            notes.append(f"{pts:.2f} pts x ${point_value:.0f} = ${pts * point_value:.0f}/ctr -> {contracts} ctr")
    return {
        "contracts": contracts,
        "risk_amount": round(risk_amt, 2),
        "max_daily_loss": round(max_daily, 2),
        "notes": notes,
    }


def main():
    st.markdown("# 🕷️ Spxder Signals")
    st.caption(
        "Consistency hybrid · HTF bias · Killzone · Liquidity sweep · Mean-reversion entry  ·  "
        "Informational only — no auto-trading"
    )

    st.sidebar.markdown("### ⚙️ Settings")
    with st.sidebar.expander("Strategy filters", expanded=True):
        require_kz = st.checkbox("Require killzone", value=DEFAULTS["require_killzone"])
        eq_tol = st.slider("Equilibrium tolerance", 0.08, 0.30, DEFAULTS["eq_tolerance"], 0.01)
        swing_lb = st.slider("Swing lookback", 2, 6, DEFAULTS["swing_lookback"])

    with st.sidebar.expander("Risk", expanded=True):
        account = st.number_input("Account size $", value=DEFAULTS["account_size"], step=1000.0)
        risk_dollars = st.number_input("Risk $ per trade", value=DEFAULTS["risk_dollars"], step=25.0)
        max_daily = st.number_input("Max daily loss %", value=DEFAULTS["max_daily_loss_pct"], step=0.5)
        min_rr = st.number_input("Min R:R to T1", value=DEFAULTS["min_rr"], step=0.1)
        sl_buf = st.number_input("SL buffer (pts)", value=DEFAULTS["sl_buffer_pts"], step=0.5)

    cfg = {
        "require_killzone": require_kz,
        "eq_tolerance": eq_tol,
        "swing_lookback": swing_lb,
        "account_size": account,
        "risk_dollars": risk_dollars,
        "max_daily_loss_pct": max_daily,
        "min_rr": min_rr,
        "sl_buffer_pts": sl_buf,
    }

    run = st.sidebar.button("🔄 Run analysis", type="primary", use_container_width=True)
    auto = st.sidebar.checkbox("Auto-refresh 60s", value=False)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Risk model**  \n"
        f"${risk_dollars:.0f} per trade  \n"
        f"Max daily ≈ ${account * max_daily / 100:,.0f}"
    )

    if "last_run" not in st.session_state:
        st.session_state["last_run"] = None
        st.session_state["results"] = {}

    if run or auto or st.session_state["last_run"] is None:
        results = {}
        data_ok = True
        with st.spinner("Scanning markets…"):
            for inst in INSTRUMENTS:
                sym = inst["symbol"]
                disp = inst["display"]
                df5 = fetch_bars(sym, "5m", 400)
                daily = fetch_daily(sym if sym != "MNQ=F" else "NQ=F")
                if df5.empty:
                    data_ok = False
                sig = build_signal(disp, df5, daily, cfg, data_ok and not df5.empty)
                risk = calc_risk(sig, inst["point_value"], cfg)
                results[disp] = {"signal": sig, "risk": risk}
        st.session_state["results"] = results
        st.session_state["last_run"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        st.session_state["data_ok"] = data_ok

    results = st.session_state.get("results", {})
    data_ok = st.session_state.get("data_ok", False)
    last = st.session_state.get("last_run", "–")

    if not data_ok:
        st.error("🔴 **DATA DISCONNECTED** — signals suppressed")
    else:
        st.success(f"🟢 Data connected  ·  Last scan: {last}")

    st.markdown("### Live prices")
    pcols = st.columns(3)
    for i, inst in enumerate(INSTRUMENTS):
        disp = inst["display"]
        sig: Signal = results.get(disp, {}).get("signal", Signal(disp))
        with pcols[i]:
            price_str = f"{sig.price:,.2f}" if sig.price else "—"
            bias_emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}.get(sig.bias, "⚪")
            kz = "Killzone" if sig.in_killzone else "Outside"
            st.metric(disp, price_str, f"{bias_emoji} {sig.bias} · {kz}")

    st.markdown("---")
    st.markdown("### Signals")

    scols = st.columns(3)
    for i, inst in enumerate(INSTRUMENTS):
        disp = inst["display"]
        r = results.get(disp, {})
        sig: Signal = r.get("signal", Signal(disp))
        risk = r.get("risk", {})
        with scols[i]:
            if sig.direction == "BUY":
                st.markdown(f'<div class="signal-buy">{disp} → BUY</div>', unsafe_allow_html=True)
            elif sig.direction == "SELL":
                st.markdown(f'<div class="signal-sell">{disp} → SELL</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="signal-none">{disp} → NO TRADE</div>', unsafe_allow_html=True)

            st.write(f"**Score** {sig.score:.0f} · {sig.confidence}")

            if sig.entry and sig.stop:
                st.write(f"Entry `{sig.entry:.2f}`")
                st.write(f"Stop `{sig.stop:.2f}`")
                st.write(f"T1 `{sig.t1:.2f}` · T2 `{sig.t2:.2f}` · T3 `{sig.t3:.2f}`")
                st.write(f"R:R **{sig.rr}** · Size **{risk.get('contracts', 0)}** ctr")
                st.caption(f"Risk ${risk.get('risk_amount', 0):.0f}")

            with st.expander("Checklist"):
                for p in sig.present:
                    st.markdown(f'<span class="step-ok">✓ {p}</span>', unsafe_allow_html=True)
                for m in sig.missing:
                    st.markdown(f'<span class="step-miss">✗ {m}</span>', unsafe_allow_html=True)
                st.markdown("---")
                for reason in sig.reasons:
                    st.caption("• " + reason)

    high = any(
        r.get("signal", Signal("")).direction in ("BUY", "SELL")
        and r.get("signal", Signal("")).score >= 80
        for r in results.values()
    )
    if high:
        st.warning("🚨 **High-quality setup active** — review the signal panel")

    st.markdown("---")
    st.markdown("""
**Spxder consistency rules**  
1. Higher-timeframe bias must be clear  
2. Only trade inside killzone (London / NY AM)  
3. Liquidity sweep required  
4. Entry only in mean-reversion / equilibrium zone  
5. All steps must pass → otherwise **NO TRADE**

Risk defaults to **$250** per trade. No orders are sent to any broker.  
Data is delayed (yfinance).
    """)

    if auto:
        import time as tm
        tm.sleep(60)
        st.rerun()


if __name__ == "__main__":
    main()
