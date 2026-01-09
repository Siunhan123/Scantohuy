import os
import json
import time
import random
import traceback
from typing import Dict, Any, Tuple, Optional, List

import requests
import pandas as pd
from tradingview_screener import Query, Column as C

STATE_FILE = "state.json"

# ====== SETTINGS ======
RSI_LEN = 14
EMA_LEN = 9

# Quét mỗi 1 giờ => giữ danh sách nhỏ để tránh 429
MAX_CANDIDATES_US = 40
MAX_CANDIDATES_VN = 40

# Nghỉ giữa request Yahoo để tránh rate-limit
SLEEP_BETWEEN_SYMBOLS = 0.5  # seconds

# Multi-timeframe & cooldown
TIMEFRAMES = [
    {"tf": "1h", "cooldown_hours": 6},
    {"tf": "4h", "cooldown_hours": 12},
    {"tf": "1d", "cooldown_hours": 48},
]

# ====== HTTP SESSION ======
SESSION = requests.Session()
SESSION.headers.update({
    "user-agent": "Mozilla/5.0",
    "accept": "application/json,text/plain,*/*",
})


# ---------------------------
# State handling (robust)
# ---------------------------
def load_state() -> Dict[str, Any]:
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if not text:
                return {}
            return json.loads(text)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(state: Dict[str, Any]) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ---------------------------
# Telegram
# ---------------------------
def send_telegram(text: str) -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = SESSION.post(url, json={
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": False
    }, timeout=30)
    # Nếu Telegram lỗi, đừng làm chết cả job
    if r.status_code >= 400:
        print("Telegram error:", r.status_code, r.text[:200])


# ---------------------------
# TradingView candidates (self-filter)
# ---------------------------
def get_candidates_tradingview() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Tự lọc vòng 1 để danh sách nhỏ + thanh khoản tốt.
    - US: NASDAQ/NYSE/AMEX, price > 2, volume > 1M, sort by volume desc
    - VN: HOSE/HNX/UPCOM, close > 1000 (VND), volume > 300k, sort by volume desc
    Có fallback nới điều kiện nếu ra quá ít mã.
    """
    # US strict
    us_df = (Query()
             .set_markets("america")
             .select("name", "exchange", "close", "volume")
             .where(C("exchange").isin(["NASDAQ", "NYSE", "AMEX"]))
             .where(C("close") > 2)
             .where(C("volume") > 1_000_000)
             .order_by("volume", ascending=False)
             .limit(MAX_CANDIDATES_US)
             .get_scanner_data()[1])

    # US fallback
    if us_df is None or len(us_df) < 10:
        us_df = (Query()
                 .set_markets("america")
                 .select("name", "exchange", "close", "volume")
                 .where(C("exchange").isin(["NASDAQ", "NYSE", "AMEX"]))
                 .where(C("close") > 1)
                 .where(C("volume") > 500_000)
                 .order_by("volume", ascending=False)
                 .limit(MAX_CANDIDATES_US)
                 .get_scanner_data()[1])

    # VN strict
    vn_df = (Query()
             .set_markets("global")
             .select("name", "exchange", "close", "volume")
             .where(C("exchange").isin(["HOSE", "HNX", "UPCOM"]))
             .where(C("close") > 1000)
             .where(C("volume") > 300_000)
             .order_by("volume", ascending=False)
             .limit(MAX_CANDIDATES_VN)
             .get_scanner_data()[1])

    # VN fallback
    if vn_df is None or len(vn_df) < 10:
        vn_df = (Query()
                 .set_markets("global")
                 .select("name", "exchange", "close", "volume")
                 .where(C("exchange").isin(["HOSE", "HNX", "UPCOM"]))
                 .where(C("close") > 500)
                 .where(C("volume") > 100_000)
                 .order_by("volume", ascending=False)
                 .limit(MAX_CANDIDATES_VN)
                 .get_scanner_data()[1])

    us = us_df.to_dict("records") if us_df is not None else []
    vn = vn_df.to_dict("records") if vn_df is not None else []
    return us, vn


def tv_to_yahoo_symbol(exchange: str, ticker: str) -> str:
    """
    TradingView trả ticker. Yahoo:
    - US: AAPL, TSLA...
    - VN: thường dùng .VN (HPG.VN, VNM.VN...)
    """
    if exchange in ["HOSE", "HNX", "UPCOM"]:
        return f"{ticker}.VN"
    return ticker


# ---------------------------
# Yahoo Finance chart (with retry/backoff + skip on error)
# ---------------------------
def yahoo_h1(symbol: str, range_: str = "60d", max_retries: int = 5) -> Optional[Tuple[pd.DataFrame, str]]:
    """
    Fetch 1h close series from Yahoo chart endpoint.
    - Handles 429 with exponential backoff
    - Returns None if failed (never raises)
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

    for attempt in range(max_retries):
        try:
            r = SESSION.get(url, params={"range": range_, "interval": "1h"}, timeout=30)

            if r.status_code == 429:
                # backoff
                wait = (2 ** attempt) + random.uniform(0.0, 1.0)
                time.sleep(wait)
                continue

            if r.status_code >= 400:
                return None

            j = r.json()
            if not j.get("chart") or not j["chart"].get("result"):
                return None

            res = j["chart"]["result"][0]
            meta = res.get("meta", {})
            tz = meta.get("timezone", "UTC")

            ts = res.get("timestamp", [])
            quote = res["indicators"]["quote"][0]
            close = quote.get("close", [])

            data = [(t, c) for t, c in zip(ts, close) if c is not None]
            if len(data) < 60:  # đủ để tính RSI/EMA multi tf
                return None

            df = pd.DataFrame(data, columns=["ts", "close"])
            df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(tz)
            df = df.set_index("dt").sort_index()
            return df[["close"]], tz

        except Exception:
            return None

    return None


# ---------------------------
# Indicators
# ---------------------------
def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def ema(series: pd.Series, length: int = 9) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def resample_close(df_h1: pd.DataFrame, tf: str) -> pd.Series:
    """
    Resample from 1H to 4H / 1D using last close.
    Note: Với stock có session break, resample vẫn dùng last của bucket.
    """
    s = df_h1["close"].dropna()
    if tf == "1h":
        return s
    if tf == "4h":
        return s.resample("4H").last().dropna()
    if tf == "1d":
        return s.resample("1D").last().dropna()
    raise ValueError("Unsupported timeframe")


def compute_cross(close: pd.Series) -> Optional[Dict[str, Any]]:
    """
    Check RSI cross up EMA(RSI) on the last CLOSED bar.
    Use [-3] and [-2] to avoid current forming bar.
    """
    if len(close) < (RSI_LEN + EMA_LEN + 10):
        return None

    r = rsi(close, RSI_LEN)
    e = ema(r, EMA_LEN)

    if len(r.dropna()) < 5 or len(e.dropna()) < 5:
        return None

    # last closed bar is -2
    r_prev, r_now = r.iloc[-3], r.iloc[-2]
    e_prev, e_now = e.iloc[-3], e.iloc[-2]
    if pd.isna(r_prev) or pd.isna(r_now) or pd.isna(e_prev) or pd.isna(e_now):
        return None

    crossed_up = (r_prev <= e_prev) and (r_now > e_now)
    return {
        "crossed_up": bool(crossed_up),
        "rsi": float(r_now),
        "ema": float(e_now),
        "bar_time": str(close.index[-2]),
    }


# ---------------------------
# Dedup / Cooldown / New-bar gating
# ---------------------------
def should_process_new_bar(state: Dict[str, Any], key: str, tf: str, bar_time: str) -> bool:
    last_bar = state.setdefault("last_bar", {}).setdefault(key, {})
    if last_bar.get(tf) == bar_time:
        return False
    last_bar[tf] = bar_time
    return True


def can_alert(state: Dict[str, Any], key: str, tf: str, cooldown_hours: int, now_ts: int) -> bool:
    last_alert = state.setdefault("last_alert", {}).setdefault(key, {})
    last = int(last_alert.get(tf, 0) or 0)
    if now_ts - last < cooldown_hours * 3600:
        return False
    last_alert[tf] = now_ts
    return True


# ---------------------------
# Main
# ---------------------------
def main():
    state = load_state()
    now_ts = int(time.time())

    try:
        us, vn = get_candidates_tradingview()
    except Exception:
        print("TradingView query failed:")
        print(traceback.format_exc())
        # vẫn lưu state để tránh crash loop
        save_state(state)
        return

    rows = [("US", r) for r in us] + [("VN", r) for r in vn]
    print(f"Candidates: US={len(us)} VN={len(vn)} total={len(rows)}")

    hits = 0
    processed = 0

    for market, r in rows:
        ticker = r.get("name")
        exchange = r.get("exchange", "")
        if not ticker:
            continue

        yahoo_sym = tv_to_yahoo_symbol(exchange, ticker)
        key = f"{market}:{exchange}:{ticker}"

        # throttle
        time.sleep(SLEEP_BETWEEN_SYMBOLS)

        data = yahoo_h1(yahoo_sym)
        if data is None:
            continue

        df_h1, _tz = data
        processed += 1

        tf_msgs = []
        for tf_cfg in TIMEFRAMES:
            tf = tf_cfg["tf"]
            close_tf = resample_close(df_h1, tf)
            out = compute_cross(close_tf)
            if not out:
                continue

            # chỉ xét khi có bar mới
            if not should_process_new_bar(state, key, tf, out["bar_time"]):
                continue

            if out["crossed_up"] and can_alert(state, key, tf, tf_cfg["cooldown_hours"], now_ts):
                tf_msgs.append(
                    f"- {tf.upper()}: RSI {out['rsi']:.2f} cắt lên EMA {out['ema']:.2f} (bar {out['bar_time']})"
                )

        if tf_msgs:
            hits += 1
            msg = (
                f"📈 RSI Cross UP (multi-TF)\n"
                f"{exchange}:{ticker} ({market})\n"
                + "\n".join(tf_msgs) + "\n"
                f"https://finance.yahoo.com/quote/{yahoo_sym}"
            )
            send_telegram(msg)

    save_state(state)
    print(f"Done. processed={processed} signals={hits}")


if __name__ == "__main__":
    main()
