import os
import json
import time
import random
import traceback
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import requests
import pandas as pd
from tradingview_screener import Query, Column as C

STATE_FILE = "state.json"

# ====== SETTINGS ======
RSI_LEN = 14
EMA_LEN = 9

MAX_CANDIDATES_US = 40
MAX_CANDIDATES_VN = 40

SLEEP_BETWEEN_SYMBOLS = 0.5  # seconds

TIMEFRAMES = [
    {"tf": "1h", "cooldown_hours": 6},
    {"tf": "4h", "cooldown_hours": 12},
    {"tf": "1d", "cooldown_hours": 48},
]

# Báo cáo mỗi lần chạy: gửi top N mã (tránh quá dài)
REPORT_TOP_N_US = 60
REPORT_TOP_N_VN = 60

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
def send_telegram(text: str, disable_preview: bool = True) -> None:
    """
    Gửi 1 message Telegram. Không raise để tránh job fail.
    """
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = SESSION.post(url, json={
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": disable_preview
    }, timeout=30)
    if r.status_code >= 400:
        print("Telegram error:", r.status_code, r.text[:300])


def send_telegram_chunked(text: str, chunk_size: int = 3500) -> None:
    """
    Telegram giới hạn ~4096 ký tự/tin, nên chia nhỏ để an toàn.
    """
    if len(text) <= chunk_size:
        send_telegram(text)
        return

    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        send_telegram(text[start:end])
        start = end


# ---------------------------
# TradingView candidates (self-filter)
# ---------------------------
def get_candidates_tradingview() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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
    if exchange in ["HOSE", "HNX", "UPCOM"]:
        return f"{ticker}.VN"
    return ticker


# ---------------------------
# Yahoo Finance chart (retry/backoff + numeric clean)
# ---------------------------
def yahoo_h1(symbol: str, range_: str = "60d", max_retries: int = 5) -> Optional[Tuple[pd.DataFrame, str]]:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

    for attempt in range(max_retries):
        try:
            r = SESSION.get(url, params={"range": range_, "interval": "1h"}, timeout=30)

            if r.status_code == 429:
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
            if len(data) < 60:
                return None

            df = pd.DataFrame(data, columns=["ts", "close"])
            df["close"] = pd.to_numeric(df["close"], errors="coerce").astype(float)
            df = df.dropna(subset=["close"])
            if len(df) < 60:
                return None

            df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(tz)
            df = df.set_index("dt").sort_index()
            return df[["close"]], tz

        except Exception:
            return None

    return None


# ---------------------------
# Indicators (force float)
# ---------------------------
def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    delta = s.diff()

    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)

    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_down = down.ewm(alpha=1/length, adjust=False).mean()

    roll_down = roll_down.replace(0.0, np.nan)

    rs = roll_up / roll_down
    out = 100.0 - (100.0 / (1.0 + rs))
    return out


def ema(series: pd.Series, length: int = 9) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    return s.ewm(span=length, adjust=False).mean()


def resample_close(df_h1: pd.DataFrame, tf: str) -> pd.Series:
    s = pd.to_numeric(df_h1["close"], errors="coerce").astype(float).dropna()
    if tf == "1h":
        return s
    if tf == "4h":
        return s.resample("4H").last().dropna().astype(float)
    if tf == "1d":
        return s.resample("1D").last().dropna().astype(float)
    raise ValueError("Unsupported timeframe")


def compute_cross(close: pd.Series) -> Optional[Dict[str, Any]]:
    close = pd.to_numeric(close, errors="coerce").astype(float).dropna()
    if len(close) < (RSI_LEN + EMA_LEN + 10):
        return None

    r = rsi(close, RSI_LEN)
    e = ema(r, EMA_LEN)

    # last closed bar is -2 (avoid forming bar)
    if len(r.dropna()) < 5 or len(e.dropna()) < 5:
        return None

    r_prev, r_now = r.iloc[-3], r.iloc[-2]
    e_prev, e_now = e.iloc[-3], e.iloc[-2]
    if np.isnan(r_prev) or np.isnan(r_now) or np.isnan(e_prev) or np.isnan(e_now):
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
        # Nếu TradingView fail, vẫn gửi report để bạn biết bot chạy nhưng không lấy được list
        err = traceback.format_exc()
        send_telegram_chunked("❌ TradingView query failed:\n" + err[:3000])
        save_state(state)
        return

    # Danh sách mã lọc ra để report
    us_list = [f"{x.get('exchange','')}:{x.get('name','')}" for x in us if x.get("name")]
    vn_list = [f"{x.get('exchange','')}:{x.get('name','')}" for x in vn if x.get("name")]

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

            if not should_process_new_bar(state, key, tf, out["bar_time"]):
                continue

            if out["crossed_up"] and can_alert(state, key, tf, tf_cfg["cooldown_hours"], now_ts):
                tf_msgs.append(
                    f"- {tf.upper()}: RSI {out['rsi']:.2f} cắt lên EMA {out['ema']:.2f} (bar {out['bar_time']})"
                )

        # Gửi tín hiệu (nếu có)
        if tf_msgs:
            hits += 1
            msg = (
                f"📈 RSI Cross UP (multi-TF)\n"
                f"{exchange}:{ticker} ({market})\n"
                + "\n".join(tf_msgs) + "\n"
                f"https://finance.yahoo.com/quote/{yahoo_sym}"
            )
            send_telegram(msg, disable_preview=True)

    # ====== ALWAYS SEND REPORT (kể cả không có tín hiệu) ======
    run_time_utc = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    report_lines = [
        f"🧾 Scan report (hourly) @ {run_time_utc} UTC",
        f"- Candidates: US={len(us_list)} | VN={len(vn_list)}",
        f"- Processed (Yahoo OK): {processed}",
        f"- Signals: {hits}",
        "",
        "🇺🇸 US list (top):",
        ", ".join(us_list[:REPORT_TOP_N_US]) if us_list else "(none)",
        "",
        "🇻🇳 VN list (top):",
        ", ".join(vn_list[:REPORT_TOP_N_VN]) if vn_list else "(none)",
    ]
    send_telegram_chunked("\n".join(report_lines))

    save_state(state)
    print(f"Done. processed={processed} signals={hits}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # tuyệt đối không làm job chết vì 1 lỗi bất ngờ
        err = traceback.format_exc()
        print("Fatal error:\n", err)
        # vẫn cố gửi 1 tin để bạn biết có lỗi
        try:
            send_telegram_chunked("❌ Fatal error in scan.py:\n" + err[:3000])
        except Exception:
            pass
        try:
            save_state(load_state())
        except Exception:
            pass
