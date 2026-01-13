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
EMA_LEN = 8

MAX_CANDIDATES_US = 40
MAX_CANDIDATES_VN = 40

SLEEP_BETWEEN_SYMBOLS = 0.5  # seconds

# Timeframes scanned
TIMEFRAMES = [
    {"tf": "1h", "cooldown_hours": 1},
    {"tf": "4h", "cooldown_hours": 4},
    {"tf": "1d", "cooldown_hours": 24},
]

# Range (more history => RSI/EMA match better)
RANGE_1H = "60d"  # You can set "730d" if you want max warmup (heavier)
RANGE_1D = "3y"

# Match TradingView "Extended hours" toggle:
# - If you use TradingView with Extended Hours OFF for stocks, set False.
# - If ON, set True (may change results).
INCLUDE_PREPOST = False

# Báo cáo mỗi lần chạy: gửi top N mã (tránh quá dài)
REPORT_TOP_N_US = 60
REPORT_TOP_N_VN = 60

SESSION = requests.Session()
SESSION.headers.update({
    "user-agent": "Mozilla/5.0",
    "accept": "application/json,text/plain,*/*",
})

# ====== EXTRA MANUAL SYMBOLS (always check) ======
# You asked: OANDA:XAUUSD and BITSTAMP:BTCUSDT
# We still fetch candles from Yahoo, so we map to Yahoo symbols with fallback list.
EXTRA_SYMBOLS = [
    {
        "market": "EXTRA",
        "exchange": "OANDA",
        "name": "XAUUSD",
        "label": "XAUUSD (OANDA)",
        "yahoo_try": ["XAUUSD=X", "GC=F"],  # spot then futures fallback
    },
    {
        "market": "EXTRA",
        "exchange": "BITSTAMP",
        "name": "BTCUSDT",
        "label": "BTCUSDT (Bitstamp)",
        "yahoo_try": ["BTC-USDT", "BTC-USD"],  # try USDT pair, fallback to BTC-USD
    },
]

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
             .set_markets("vietnam")
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


# ---------------------------
# Yahoo symbol mapping
# ---------------------------
def tv_to_yahoo_symbol(exchange: str, ticker: str) -> str:
    # Stocks VN
    if exchange in ["HOSE", "HNX", "UPCOM"]:
        return f"{ticker}.VN"
    # Default US stock
    return ticker


def resolve_yahoo_try_list(market: str, exchange: str, ticker: str) -> List[str]:
    """
    Return a list of Yahoo symbols to try (fallback supported).
    """
    # Manual extras
    if market == "EXTRA":
        for it in EXTRA_SYMBOLS:
            if it["exchange"] == exchange and it["name"] == ticker:
                return it.get("yahoo_try", [])

    # Normal stocks: single yahoo symbol
    return [tv_to_yahoo_symbol(exchange, ticker)]


# ---------------------------
# Yahoo Finance chart (retry/backoff + numeric clean)
# ---------------------------
def yahoo_chart(symbol: str, interval: str, range_: str, include_prepost: bool, max_retries: int = 5) -> Optional[Tuple[pd.DataFrame, str]]:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

    for attempt in range(max_retries):
        try:
            r = SESSION.get(
                url,
                params={
                    "range": range_,
                    "interval": interval,
                    "includePrePost": "true" if include_prepost else "false",
                },
                timeout=30
            )

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
            quote = res.get("indicators", {}).get("quote", [{}])[0]
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


def yahoo_fetch_with_fallback(symbols_to_try: List[str], interval: str, range_: str, include_prepost: bool) -> Optional[Tuple[pd.DataFrame, str, str]]:
    """
    Try multiple Yahoo tickers until one works.
    Returns (df, tz, used_symbol)
    """
    for sym in symbols_to_try:
        out = yahoo_chart(sym, interval=interval, range_=range_, include_prepost=include_prepost)
        if out is not None:
            df, tz = out
            return df, tz, sym
    return None


# ---------------------------
# Indicators (closer to TradingView)
# ---------------------------
def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """
    Wilder RSI: close to TradingView ta.rsi.
    Handle extremes (avgLoss=0 => RSI=100, avgGain=0 => RSI=0) to avoid NaN drift.
    """
    s = pd.to_numeric(series, errors="coerce").astype(float)
    delta = s.diff()

    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)

    avg_gain = up.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = down.ewm(alpha=1/length, adjust=False).mean()

    rs = avg_gain / avg_loss
    out = 100.0 - (100.0 / (1.0 + rs))

    # match TV behavior on extremes
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(avg_gain != 0, 0.0)
    return out


def ema(series: pd.Series, length: int = 9) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    return s.ewm(span=length, adjust=False).mean()


# ---------------------------
# Timeframe helpers
# ---------------------------
def resample_4h_from_1h(close_1h: pd.Series) -> pd.Series:
    s = pd.to_numeric(close_1h, errors="coerce").astype(float).dropna()
    # "right/right" makes close at the bar end more consistent
    return s.resample("4H", label="right", closed="right").last().dropna().astype(float)


def compute_cross(close: pd.Series) -> Optional[Dict[str, Any]]:
    close = pd.to_numeric(close, errors="coerce").astype(float).dropna()

    # warmup: need more than just 14/9 to converge
    min_len = max(200, (RSI_LEN + EMA_LEN + 50))
    if len(close) < min_len:
        return None

    r = rsi(close, RSI_LEN)
    e = ema(r, EMA_LEN)

    r = r.dropna()
    e = e.dropna()
    if len(r) < 10 or len(e) < 10:
        return None

    # last closed bar = -2 (avoid forming bar)
    if len(close) < 3:
        return None

    r_prev, r_now = r.iloc[-3], r.iloc[-2]
    e_prev, e_now = e.iloc[-3], e.iloc[-2]

    if np.isnan(r_prev) or np.isnan(r_now) or np.isnan(e_prev) or np.isnan(e_now):
        return None

    crossed_up = (r_prev <= e_prev) and (r_now > e_now)

    # bar time taken from close index (-2)
    bar_time = str(close.index[-2])

    return {
        "crossed_up": bool(crossed_up),
        "rsi": float(r_now),
        "ema": float(e_now),
        "bar_time": bar_time,
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
        err = traceback.format_exc()
        send_telegram_chunked("❌ TradingView query failed:\n" + err[:3000])
        save_state(state)
        return

    # Append manual symbols (avoid duplicates)
    existing = set()
    for r in us:
        if r.get("exchange") and r.get("name"):
            existing.add(("US", r["exchange"], r["name"]))
    for r in vn:
        if r.get("exchange") and r.get("name"):
            existing.add(("VN", r["exchange"], r["name"]))

    extras_records = []
    for it in EXTRA_SYMBOLS:
        tup = (it["market"], it["exchange"], it["name"])
        if tup not in existing:
            extras_records.append({"exchange": it["exchange"], "name": it["name"], "close": None, "volume": None, "label": it.get("label")})

    # Build report lists
    us_list = [f"{x.get('exchange','')}:{x.get('name','')}" for x in us if x.get("name")]
    vn_list = [f"{x.get('exchange','')}:{x.get('name','')}" for x in vn if x.get("name")]
    extra_list = [f"{x.get('exchange','')}:{x.get('name','')}" for x in extras_records if x.get("name")]

    rows = [("US", r) for r in us] + [("VN", r) for r in vn] + [("EXTRA", r) for r in extras_records]
    print(f"Candidates: US={len(us)} VN={len(vn)} EXTRA={len(extras_records)} total={len(rows)}")

    hits = 0
    processed = 0

    for market, r in rows:
        ticker = r.get("name")
        exchange = r.get("exchange", "")
        label = r.get("label") or f"{exchange}:{ticker}"

        if not ticker:
            continue

        key = f"{market}:{exchange}:{ticker}"
        time.sleep(SLEEP_BETWEEN_SYMBOLS)

        # Decide needed data per TF:
        need_1h = any(x["tf"] in ("1h", "4h") for x in TIMEFRAMES)
        need_1d = any(x["tf"] == "1d" for x in TIMEFRAMES)

        yahoo_try = resolve_yahoo_try_list(market, exchange, ticker)

        df_1h = None
        used_sym_1h = None
        tz_1h = None

        df_1d = None
        used_sym_1d = None
        tz_1d = None

        # fetch 1H (for 1H + 4H)
        if need_1h:
            out = yahoo_fetch_with_fallback(yahoo_try, interval="1h", range_=RANGE_1H, include_prepost=INCLUDE_PREPOST)
            if out is not None:
                df_1h, tz_1h, used_sym_1h = out

        # fetch 1D (native daily, better match TV daily)
        if need_1d:
            out = yahoo_fetch_with_fallback(yahoo_try, interval="1d", range_=RANGE_1D, include_prepost=INCLUDE_PREPOST)
            if out is not None:
                df_1d, tz_1d, used_sym_1d = out

        if df_1h is None and df_1d is None:
            continue

        processed += 1

        tf_msgs = []
        for tf_cfg in TIMEFRAMES:
            tf = tf_cfg["tf"]

            try:
                if tf == "1h":
                    if df_1h is None:
                        continue
                    close_tf = pd.to_numeric(df_1h["close"], errors="coerce").astype(float).dropna()

                elif tf == "4h":
                    if df_1h is None:
                        continue
                    close_1h = pd.to_numeric(df_1h["close"], errors="coerce").astype(float).dropna()
                    close_tf = resample_4h_from_1h(close_1h)

                elif tf == "1d":
                    if df_1d is None:
                        continue
                    close_tf = pd.to_numeric(df_1d["close"], errors="coerce").astype(float).dropna()

                else:
                    continue

                out = compute_cross(close_tf)
                if not out:
                    continue

                if not should_process_new_bar(state, key, tf, out["bar_time"]):
                    continue

                if out["crossed_up"] and can_alert(state, key, tf, tf_cfg["cooldown_hours"], now_ts):
                    tf_msgs.append(
                        f"- {tf.upper()}: RSI {out['rsi']:.2f} cắt lên EMA {out['ema']:.2f} (bar {out['bar_time']})"
                    )

            except Exception:
                continue

        # Gửi tín hiệu (nếu có)
        if tf_msgs:
            hits += 1

            # Choose a yahoo link if available
            used_link_sym = used_sym_1h or used_sym_1d or (yahoo_try[0] if yahoo_try else "")
            yahoo_link = f"https://finance.yahoo.com/quote/{used_link_sym}" if used_link_sym else ""

            tv_link = f"https://www.tradingview.com/symbols/{ticker}/"  # generic TV symbol page

            msg = (
                f"📈 RSI Cross UP (multi-TF)\n"
                f"{label} ({market})\n"
                + "\n".join(tf_msgs) + "\n"
                + (yahoo_link + "\n" if yahoo_link else "")
                + tv_link
            )
            send_telegram(msg, disable_preview=True)

    # ====== ALWAYS SEND REPORT (kể cả không có tín hiệu) ======
    run_time_utc = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    report_lines = [
        f"🧾 Scan report (hourly) @ {run_time_utc} UTC",
        f"- Candidates: US={len(us_list)} | VN={len(vn_list)} | EXTRA={len(extra_list)}",
        f"- Processed (Yahoo OK): {processed}",
        f"- Signals: {hits}",
        "",
        "🇺🇸 US list (top):",
        ", ".join(us_list[:REPORT_TOP_N_US]) if us_list else "(none)",
        "",
        "🇻🇳 VN list (top):",
        ", ".join(vn_list[:REPORT_TOP_N_VN]) if vn_list else "(none)",
        "",
        "🧩 EXTRA symbols:",
        ", ".join(extra_list) if extra_list else "(none)",
    ]
    send_telegram_chunked("\n".join(report_lines))

    save_state(state)
    print(f"Done. processed={processed} signals={hits}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        print("Fatal error:\n", err)
        try:
            send_telegram_chunked("❌ Fatal error in scan.py:\n" + err[:3000])
        except Exception:
            pass
        try:
            save_state(load_state())
        except Exception:
            pass
