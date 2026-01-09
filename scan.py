import os, json, time
import requests
import pandas as pd
from tradingview_screener import Query, Column as C

STATE_FILE = "state.json"

# ====== CONFIG ======
TIMEFRAMES = [
    {"tf": "1h", "cooldown_hours": 6},
    {"tf": "4h", "cooldown_hours": 12},
    {"tf": "1d", "cooldown_hours": 48},
]
RSI_LEN = 14
EMA_LEN = 9

MAX_CANDIDATES_US = 200
MAX_CANDIDATES_VN = 200

# Bạn chỉnh filter vòng 1 tại đây
def get_candidates_tradingview():
    # US (America market)
    us_df = (Query()
             .set_markets("america")
             .select("name", "exchange", "close", "volume")
             .where(C("close") > 1)
             .where(C("volume") > 500_000)
             .order_by("volume", ascending=False)
             .limit(MAX_CANDIDATES_US)
             .get_scanner_data()[1])

    # VN (Global market + lọc exchange)
    vn_df = (Query()
             .set_markets("global")
             .select("name", "exchange", "close", "volume")
             .where(C("exchange").isin(["HOSE", "HNX", "UPCOM"]))
             .where(C("close") > 1)
             .where(C("volume") > 100_000)
             .order_by("volume", ascending=False)
             .limit(MAX_CANDIDATES_VN)
             .get_scanner_data()[1])

    us = us_df.to_dict("records") if us_df is not None else []
    vn = vn_df.to_dict("records") if vn_df is not None else []
    return us, vn

def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def send_telegram(text: str):
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(url, json={
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": False
    }, timeout=20)
    r.raise_for_status()

def tv_to_yahoo_symbol(exchange: str, ticker: str):
    # US: AAPL, TSLA...
    # VN: thường là HPG.VN, VNM.VN...
    if exchange in ["HOSE", "HNX", "UPCOM"]:
        return f"{ticker}.VN"
    return ticker

def yahoo_h1(symbol: str, range_="60d"):
    # Yahoo chart endpoint (free)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    r = requests.get(url, params={"range": range_, "interval": "1h"}, timeout=20)
    r.raise_for_status()
    j = r.json()
    res = j["chart"]["result"][0]
    meta = res.get("meta", {})
    tz = meta.get("timezone", "UTC")

    ts = res.get("timestamp", [])
    quote = res["indicators"]["quote"][0]
    close = quote.get("close", [])

    data = [(t, c) for t, c in zip(ts, close) if c is not None]
    if len(data) < 50:
        return None

    df = pd.DataFrame(data, columns=["ts", "close"])
    df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(tz)
    df = df.set_index("dt").sort_index()
    return df[["close"]], tz

def rsi(series: pd.Series, length=14):
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))

def ema(series: pd.Series, length=9):
    return series.ewm(span=length, adjust=False).mean()

def resample_close(df_h1: pd.DataFrame, tf: str):
    # df_h1 index đã theo timezone của sàn
    if tf == "1h":
        return df_h1["close"].dropna()

    if tf == "4h":
        # lấy close của cây 4h
        return df_h1["close"].resample("4H").last().dropna()

    if tf == "1d":
        return df_h1["close"].resample("1D").last().dropna()

    raise ValueError("Unsupported tf")

def check_cross(close: pd.Series):
    r = rsi(close, RSI_LEN)
    e = ema(r, EMA_LEN)

    # Dùng 2 nến đã đóng gần nhất => lấy [-3],[-2] để tránh “nến cuối đang hình thành”
    if len(r.dropna()) < 5 or len(e.dropna()) < 5:
        return None

    r_prev, r_now = r.iloc[-3], r.iloc[-2]
    e_prev, e_now = e.iloc[-3], e.iloc[-2]

    if pd.isna(r_prev) or pd.isna(r_now) or pd.isna(e_prev) or pd.isna(e_now):
        return None

    crossed_up = (r_prev <= e_prev) and (r_now > e_now)
    return {
        "crossed_up": bool(crossed_up),
        "rsi": float(r_now),
        "ema": float(e_now),
        "bar_time": str(close.index[-2])  # thời điểm nến đã đóng
    }

def should_process_new_bar(state, key, tf, bar_time_str):
    last_bar = state.setdefault("last_bar", {}).setdefault(key, {})
    if last_bar.get(tf) == bar_time_str:
        return False
    last_bar[tf] = bar_time_str
    return True

def can_alert(state, key, tf, cooldown_hours, now_ts):
    last_alert = state.setdefault("last_alert", {}).setdefault(key, {})
    last = last_alert.get(tf, 0)
    if now_ts - last < cooldown_hours * 3600:
        return False
    last_alert[tf] = now_ts
    return True

def main():
    state = load_state()
    now_ts = int(time.time())

    us, vn = get_candidates_tradingview()
    rows = [("US", r) for r in us] + [("VN", r) for r in vn]

    hits = 0
    for market, r in rows:
        ticker = r.get("name")
        exchange = r.get("exchange", "")
        if not ticker:
            continue

        yahoo_sym = tv_to_yahoo_symbol(exchange, ticker)
        data = yahoo_h1(yahoo_sym)
        if data is None:
            continue

        df_h1, _tz = data
        key = f"{market}:{exchange}:{ticker}"

        # Check từng timeframe
        tf_msgs = []
        for tf_cfg in TIMEFRAMES:
            tf = tf_cfg["tf"]
            close_tf = resample_close(df_h1, tf)
            out = check_cross(close_tf)
            if not out:
                continue

            # chỉ xử lý khi có nến mới
            if not should_process_new_bar(state, key, tf, out["bar_time"]):
                continue

            if out["crossed_up"]:
                if can_alert(state, key, tf, tf_cfg["cooldown_hours"], now_ts):
                    tf_msgs.append(
                        f"- {tf.upper()}: RSI {out['rsi']:.2f} cắt lên EMA {out['ema']:.2f} (bar {out['bar_time']})"
                    )

        if tf_msgs:
            hits += 1
            msg = (
                f"📈 RSI Cross UP (multi-TF)\n"
                f"{exchange}:{ticker}  ({market})\n"
                + "\n".join(tf_msgs) + "\n"
                f"https://finance.yahoo.com/quote/{yahoo_sym}"
            )
            send_telegram(msg)

    save_state(state)
    print("Done. Signals:", hits)

if __name__ == "__main__":
    main()
