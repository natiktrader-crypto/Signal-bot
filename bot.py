"""
Телеграм-бот сигналов v2: Binance + Deriv.
Индикаторы: EMA-тренд, MACD, RSI, Bollinger, уровни Фибоначчи,
свечные паттерны, старший таймфрейм + проверка правил на истории.

pip install python-telegram-bot pandas requests websockets
Команды:
  /signal binance BTCUSDT        (по умолчанию 5m)
  /signal deriv frxEURUSD 15m
"""
import asyncio
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pandas as pd
import requests
import websockets
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

TOKEN = os.getenv("TG_TOKEN")
DERIV_WS_LIST = [
    "wss://api.derivws.com/trading/v1/options/ws/public",  # новый публичный
    "wss://ws.derivws.com/websockets/v3?app_id=1089",      # старый
]
TF_SECONDS = {"1m": 60, "2m": 120, "3m": 180, "5m": 300,
              "15m": 900, "1h": 3600, "4h": 14400}
HTF = {"1m": "5m", "5m": "15m", "15m": "1h", "1h": "4h"}  # старший таймфрейм
BREAKEVEN = 52.6  # % верных сделок для нуля при выплате ~90%


# ---------- данные ----------
def binance_candles(symbol: str, tf: str, limit: int = 500) -> pd.DataFrame:
    last_err = None
    for base in ("https://data-api.binance.vision", "https://api.binance.com"):
        try:
            r = requests.get(
                f"{base}/api/v3/klines",
                params={"symbol": symbol.upper(), "interval": tf, "limit": limit},
                timeout=10,
            )
            r.raise_for_status()
            break
        except Exception as e:
            last_err = e
    else:
        raise last_err
    df = pd.DataFrame(r.json()).iloc[:, :6]
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    return df.astype({"open": float, "high": float, "low": float, "close": float})


async def deriv_candles(symbol: str, tf: str, count: int = 500) -> pd.DataFrame:
    req = {
        "ticks_history": symbol,
        "style": "candles",
        "granularity": TF_SECONDS[tf],
        "count": count,
        "end": "latest",
    }
    data, last_err = None, None
    for url in DERIV_WS_LIST:
        try:
            async with websockets.connect(url, open_timeout=10) as ws:
                await ws.send(json.dumps(req))
                data = json.loads(await asyncio.wait_for(ws.recv(), 15))
            if "error" in data:
                raise ValueError(data["error"].get("message", "ошибка Deriv"))
            if "candles" not in data:
                raise ValueError(f"Нет свечей в ответе: {str(data)[:150]}")
            break
        except Exception as e:
            last_err = f"{url.split('/')[2]}: {e}"
            data = None
    if data is None:
        raise ValueError(last_err)
    df = pd.DataFrame(data["candles"])
    return df.astype({"open": float, "high": float, "low": float, "close": float})


async def fetch(source: str, symbol: str, tf: str, limit: int = 500) -> pd.DataFrame:
    if source == "binance":
        return await asyncio.to_thread(binance_candles, symbol, tf, limit)
    if source == "deriv":
        return await deriv_candles(symbol, tf, limit)
    raise ValueError("Источник: binance или deriv")


# ---------- индикаторы ----------
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]
    df["ema20"] = c.ewm(span=20, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["macd_h"] = macd - macd.ewm(span=9, adjust=False).mean()
    df["rsi"] = rsi(c)
    ma, sd = c.rolling(20).mean(), c.rolling(20).std()
    df["bb_up"], df["bb_lo"] = ma + 2 * sd, ma - 2 * sd
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    return df


def candle_pattern(df: pd.DataFrame, i: int):
    o, h, l, c = (df[k].iloc[i] for k in ("open", "high", "low", "close"))
    po, pc = df["open"].iloc[i - 1], df["close"].iloc[i - 1]
    body, rng = abs(c - o), h - l
    if rng == 0:
        return 0, ""
    up, lo = h - max(o, c), min(o, c) - l
    if pc < po and c > o and c >= po and o <= pc:
        return 1, "бычье поглощение"
    if pc > po and c < o and c <= po and o >= pc:
        return -1, "медвежье поглощение"
    if body / rng < 0.4 and lo >= 2 * body and up <= body:
        return 1, "молот"
    if body / rng < 0.4 and up >= 2 * body and lo <= body:
        return -1, "падающая звезда"
    return 0, ""


def fib_signal(df: pd.DataFrame, i: int, lookback: int = 100):
    w = df.iloc[max(0, i - lookback + 1): i + 1]
    hi, lo = w["high"].max(), w["low"].min()
    atr = df["atr"].iloc[i]
    if hi == lo or pd.isna(atr):
        return 0, ""
    up_swing = w["high"].values.argmax() > w["low"].values.argmin()
    c = df["close"].iloc[i]
    for r in (0.382, 0.5, 0.618):
        if up_swing:      # рост, откат вниз к уровню -> отскок вверх
            level, side = hi - r * (hi - lo), 1
        else:             # падение, откат вверх к уровню -> отскок вниз
            level, side = lo + r * (hi - lo), -1
        if abs(c - level) <= 0.5 * atr:
            return side, f"Фибо {r}"
    return 0, ""


def score_at(df: pd.DataFrame, i: int):
    r, p = df.iloc[i], df.iloc[i - 1]
    parts = {}
    parts["Тренд"] = 1 if r["ema20"] > r["ema50"] else -1
    if r["macd_h"] > 0 and r["macd_h"] > p["macd_h"]:
        parts["MACD"] = 1
    elif r["macd_h"] < 0 and r["macd_h"] < p["macd_h"]:
        parts["MACD"] = -1
    else:
        parts["MACD"] = 0
    parts["RSI"] = 1 if r["rsi"] < 35 else -1 if r["rsi"] > 65 else 0
    parts["Bollinger"] = 1 if r["close"] < r["bb_lo"] else -1 if r["close"] > r["bb_up"] else 0
    pat, pname = candle_pattern(df, i)
    parts["Свеча"] = pat
    fib, fname = fib_signal(df, i)
    parts["Фибо"] = fib
    return sum(parts.values()), parts, pname, fname


def analyze(df: pd.DataFrame, htf: pd.DataFrame = None) -> dict:
    df = add_indicators(df)
    i = len(df) - 1
    s, parts, pname, fname = score_at(df, i)

    htf_trend = 0
    if htf is not None and len(htf) > 55:
        h = add_indicators(htf)
        htf_trend = 1 if h["ema20"].iloc[-1] > h["ema50"].iloc[-1] else -1
        s += htf_trend

    # проверка тех же правил на истории (без старшего ТФ)
    n = wins = 0
    for j in range(60, len(df) - 1):
        sj = score_at(df, j)[0]
        if abs(sj) >= 3:
            n += 1
            if (df["close"].iloc[j + 1] - df["close"].iloc[j]) * sj > 0:
                wins += 1

    if s >= 3:
        side = "⬆️ ПОКУПКА (Rise)"
    elif s <= -3:
        side = "⬇️ ПРОДАЖА (Fall)"
    else:
        side = "⏸ НЕТ СИГНАЛА"
    strength = "Сильный" if abs(s) >= 5 else "Средний" if abs(s) == 4 else "Слабый"
    return {
        "side": side, "strength": strength, "score": s, "parts": parts,
        "pname": pname, "fname": fname, "htf_trend": htf_trend,
        "price": df["close"].iloc[-1], "rsi": df["rsi"].iloc[-1],
        "bt_n": n, "bt_win": (wins / n * 100) if n else 0.0,
    }


def fmt(v: int) -> str:
    return f"+{v}" if v > 0 else str(v)


# ---------- бот ----------
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Команды:\n/signal binance BTCUSDT\n/signal deriv frxEURUSD\n"
        "Таймфрейм по умолчанию 5m. Можно: /signal deriv frxEURUSD 15m"
    )


async def signal(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        source, symbol = ctx.args[0].lower(), ctx.args[1]
        tf = ctx.args[2] if len(ctx.args) > 2 else "5m"
        if tf not in TF_SECONDS:
            raise ValueError(f"Таймфрейм: {', '.join(TF_SECONDS)}")
        df = await fetch(source, symbol, tf, 500)
        df = df.iloc[:-1].reset_index(drop=True)  # без незакрытой свечи
        if len(df) < 80:
            raise ValueError("Слишком мало данных по этой паре")
        htf_tf, htf = HTF.get(tf), None
        if htf_tf:
            try:
                htf = (await fetch(source, symbol, htf_tf, 200)).iloc[:-1]
            except Exception:
                htf = None
        a = await asyncio.to_thread(analyze, df, htf)
    except IndexError:
        await update.message.reply_text("Пример: /signal binance BTCUSDT")
        return
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")
        return

    p = a["parts"]
    candle = f"{fmt(p['Свеча'])}" + (f" ({a['pname']})" if a["pname"] else "")
    fibo = f"{fmt(p['Фибо'])}" + (f" ({a['fname']})" if a["fname"] else "")
    htf_line = f"Старший ТФ {htf_tf}: {fmt(a['htf_trend'])}\n" if htf_tf and a["htf_trend"] else ""
    if a["bt_n"]:
        warn = " (мало данных)" if a["bt_n"] < 30 else ""
        verdict = "выше" if a["bt_win"] > BREAKEVEN else "НИЖЕ"
        bt = (f"История: {a['bt_n']} сигналов, верных {a['bt_win']:.0f}%{warn}\n"
              f"Для нуля нужно ≥{BREAKEVEN}% — результат {verdict} порога")
    else:
        bt = "История: сигналов по этим правилам не было"

    await update.message.reply_text(
        f"{symbol.upper()} · {tf} · {source}\n"
        f"Сигнал: {a['side']} ({a['strength']}, счёт {fmt(a['score'])})\n"
        f"Цена: {a['price']:.5f} · RSI {a['rsi']:.0f}\n\n"
        f"Тренд {fmt(p['Тренд'])} · MACD {fmt(p['MACD'])} · RSI {fmt(p['RSI'])}\n"
        f"Bollinger {fmt(p['Bollinger'])} · Свеча {candle} · Фибо {fibo}\n"
        f"{htf_line}\n{bt}\n\n"
        "⚠️ Не финансовый совет. Гарантий прибыли нет."
    )


class _Ping(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        pass


def run_web():
    port = int(os.getenv("PORT", 10000))
    HTTPServer(("0.0.0.0", port), _Ping).serve_forever()


def main():
    threading.Thread(target=run_web, daemon=True).start()
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signal", signal))
    app.run_polling()


if __name__ == "__main__":
    main()
