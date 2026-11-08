"""
Телеграм-бот сигналов: Binance + Deriv.
Установка:  pip install python-telegram-bot pandas requests websockets
Запуск:     TG_TOKEN=твой_токен python signal_bot.py

Команды:
  /signal binance BTCUSDT        (по умолчанию 5m)
  /signal deriv frxEURUSD        (по умолчанию 5m)
  /signal deriv frxEURUSD 15m
Символы Deriv: frxEURUSD, frxGBPUSD, cryBTCUSD, R_100 (синтетика) и т.д.
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
TF_SECONDS = {"1m": 60, "2m": 120, "3m": 180, "5m": 300, "15m": 900, "1h": 3600}


# ---------- данные ----------
def binance_candles(symbol: str, tf: str, limit: int = 200) -> pd.DataFrame:
    # api.binance.com блокирует часть серверов (ошибка 451),
    # поэтому сначала пробуем публичное зеркало для рыночных данных
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


async def deriv_candles(symbol: str, tf: str, count: int = 200) -> pd.DataFrame:
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


# ---------- индикаторы ----------
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def analyze(df: pd.DataFrame) -> dict:
    c = df["close"]
    ema20, ema50 = c.ewm(span=20).mean(), c.ewm(span=50).mean()
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    hist = macd - macd.ewm(span=9).mean()
    r = rsi(c)
    atr = (df["high"] - df["low"]).rolling(14).mean()

    trend = 1 if ema20.iloc[-1] > ema50.iloc[-1] else -1
    macd_s = 0
    if hist.iloc[-1] > 0 and hist.iloc[-1] > hist.iloc[-2]:
        macd_s = 1
    elif hist.iloc[-1] < 0 and hist.iloc[-1] < hist.iloc[-2]:
        macd_s = -1
    rsi_s = 1 if r.iloc[-1] < 35 else -1 if r.iloc[-1] > 65 else 0

    score = trend + macd_s + rsi_s
    if score >= 2:
        side = "⬆️ ПОКУПКА (Rise)"
    elif score <= -2:
        side = "⬇️ ПРОДАЖА (Fall)"
    else:
        side = "⏸ НЕТ СИГНАЛА"
    strength = {3: "Сильный", 2: "Средний"}.get(abs(score), "Слабый")

    return {
        "side": side,
        "strength": strength,
        "price": c.iloc[-1],
        "rsi": r.iloc[-1],
        "trend": "восходящий" if trend > 0 else "нисходящий",
        "macd": "бычий" if hist.iloc[-1] > 0 else "медвежий",
        "vol": atr.iloc[-1],
    }


# ---------- бот ----------
async def signal(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        source, symbol = ctx.args[0].lower(), ctx.args[1]
        tf = ctx.args[2] if len(ctx.args) > 2 else "5m"
        if tf not in TF_SECONDS:
            raise ValueError(f"Таймфрейм: {', '.join(TF_SECONDS)}")
        if source == "binance":
            df = await asyncio.to_thread(binance_candles, symbol, tf)
        elif source == "deriv":
            df = await deriv_candles(symbol, tf)
        else:
            raise ValueError("Источник: binance или deriv")
        df = df.iloc[:-1]  # последняя свеча ещё не закрыта, не берём её
        a = analyze(df)
    except IndexError:
        await update.message.reply_text("Пример: /signal binance BTCUSDT 1m")
        return
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")
        return

    await update.message.reply_text(
        f"{symbol.upper()} · {tf} · {source}\n"
        f"Сигнал: {a['side']} ({a['strength']})\n"
        f"Цена: {a['price']:.5f}\n"
        f"Тренд: {a['trend']}\n"
        f"MACD: {a['macd']}\n"
        f"RSI: {a['rsi']:.1f}\n"
        f"Волатильность (ATR): {a['vol']:.5f}\n"
        f"Сигнал по закрытой свече, срок: 1 свеча ({tf})\n\n"
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
    # Render требует открытый порт у Web Service
    port = int(os.getenv("PORT", 10000))
    HTTPServer(("0.0.0.0", port), _Ping).serve_forever()


def main():
    threading.Thread(target=run_web, daemon=True).start()
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("signal", signal))
    app.run_polling()


if __name__ == "__main__":
    main()
