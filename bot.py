"""
Телеграм-бот сигналов v4 (отчёт): Binance + Deriv.
/signal binance BTCUSDT        (по умолчанию 5m)
/signal deriv frxEURUSD 15m

pip install python-telegram-bot pandas requests websockets
Переменная окружения: TG_TOKEN
"""
import asyncio
import html
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional, Tuple

import pandas as pd
import requests
import websockets
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ====================== КОНФИГ ======================
TOKEN = os.getenv("TG_TOKEN")
DERIV_WS_LIST = [
    "wss://api.derivws.com/trading/v1/options/ws/public",
    "wss://ws.derivws.com/websockets/v3?app_id=1089",
]
TF_SECONDS = {"1m": 60, "2m": 120, "3m": 180, "5m": 300,
              "15m": 900, "1h": 3600, "4h": 14400}
HTF_MAP = {"1m": "5m", "5m": "15m", "15m": "1h", "1h": "4h"}

MIN_SCORE = 3.5
BREAKEVEN = 52.6          # % верных для нуля при выплате ~90%
MIN_VOLUME_RATIO = 0.75   # объём >= 75% от средней за 20 свечей
ATR_REL_MIN, ATR_REL_MAX = 0.5, 3.0  # волатильность относительно нормы пары
MIN_HISTORY = 30          # меньше сигналов -> выводам по истории не верим

WEIGHTS = {"trend": 1.6, "macd": 1.3, "rsi": 1.0,
           "bollinger": 0.9, "candle": 1.4, "fib": 1.1}


# ====================== ДАННЫЕ ======================
def binance_candles(symbol: str, tf: str, limit: int = 500) -> pd.DataFrame:
    last_err = None
    for base in ("https://data-api.binance.vision", "https://api.binance.com"):
        try:
            r = requests.get(
                f"{base}/api/v3/klines",
                params={"symbol": symbol.upper(), "interval": tf, "limit": limit},
                timeout=12,
            )
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            last_err = e
    else:
        raise last_err
    df = pd.DataFrame(data).iloc[:, :6]
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    return df.astype({k: float for k in ("open", "high", "low", "close", "volume")})


async def deriv_candles(symbol: str, tf: str, count: int = 500) -> pd.DataFrame:
    req = {"ticks_history": symbol, "style": "candles",
           "granularity": TF_SECONDS[tf], "count": count, "end": "latest"}
    data, last_err = None, None
    for url in DERIV_WS_LIST:
        try:
            async with websockets.connect(url, open_timeout=10) as ws:
                await ws.send(json.dumps(req))
                data = json.loads(await asyncio.wait_for(ws.recv(), 15))
            if "error" in data:
                raise ValueError(data["error"].get("message", "ошибка Deriv"))
            if "candles" not in data:
                raise ValueError(f"Нет свечей: {str(data)[:120]}")
            break
        except Exception as e:
            last_err = f"{url.split('/')[2]}: {e}"
            data = None
    if data is None:
        raise ValueError(last_err)
    df = pd.DataFrame(data["candles"])
    return df.astype({k: float for k in ("open", "high", "low", "close")})


async def fetch(source: str, symbol: str, tf: str, limit: int = 500) -> pd.DataFrame:
    if source == "binance":
        return await asyncio.to_thread(binance_candles, symbol, tf, limit)
    if source == "deriv":
        return await deriv_candles(symbol, tf, limit)
    raise ValueError("Источник: binance или deriv")


# ====================== ИНДИКАТОРЫ ======================
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
    df["atr_pct"] = df["atr"] / c * 100
    df["atr_rel"] = df["atr_pct"] / df["atr_pct"].rolling(100, min_periods=30).median()
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["vol_sma"] = df["volume"].rolling(20).mean()
    return df


def candle_pattern(df: pd.DataFrame, i: int) -> Tuple[int, str]:
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
    if body / rng < 0.35 and lo >= 2 * body and up <= body * 0.6:
        return 1, "молот"
    if body / rng < 0.35 and up >= 2 * body and lo <= body * 0.6:
        return -1, "падающая звезда"
    return 0, ""


def fib_signal(df: pd.DataFrame, i: int, lookback: int = 100) -> Tuple[int, str]:
    w = df.iloc[max(0, i - lookback + 1): i + 1]
    hi, lo = w["high"].max(), w["low"].min()
    atr = df["atr"].iloc[i]
    if hi == lo or pd.isna(atr):
        return 0, ""
    up_swing = w["high"].values.argmax() > w["low"].values.argmin()
    c = df["close"].iloc[i]
    for ratio in (0.382, 0.5, 0.618):
        if up_swing:
            level, side = hi - ratio * (hi - lo), 1
        else:
            level, side = lo + ratio * (hi - lo), -1
        if abs(c - level) <= 0.45 * atr:
            return side, f"уровень {ratio}"
    return 0, ""


def score_at(df: pd.DataFrame, i: int):
    r, p = df.iloc[i], df.iloc[i - 1]
    parts = {"trend": 1 if r["ema20"] > r["ema50"] else -1}
    if r["macd_h"] > 0 and r["macd_h"] > p["macd_h"]:
        parts["macd"] = 1
    elif r["macd_h"] < 0 and r["macd_h"] < p["macd_h"]:
        parts["macd"] = -1
    else:
        parts["macd"] = 0
    parts["rsi"] = 1 if r["rsi"] < 32 else -1 if r["rsi"] > 68 else 0
    parts["bollinger"] = 1 if r["close"] < r["bb_lo"] else -1 if r["close"] > r["bb_up"] else 0
    pat, pname = candle_pattern(df, i)
    parts["candle"] = pat
    fib, fname = fib_signal(df, i)
    parts["fib"] = fib
    score = sum(parts[k] * WEIGHTS[k] for k in parts)
    return score, parts, pname, fname


def filters_ok(df: pd.DataFrame, i: int) -> Tuple[bool, bool]:
    r = df.iloc[i]
    vol_ok = True
    if r["vol_sma"] > 0:
        vol_ok = r["volume"] >= r["vol_sma"] * MIN_VOLUME_RATIO
    rel = r["atr_rel"]
    atr_ok = (not pd.isna(rel)) and ATR_REL_MIN <= rel <= ATR_REL_MAX
    return vol_ok, atr_ok


def analyze(df: pd.DataFrame, htf: Optional[pd.DataFrame] = None) -> dict:
    df = add_indicators(df)
    i = len(df) - 1
    raw, parts, pname, fname = score_at(df, i)
    score, notes = raw, []
    r = df.iloc[i]
    vol_ok, atr_ok = filters_ok(df, i)

    htf_trend = 0
    if htf is not None and len(htf) > 60:
        h = add_indicators(htf)
        htf_trend = 1 if h["ema20"].iloc[-1] > h["ema50"].iloc[-1] else -1

    if abs(score) >= MIN_SCORE:  # причины отмены показываем, только если сигнал был
        if htf_trend and (score > 0) != (htf_trend > 0):
            score = 0
            notes.append("старший таймфрейм против")
        if not vol_ok:
            score = 0
            notes.append("низкий объём")
        if not atr_ok:
            score = 0
            notes.append("волатильность вне нормы")

    # история: те же правила и фильтры, сделка на 1 свечу (вход по закрытию)
    n = wins = 0
    for j in range(70, len(df) - 1):
        sj = score_at(df, j)[0]
        if abs(sj) < MIN_SCORE or not all(filters_ok(df, j)):
            continue
        n += 1
        if (df["close"].iloc[j + 1] - df["close"].iloc[j]) * sj > 0:
            wins += 1
    winrate = wins / n * 100 if n else 0.0

    if score >= MIN_SCORE:
        side = "⬆️ ВВЕРХ (Rise)"
    elif score <= -MIN_SCORE:
        side = "⬇️ ВНИЗ (Fall)"
    else:
        side = "⏸ НЕТ СИГНАЛА"

    level = 0
    if side != "⏸ НЕТ СИГНАЛА":
        level = 2 if abs(score) >= 5.5 else 1 if abs(score) >= 4.2 else 0
        if n < MIN_HISTORY or winrate < BREAKEVEN:
            level = max(0, level - 1)
    conf = ["низкая", "средняя", "высокая"][level]

    return {
        "side": side, "conf": conf, "score": score, "raw": raw, "parts": parts,
        "pname": pname, "fname": fname, "htf_trend": htf_trend, "notes": notes,
        "price": r["close"], "rsi": r["rsi"], "atr_rel": r["atr_rel"],
        "bt_n": n, "bt_win": winrate,
    }


# ====================== ОТЧЁТ ======================
def lab(v, pos, neg, zero="нейтрально"):
    return pos if v > 0 else neg if v < 0 else zero


def build_report(a: dict, symbol: str, tf: str, source: str, htf_tf: Optional[str]) -> str:
    p = a["parts"]
    sym, src = html.escape(symbol.upper()), html.escape(source)
    lines = [f"📊 <b>ОТЧЁТ · {sym} · {tf} · {src}</b>",
             f"Цена: {a['price']:.8f}", ""]

    trend = f"1) Тренд: {lab(p['trend'], 'восходящий', 'нисходящий')} (EMA20/50)"
    if htf_tf and a["htf_trend"]:
        trend += f"; старший ТФ {htf_tf}: {lab(a['htf_trend'], 'восходящий', 'нисходящий')}"
    lines.append(trend)

    lines.append(f"2) Импульс: MACD {lab(p['macd'], 'бычий', 'медвежий')}; "
                 f"RSI {a['rsi']:.0f} ({lab(p['rsi'], 'перепродан', 'перекуплен')})")

    rel = a["atr_rel"]
    rel_txt = f"{rel:.1f}× от нормы" if not pd.isna(rel) else "н/д"
    lines.append(f"3) Bollinger: {lab(p['bollinger'], 'ниже нижней полосы', 'выше верхней полосы', 'внутри полос')}; "
                 f"волатильность {rel_txt}")

    if a["pname"]:
        lines.append(f"4) Свечи: {html.escape(a['pname'])} ({lab(p['candle'], 'бычий', 'медвежий')})")
    else:
        lines.append("4) Свечи: паттерна нет")

    if a["fname"]:
        lines.append(f"5) Фибоначчи: цена у {html.escape(a['fname'])} "
                     f"({lab(p['fib'], 'ждём отскок вверх', 'ждём отскок вниз')})")
    else:
        lines.append("5) Фибоначчи: цена не у ключевого уровня")

    if a["bt_n"]:
        verdict = "выше порога" if a["bt_win"] >= BREAKEVEN else "НИЖЕ порога"
        few = ", мало данных" if a["bt_n"] < MIN_HISTORY else ""
        lines.append(f"6) История: {a['bt_n']} сигналов, верных {a['bt_win']:.0f}% "
                     f"(нужно ≥{BREAKEVEN}%) — {verdict}{few}")
    else:
        lines.append("6) История: сигналов по этим правилам не было")

    lines += ["", f"<b>ИТОГ: {a['side']}</b>"]
    if a["side"] == "⏸ НЕТ СИГНАЛА":
        lean = lab(a["raw"], "вверх", "вниз", "нет")
        if a["notes"]:
            lines.append("Причина: " + ", ".join(a["notes"]))
        lines.append(f"Склонность: {lean} (счёт {a['raw']:+.1f}, нужно ±{MIN_SCORE}) — слишком слабая для входа")
    else:
        lines.append(f"Уверенность: {a['conf']} (счёт {a['score']:+.1f})")
        lines.append(f"Срок: 1 свеча ({tf}), вход по закрытию свечи")
        if a["bt_n"] < MIN_HISTORY or a["bt_win"] < BREAKEVEN:
            lines.append("⚠️ История не подтверждает сигнал, доверие понижено")
    lines += ["", "Не финансовый совет. Гарантий прибыли нет."]
    return "\n".join(lines)


# ====================== БОТ ======================
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Команды:\n/signal binance BTCUSDT\n/signal deriv frxEURUSD\n"
        "/signal deriv frxEURUSD 15m\nТаймфрейм по умолчанию: 5m"
    )


async def signal(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        if len(ctx.args) < 2:
            raise IndexError
        source, symbol = ctx.args[0].lower(), ctx.args[1]
        tf = ctx.args[2] if len(ctx.args) > 2 else "5m"
        if tf not in TF_SECONDS:
            raise ValueError(f"Таймфрейм: {', '.join(TF_SECONDS)}")
        df = await fetch(source, symbol, tf, 500)
        df = df.iloc[:-1].reset_index(drop=True)  # без незакрытой свечи
        if len(df) < 100:
            raise ValueError("Слишком мало данных по этой паре")
        htf_tf, htf = HTF_MAP.get(tf), None
        if htf_tf:
            try:
                htf = (await fetch(source, symbol, htf_tf, 250)).iloc[:-1]
            except Exception:
                htf = None
        a = await asyncio.to_thread(analyze, df, htf)
    except IndexError:
        await update.message.reply_text("Пример: /signal binance BTCUSDT")
        return
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")
        return
    await update.message.reply_text(
        build_report(a, symbol, tf, source, htf_tf), parse_mode="HTML"
    )


class _Ping(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        pass


def run_web():
    HTTPServer(("0.0.0.0", int(os.getenv("PORT", 10000))), _Ping).serve_forever()


def main():
    if not TOKEN:
        raise RuntimeError("Не задан TG_TOKEN")
    threading.Thread(target=run_web, daemon=True).start()
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signal", signal))
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
