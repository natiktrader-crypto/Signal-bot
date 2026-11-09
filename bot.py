"""
Телеграм-бот сигналов v5: Binance + Deriv, авто-сигнал по закрытию каждой свечи.

/signal deriv frxAUDCAD 15m   - разовый отчёт
/watch  deriv frxAUDCAD 15m   - присылать сигнал после закрытия КАЖДОЙ свечи
/unwatch [deriv frxAUDCAD 15m] - остановить (без параметров - всё)
/watchlist                    - список слежения

pip install python-telegram-bot pandas requests websockets
Переменные окружения: TG_TOKEN, TZ_OFFSET (часы от UTC для времени в сообщениях, по умолчанию 4)
"""
import asyncio
import html
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional, Tuple

import pandas as pd
import requests
import websockets
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("signal_bot")

# ====================== КОНФИГ ======================
TOKEN = os.getenv("TG_TOKEN")
TZ_OFFSET = int(os.getenv("TZ_OFFSET", "4"))
DERIV_WS_LIST = [
    "wss://api.derivws.com/trading/v1/options/ws/public",
    "wss://ws.derivws.com/websockets/v3?app_id=1089",
]
TF_SECONDS = {"1m": 60, "2m": 120, "3m": 180, "5m": 300,
              "15m": 900, "1h": 3600, "4h": 14400}
HTF_MAP = {"1m": "5m", "5m": "15m", "15m": "1h", "1h": "4h"}

MIN_SCORE = 3.5
BREAKEVEN = 52.6          # % верных для нуля при выплате ~90% (при 86% нужно ~54%)
MIN_VOLUME_RATIO = 0.75
ATR_REL_MIN, ATR_REL_MAX = 0.5, 3.0
MIN_HISTORY = 30
CLOSE_DELAY = 5           # секунд после закрытия свечи до запроса данных
MAX_WATCHES = 10          # всего слежений на сервере

WEIGHTS = {"trend": 1.6, "macd": 1.3, "rsi": 1.0,
           "bollinger": 0.9, "candle": 1.4, "fib": 1.1}
NO_SIGNAL = "⏸ НЕТ СИГНАЛА"


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
    df = df.astype({k: float for k in ("open", "high", "low", "close", "volume")})
    df["t"] = df["time"].astype(float) / 1000  # время открытия свечи, сек
    return df


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
    df = df.astype({k: float for k in ("open", "high", "low", "close")})
    if "epoch" in df.columns:
        df["t"] = df["epoch"].astype(float)
    return df


async def fetch(source: str, symbol: str, tf: str, limit: int = 500) -> pd.DataFrame:
    if source == "binance":
        return await asyncio.to_thread(binance_candles, symbol, tf, limit)
    if source == "deriv":
        return await deriv_candles(symbol, tf, limit)
    raise ValueError("Источник: binance или deriv")


def closed_only(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Оставляет только закрытые свечи (открывшиеся до текущего интервала)."""
    if "t" not in df.columns:
        return df.iloc[:-1].reset_index(drop=True)
    cur_open = int(time.time() // period) * period
    return df[df["t"] < cur_open].reset_index(drop=True)


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
            return side, f"{ratio}"
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

    if abs(score) >= MIN_SCORE:
        if htf_trend and (score > 0) != (htf_trend > 0):
            score = 0
            notes.append("старший таймфрейм против")
        if not vol_ok:
            score = 0
            notes.append("низкий объём")
        if not atr_ok:
            score = 0
            notes.append("волатильность вне нормы")

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
        side = NO_SIGNAL

    level = 0
    if side != NO_SIGNAL:
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


async def get_analysis(source: str, symbol: str, tf: str):
    df = closed_only(await fetch(source, symbol, tf, 500), TF_SECONDS[tf])
    if len(df) < 100:
        raise ValueError("Слишком мало данных по этой паре")
    htf_tf, htf = HTF_MAP.get(tf), None
    if htf_tf:
        try:
            htf = closed_only(await fetch(source, symbol, htf_tf, 250), TF_SECONDS[htf_tf])
        except Exception:
            htf = None
    a = await asyncio.to_thread(analyze, df, htf)
    return a, htf_tf


# ====================== ОТЧЁТ ======================
def lab(v, pos, neg, zero="нейтрально"):
    return pos if v > 0 else neg if v < 0 else zero


def verdict(a: dict):
    """Возвращает (стрелка, уверенность, рекомендация, ключ рекомендации)."""
    p = a["parts"]
    d = a["raw"] if a["raw"] != 0 else p["trend"]
    arrow = "⬆️ ВВЕРХ (Rise)" if d > 0 else "⬇️ ВНИЗ (Fall)"
    if a["side"] != NO_SIGNAL:
        conf = f"{a['conf']} (счёт {a['score']:+.1f})"
        if a["bt_n"] < MIN_HISTORY or a["bt_win"] < BREAKEVEN:
            return arrow, conf, "осторожно, история не подтверждает сигнал", "caution"
        return arrow, conf, "можно входить", "enter"
    conf = f"слабая (счёт {a['raw']:+.1f}, для входа нужно ±{MIN_SCORE})"
    return arrow, conf, "пропустить, сигнал слабый", "skip"


def build_report(a: dict, symbol: str, tf: str, source: str, htf_tf: Optional[str]) -> str:
    p = a["parts"]
    sym, src = html.escape(symbol.upper()), html.escape(source)
    price = a["price"]
    dec = 2 if price >= 1000 else 4 if price >= 1 else 8
    lines = [f"📊 <b>ОТЧЁТ · {sym} · {tf} · {src}</b>", f"Цена: {price:.{dec}f}", ""]

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
        lines.append(f"5) Фибоначчи: цена у уровня {html.escape(a['fname'])} "
                     f"({lab(p['fib'], 'ждём отскок вверх', 'ждём отскок вниз')})")
    else:
        lines.append("5) Фибоначчи: цена не у ключевого уровня")
    if a["bt_n"]:
        vd = "выше порога" if a["bt_win"] >= BREAKEVEN else "НИЖЕ порога"
        few = ", мало данных" if a["bt_n"] < MIN_HISTORY else ""
        lines.append(f"6) История: {a['bt_n']} сигналов, верных {a['bt_win']:.0f}% "
                     f"(нужно ≥{BREAKEVEN}%) — {vd}{few}")
    else:
        lines.append("6) История: сигналов по этим правилам не было")

    arrow, conf, rec, _ = verdict(a)
    lines += ["", f"<b>ИТОГ: {arrow}</b>", f"Уверенность: {conf}"]
    if a["side"] == NO_SIGNAL and a["notes"]:
        lines.append("Причина: " + ", ".join(a["notes"]))
    elif a["notes"]:
        lines.append("Причина: " + ", ".join(a["notes"]))
    lines += [f"Рекомендация: {rec}", f"Срок: 1 свеча ({tf}), вход по закрытию свечи",
              "", "Не финансовый совет. Гарантий прибыли нет."]
    return "\n".join(lines)


def local_hhmm(ts: float) -> str:
    return time.strftime("%H:%M", time.gmtime(ts + TZ_OFFSET * 3600))


def build_watch_message(a: dict, symbol: str, tf: str, source: str, htf_tf) -> str:
    period = TF_SECONDS[tf]
    close_ts = int(time.time() // period) * period
    arrow, conf, rec, key = verdict(a)
    icon = {"enter": "🟢", "caution": "🟡", "skip": "⚪"}[key]
    head = (f"🕒 {local_hhmm(close_ts)} (UTC{TZ_OFFSET:+d}) · {html.escape(symbol.upper())} · {tf}\n"
            f"{icon} <b>{arrow}</b> · {rec}")
    if key == "skip":
        hist = f"{a['bt_win']:.0f}% на {a['bt_n']}" if a["bt_n"] else "нет"
        return head + f"\nСчёт {a['raw']:+.1f} из ±{MIN_SCORE}; история: {hist}"
    return head + "\n\n" + build_report(a, symbol, tf, source, htf_tf)


# ====================== КОМАНДЫ ======================
WATCHES: dict = {}  # (chat_id, source, SYMBOL, tf) -> asyncio.Task


def parse_args(args):
    if len(args) < 2:
        raise IndexError
    source, symbol = args[0].lower(), args[1]
    tf = args[2] if len(args) > 2 else "5m"
    if source not in ("binance", "deriv"):
        raise ValueError("Источник: binance или deriv")
    if tf not in TF_SECONDS:
        raise ValueError(f"Таймфрейм: {', '.join(TF_SECONDS)}")
    return source, symbol, tf


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Команды:\n"
        "/signal deriv frxAUDCAD 15m - разовый отчёт\n"
        "/watch deriv frxAUDCAD 15m - сигнал после закрытия каждой свечи\n"
        "/unwatch - остановить слежение\n"
        "/watchlist - что отслеживается\n"
        "Пары Binance: BTCUSDT, ETHUSDT. Таймфрейм по умолчанию 5m."
    )


async def signal(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        source, symbol, tf = parse_args(ctx.args)
        a, htf_tf = await get_analysis(source, symbol, tf)
    except IndexError:
        await update.message.reply_text("Пример: /signal deriv frxAUDCAD 15m")
        return
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")
        return
    await update.message.reply_text(build_report(a, symbol, tf, source, htf_tf), parse_mode="HTML")


async def watch_loop(bot, chat_id: int, source: str, symbol: str, tf: str):
    period = TF_SECONDS[tf]
    while True:
        now = time.time()
        base = (now // period) * period
        target = base + CLOSE_DELAY if now < base + CLOSE_DELAY else base + period + CLOSE_DELAY
        await asyncio.sleep(max(0, target - now))
        try:
            a, htf_tf = await get_analysis(source, symbol, tf)
            text = build_watch_message(a, symbol, tf, source, htf_tf)
        except Exception as e:
            text = f"⚠️ {html.escape(symbol.upper())} {tf}: ошибка {html.escape(str(e))}"
        try:
            await bot.send_message(chat_id, text, parse_mode="HTML")
        except Exception as e:
            log.warning("send failed: %s", e)


async def watch(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        source, symbol, tf = parse_args(ctx.args)
    except IndexError:
        await update.message.reply_text("Пример: /watch deriv frxAUDCAD 15m")
        return
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")
        return
    chat_id = update.effective_chat.id
    key = (chat_id, source, symbol.upper(), tf)
    if key in WATCHES and not WATCHES[key].done():
        await update.message.reply_text("Это слежение уже запущено.")
        return
    if sum(1 for t in WATCHES.values() if not t.done()) >= MAX_WATCHES:
        await update.message.reply_text("Слишком много слежений на сервере, остановите лишнее: /unwatch")
        return
    WATCHES[key] = asyncio.create_task(watch_loop(ctx.bot, chat_id, source, symbol, tf))
    period = TF_SECONDS[tf]
    now = time.time()
    base = (now // period) * period
    nxt = base + period + CLOSE_DELAY if now >= base + CLOSE_DELAY else base + CLOSE_DELAY
    await update.message.reply_text(
        f"Слежу за {symbol.upper()} {tf} ({source}). Следующий сигнал в {local_hhmm(nxt)} (UTC{TZ_OFFSET:+d}), "
        f"дальше после закрытия каждой свечи. Остановить: /unwatch"
    )


async def unwatch(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if ctx.args:
        try:
            source, symbol, tf = parse_args(ctx.args)
        except Exception:
            await update.message.reply_text("Пример: /unwatch deriv frxAUDCAD 15m")
            return
        keys = [(chat_id, source, symbol.upper(), tf)]
    else:
        keys = [k for k in WATCHES if k[0] == chat_id]
    stopped = 0
    for k in keys:
        t = WATCHES.pop(k, None)
        if t:
            t.cancel()
            stopped += 1
    await update.message.reply_text(f"Остановлено слежений: {stopped}")


async def watchlist(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    items = [f"{k[2]} {k[3]} ({k[1]})" for k, t in WATCHES.items() if k[0] == chat_id and not t.done()]
    await update.message.reply_text("Слежу за:\n" + "\n".join(items) if items else "Слежений нет.")


# ====================== KEEP-ALIVE ======================
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
    app.add_handler(CommandHandler("watch", watch))
    app.add_handler(CommandHandler("unwatch", unwatch))
    app.add_handler(CommandHandler("watchlist", watchlist))
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
