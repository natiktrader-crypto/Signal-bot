"""
Telegram Signal Bot v6.2 (Fixed & Production Ready)
Binance Spot + Deriv Forex
Commands:
  /signal deriv frxAUDCAD 15m
  /watch deriv frxAUDCAD 15m
  /unwatch [deriv frxAUDCAD 15m]
  /watchlist
  /volatility [forex|crypto|all] [1m|2m|3m|5m|15m|1h|4h]

Environment:
  TG_TOKEN  - Telegram bot token (required)
  TZ_OFFSET - local UTC offset in hours (default: 4)
"""

import asyncio
import html
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Tuple

import pandas as pd
import requests
import websockets
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.request import HTTPXRequest

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("signal_bot")

TOKEN = os.getenv("TG_TOKEN", "").strip()
TZ_OFFSET = int(os.getenv("TZ_OFFSET", "4"))

DERIV_WS_URL = "wss://ws.derivws.com/websockets/v3?app_id=1089"
BINANCE_BASES = (
    "https://data-api.binance.vision",
    "https://api.binance.com",
)

TF_SECONDS = {
    "1m": 60, "2m": 120, "3m": 180, "5m": 300,
    "15m": 900, "1h": 3600, "4h": 14400,
}
BINANCE_TFS = {"1m", "3m", "5m", "15m", "1h", "4h"}

PERIOD_RU = {
    "1m": "1 мин", "2m": "2 мин", "3m": "3 мин",
    "5m": "5 мин", "15m": "15 мин", "1h": "1 час", "4h": "4 часа",
}
HTF_MAP = {"1m": "5m", "5m": "15m", "15m": "1h", "1h": "4h"}

MIN_SCORE = 3.5
PAYOUT = float(os.getenv("PAYOUT", "0.86"))
BREAKEVEN = 100.0 / (1.0 + PAYOUT) if PAYOUT > 0 else 100.0
MIN_VOLUME_RATIO = 0.75
ATR_REL_MIN, ATR_REL_MAX = 0.5, 3.0
MIN_HISTORY = 30
CLOSE_DELAY = 5
MAX_WATCHES = 10
ADX_TREND = 22

WEIGHTS_TREND = {
    "trend": 1.6, "macd": 1.3, "candle": 1.2, "fib": 0.8,
    "rsi": 0.0, "bollinger": 0.0, "stoch": 0.0,
}
WEIGHTS_RANGE = {
    "trend": 0.4, "macd": 0.4, "candle": 1.4, "fib": 1.1,
    "rsi": 1.4, "bollinger": 1.3, "stoch": 1.1,
}
NO_SIGNAL = "⏸ НЕТ СИГНАЛА"

FOREX_LIST = [
    "frxEURUSD", "frxGBPUSD", "frxUSDJPY", "frxAUDUSD", "frxUSDCAD",
    "frxUSDCHF", "frxNZDUSD", "frxEURGBP", "frxEURJPY", "frxGBPJPY",
    "frxAUDJPY", "frxAUDCAD", "frxEURAUD", "frxEURCAD", "frxEURCHF",
    "frxGBPAUD", "frxGBPCAD", "frxGBPCHF", "frxAUDNZD", "frxCADJPY",
    "frxCHFJPY", "frxNZDJPY", "frxXAUUSD",
]

VOL_WINDOW, VOL_BASE = 8, 96
WATCHES: dict = {}


# ---------------------- DATA ----------------------

def binance_get(path: str, params: Optional[dict] = None):
    last_err = None
    for base in BINANCE_BASES:
        try:
            response = requests.get(base + path, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_err = exc
            log.warning("Binance endpoint failed (%s): %s", base, exc)
    raise RuntimeError(f"Binance API недоступен: {last_err}")


def binance_candles(symbol: str, tf: str, limit: int = 500) -> pd.DataFrame:
    if tf not in BINANCE_TFS:
        raise ValueError(
            f"Binance Spot не поддерживает {tf}. "
            f"Доступные интервалы: {', '.join(sorted(BINANCE_TFS))}"
        )
    data = binance_get(
        "/api/v3/klines",
        {"symbol": symbol.upper(), "interval": tf, "limit": min(limit, 1000)},
    )
    if not isinstance(data, list) or not data:
        raise ValueError(f"Binance не вернул свечи для {symbol}")
    df = pd.DataFrame(data).iloc[:, :6].copy()
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["t"] = pd.to_numeric(df["time"], errors="coerce") / 1000.0
    return df.dropna(subset=["open", "high", "low", "close", "t"]).reset_index(drop=True)


def _deriv_df(candles) -> pd.DataFrame:
    if not candles:
        raise ValueError("Deriv вернул пустой список свечей")
    df = pd.DataFrame(candles).copy()
    required = {"open", "high", "low", "close", "epoch"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"В ответе Deriv отсутствуют поля: {', '.join(sorted(missing))}")
    for col in ("open", "high", "low", "close", "epoch"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["t"] = df["epoch"]
    df["volume"] = 0.0
    return df.dropna(subset=["open", "high", "low", "close", "t"]).reset_index(drop=True)


async def _deriv_request(ws, request: dict, timeout: int = 20) -> dict:
    await ws.send(json.dumps(request))
    deadline = time.monotonic() + timeout
    req_id = request.get("req_id")
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Тайм-аут ответа Deriv")
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
        except asyncio.TimeoutError:
            raise TimeoutError("Тайм-аут ответа Deriv") from None
        data = json.loads(raw)
        if req_id is not None and data.get("req_id") != req_id:
            continue
        if "error" in data:
            message = data["error"].get("message", "ошибка Deriv")
            raise ValueError(message)
        return data


async def deriv_candles(symbol: str, tf: str, count: int = 500) -> pd.DataFrame:
    if tf not in TF_SECONDS:
        raise ValueError(f"Неизвестный таймфрейм: {tf}")
    request = {
        "ticks_history": symbol,
        "style": "candles",
        "granularity": TF_SECONDS[tf],
        "count": min(count, 5000),
        "end": "latest",
    }
    try:
        async with websockets.connect(
            DERIV_WS_URL, open_timeout=12, close_timeout=5, ping_interval=20
        ) as ws:
            data = await _deriv_request(ws, request, timeout=20)
    except Exception as exc:
        raise RuntimeError(f"Deriv WebSocket: {exc}") from exc
    if "candles" not in data:
        raise ValueError(f"Deriv не вернул свечи: {str(data)[:180]}")
    return _deriv_df(data["candles"])


async def deriv_multi(symbols, tf: str, count: int = 120) -> dict:
    result = {}
    try:
        async with websockets.connect(
            DERIV_WS_URL, open_timeout=12, close_timeout=5, ping_interval=20
        ) as ws:
            for req_id, symbol in enumerate(symbols, start=1):
                request = {
                    "ticks_history": symbol,
                    "style": "candles",
                    "granularity": TF_SECONDS[tf],
                    "count": count,
                    "end": "latest",
                    "req_id": req_id,
                }
                try:
                    data = await _deriv_request(ws, request, timeout=15)
                    if "candles" in data:
                        result[symbol] = _deriv_df(data["candles"])
                    else:
                        log.warning("No candles for %s: %s", symbol, data)
                except Exception as exc:
                    log.warning("Deriv symbol %s skipped: %s", symbol, exc)
    except Exception as exc:
        raise RuntimeError(f"Deriv WebSocket: {exc}") from exc
    if not result:
        raise ValueError("Deriv не вернул свечи ни по одной валютной паре")
    return result


async def fetch(source: str, symbol: str, tf: str, limit: int = 500) -> pd.DataFrame:
    if source == "binance":
        if tf not in BINANCE_TFS:
            raise ValueError(f"Binance не поддерживает {tf}")
        return await asyncio.to_thread(binance_candles, symbol, tf, limit)
    if source == "deriv":
        return await deriv_candles(symbol, tf, limit)
    raise ValueError("Источник: binance или deriv")


def closed_only(df: pd.DataFrame, period: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if "t" not in df.columns:
        return df.iloc[:-1].reset_index(drop=True)
    current_open = int(time.time() // period) * period
    return df[df["t"] < current_open].reset_index(drop=True)


# ---------------------- INDICATORS ----------------------

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    out = 100 - 100 / (1 + rs)
    return out.fillna(50)


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up, down = high.diff(), -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    atr_n = tr.ewm(alpha=1 / n, adjust=False).mean().replace(0, float("nan"))
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr_n
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr_n
    denom = (plus_di + minus_di).replace(0, float("nan"))
    dx = 100 * (plus_di - minus_di).abs() / denom
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()

    macd_line = (
        close.ewm(span=12, adjust=False).mean()
        - close.ewm(span=26, adjust=False).mean()
    )
    df["macd_h"] = macd_line - macd_line.ewm(span=9, adjust=False).mean()
    df["rsi"] = rsi(close)

    middle = close.rolling(20).mean()
    deviation = close.rolling(20).std()
    df["bb_up"] = middle + 2 * deviation
    df["bb_lo"] = middle - 2 * deviation

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = true_range.ewm(alpha=1 / 14, adjust=False).mean()
    df["atr_pct"] = df["atr"] / close.replace(0, float("nan")) * 100
    baseline = df["atr_pct"].rolling(100, min_periods=30).median()
    df["atr_rel"] = df["atr_pct"] / baseline.replace(0, float("nan"))
    df["adx"] = adx(df)

    lowest = df["low"].rolling(14).min()
    highest = df["high"].rolling(14).max()
    raw_stoch = (close - lowest) / (highest - lowest).replace(0, float("nan")) * 100
    df["stoch"] = raw_stoch.rolling(3).mean()

    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["vol_sma"] = df["volume"].rolling(20).mean()
    return df


def candle_pattern(df: pd.DataFrame, i: int) -> Tuple[int, str]:
    if i < 1:
        return 0, ""
    o, h, low, close = (float(df[k].iloc[i]) for k in ("open", "high", "low", "close"))
    prev_o, prev_c = float(df["open"].iloc[i - 1]), float(df["close"].iloc[i - 1])
    body, candle_range = abs(close - o), h - low
    if candle_range <= 0:
        return 0, ""
    upper = h - max(o, close)
    lower = min(o, close) - low

    if prev_c < prev_o and close > o and close >= prev_o and o <= prev_c:
        return 1, "бычье поглощение"
    if prev_c > prev_o and close < o and close <= prev_o and o >= prev_c:
        return -1, "медвежье поглощение"
    if body / candle_range < 0.35 and lower >= 2 * body and upper <= body * 0.6:
        return 1, "молот"
    if body / candle_range < 0.35 and upper >= 2 * body and lower <= body * 0.6:
        return -1, "падающая звезда"
    return 0, ""


def fib_signal(df: pd.DataFrame, i: int, lookback: int = 100) -> Tuple[int, str]:
    window = df.iloc[max(0, i - lookback + 1): i + 1]
    if len(window) < 10:
        return 0, ""
    high, low = window["high"].max(), window["low"].min()
    atr_value = df["atr"].iloc[i]
    if high == low or pd.isna(atr_value):
        return 0, ""
    
    high_pos = int(window["high"].values.argmax())
    low_pos = int(window["low"].values.argmin())  # Исправлено: argmin вместо argmax для поиска минимума
    close = df["close"].iloc[i]
    
    rising_swing = low_pos < high_pos
    for ratio in (0.382, 0.5, 0.618):
        if rising_swing:
            level = high - ratio * (high - low)
            side = 1
        else:
            level = low + ratio * (high - low)
            side = -1
        if abs(close - level) <= 0.45 * atr_value:
            return side, f"{ratio}"
    return 0, ""


def weights_for(row) -> Tuple[dict, str]:
    value = row["adx"]
    if not pd.isna(value) and value >= ADX_TREND:
        return WEIGHTS_TREND, "тренд"
    return WEIGHTS_RANGE, "флэт"


def score_at(df: pd.DataFrame, i: int):
    row, previous = df.iloc[i], df.iloc[i - 1]
    parts = {"trend": 1 if row["ema20"] > row["ema50"] else -1}

    if row["macd_h"] > 0 and row["macd_h"] > previous["macd_h"]:
        parts["macd"] = 1
    elif row["macd_h"] < 0 and row["macd_h"] < previous["macd_h"]:
        parts["macd"] = -1
    else:
        parts["macd"] = 0

    parts["rsi"] = 1 if row["rsi"] < 32 else -1 if row["rsi"] > 68 else 0
    parts["bollinger"] = (
        1 if row["close"] < row["bb_lo"]
        else -1 if row["close"] > row["bb_up"] else 0
    )
    parts["stoch"] = 1 if row["stoch"] < 20 else -1 if row["stoch"] > 80 else 0
    pattern, pattern_name = candle_pattern(df, i)
    parts["candle"] = pattern
    fib, fib_name = fib_signal(df, i)
    parts["fib"] = fib

    weights, _ = weights_for(row)
    score = sum(parts[key] * weights[key] for key in weights)
    return score, parts, pattern_name, fib_name


def filters_ok(df: pd.DataFrame, i: int) -> Tuple[bool, bool]:
    row = df.iloc[i]
    volume_ok = True
    if row["vol_sma"] > 0:
        volume_ok = row["volume"] >= row["vol_sma"] * MIN_VOLUME_RATIO
    relative_atr = row["atr_rel"]
    atr_ok = (
        not pd.isna(relative_atr)
        and ATR_REL_MIN <= relative_atr <= ATR_REL_MAX
    )
    return volume_ok, atr_ok


def analyze(raw_df: pd.DataFrame, htf: Optional[pd.DataFrame] = None) -> dict:
    df = add_indicators(raw_df)
    if len(df) < 100:
        raise ValueError("Для анализа требуется минимум 100 закрытых свечей")
    i = len(df) - 1
    raw_score, parts, pattern_name, fib_name = score_at(df, i)
    score = raw_score
    notes = []
    row = df.iloc[i]
    volume_ok, atr_ok = filters_ok(df, i)
    weights, regime = weights_for(row)
    max_weight = sum(weights.values())

    htf_trend = 0
    if htf is not None and len(htf) >= 60:
        higher = add_indicators(htf)
        htf_trend = (
            1 if higher["ema20"].iloc[-1] > higher["ema50"].iloc[-1] else -1
        )

    if abs(score) >= MIN_SCORE:
        if htf_trend and ((score > 0) != (htf_trend > 0)):
            score = 0
            notes.append("старший таймфрейм против")
        if not volume_ok:
            score = 0
            notes.append("низкий объём")
        if not atr_ok:
            score = 0
            notes.append("волатильность вне нормы")

    sample_count = wins = 0
    for j in range(70, len(df) - 1):
        historical_score = score_at(df, j)[0]
        if abs(historical_score) < MIN_SCORE or not all(filters_ok(df, j)):
            continue
        move = df["close"].iloc[j + 1] - df["open"].iloc[j + 1]
        if move == 0:
            continue
        sample_count += 1
        if move * historical_score > 0:
            wins += 1
    win_rate = wins / sample_count * 100 if sample_count else 0.0

    if score >= MIN_SCORE:
        side = "⬆️ ВВЕРХ (Rise)"
    elif score <= -MIN_SCORE:
        side = "⬇️ ВНИЗ (Fall)"
    else:
        side = NO_SIGNAL

    level = 0
    if side != NO_SIGNAL:
        strength = abs(score) / max_weight if max_weight else 0
        level = 2 if strength >= 0.8 else 1 if strength >= 0.65 else 0
        if sample_count < MIN_HISTORY or win_rate < BREAKEVEN:
            level = max(0, level - 1)
    confidence = ("низкая", "средняя", "высокая")[level]

    return {
        "side": side, "conf": confidence, "score": score, "raw": raw_score,
        "parts": parts, "pname": pattern_name, "fname": fib_name,
        "htf_trend": htf_trend, "notes": notes, "price": row["close"],
        "rsi": row["rsi"], "stoch": row["stoch"], "adx": row["adx"],
        "regime": regime, "atr_rel": row["atr_rel"],
        "bt_n": sample_count, "bt_win": win_rate,
    }


async def get_analysis(source: str, symbol: str, tf: str):
    period = TF_SECONDS[tf]
    candles = await fetch(source, symbol, tf, 500)
    df = closed_only(candles, period)
    if len(df) < 100:
        raise ValueError("Слишком мало закрытых свечей. Проверьте символ и таймфрейм")

    last_close = float(df["t"].iloc[-1]) + period
    if time.time() - last_close > period * 4:
        raise ValueError(
            f"Нет свежих данных: последняя свеча закрылась в "
            f"{local_hhmm(last_close)}. Возможно, рынок закрыт"
        )

    higher_tf = HTF_MAP.get(tf)
    higher_df = None
    if higher_tf:
        try:
            higher_df = closed_only(
                await fetch(source, symbol, higher_tf, 250),
                TF_SECONDS[higher_tf],
            )
        except Exception as exc:
            log.warning("Higher timeframe unavailable for %s: %s", symbol, exc)

    result = await asyncio.to_thread(analyze, df, higher_df)
    return result, higher_tf


# ---------------------- REPORTS ----------------------

def local_hhmm(timestamp: float) -> str:
    return time.strftime("%H:%M", time.gmtime(timestamp + TZ_OFFSET * 3600))


def lab(value, positive, negative, neutral="нейтрально"):
    return positive if value > 0 else negative if value < 0 else neutral


def verdict(result: dict):
    parts = result["parts"]
    direction = result["raw"] if result["raw"] != 0 else parts["trend"]
    arrow = "⬆️ ВВЕРХ (Rise)" if direction > 0 else "⬇️ ВНИЗ (Fall)"

    if result["side"] != NO_SIGNAL:
        confidence = f"{result['conf']} (счёт {result['score']:+.1f})"
        if result["bt_n"] < MIN_HISTORY or result["bt_win"] < BREAKEVEN:
            return arrow, confidence, "осторожно, история не подтверждает сигнал", "caution"
        return arrow, confidence, "сигнал прошёл фильтры, проверьте условия контракта", "enter"

    confidence = f"слабая (счёт {result['raw']:+.1f}, для входа нужно ±{MIN_SCORE})"
    return arrow, confidence, "пропустить, сигнал слабый", "skip"


def action_line(result: dict, tf: str) -> str:
    arrow, _, _, key = verdict(result)
    word = "ВВЕРХ" if "ВВЕРХ" in arrow else "ВНИЗ"
    if key == "skip":
        return f"👉 ДЕЙСТВИЕ: НЕ ОТКРЫВАТЬ (слабое направление {word.lower()})"

    period = TF_SECONDS[tf]
    now = time.time()
    candle_open = int(now // period) * period
    age = now - candle_open
    expiry = candle_open + period

    if age > min(120, period * 0.25):
        return (
            f"👉 ДЕЙСТВИЕ: ЖДАТЬ СЛЕДУЮЩУЮ СВЕЧУ. "
            f"Текущий сигнал устаревает; следующий интервал начнётся "
            f"в {local_hhmm(expiry)} (UTC{TZ_OFFSET:+d})"
        )

    prefix = "СИГНАЛ" if key == "enter" else "СИГНАЛ С ОСТОРОЖНОСТЬЮ"
    return (
        f"👉 ДЕЙСТВИЕ: {prefix} {word} на {PERIOD_RU[tf]}\n"
        f"Ориентировочное закрытие следующей свечи: "
        f"{local_hhmm(expiry)} (UTC{TZ_OFFSET:+d})"
    )


def build_report(result: dict, symbol: str, tf: str, source: str, higher_tf):
    parts = result["parts"]
    symbol_safe = html.escape(symbol.upper())
    source_safe = html.escape(source)
    price = float(result["price"])
    decimals = 2 if price >= 1000 else 4 if price >= 1 else 8
    lines = [
        f"📊 <b>ОТЧЁТ · {symbol_safe} · {tf} · {source_safe}</b>",
        f"Цена: {price:.{decimals}f}", "",
    ]

    trend = (
        f"1) Тренд: {lab(parts['trend'], 'восходящий', 'нисходящий')} (EMA20/50)"
    )
    if higher_tf and result["htf_trend"]:
        trend += (
            f"; старший ТФ {higher_tf}: "
            f"{lab(result['htf_trend'], 'восходящий', 'нисходящий')}"
        )
    if not pd.isna(result["adx"]):
        trend += f"\n   Режим рынка: {result['regime']} (ADX {result['adx']:.0f})"
    lines.append(trend)

    stochastic = (
        f"{result['stoch']:.0f}" if not pd.isna(result["stoch"]) else "н/д"
    )
    lines.append(
        f"2) Импульс: MACD {lab(parts['macd'], 'бычий', 'медвежий')}; "
        f"RSI {result['rsi']:.0f}; Stochastic {stochastic}"
    )
    relative_atr = result["atr_rel"]
    relative_text = (
        f"{relative_atr:.1f}× от нормы"
        if not pd.isna(relative_atr) else "н/д"
    )
    lines.append(
        f"3) Bollinger: "
        f"{lab(parts['bollinger'], 'ниже нижней полосы', 'выше верхней полосы', 'внутри полос')}; "
        f"волатильность {relative_text}"
    )
    lines.append(
        f"4) Свечи: {html.escape(result['pname'])}"
        if result["pname"] else "4) Свечи: паттерна нет"
    )
    lines.append(
        f"5) Фибоначчи: цена у уровня {html.escape(result['fname'])}"
        if result["fname"] else "5) Фибоначчи: цена не у ключевого уровня"
    )

    if result["bt_n"]:
        status = "выше порога" if result["bt_win"] >= BREAKEVEN else "НИЖЕ порога"
        few = ", мало наблюдений" if result["bt_n"] < MIN_HISTORY else ""
        lines.append(
            f"6) Историческая оценка: {result['bt_n']} сигналов, "
            f"совпадений {result['bt_win']:.1f}% "
            f"(ориентир безубыточности {BREAKEVEN:.2f}%) - {status}{few}"
        )
    else:
        lines.append("6) Историческая оценка: недостаточно сигналов")

    arrow, confidence, recommendation, _ = verdict(result)
    lines += [
        "", f"<b>ИТОГ: {arrow}</b>",
        f"Уверенность: {confidence}",
    ]
    if result["notes"]:
        lines.append("Причина фильтрации: " + ", ".join(result["notes"]))
    lines += [
        f"Рекомендация: {recommendation}", "",
        f"<b>{action_line(result, tf)}</b>", "",
        "Историческая оценка не гарантирует результат. "
        "Не финансовый совет. Проверьте выплату, спред и риск.",
    ]
    return "\n".join(lines)


def build_watch_message(result: dict, symbol: str, tf: str, source: str, higher_tf):
    close_timestamp = int(time.time() // TF_SECONDS[tf]) * TF_SECONDS[tf]
    _, _, _, key = verdict(result)
    icon = {"enter": "🟢", "caution": "🟡", "skip": "⚪"}[key]
    header = (
        f"{icon} 🕒 {local_hhmm(close_timestamp)} (UTC{TZ_OFFSET:+d}) · "
        f"{html.escape(symbol.upper())} · {tf}"
    )
    if key == "skip":
        historical = (
            f"{result['bt_win']:.0f}% на {result['bt_n']}"
            if result["bt_n"] else "недостаточно данных"
        )
        return (
            f"{header}\n<b>{action_line(result, tf)}</b>\n"
            f"Счёт {result['raw']:+.1f} из ±{MIN_SCORE}; "
            f"история: {historical}"
        )
    return header + "\n\n" + build_report(result, symbol, tf, source, higher_tf)


# ---------------------- VOLATILITY ----------------------

def range_stats(df: pd.DataFrame) -> Tuple[float, float]:
    ranges = (df["high"] - df["low"]) / df["close"].replace(0, float("nan")) * 100
    recent = ranges.iloc[-VOL_WINDOW:].mean()
    baseline = ranges.iloc[-VOL_BASE:].mean()
    if pd.isna(recent) or pd.isna(baseline) or baseline <= 0:
        return 0.0, 0.0
    return float(recent), float(recent / baseline)


def crypto_volatility(tf: str, top: int = 5):
    tickers = binance_get("/api/v3/ticker/24hr")
    candidates = []
    for ticker in tickers:
        symbol = ticker.get("symbol", "")
        if not symbol.endswith("USDT"):
            continue
        base = symbol[:-4]
        if base.endswith(("UP", "DOWN", "BULL", "BEAR")):
            continue
        try:
            quote_volume = float(ticker.get("quoteVolume", 0))
        except (TypeError, ValueError):
            continue
        if quote_volume >= 20_000_000:
            candidates.append((quote_volume, symbol))

    symbols = [symbol for _, symbol in sorted(candidates, reverse=True)[:30]]

    def one(symbol):
        try:
            df = closed_only(
                binance_candles(symbol, tf, 120), TF_SECONDS[tf]
            )
            if len(df) < VOL_BASE:
                return None
            recent, ratio = range_stats(df)
            return symbol[:-4] + "/USDT", recent, ratio, symbol
        except Exception as exc:
            log.debug("Crypto volatility skipped %s: %s", symbol, exc)
            return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        rows = [row for row in executor.map(one, symbols) if row]
    rows.sort(key=lambda row: row[1], reverse=True)
    return rows[:top]


async def forex_volatility(tf: str, top: int = 5):
    data = await deriv_multi(FOREX_LIST, tf, 120)
    period = TF_SECONDS[tf]
    rows = []
    for symbol, candles in data.items():
        df = closed_only(candles, period)
        if len(df) < VOL_BASE:
            continue
        if time.time() - (df["t"].iloc[-1] + period) > period * 4:
            continue
        recent, ratio = range_stats(df)
        rows.append((symbol[3:6] + "/" + symbol[6:], recent, ratio, symbol))
    rows.sort(key=lambda row: row[1], reverse=True)
    return rows[:top]


def format_volatility(title: str, rows, tf: str, source: str):
    if not rows:
        return f"{title}: нет свежих данных (возможно, рынок закрыт)"
    lines = [f"🔥 <b>{html.escape(title)}</b> · {tf}, последние {VOL_WINDOW} свечей"]
    for index, (name, recent, ratio, symbol) in enumerate(rows, start=1):
        heat = "🔥" if ratio >= 1.5 else "↗️" if ratio >= 1.0 else "↘️"
        lines.append(
            f"{index}. {html.escape(name)} - {recent:.3f}% за свечу, "
            f"{ratio:.1f}× от нормы {heat}"
        )
    lines.append(f"Проверить сигнал: /signal {source} {rows[0][3]} {tf}")
    return "\n".join(lines)


# ---------------------- TELEGRAM COMMANDS ----------------------

def parse_args(args):
    if len(args) < 2:
        raise IndexError
    source = args[0].lower().strip()
    symbol = args[1].strip()
    tf = args[2].lower().strip() if len(args) > 2 else "5m"

    if source not in ("binance", "deriv"):
        raise ValueError("Источник должен быть binance или deriv")
    if not symbol:
        raise ValueError("Не указана торговая пара")
    if tf not in TF_SECONDS:
        raise ValueError(f"Таймфрейм: {', '.join(TF_SECONDS)}")
    if source == "binance":
        symbol = symbol.upper()
        if tf not in BINANCE_TFS:
            raise ValueError(f"Binance не поддерживает {tf}")
    return source, symbol, tf


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = (
        "🤖 <b>Signal Bot v6.2 (Fixed)</b>\n\n"
        "/signal deriv frxAUDCAD 15m - разовый отчёт\n"
        "/watch deriv frxAUDCAD 15m - слежение после закрытия свечи\n"
        "/unwatch - остановить все слежения\n"
        "/unwatch deriv frxAUDCAD 15m - остановить одну пару\n"
        "/watchlist - активные слежения\n"
        "/volatility forex 15m - волатильность Forex\n"
        "/volatility crypto 15m - волатильность криптовалют\n\n"
        "Binance: BTCUSDT, ETHUSDT. Таймфрейм по умолчанию 5m.\n"
        "Deriv: frxAUDCAD, frxEURUSD и другие доступные символы."
    )
    await update.effective_message.reply_text(message, parse_mode="HTML")


async def signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        source, symbol, tf = parse_args(context.args)
        result, higher_tf = await get_analysis(source, symbol, tf)
        report = build_report(result, symbol, tf, source, higher_tf)
    except IndexError:
        await update.effective_message.reply_text(
            "Пример: /signal deriv frxAUDCAD 15m"
        )
        return
    except Exception as exc:
        log.exception("Signal command failed")
        await update.effective_message.reply_text(
            f"Ошибка: {html.escape(str(exc))}"
        )
        return
    await update.effective_message.reply_text(report, parse_mode="HTML")


async def watch_loop(bot, chat_id: int, source: str, symbol: str, tf: str):
    period = TF_SECONDS[tf]
    while True:
        now = time.time()
        current_open = int(now // period) * period
        target = (
            current_open + CLOSE_DELAY
            if now < current_open + CLOSE_DELAY
            else current_open + period + CLOSE_DELAY
        )
        await asyncio.sleep(max(0, target - time.time()))

        try:
            result, higher_tf = await get_analysis(source, symbol, tf)
            message = build_watch_message(
                result, symbol, tf, source, higher_tf
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("Watch analysis failed")
            message = (
                f"⚠️ {html.escape(symbol.upper())} {tf}: "
                f"ошибка {html.escape(str(exc))}"
            )

        try:
            await bot.send_message(chat_id, message, parse_mode="HTML")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("Telegram send failed: %s", exc)


async def watch_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        source, symbol, tf = parse_args(context.args)
    except IndexError:
        await update.effective_message.reply_text(
            "Пример: /watch deriv frxAUDCAD 15m"
        )
        return
    except Exception as exc:
        await update.effective_message.reply_text(
            f"Ошибка: {html.escape(str(exc))}"
        )
        return

    chat_id = update.effective_chat.id
    key = (chat_id, source, symbol.upper(), tf)
    existing = WATCHES.get(key)
    if existing and not existing.done():
        await update.effective_message.reply_text("Это слежение уже запущено.")
        return

    active_count = sum(1 for task in WATCHES.values() if not task.done())
    if active_count >= MAX_WATCHES:
        await update.effective_message.reply_text(
            f"Достигнут лимит {MAX_WATCHES} слежений на сервере. "
            "Остановите ненужные через /unwatch."
        )
        return

    task = asyncio.create_task(
        watch_loop(context.bot, chat_id, source, symbol, tf)
    )
    WATCHES[key] = task

    period = TF_SECONDS[tf]
    now = time.time()
    current_open = int(now // period) * period
    next_run = (
        current_open + CLOSE_DELAY
        if now < current_open + CLOSE_DELAY
        else current_open + period + CLOSE_DELAY
    )
    await update.effective_message.reply_text(
        f"Слежу за {symbol.upper()} {tf} ({source}). "
        f"Следующая проверка примерно в {local_hhmm(next_run)} "
        f"(UTC{TZ_OFFSET:+d})."
    )


async def unwatch_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if context.args:
        try:
            source, symbol, tf = parse_args(context.args)
        except Exception:
            await update.effective_message.reply_text(
                "Пример: /unwatch deriv frxAUDCAD 15m"
            )
            return
        keys = [(chat_id, source, symbol.upper(), tf)]
    else:
        keys = [key for key in list(WATCHES) if key[0] == chat_id]

    stopped = 0
    for key in keys:
        task = WATCHES.pop(key, None)
        if task:
            task.cancel()
            stopped += 1
    await update.effective_message.reply_text(f"Остановлено слежений: {stopped}")


async def watchlist_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    items = [
        f"• {key[2]} {key[3]} ({key[1]})"
        for key, task in WATCHES.items()
        if key[0] == chat_id and not task.done()
    ]
    await update.effective_message.reply_text(
        "Слежу за:\n" + "\n".join(items) if items else "Активных слежений нет."
    )


async def volatility_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kind = context.args[0].lower() if context.args else "all"
    tf = context.args[1].lower() if len(context.args) > 1 else "15m"

    if kind not in ("forex", "crypto", "all") or tf not in TF_SECONDS:
        await update.effective_message.reply_text(
            "Пример: /volatility forex 15m или /volatility crypto 5m"
        )
        return
    if kind in ("crypto", "all") and tf not in BINANCE_TFS:
        await update.effective_message.reply_text(
            f"Binance не поддерживает {tf}. Для crypto выберите: "
            f"{', '.join(sorted(BINANCE_TFS))}"
        )
        return

    await update.effective_message.reply_text(
        "Считаю волатильность. Это может занять 15-30 секунд..."
    )
    reports = []

    if kind in ("forex", "all"):
        try:
            rows = await forex_volatility(tf)
            reports.append(format_volatility("Форекс", rows, tf, "deriv"))
        except Exception as exc:
            log.exception("Forex volatility failed")
            reports.append(f"Форекс: ошибка {html.escape(str(exc))}")

    if kind in ("crypto", "all"):
        try:
            rows = await asyncio.to_thread(crypto_volatility, tf)
            reports.append(format_volatility("Криптовалюта", rows, tf, "binance"))
        except Exception as exc:
            log.exception("Crypto volatility failed")
            reports.append(f"Крипта: ошибка {html.escape(str(exc))}")

    reports.append(
        "Волатильность показывает размер движения, но не его направление. "
        "Список не является торговой рекомендацией."
    )
    await update.effective_message.reply_text(
        "\n\n".join(reports), parse_mode="HTML"
    )


# ---------------------- HEALTH CHECK / STARTUP ----------------------

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"Signal bot is running")

    def log_message(self, *_args):
        return


def run_health_server():
    port = int(os.getenv("PORT", "10000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), HealthHandler)
    server.serve_forever()


def main():
    if not TOKEN:
        raise RuntimeError(
            "Не задан TG_TOKEN. Добавьте токен в Render Environment."
        )

    if not 0 < PAYOUT <= 10:
        raise RuntimeError(
            "PAYOUT должен быть задан долей, например 0.86"
        )

    # Основное соединение Telegram API
    telegram_request = HTTPXRequest(
        connection_pool_size=16,
        connect_timeout=30,
        read_timeout=60,
        write_timeout=30,
        pool_timeout=30,
    )

    # Отдельное соединение для получения обновлений
    polling_request = HTTPXRequest(
        connection_pool_size=4,
        connect_timeout=30,
        read_timeout=60,
        write_timeout=30,
        pool_timeout=30,
    )

    # Health check для Render
    threading.Thread(
        target=run_health_server,
        daemon=True,
        name="health-server"
    ).start()

    app = (
        Application.builder()
        .token(TOKEN)
        .request(telegram_request)
        .get_updates_request(polling_request)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signal", signal_command))
    app.add_handler(CommandHandler("watch", watch_command))
    app.add_handler(CommandHandler("unwatch", unwatch_command))
    app.add_handler(CommandHandler("watchlist", watchlist_command))
    app.add_handler(CommandHandler("volatility", volatility_command))

    log.info(
        "Signal Bot v6.2 starting; timezone UTC%+d",
        TZ_OFFSET
    )

    # Автоматические повторные попытки при ошибках запуска
    app.run_polling(
        poll_interval=1.0,
        timeout=30,
        bootstrap_retries=-1,
        drop_pending_updates=False,
        close_loop=True,
    )


if __name__ == "__main__":
    main()
