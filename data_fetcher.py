"""
data_fetcher.py — Data crypto realtime dari Binance (public API + ccxt)
Semua data tanpa login untuk market data publik.
API key Binance hanya dibutuhkan untuk eksekusi order live.
"""
import asyncio, threading
import pandas as pd
import numpy as np
import ccxt, requests
from datetime import datetime
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor
from rich.console import Console
import warnings; warnings.filterwarnings("ignore")

from config import (
    EXCHANGE_NAME, EXCHANGE_API_KEY, EXCHANGE_API_SECRET, USE_TESTNET,
    PRIMARY_TIMEFRAME, TREND_TIMEFRAME, CONTEXT_TIMEFRAME,
    BINANCE_DEPTH_URL, BINANCE_TRADES_URL, BINANCE_TICKER_URL,
    BINANCE_FUNDING_URL, BINANCE_OI_URL,
)
console = Console()

_ohlcv_cache   = TTLCache(maxsize=500, ttl=90)
_ticker_cache  = TTLCache(maxsize=200, ttl=30)
_metrics_cache = TTLCache(maxsize=200, ttl=60)
_ob_cache      = TTLCache(maxsize=200, ttl=15)   # orderbook 15 detik

_exchange_lock = threading.Lock()
_exchange_inst = None

HEADERS = {"User-Agent": "Mozilla/5.0 TradingBot/1.0"}

def get_exchange():
    global _exchange_inst
    with _exchange_lock:
        if _exchange_inst is None:
            cls = getattr(ccxt, EXCHANGE_NAME)
            _exchange_inst = cls({
                "apiKey": EXCHANGE_API_KEY,
                "secret": EXCHANGE_API_SECRET,
                "enableRateLimit": True,
                "options": {"defaultType": "future", "adjustForTimeDifference": True},
            })
            if USE_TESTNET:
                try: _exchange_inst.set_sandbox_mode(True)
                except Exception: pass
            try: _exchange_inst.load_markets()
            except Exception: pass
        return _exchange_inst

# ═══════════════════════════════════════════════════════════════
#  OHLCV — via ccxt (Binance Futures)
# ═══════════════════════════════════════════════════════════════
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4))
def fetch_ohlcv(symbol: str, timeframe: str = "1h", limit: int = 300) -> pd.DataFrame:
    key = f"{symbol}_{timeframe}_{limit}"
    if key in _ohlcv_cache:
        return _ohlcv_cache[key]
    ex  = get_exchange()
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not raw or len(raw) < 20:
        raise ValueError(f"Data tidak cukup: {symbol} [{timeframe}]")
    df = pd.DataFrame(raw, columns=["ts","Open","High","Low","Close","Volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    df.dropna(inplace=True)
    _ohlcv_cache[key] = df
    return df

def fetch_multi_timeframe(symbol: str) -> dict[str, pd.DataFrame]:
    """Ambil OHLCV untuk semua timeframe analisis sekaligus."""
    tf_config = {
        "1m":  300, "5m":  300, "15m": 300, "30m": 300,
        "1h":  300, "4h":  300, "1d":  365, "1w":  104, "1M": 48,
    }
    frames = {}
    for tf, limit in tf_config.items():
        try:
            frames[tf] = fetch_ohlcv(symbol, tf, limit)
        except Exception as e:
            console.print(f"[dim]  Skip {symbol}[{tf}]: {e}[/dim]")
    return frames

# ═══════════════════════════════════════════════════════════════
#  TICKER & METRICS — Binance public REST (lebih cepat dari ccxt)
# ═══════════════════════════════════════════════════════════════
def fetch_ticker_binance(symbol: str) -> dict:
    """Ambil ticker 24h dari Binance public REST."""
    if symbol in _ticker_cache:
        return _ticker_cache[symbol]
    clean = symbol.replace("/","")
    try:
        r = requests.get(BINANCE_TICKER_URL, params={"symbol": clean},
                         headers=HEADERS, timeout=6)
        if r.status_code == 200:
            d = r.json()
            t = {
                "price":         float(d.get("lastPrice",0)),
                "high_24h":      float(d.get("highPrice",0)),
                "low_24h":       float(d.get("lowPrice",0)),
                "change_24h_pct":float(d.get("priceChangePercent",0)),
                "volume_24h":    float(d.get("quoteVolume",0)),
                "count":         int(d.get("count",0)),
            }
            _ticker_cache[symbol] = t
            return t
    except Exception: pass
    # Fallback ccxt
    try:
        ex = get_exchange()
        td = ex.fetch_ticker(symbol)
        t  = {
            "price":         td.get("last",0) or 0,
            "high_24h":      td.get("high",0) or 0,
            "low_24h":       td.get("low",0) or 0,
            "change_24h_pct":td.get("percentage",0) or 0,
            "volume_24h":    td.get("quoteVolume",0) or 0,
            "count":         0,
        }
        _ticker_cache[symbol] = t
        return t
    except Exception:
        return {}

def fetch_funding_rate(symbol: str) -> float:
    clean = symbol.replace("/","")
    try:
        r = requests.get(BINANCE_FUNDING_URL, params={"symbol":clean},
                         headers=HEADERS, timeout=5)
        if r.status_code == 200:
            return float(r.json().get("lastFundingRate",0))
    except Exception: pass
    try:
        return float(get_exchange().fetch_funding_rate(symbol).get("fundingRate",0) or 0)
    except Exception: return 0.0

def fetch_open_interest(symbol: str) -> float:
    clean = symbol.replace("/","")
    try:
        r = requests.get(BINANCE_OI_URL, params={"symbol":clean},
                         headers=HEADERS, timeout=5)
        if r.status_code == 200:
            return float(r.json().get("openInterest",0))
    except Exception: pass
    return 0.0

def fetch_market_metrics(symbol: str) -> dict:
    key = f"metrics_{symbol}"
    if key in _metrics_cache:
        return _metrics_cache[key]
    ticker  = fetch_ticker_binance(symbol)
    funding = fetch_funding_rate(symbol)
    oi      = fetch_open_interest(symbol)
    m = {
        "symbol": symbol,
        "price":         ticker.get("price",0),
        "high_24h":      ticker.get("high_24h",0),
        "low_24h":       ticker.get("low_24h",0),
        "change_24h_pct":ticker.get("change_24h_pct",0),
        "volume_24h":    ticker.get("volume_24h",0),
        "funding_rate":  funding,
        "open_interest": oi,
        "timestamp":     datetime.now().isoformat(),
    }
    _metrics_cache[key] = m
    return m

# ═══════════════════════════════════════════════════════════════
#  ORDER BOOK & RECENT TRADES — Binance public (realtime)
# ═══════════════════════════════════════════════════════════════
def fetch_order_book(symbol: str, depth: int = 20) -> dict:
    """Order book realtime dari Binance public REST — tanpa auth."""
    key = f"ob_{symbol}"
    if key in _ob_cache:
        return _ob_cache[key]
    clean = symbol.replace("/","")
    try:
        r = requests.get(BINANCE_DEPTH_URL,
                         params={"symbol":clean,"limit":depth},
                         headers=HEADERS, timeout=5)
        if r.status_code == 200:
            d    = r.json()
            bids = [[float(x[0]),float(x[1])] for x in d.get("bids",[])]
            asks = [[float(x[0]),float(x[1])] for x in d.get("asks",[])]
            bid_vol = sum(b[1] for b in bids)
            ask_vol = sum(a[1] for a in asks)
            total   = bid_vol + ask_vol
            imb     = (bid_vol - ask_vol) / max(total, 1) * 100
            spread  = ((asks[0][0] - bids[0][0]) / bids[0][0] * 100) if bids and asks else 0
            result  = {
                "best_bid":       bids[0][0] if bids else 0,
                "best_ask":       asks[0][0] if asks else 0,
                "bid_volume":     round(bid_vol, 2),
                "ask_volume":     round(ask_vol, 2),
                "depth_imbalance":round(imb, 2),
                "spread_pct":     round(spread, 5),
            }
            _ob_cache[key] = result
            return result
    except Exception: pass
    return {}

def fetch_recent_trades(symbol: str, limit: int = 50) -> dict:
    """Recent trades dari Binance public — tanpa auth."""
    clean = symbol.replace("/","")
    try:
        r = requests.get(BINANCE_TRADES_URL,
                         params={"symbol":clean,"limit":limit},
                         headers=HEADERS, timeout=5)
        if r.status_code == 200:
            trades = r.json()
            buys   = sum(1 for t in trades if not t.get("isBuyerMaker"))
            sells  = len(trades) - buys
            return {
                "buy_ratio_pct":  round(buys/max(len(trades),1)*100, 1),
                "sell_ratio_pct": round(sells/max(len(trades),1)*100, 1),
                "volume_ratio":   1.0,
                "change_pct":     0.0,
                "sample":         len(trades),
            }
    except Exception: pass
    return {}

# ═══════════════════════════════════════════════════════════════
#  BATCH FETCH — semua pasangan paralel
# ═══════════════════════════════════════════════════════════════
def _fetch_one(symbol: str) -> tuple:
    try:
        frames  = fetch_multi_timeframe(symbol)
        metrics = fetch_market_metrics(symbol)
        ob      = fetch_order_book(symbol)
        trades  = fetch_recent_trades(symbol)
        if not frames or not any(tf in frames for tf in [PRIMARY_TIMEFRAME,"1h","1d"]):
            return symbol, None
        return symbol, {"dataframes":frames,"metrics":metrics,"order_book":ob,"trades":trades}
    except Exception as e:
        console.print(f"[red]Error {symbol}: {e}[/red]")
        return symbol, None

def fetch_all_market_data(symbols: list[str]) -> dict:
    results = {}
    with ThreadPoolExecutor(max_workers=min(10, len(symbols))) as pool:
        for sym, data in pool.map(_fetch_one, symbols):
            if data: results[sym] = data
    return results

async def fetch_all_market_data_async(symbols: list[str]) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fetch_all_market_data, symbols)

def fetch_current_prices(symbols: list[str]) -> dict[str, float]:
    prices = {}
    def _p(sym):
        try:
            if sym in _ticker_cache:
                return sym, _ticker_cache[sym].get("price",0)
            return sym, fetch_ticker_binance(sym).get("price",0)
        except Exception:
            return sym, 0.0
    with ThreadPoolExecutor(max_workers=min(8,len(symbols))) as pool:
        for sym, price in pool.map(_p, symbols):
            if price > 0: prices[sym] = price
    return prices

def get_market_status() -> str:
    return f"🟢 CRYPTO PASAR BUKA 24/7 — {datetime.utcnow().strftime('%H:%M:%S')} UTC"
