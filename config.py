"""
config.py — Konfigurasi pusat AI Crypto Futures Trading Bot
Exchange   : Binance USDT-M Futures (long + short + leverage)
AI Primary : Google Gemini 1.5 Pro (wajib) + GPT-4o & Claude (ensemble opsional)
Data       : Binance realtime + CoinGecko fundamental + CryptoPanic news +
             Fear&Greed + Google News RSS + CoinDesk RSS
Charts     : mplfinance — semua timeframe 1m s/d 1M disimpan + galeri browser
"""
import os, sys
from dotenv import load_dotenv
load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ═══════════════════════════════════════════════════════════════
#  AI API KEYS
# ═══════════════════════════════════════════════════════════════
GOOGLE_API_KEY     = os.getenv("GOOGLE_API_KEY", "")       # ← AI UTAMA wajib isi
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")       # opsional ensemble
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")    # opsional ensemble

# ═══════════════════════════════════════════════════════════════
#  EXCHANGE — Binance USDT-M Futures
# ═══════════════════════════════════════════════════════════════
# ── Binance Exchange ─────────────────────────────────────────
# Untuk live trading isi dari: binance.com → API Management
# WAJIB: aktifkan "Enable Futures", JANGAN aktifkan "Enable Withdrawals"
EXCHANGE_NAME       = os.getenv("EXCHANGE_NAME", "binanceusdm")
EXCHANGE_API_KEY    = os.getenv("EXCHANGE_API_KEY", "")
EXCHANGE_API_SECRET = os.getenv("EXCHANGE_API_SECRET", "")
USE_TESTNET         = os.getenv("USE_TESTNET", "false").lower() == "true"

# PENTING: Mulai dari "paper", ganti ke "live" HANYA setelah
# paper trading profit konsisten minimal 2 minggu
TRADING_MODE        = os.getenv("TRADING_MODE", "paper")   # paper | live

# ═══════════════════════════════════════════════════════════════
#  DATA SOURCES — Bloomberg-grade, semua gratis/free-tier
# ═══════════════════════════════════════════════════════════════
# CoinGecko — fundamental crypto (market cap, supply, dev activity)
COINGECKO_API_KEY  = os.getenv("COINGECKO_API_KEY", "")    # opsional, tanpa key = free tier
COINGECKO_BASE     = "https://api.coingecko.com/api/v3"
COINGECKO_PRO_BASE = "https://pro-api.coingecko.com/api/v3"

# CryptoPanic — news aggregator crypto terbaik (free tier tersedia)
CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "")  # opsional
CRYPTOPANIC_BASE   = "https://cryptopanic.com/api/v1/posts"

# Fear & Greed Index (alternative.me) — gratis tanpa key
FEAR_GREED_URL     = "https://api.alternative.me/fng/?limit=7"

# Binance market data (public, tanpa key)
BINANCE_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
BINANCE_TICKER_URL = "https://fapi.binance.com/fapi/v1/ticker/24hr"
BINANCE_DEPTH_URL  = "https://fapi.binance.com/fapi/v1/depth"
BINANCE_TRADES_URL = "https://fapi.binance.com/fapi/v1/trades"
BINANCE_OI_URL     = "https://fapi.binance.com/fapi/v1/openInterest"
BINANCE_FUNDING_URL= "https://fapi.binance.com/fapi/v1/premiumIndex"

# RSS Berita Crypto (tanpa key)
CRYPTO_NEWS_RSS = [
    "https://cointelegraph.com/rss",
    "https://coindesk.com/arc/outboundfeeds/rss/",
    "https://decrypt.co/feed",
    "https://www.theblock.co/rss.xml",
    "https://cryptopotato.com/feed/",
    "https://beincrypto.com/feed/",
    "https://ambcrypto.com/feed/",
]
GOOGLE_NEWS_CRYPTO = "https://news.google.com/rss/search?q={query}+crypto+price&hl=en&gl=US&ceid=US:en"

NEWS_MAX_ITEMS     = int(os.getenv("NEWS_MAX_ITEMS", 15))
NEWS_TTL_MIN       = int(os.getenv("NEWS_TTL_MIN", 10))    # cache berita 10 menit
INCLUDE_NEWS_IN_AI = os.getenv("INCLUDE_NEWS_IN_AI", "true").lower() == "true"

# ═══════════════════════════════════════════════════════════════
#  CHART CONFIG
# ═══════════════════════════════════════════════════════════════
CHART_DIR         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
CHART_SERVER_PORT = int(os.getenv("CHART_SERVER_PORT", 8008))
# Semua timeframe yang digenerate (Binance interval string)
CHART_TIMEFRAMES  = ["1m","5m","15m","30m","1h","4h","1d","1w","1M"]

# ═══════════════════════════════════════════════════════════════
#  TRADING PARAMETERS
# ═══════════════════════════════════════════════════════════════
INITIAL_CAPITAL_USDT   = float(os.getenv("INITIAL_CAPITAL_USDT", 1000))
DEFAULT_LEVERAGE       = float(os.getenv("DEFAULT_LEVERAGE", 3))      # konservatif
MAX_LEVERAGE           = float(os.getenv("MAX_LEVERAGE", 10))          # hard cap 10x
MAX_RISK_PER_TRADE     = float(os.getenv("MAX_RISK_PER_TRADE", 1.0))  # 1% equity per trade
MAX_POSITIONS          = int(os.getenv("MAX_POSITIONS", 4))            # fokus, tidak scatter
ANALYSIS_INTERVAL_MIN  = int(os.getenv("ANALYSIS_INTERVAL_MINUTES", 15))
MIN_AI_SCORE_TO_TRADE  = float(os.getenv("MIN_AI_SCORE_TO_TRADE", 72)) # threshold tinggi
TAKER_FEE_RATE         = 0.0004   # 0.04% Binance Futures

# Monitor posisi aktif lebih sering dari siklus analisis
POSITION_MONITOR_SEC   = int(os.getenv("POSITION_MONITOR_SEC", 30))   # setiap 30 detik

# Timeframe analisis
PRIMARY_TIMEFRAME  = os.getenv("PRIMARY_TIMEFRAME",  "1h")
TREND_TIMEFRAME    = os.getenv("TREND_TIMEFRAME",    "4h")
CONTEXT_TIMEFRAME  = os.getenv("CONTEXT_TIMEFRAME",  "1d")

# ═══════════════════════════════════════════════════════════════
#  WATCHLIST CRYPTO (Binance USDT-M Futures)
# ═══════════════════════════════════════════════════════════════
CRYPTO_WATCHLIST = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "LINK/USDT", "DOT/USDT",
    "MATIC/USDT", "LTC/USDT", "BCH/USDT", "ATOM/USDT", "NEAR/USDT",
    "APT/USDT", "ARB/USDT", "OP/USDT", "SUI/USDT", "INJ/USDT",
]

# CoinGecko ID per pasangan (untuk fundamental)
COINGECKO_IDS = {
    "BTC/USDT": "bitcoin",     "ETH/USDT": "ethereum",
    "BNB/USDT": "binancecoin", "SOL/USDT": "solana",
    "XRP/USDT": "ripple",      "ADA/USDT": "cardano",
    "AVAX/USDT":"avalanche-2", "DOGE/USDT":"dogecoin",
    "LINK/USDT":"chainlink",   "DOT/USDT": "polkadot",
    "MATIC/USDT":"matic-network","LTC/USDT":"litecoin",
    "BCH/USDT": "bitcoin-cash","ATOM/USDT":"cosmos",
    "NEAR/USDT":"near",        "APT/USDT": "aptos",
    "ARB/USDT": "arbitrum",    "OP/USDT":  "optimism",
    "SUI/USDT": "sui",         "INJ/USDT": "injective-protocol",
}

# Leverage maksimum per simbol — KONSERVATIF untuk profitabilitas konsisten
# Rumus: semakin liquid & stabil, semakin tinggi leverage yang AMAN
SYMBOL_MAX_LEVERAGE = {
    # BTC & ETH — liquid tertinggi, leverage lebih longgar
    "BTC/USDT": 10, "ETH/USDT": 10,
    # Large cap — cukup liquid
    "BNB/USDT": 7,  "SOL/USDT": 7,  "XRP/USDT": 7,
    # Mid cap — volatil, leverage rendah
    "ADA/USDT": 5,  "AVAX/USDT": 5, "DOGE/USDT": 5,
    "LINK/USDT": 5, "DOT/USDT":  5, "LTC/USDT":  5,
    "MATIC/USDT":5, "BCH/USDT":  5, "ATOM/USDT": 5,
    # Small-mid cap — paling berisiko, leverage minimal
    "NEAR/USDT": 3, "APT/USDT":  3, "ARB/USDT":  3,
    "OP/USDT":   3, "SUI/USDT":  3, "INJ/USDT":  3,
}

# ═══════════════════════════════════════════════════════════════
#  AI MODEL CONFIG — Gemini PRIMARY
# ═══════════════════════════════════════════════════════════════
AI_PRIMARY = "gemini"

AI_MODELS = {
    "gemini": {
        "model":       "gemini-2.0-flash-lite",    # quota lebih besar untuk free tier
        "fallback":    "gemini-2.0-flash",          # fallback jika lite habis
        "max_tokens":  3500,
        "temperature": 0.05,
        "description": "Gemini 2.0 Flash Lite — cepat, hemat quota, akurat",
    },
    "openai": {
        "model":       "gpt-4o",
        "fallback":    "gpt-4o-mini",
        "max_tokens":  2500,
        "temperature": 0.05,
        "description": "GPT-4o — reasoning & fundamental",
    },
    "anthropic": {
        "model":       "claude-3-5-sonnet-20241022",
        "fallback":    "claude-3-haiku-20240307",
        "max_tokens":  2500,
        "temperature": 0.05,
        "description": "Claude — risk assessment konservatif",
    },
}

# ═══════════════════════════════════════════════════════════════
#  TECHNICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════
TA_CONFIG = {
    "rsi_period":        14,
    "rsi_ob":            70,
    "rsi_os":            30,
    "macd_fast":         12,
    "macd_slow":         26,
    "macd_signal":       9,
    "bb_period":         20,
    "bb_std":            2,
    "stoch_k":           14,
    "stoch_d":           3,
    "atr_period":        14,
    "adx_period":        14,
    "volume_sma_period": 20,     # ← key yang dipakai technical_analyzer
    "ema_periods":       [9, 21, 50, 200],
}

# ═══════════════════════════════════════════════════════════════
#  SIGNAL LEVELS & WARNA
# ═══════════════════════════════════════════════════════════════
SIGNAL_LEVELS = {
    "STRONG_LONG":  (75, 100),
    "LONG":         (62, 75),
    "NEUTRAL":      (38, 62),
    "SHORT":        (25, 38),
    "STRONG_SHORT": (0,  25),
}
COLORS = {
    "STRONG_LONG":  "bold green",
    "LONG":         "green",
    "NEUTRAL":      "yellow",
    "SHORT":        "red",
    "STRONG_SHORT": "bold red",
}

# ═══════════════════════════════════════════════════════════════
#  TELEGRAM
# ═══════════════════════════════════════════════════════════════
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
