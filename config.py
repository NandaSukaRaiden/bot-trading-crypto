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
KENARI_API_KEY     = os.getenv("KENARI_API_KEY", "")        # kenari.id — 49 model via 1 key
KENARI_MODEL       = os.getenv("KENARI_MODEL", "laguna-s-2-1:free")  # model Kenari default (legacy)

# ── Model GRATIS Kenari — semua dipakai paralel sebagai ensemble ──
# Dipanggil sekaligus, hasilnya di-voting untuk keputusan final
KENARI_FREE_MODELS_RAW = os.getenv(
    "KENARI_FREE_MODELS",
    "laguna-s-2-1:free,deepseek-v4-flash:free,glm-4-7-flash:free,"
    "ling-3-0-flash:free,longcat-2-0:free,mimo-v2-5:free,"
    "nemotron-3-super-120b-a12b:free,nemotron-3-ultra-550b-a55b:free,"
    "step-3-7-flash:free"
)
KENARI_FREE_MODELS = [m.strip() for m in KENARI_FREE_MODELS_RAW.split(",") if m.strip()]
# Model utama Kenari (dipanggil pertama, langsung final jika berhasil)
KENARI_PRIMARY_MODEL = os.getenv("KENARI_PRIMARY_MODEL", "deepseek-v4-flash:free")

# ═══════════════════════════════════════════════════════════════
#  AI PROVIDER — SEMUA keputusan trading ada di tangan AI ini
# ═══════════════════════════════════════════════════════════════
# Pilih provider AI pengambil keputusan. Semua AI dipanggil via API
# yang key-nya diisi lewat .env. Tidak perlu SDK khusus untuk
# provider "custom" — cukup API yang kompatibel dengan OpenAI format
# (kenari.id, openrouter, groq, deepseek, vLLM, dll).
#
#   auto     → pakai SEMUA provider yang punya key (ensemble terbaik)
#   gemini   → hanya Google Gemini
#   openai   → hanya OpenAI
#   anthropic→ hanya Anthropic Claude
#   kenari   → hanya Kenari.id
#   custom   → provider OpenAI-compatible dari AI_BASE_URL + AI_API_KEY
AI_PROVIDER       = os.getenv("AI_PROVIDER", "auto")

# AI UTAMA (primary) — provider yang dipanggil LEBIH DULU dan langsung
# dijadikan keputusan final jika berhasil. Jika gagal, fallback ke AI lain.
#   kenari   → Kenari.id jadi pengambil keputusan utama
#   gemini/openai/anthropic/custom → pilih yang lain
#   (kosong) → semua provider dipanggil paralel (ensembel biasa)
AI_PRIMARY_PROVIDER = os.getenv("AI_PRIMARY_PROVIDER", "kenari")
AI_API_KEY        = os.getenv("AI_API_KEY", "")             # key provider custom
AI_BASE_URL       = os.getenv("AI_BASE_URL", "")            # contoh: https://openrouter.ai/api/v1
AI_MODEL          = os.getenv("AI_MODEL", "")               # contoh: deepseek/deepseek-chat
AI_FALLBACK_MODEL = os.getenv("AI_FALLBACK_MODEL", "")      # cadangan jika model utama gagal
AI_MAX_TOKENS     = int(os.getenv("AI_MAX_TOKENS", 4000))
AI_TEMPERATURE    = float(os.getenv("AI_TEMPERATURE", 0.05))

# ── SEMUA keputusan dari AI ──────────────────────────────────
# true  → AI menganalisis SEMUA pair di watchlist, tidak ada filter
#         teknikal yang menutup jalan sebelum AI melihat datanya.
# false → filter teknikal ringan dulu (hemat quota API).
AI_ANALYZE_ALL_PAIRS = os.getenv("AI_ANALYZE_ALL_PAIRS", "true").lower() == "true"

# AI juga yang memutuskan MANAJEMEN posisi terbuka:
# hold / close / partial / geser SL-TP (dipanggil tiap siklus analisis)
AI_MANAGE_POSITIONS       = os.getenv("AI_MANAGE_POSITIONS", "true").lower() == "true"
# Posisi hanya dianalisis AI jika pergerakan harga ≥ x% dari entry
# (hemat API — tidak memanggil AI kalau pasar diam)
AI_MANAGE_MIN_MOVE_PCT    = float(os.getenv("AI_MANAGE_MIN_MOVE_PCT", 0.35))

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
#  TRADING STYLE — scalper (default) | swing
# ═══════════════════════════════════════════════════════════════
# scalper → timeframe kecil (1m/5m/15m), leverage 20x+, holding menit-jam,
#           profit dari banyak trade kecil cepat (butuh berita/fundamental tetap)
# swing   → timeframe besar (1h/4h/1d), leverage rendah, holding 1-7 hari
TRADING_STYLE = os.getenv("TRADING_STYLE", "scalper").lower()

# Timeframe yang dipakai tergantung gaya trading
if TRADING_STYLE == "scalper":
    PRIMARY_TIMEFRAME  = os.getenv("PRIMARY_TIMEFRAME",  "5m")
    TREND_TIMEFRAME    = os.getenv("TREND_TIMEFRAME",    "1h")
    CONTEXT_TIMEFRAME  = os.getenv("CONTEXT_TIMEFRAME",  "4h")
else:
    PRIMARY_TIMEFRAME  = os.getenv("PRIMARY_TIMEFRAME",  "1h")
    TREND_TIMEFRAME    = os.getenv("TREND_TIMEFRAME",    "4h")
    CONTEXT_TIMEFRAME  = os.getenv("CONTEXT_TIMEFRAME",  "1d")

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
# BTC scalper: leverage 20x-25x, risiko per trade dikunci kecil (0.5%)
DEFAULT_LEVERAGE       = float(os.getenv("DEFAULT_LEVERAGE", 20))    # default 20x
MAX_LEVERAGE           = float(os.getenv("MAX_LEVERAGE", 25))         # hard cap 25x (BTC)
MIN_LEVERAGE           = 20                                            # minimum leverage BTC
MAX_RISK_PER_TRADE     = float(os.getenv("MAX_RISK_PER_TRADE", 0.5)) # 0.5% equity per trade
MAX_POSITIONS          = int(os.getenv("MAX_POSITIONS", 3))           # BTC-only: cukup 3
ANALYSIS_INTERVAL_MIN  = int(os.getenv("ANALYSIS_INTERVAL_MINUTES", 2))
MIN_AI_SCORE_TO_TRADE  = float(os.getenv("MIN_AI_SCORE_TO_TRADE", 65))
MIN_RRR                = float(os.getenv("MIN_RRR", 1.5))
MIN_SL_DISTANCE_PCT    = float(os.getenv("MIN_SL_DISTANCE_PCT", 0.1))
TAKER_FEE_RATE         = 0.0004   # 0.04% Binance Futures testnet

# Batas kerugian (membatasi drawdown — proteksi utama scalper)
DAILY_LOSS_LIMIT_PCT   = float(os.getenv("DAILY_LOSS_LIMIT_PCT", 2.0))
WEEKLY_LOSS_LIMIT_PCT  = float(os.getenv("WEEKLY_LOSS_LIMIT_PCT", 5.0))
MAX_DRAWDOWN_PCT       = float(os.getenv("MAX_DRAWDOWN_PCT", 10.0))

# Monitor posisi aktif — scalper butuh respons cepat (setiap 10 detik)
POSITION_MONITOR_SEC   = int(os.getenv("POSITION_MONITOR_SEC", 10))

# Timeframe analisis
PRIMARY_TIMEFRAME  = os.getenv("PRIMARY_TIMEFRAME",  PRIMARY_TIMEFRAME)
TREND_TIMEFRAME    = os.getenv("TREND_TIMEFRAME",    TREND_TIMEFRAME)
CONTEXT_TIMEFRAME  = os.getenv("CONTEXT_TIMEFRAME",  CONTEXT_TIMEFRAME)

# ═══════════════════════════════════════════════════════════════
#  WATCHLIST CRYPTO — BTC ONLY (Binance USDT-M Futures Testnet)
# ═══════════════════════════════════════════════════════════════
# Fokus BTC/USDT saja: likuiditas tertinggi, spread terkecil,
# leverage 20x-25x aman, data paling banyak untuk AI.
CRYPTO_WATCHLIST = [
    "BTC/USDT",
]

# CoinGecko ID per pasangan (untuk fundamental)
COINGECKO_IDS = {
    "BTC/USDT": "bitcoin",
}

# Leverage per simbol — BTC 25x (testnet)
# Leverage tinggi aman karena risiko dikunci via jarak SL kecil.
SYMBOL_MAX_LEVERAGE = {
    "BTC/USDT": 25,
}

# ═══════════════════════════════════════════════════════════════
#  AI MODEL CONFIG — Gemini PRIMARY
# ═══════════════════════════════════════════════════════════════
AI_PRIMARY = os.getenv("AI_PRIMARY", "kenari")

AI_MODELS = {
    "gemini": {
        "models":      [os.getenv("GEMINI_MODEL", ""),
                        "gemini-2.0-flash-lite",
                        "gemini-2.0-flash"],
        "max_tokens":  AI_MAX_TOKENS,
        "temperature": AI_TEMPERATURE,
        "description": "Google Gemini — AI utama (gratis free tier)",
    },
    "openai": {
        "model":       os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "fallback":    os.getenv("OPENAI_MODEL_FALLBACK", "gpt-4o-mini"),
        "max_tokens":  AI_MAX_TOKENS,
        "temperature": AI_TEMPERATURE,
        "description": "OpenAI — reasoning & fundamental",
    },
    "anthropic": {
        "model":       os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
        "fallback":    os.getenv("ANTHROPIC_MODEL_FALLBACK", "claude-3-haiku-20240307"),
        "max_tokens":  AI_MAX_TOKENS,
        "temperature": AI_TEMPERATURE,
        "description": "Claude — risk assessment konservatif",
    },
    "kenari": {
        "primary_model":    KENARI_PRIMARY_MODEL,
        "free_models":      KENARI_FREE_MODELS,
        "max_tokens":       AI_MAX_TOKENS,
        "temperature":      AI_TEMPERATURE,
        "base_url":         "https://kenari.id/v1",
        "description":      f"Kenari.id — {len(KENARI_FREE_MODELS)} free models ensemble",
    },
    "custom": {
        "model":       AI_MODEL,
        "fallback":    AI_FALLBACK_MODEL,
        "max_tokens":  AI_MAX_TOKENS,
        "temperature": AI_TEMPERATURE,
        "base_url":    AI_BASE_URL,
        "description": "Custom OpenAI-compatible provider dari .env",
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
