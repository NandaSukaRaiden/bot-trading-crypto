# 🤖 AI Crypto Futures Trading Bot

Bot trading crypto **Binance USDT-M Futures** bertenaga **Gemini 1.5 Pro** sebagai AI utama, dengan GPT-4o dan Claude 3.5 sebagai ensemble opsional. Bot beroperasi **sepenuhnya mandiri** seperti trader profesional.

---

## 🧠 Arsitektur

```
Binance Realtime ──┐
CoinGecko Fundamental ──┤
CryptoPanic + RSS News ──┤──→ Gemini 1.5 Pro ──┐
Fear & Greed Index ──┤       GPT-4o (opt) ──┤──→ Ensemble → Risk Manager → Execute
Makro Global (DXY,VIX) ──┤   Claude 3.5 (opt) ──┘
Chart 1m-1M (Binance) ──┘
```

---

## 📰 News: Semua Sumber TANPA LOGIN

| Sumber | Data | Auth |
|--------|------|------|
| CryptoPanic | News aggregator terbaik crypto | Free (no key needed) |
| CoinTelegraph RSS | Breaking news crypto | Tidak perlu |
| CoinDesk RSS | Institutional news | Tidak perlu |
| Decrypt RSS | DeFi & Web3 news | Tidak perlu |
| The Block RSS | On-chain & market data | Tidak perlu |
| BeInCrypto RSS | Asia crypto news | Tidak perlu |
| Reddit RSS | r/CryptoCurrency, r/Bitcoin | Tidak perlu |
| Google News RSS | Real-time keyword search | Tidak perlu |
| Fear & Greed Index | alternative.me API | Tidak perlu |
| CoinGecko API | Fundamental data | Tidak perlu |
| Binance Public API | Funding, OI, orderbook | Tidak perlu |
| Makro: DXY, VIX, SP500 | yfinance public | Tidak perlu |

---

## � Charts: Semua Timeframe dari Binance

| Timeframe | Deskripsi | Bars |
|-----------|-----------|------|
| 1m | 1 Menit (scalping) | 300 |
| 5m | 5 Menit | 300 |
| 15m | 15 Menit | 300 |
| 30m | 30 Menit | 300 |
| 1h | 1 Jam (primary) | 300 |
| 4h | 4 Jam (trend) | 300 |
| 1d | Harian (context) | 365 |
| 1w | Mingguan | 104 |
| 1M | Bulanan | 48 |

**Indikator di setiap chart:** EMA9, EMA21, SMA50, SMA200, Bollinger Bands, RSI, MACD

Buka di browser: `http://localhost:8008`

---

## 🤖 AI Bekerja Mandiri — Chain of Thought

Setiap siklus, AI melakukan:

1. **Baca tren utama** dari chart 4h dan 1d
2. **Konfirmasi** dengan chart 1h (multi-timeframe alignment)
3. **Analisis orderbook** — tekanan beli vs jual realtime
4. **Baca berita terbaru** — ada catalyst atau risiko?
5. **Cek Fear & Greed + Funding Rate** — pasar overcrowded?
6. **Hitung entry, SL (ATR-based), TP1, TP2** (min RRR 1:2)
7. **Sizing berbasis % risiko** — bukan % posisi
8. **Leverage aman** — SL harus hit sebelum likuidasi
9. **Tulis reasoning** 4-5 paragraf + trade plan

---

## �️ Risk Management (5 Lapis)

| Lapis | Perlindungan |
|-------|-------------|
| 1 | Daily loss 3% → stop trading hari itu |
| 2 | Weekly loss 7% → stop trading seminggu |
| 3 | Drawdown 15% → circuit breaker |
| 4 | AI voting (Gemini 2x bobot) minimal setuju |
| 5 | Trailing stop otomatis setelah profit >6% |

**Extra crypto protections:**
- Funding rate >+0.2% → tidak buka LONG baru
- Funding rate <-0.2% → tidak buka SHORT baru
- Fear & Greed >85 → tidak LONG (extreme greed)
- Fear & Greed <15 → tidak SHORT (extreme fear)
- Berita hack/SEC/ban → AVOID 24 jam

---

## 🚀 Quick Start

```batch
# 1. Setup (sekali)
setup.bat

# 2. Isi .env — minimal wajib:
GOOGLE_API_KEY=AIzaSy...   (dari aistudio.google.com — GRATIS)

# 3. Jalankan bot
venv\Scripts\activate
python trading_bot.py

# 4. Lihat charts di browser
python charts.py BTC/USDT

# 5. Monitor dashboard (terminal lain)
python dashboard.py
```

---

## 📁 Struktur File

```
├── trading_bot.py       ← Bot utama + orkestrator
├── ai_analyzer.py       ← Gemini + GPT + Claude (paralel)
├── news_fetcher.py      ← Semua news TANPA LOGIN
├── charts.py            ← Chart 1m-1M dari Binance + HTML gallery
├── data_fetcher.py      ← Binance realtime OHLCV + orderbook
├── technical_analyzer.py← RSI, MACD, ADX, BB, OBV, dll
├── risk_manager.py      ← Guardian + circuit breaker
├── portfolio.py         ← Paper/live P&L + trailing stop
├── dashboard.py         ← Terminal monitor realtime
├── notifier.py          ← Telegram alert
├── config.py            ← Semua konfigurasi
├── requirements.txt     ← Dependencies
├── setup.bat            ← Windows installer
└── .env                 ← API keys (jangan di-commit!)
```

---

## ⚠️ Disclaimer

Bot ini untuk **edukasi dan riset**. Trading futures dengan leverage mengandung risiko kehilangan seluruh modal. Gunakan **paper mode** minimal 1-2 bulan sebelum live. Keputusan investasi sepenuhnya tanggung jawab pengguna.
