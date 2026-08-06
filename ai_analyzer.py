"""
ai_analyzer.py — AI Trader Mandiri: SEMUA keputusan trading ada di tangan AI.
AI dipanggil via API yang key-nya diisi di .env. Mendukung banyak provider:

  • Kenari.id FREE ENSEMBLE — 9 model gratis dipanggil PARALEL sekaligus,
    hasilnya di-voting untuk keputusan final (akurasi jauh lebih tinggi)
  • Gemini (google-genai SDK)          — ensemble opsional
  • OpenAI (GPT)                       — ensemble opsional
  • Anthropic (Claude)                 — ensemble opsional
  • Custom (AI_BASE_URL + AI_API_KEY)  — provider OpenAI-compatible apa pun

Fitur:
  1. Analisis ENTRY sangat rinci (multi-TF + orderbook + news + fundamental +
     makro + chart) dengan scoring per komponen.
  2. Manajemen POSISI TERBUKA oleh AI (hold/close/partial/trail/geser TP).
  3. AI UTAMA (KENARI_PRIMARY_MODEL) — dipanggil lebih dulu, jika berhasil
     langsung final; lalu sisa model gratis dipanggil paralel sebagai verifikasi.
  4. Fallback ke teknikal jika semua AI gagal.
  5. Semua 9 model gratis Kenari berjalan bersamaan → voting mayoritas.
"""
import asyncio, json, re
from collections import Counter
from datetime import datetime
from typing import Optional

# ── google-genai SDK (v2.x) ───────────────────────────────────
try:
    from google import genai as google_genai
    from google.genai import types as genai_types
    GENAI_NEW = True
except ImportError:
    google_genai = None
    genai_types  = None
    GENAI_NEW    = False

# ── OpenAI SDK (untuk GPT + Kenari + Custom) ──────────────────
try:
    import openai
except ImportError:
    openai = None

# ── Anthropic SDK (Claude) ────────────────────────────────────
try:
    import anthropic
except ImportError:
    anthropic = None

from config import (
    GOOGLE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY,
    KENARI_API_KEY, AI_PROVIDER, AI_PRIMARY_PROVIDER,
    AI_API_KEY, AI_BASE_URL, AI_MODEL, AI_FALLBACK_MODEL, AI_MODELS,
    DEFAULT_LEVERAGE, MAX_LEVERAGE, MIN_RRR,
    KENARI_FREE_MODELS, KENARI_PRIMARY_MODEL,
)
from rich.console import Console
console = Console()


def _split_keys(raw) -> list:
    """Pecah nilai env yang berisi banyak key dipisah koma."""
    if not raw:
        return []
    if isinstance(raw, list):
        keys = [str(k).strip() for k in raw]
    else:
        keys = [k.strip() for k in str(raw).split(",")]
    return [k for k in keys if k]


class _KeyRateLimited(Exception):
    """Sinyal internal: key ini kena rate limit → rotasi ke key berikutnya."""


# Indeks rotasi per provider
_rot = {"gemini": 0, "openai": 0, "anthropic": 0, "kenari": 0, "custom": 0}

# ── Init AI Clients ───────────────────────────────────────────
_gemini_clients = []
_gemini_key_tails = []
if GENAI_NEW:
    for k in _split_keys(GOOGLE_API_KEY):
        try:
            _gemini_clients.append(google_genai.Client(api_key=k))
            _gemini_key_tails.append(str(k)[-6:])
        except Exception as _e:
            console.print(f"[yellow]Gemini init error (key {str(k)[:8]}...): {_e}[/yellow]")

openai_clients = []
if openai:
    for k in _split_keys(OPENAI_API_KEY):
        try:
            openai_clients.append(openai.AsyncOpenAI(api_key=k))
        except Exception:
            pass

anthropic_clients = []
if anthropic:
    for k in _split_keys(ANTHROPIC_API_KEY):
        try:
            anthropic_clients.append(anthropic.AsyncAnthropic(api_key=k))
        except Exception:
            pass

def _make_openai_compatible(api_key: str, base_url: str):
    if not (openai and api_key and base_url):
        return None
    try:
        return openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
    except Exception:
        return None

# ── Kenari clients — satu client per key ─────────────────────
# (biasanya 1 key, tapi bisa multi-key via koma di .env)
_kenari_clients = [
    c for c in (_make_openai_compatible(k, "https://kenari.id/v1")
                for k in _split_keys(KENARI_API_KEY)) if c
]

_custom_clients = []
if AI_API_KEY and AI_BASE_URL and AI_MODEL:
    for k in _split_keys(AI_API_KEY):
        c = _make_openai_compatible(k, AI_BASE_URL)
        if c:
            _custom_clients.append(c)

GEMINI_OK = bool(_gemini_clients)
GPT_OK    = bool(openai_clients)
CLAUDE_OK = bool(anthropic_clients)
KENARI_OK = bool(_kenari_clients)
CUSTOM_OK = bool(_custom_clients)


def _selected_providers() -> dict:
    """
    Provider mana yang dipakai berdasarkan AI_PROVIDER di .env:
      auto   → semua provider yang punya key
      kenari → hanya Kenari (direkomendasikan — ensemble 9 model gratis)
      X      → hanya provider X
    """
    p = (AI_PROVIDER or "auto").lower()
    if p == "auto":
        return {
            "gemini":    GEMINI_OK,
            "openai":    GPT_OK,
            "anthropic": CLAUDE_OK,
            "kenari":    KENARI_OK,
            "custom":    CUSTOM_OK,
        }
    return {
        "gemini":    p == "gemini" and GEMINI_OK,
        "openai":    p == "openai" and GPT_OK,
        "anthropic": p == "anthropic" and CLAUDE_OK,
        "kenari":    p == "kenari" and KENARI_OK,
        "custom":    p == "custom" and CUSTOM_OK,
    }


AI_AVAILABLE = any(_selected_providers().values())


def get_active_providers() -> list[str]:
    """Nama provider yang aktif untuk banner startup."""
    primary = (AI_PRIMARY_PROVIDER or "").lower()
    out = []
    sel = _selected_providers()

    if sel.get("kenari") and KENARI_OK:
        n_models = len(KENARI_FREE_MODELS)
        n_keys   = len(_kenari_clients)
        label = f"Kenari FREE ×{n_models} models ×{n_keys} key"
        if primary == "kenari":
            label = f"★ {label} (PRIMARY={KENARI_PRIMARY_MODEL})"
        out.append(label)
    if sel.get("gemini") and GEMINI_OK:
        out.append(f"Gemini×{len(_gemini_clients)}")
    if sel.get("openai") and GPT_OK:
        out.append(f"GPT×{len(openai_clients)}")
    if sel.get("anthropic") and CLAUDE_OK:
        out.append(f"Claude×{len(anthropic_clients)}")
    if sel.get("custom") and CUSTOM_OK:
        out.append(f"Custom({AI_MODEL})×{len(_custom_clients)}")
    return out


# ═══════════════════════════════════════════════════════════════
#  SYSTEM PROMPT — AI Trader Profesional Mandiri (ENTRY)
# ═══════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """
Kamu adalah AI Crypto Futures SCALPER kelas dunia. Kamu melakukan 100+ trade
per minggu selama bertahun-tahun dengan leverage TINGGI (20x+) dan kamu TIDAK
pernah blown account. Rahasiamu: leverage tinggi untuk memperbesar notional,
TAPI risiko per trade dikunci kecil (0.5% equity) lewat jarak SL yang sangat
ketat. Kamu adalah pengambil keputusan FINAL dan SATU-SATUNYA untuk semua trade.

Kamu TETAP membaca berita, fundamental, dan konteks pasar secara menyeluruh
sebelum menembak — scalper sejati tidak menembak buta. Berita besar (CPI/FOMC/
hack/ETF) bisa membalikkan arah dalam hitungan detik.

═══════════════════════════════════════════════════════
GAYA TRADING KAMU (SCALPING):
═══════════════════════════════════════════════════════
• Timeframe entry : 1m/5m — sinyal muncul, langsung eksekusi cepat.
• Konfirmasi      : 15m — arah harus sejalan dengan 5m.
• Tren utama      : 1h — JANGAN pernah melawan tren 1h untuk scalping.
• Konteks makro   : 4h/1d — hanya untuk menilai kondisi pasar, bukan entry.
• Holding         : menit sampai beberapa jam (scalp). Jangan tahan berhari-hari.
• Target profit   : 0.5% - 2.5% per trade — kecil, cepat, sering.
• Stop loss       : 0.3% - 1.0% — TIPIS dan WAJIB di level valid (ATR/swing).
• Leverage        : 20x - 25x. Leverage memperbesar notional, BUKAN risiko.
• Risiko per trade: 0.5% equity. HITUNG leverage dari jarak SL agar SL
  selalu menang sebelum likuidasi.

═══════════════════════════════════════════════════════
SISTEM SCORING — 5 FAKTOR CONFLUENCE (WAJIB HITUNG DULU):
═══════════════════════════════════════════════════════
1. TREND (bobot 25%)  — 5m/15m searah dengan 1h. Jika 1h berlawanan → HOLD.
2. MOMENTUM (bobot 20%) — RSI + MACD histogram + EMA alignment searah.
3. STRUKTUR HARGA (bobot 20%) — entry di dekat support/resistance VWAP/volume
   profile PoC. Beli di support, jual di resistance — BUKAN di tengah range.
4. VOLUME (bobot 20%) — volume breakout > 1.5x rata-rata = WAJIB untuk scalper.
   Scalping tanpa volume = sinyal palsu.
5. KONTEKS PASAR (bobot 15%) — Fear&Greed, funding rate, berita, fundamental,
   makro, orderbook, tape tidak berlawanan dengan arah trade.

Hitung kelima skor komponen (0-100) secara eksplisit dan isi di JSON:
trend_score, momentum_score, structure_score, volume_score, context_score.
Confluence score = rata-rata tertimbang dari kelima komponen.
Jika confluence < 60 → HOLD atau AVOID.

═══════════════════════════════════════════════════════
ATURAN ENTRY YANG TIDAK BISA DILANGGAR (SCALPER):
═══════════════════════════════════════════════════════
1. RRR minimum 1:1.5 — jika TP1 tidak bisa 1.5x jarak SL, SKIP.
2. SL WAJIB di bawah/atas level support/resistance VALID, minimal ATR × 1.2
   dari entry. Scalping SL TIPIS: 0.3% - 1.0%. JANGAN SL lebih jauh 1.5%.
3. Leverage 20x-25x HANYA jika jarak SL cukup tipis. Semakin lebar SL,
   semakin kecil leverage yang aman (SL harus menang sebelum likuidasi).
4. Funding rate:
   • > +0.10% → hindari LONG (overcrowded, risiko squeeze)
   • < -0.10% → hindari SHORT (overcrowded, risiko squeeze ke atas)
5. Fear & Greed:
   • > 80 (Extreme Greed) → TIDAK buka LONG baru kecuali breakout ATH baru
   • < 20 (Extreme Fear) → TIDAK buka SHORT baru kecuali breakdown support mayor
6. Berita negatif BERAT (hack > $10M, exchange collapse, ban negara besar)
   → AVOID semua trade 24 jam. Berita ekonomi terjadwal besar (CPI/FOMC/NFP)
   → jangan buka posisi baru 30 menit sebelum, kecilkan size setelah.
7. TF conflict (5m vs 1h berlawanan) → HOLD.
8. Volume ratio < 0.7x rata-rata → SKIP, sinyal tidak valid tanpa volume.
9. Volatilitas (ATR%):
   • ATR% > 1.5% (timeframe kecil) → leverage turunkan ke 10x, size 75%
   • ATR% > 3%  → leverage turunkan ke 5x, size 50%
   • ATR% > 5%  → AVOID scalping (pasar tidak terkendali untuk SL tipis)
10. Jangan buka posisi baru jika sudah ada MAX_POSITIONS posisi.
11. TIDAK pernah average down / menambah posisi yang sedang rugi.
12. Sudah ada posisi di symbol yang sama → JANGAN buka lagi.
13. Spread/orderbook: jika spread > 0.05% atau tekanan orderbook berlawanan
    dengan arah entry → skip. Slippage adalah musuh scalper.

═══════════════════════════════════════════════════════
MANAJEMEN POSISI AKTIF (SCALP):
═══════════════════════════════════════════════════════
• TP1 (50% posisi): 2x jarak SL — ambil profit, geser SL ke breakeven.
• TP2 (50% sisa)  : 3x jarak SL atau sinyal reversal — trailing stop.
• Jika profit sudah > 1.5% dan momentum melemah → TUTUP, jangan serakah.
• Jika trade berjalan melawan 0.5x jarak SL dan invalidation terpenuhi
  → TUTUP lebih awal, jangan tunggu SL.

═══════════════════════════════════════════════════════
KAPAN BILANG HOLD (LEBIH SERING DARI LONG/SHORT):
═══════════════════════════════════════════════════════
• Confluence < 60%          • Volume tidak ada / range ketat
• Harga di tengah range     • Berita campur aduk
• Funding mendekati threshold • Sudah ada posisi di coin ini
• Spread lebar / ATR% terlalu tinggi untuk SL tipis

TAPI INGAT: Jangan selalu HOLD. Jika confluence ≥ 60, volume kuat, RRR ≥ 1.5,
dan SL tipis valid — AMBIL TRADE dengan percaya diri dan eksekusi cepat.
Scalper menghasilkan uang dari kecepatan + konsistensi setup, bukan dari takut.

═══════════════════════════════════════════════════════
CHAIN OF THOUGHT (Tulis berurutan di chain_of_thought):
═══════════════════════════════════════════════════════
1. [TREN]  Apa kata chart 1h dan 15m? Bullish/bearish/ranging?
2. [STRUKTUR] Di mana harga vs support/resistance, VWAP, volume profile PoC?
3. [MOMENTUM] RSI, MACD, EMA, Stochastic — konfirmasi atau divergen?
4. [VOLUME] Apakah volume + OBV + CMF mendukung? (WAJIB untuk scalper)
5. [KONTEKS] Fear&Greed, funding, berita, fundamental, makro, orderbook, tape.
6. [SCORE PER KOMPONEN] trend/momentum/structure/volume/context → confluence.
7. [KEPUTUSAN] LONG/SHORT/HOLD/AVOID berdasarkan skor.
8. [TRADE PLAN] Entry di ... → SL tipis di ... → TP1 ... TP2 ...
9. [SIZING] Leverage berapa? SL berapa %? Kenapa aman mengingat ATR%?
10. [RISIKO] Apa yang bisa membuat trade ini salah? Failure modes?

═══════════════════════════════════════════════════════
FORMAT OUTPUT (WAJIB JSON VALID, TIDAK ADA TEKS LAIN):
═══════════════════════════════════════════════════════
{
  "action": "LONG" | "SHORT" | "HOLD" | "AVOID",
  "confidence": 0-100,
  "risk_level": "LOW" | "MEDIUM" | "HIGH" | "VERY_HIGH",
  "confluence_score": 0-100,
  "trend_score": 0-100,
  "momentum_score": 0-100,
  "structure_score": 0-100,
  "volume_score": 0-100,
  "context_score": 0-100,
  "entry_price": number,
  "stop_loss": number,
  "stop_loss_reason": "mengapa SL di level ini (support/ATR/dll)",
  "take_profit_1": number,
  "take_profit_2": number,
  "take_profit_3": number,
  "risk_reward_ratio": number,
  "position_size_pct": number,
  "leverage": number,
  "leverage_reason": "mengapa leverage ini aman & SL menang sebelum likuidasi",
  "holding_period": "scalp_menit" | "scalp_jam" | "intraday",
  "overall_score": 0-100,
  "technical_score": 0-100,
  "fundamental_score": 0-100,
  "sentiment_score": 0-100,
  "macro_score": 0-100,
  "suggested_risk_multiplier": 0.5-1.5,
  "funding_analysis": "analisis funding rate & implikasinya",
  "volume_analysis": "analisis volume, OBV, CMF & implikasinya",
  "support_resistance_analysis": "analisis level SR & di mana entry terbaik",
  "chain_of_thought": "reasoning 10 langkah seperti di atas",
  "key_risks": ["risiko spesifik 1", "risiko 2", "risiko 3"],
  "failure_modes": ["cara paling mungkin trade ini rugi"],
  "invalidation": "kondisi yang akan membuat analisis ini salah",
  "catalysts": ["katalis positif konkret 1", "katalis 2"],
  "trade_plan": "entry X → SL Y (-Z%) → TP1 A (+B%) → TP2 C (+D%)",
  "verdict": "keputusan final dalam 1 kalimat tegas"
}

PENTING:
- Jika confluence_score < 60 → action WAJIB "HOLD" atau "AVOID"
- Jika risk_level = "VERY_HIGH" → action WAJIB "AVOID"
- leverage maks 25x untuk BTC/ETH, 20x untuk altcoin
- position_size_pct TIDAK BOLEH lebih dari 1% equity (scalper)
- stop_loss untuk LONG WAJIB lebih kecil dari entry_price
- stop_loss untuk SHORT WAJIB lebih besar dari entry_price
- SL tipis (0.3-1.0%) — leverage harus dihitung agar SL menang sebelum likuidasi
- leverage & size harus menyesuaikan ATR% (lihat aturan 9)
- isi kelima skor komponen secara jujur — jangan asal 50
- news & fundamental TETAP dianalisis di context_score — jangan pernah abaikan
- Beri trade_plan yang konkret dengan angka, bukan kalimat umum
"""


# ═══════════════════════════════════════════════════════════════
#  SYSTEM PROMPT — AI Portfolio Manager (POSISI TERBUKA)
# ═══════════════════════════════════════════════════════════════
MANAGEMENT_SYSTEM_PROMPT = """
Kamu adalah AI Portfolio Manager SCALPER crypto futures kelas dunia. Tugasmu
MENGELOLA posisi yang SUDAH terbuka di portfolio — bukan membuka posisi baru.

FILOSOFI MANAJEMEN SCALP:
• "Cut losses short, let winners run." — scalper tidak pernah menunggu SL besar.
• SL HANYA boleh digeser searah profit (LOCK profit), TIDAK PERNAH menjauh.
• Posisi yang profit ≥ 1.5% dan momentum melemah → ambil profit, jangan serakah.
• Jika trade melawan rencana dan invalidation terpenuhi → CLOSE lebih awal.
• Jangan pernah average down posisi yang rugi.
• ADD hanya jika sudah profit > 2% DAN sinyal baru sangat kuat (confluence ≥ 70).
• Berita negatif berat / pasar akan rilis data ekonomi besar → tutup posisi.
• Data lebih penting dari harapan.

INPUT per posisi: harga entry, harga sekarang, P&L%, SL, TP1, TP2, liq,
leverage, jam holding, dan analisis teknikal terbaru + berita + sentimen pasar.

═══════════════════════════════════════════════════════
FORMAT OUTPUT (WAJIB JSON VALID):
═══════════════════════════════════════════════════════
{
  "market_assessment": "penilaian kondisi pasar keseluruhan 1-2 kalimat",
  "recommended_risk_multiplier": 0.5-1.5,
  "positions": [
    {
      "symbol": "BTC/USDT",
      "action": "HOLD" | "CLOSE" | "PARTIAL_CLOSE" | "TRAIL_STOP" | "MOVE_TP" | "ADD",
      "reason": "alasan berbasis data (max 200 char)",
      "new_stop_loss": null,
      "new_take_profit": null,
      "close_fraction": 0.0,
      "urgency": "LOW" | "MEDIUM" | "HIGH"
    }
  ]
}

ATURAN AKSI:
- CLOSE          → tutup 100%. Gunakan jika invalidation terpenuhi, berita
                   negatif berat, tren berbalik kuat, atau risiko sangat tinggi.
- PARTIAL_CLOSE  → tutup close_fraction (misal 0.5). Gunakan jika sudah profit
                   dan momentum mulai lemah, atau menjelang berita besar.
- TRAIL_STOP     → geser SL mengunci profit. new_stop_loss HARUS lebih baik:
                   LONG: new_stop_loss > stop_loss lama.
                   SHORT: new_stop_loss < stop_loss lama.
- MOVE_TP        → geser TP lebih jauh searah profit bila tren sangat kuat.
- HOLD           → biarkan berjalan, tidak ada aksi.
- ADD            → hanya jika profit > 2% dan confluence baru ≥ 70.

Keselamatan:
- JANGAN rekomendasikan new_stop_loss yang MERUGIKAN (menjauh dari profit).
- JANGAN menaikkan leverage atau ukuran posisi saat rugi.
- Untuk scalper: posisi yang tidak bergerak sesuai rencana dalam 1-2 jam
  sebaiknya ditutup (waktu adalah biaya).
- recommended_risk_multiplier = 1.0 artinya risk normal; 0.5 artinya pasar
  berbahaya (kecilkan semua posisi baru); 1.5 artinya tren sangat jelas.
"""


# ═══════════════════════════════════════════════════════════════
#  BUILD PROMPT — ENTRY (semua data pasar ke AI)
# ═══════════════════════════════════════════════════════════════
def build_prompt(
    symbol: str,
    technical: dict,
    metrics: dict,
    portfolio_ctx: dict,
    market_context: dict,    # dari news_fetcher.get_full_market_context
    order_book: dict = None,
    recent_trades: dict = None,
    chart_paths: dict = None,
) -> str:
    tf = technical.get("per_timeframe", {})
    tf_text = ""
    for label, r in tf.items():
        if isinstance(r, dict):
            e9, e21, e50 = r.get("ema_9", 0), r.get("ema_21", 0), r.get("ema_50", 0)
            if e9 and e9 > e21 > e50:
                ema_align = "9>21>50 (bull)"
            elif e9 and e9 < e21 < e50:
                ema_align = "9<21<50 (bear)"
            else:
                ema_align = "mixed"
            tf_text += (f"  [{label}] score={r.get('score',0):+.0f} | dir={r.get('direction_bias','?')} | "
                        f"RSI={r.get('rsi',50):.1f} | RSI7={r.get('rsi7',50):.1f} | ADX={r.get('adx',0):.1f} | "
                        f"ATR%={r.get('atr_pct',0):.2f}% | vol={r.get('volume_ratio',1):.2f}x | "
                        f"BB%b={r.get('bb_pb',0.5):.2f} | Stoch={r.get('stoch_k',50):.0f} | "
                        f"CMF={r.get('cmf',0):.2f} | EMA:{ema_align} | "
                        f"ret1/5/20={r.get('return_1',0):+.2f}/{r.get('return_5',0):+.2f}/{r.get('return_20',0):+.2f}%\n")

    sr   = technical.get("support_resistance", {})
    sigs = "\n".join(f"  • {s}" for s in technical.get("signals_primary", [])[:25])

    # Level-level penting untuk keputusan SL/TP
    levels = []
    for key, label in [("ema9","EMA9"),("ema21","EMA21"),("ema50","EMA50"),
                       ("ema200","EMA200"),("vwap","VWAP"),("vp_poc","VP PoC"),
                       ("vp_vah","VP VAH"),("vp_val","VP VAL"),
                       ("ichi_tenkan","Ichimoku Tenkan"),("ichi_kijun","Ichimoku Kijun")]:
        v = technical.get(key)
        if v:
            dist = (technical.get("current_price", 0) - v) / v * 100 if v else 0
            levels.append(f"{label}=${v:,.4f} ({dist:+.1f}% dari harga)")
    if sr:
        for k, lab in [("support1","Sup1"),("support2","Sup2"),
                       ("resistance1","Res1"),("resistance2","Res2"),("pivot","Pivot")]:
            if sr.get(k):
                levels.append(f"{lab}=${sr[k]:,.4f}")
    levels_text = " | ".join(levels) if levels else "tidak tersedia"

    # Orderbook
    ob_text = "tidak tersedia"
    if order_book:
        imb = order_book.get("depth_imbalance", 0)
        direction = "TEKANAN BELI DOMINAN" if imb > 10 else ("TEKANAN JUAL DOMINAN" if imb < -10 else "SEIMBANG")
        ob_text = (f"Bid={order_book.get('best_bid',0):,.4f} Ask={order_book.get('best_ask',0):,.4f} "
                   f"Spread={order_book.get('spread_pct',0):.4f}% | "
                   f"BidVol={order_book.get('bid_volume',0):,.0f} AskVol={order_book.get('ask_volume',0):,.0f} "
                   f"Imbalance={imb:+.1f}% → {direction}")

    tape_text = "tidak tersedia"
    if recent_trades:
        tape_text = (f"Buy={recent_trades.get('buy_ratio_pct',50):.0f}% "
                     f"Sell={100-recent_trades.get('buy_ratio_pct',50):.0f}% "
                     f"VolumeRatio={recent_trades.get('volume_ratio',1):.1f}x "
                     f"Change={recent_trades.get('change_pct',0):+.2f}%")

    chart_text = "tidak ada chart tersedia"
    if chart_paths:
        chart_text = "Chart tersedia: " + ", ".join(f"{tf}={p}" for tf,p in chart_paths.items())

    news_text = market_context.get("text", "data berita tidak tersedia") if market_context else ""

    # Posisi terbuka lain (untuk manajemen eksposur & korelasi)
    open_pos_text = "tidak ada"
    open_detail = portfolio_ctx.get("open_positions_detail")
    if open_detail:
        rows = []
        for p in open_detail:
            rows.append(f"{p.get('symbol')} {p.get('side')} entry=${p.get('entry_price',0):,.4f} "
                        f"pnl={p.get('pnl_pct',0):+.2f}% lev={p.get('leverage',0):.0f}x "
                        f"sl={p.get('stop_loss',0):,.4f}")
        open_pos_text = "\n".join("  • " + r for r in rows)

    prompt = f"""
# CRYPTO FUTURES ANALYSIS REQUEST: {symbol}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC

## PORTFOLIO CONTEXT
  Equity        : ${portfolio_ctx.get('equity',0):,.2f}
  Available Margin: ${portfolio_ctx.get('available_margin',0):,.2f}
  Open Positions: {portfolio_ctx.get('open_positions',0)} / {portfolio_ctx.get('max_positions','?')}
  Total P&L     : {portfolio_ctx.get('total_pnl_pct',0):+.2f}%
  Drawdown      : {portfolio_ctx.get('drawdown_pct',0):.2f}%
  Risk/trade    : {portfolio_ctx.get('risk_per_trade_pct','?')}% equity
  Already holds : {portfolio_ctx.get('has_position',False)}
  Other positions:
{open_pos_text}

## BINANCE REALTIME MARKET DATA
  Price         : ${metrics.get('price',0):,.6f}
  24h High/Low  : ${metrics.get('high_24h',0):,.6f} / ${metrics.get('low_24h',0):,.6f}
  24h Change    : {metrics.get('change_24h_pct',0):+.2f}%
  24h Volume    : ${metrics.get('volume_24h',0):,.0f}
  Funding Rate  : {metrics.get('funding_rate',0):+.6f}
  Open Interest : {metrics.get('open_interest',0):,.0f}
  Order Book    : {ob_text}
  Recent Tape   : {tape_text}
  Charts        : {chart_text}

## TECHNICAL ANALYSIS — MULTI-TIMEFRAME
  Combined Score: {technical.get('score',0):+.1f}/100 | Direction: {technical.get('direction','?')}
  Market Regime : {technical.get('market_regime','?')} | Confidence: {technical.get('confidence',0):.0f}%
  TF Conflict   : {technical.get('tf_conflict',False)}
  RSI (primary) : {technical.get('primary_rsi',50):.1f}
  ADX (primary) : {technical.get('primary_adx',0):.1f}
  ATR %/bar     : {technical.get('atr_pct',0):.3f}%
  Volume Ratio  : {technical.get('volume_ratio',1):.1f}x vs avg
  Realized Vol  : {technical.get('realized_vol_annual_pct',0):.0f}%/tahun
  Candle Patterns: {', '.join(technical.get('candlestick_patterns',[])) or 'none'}
  Suggested SL  : {technical.get('suggested_sl_pct',0):.2f}% | TP1: {technical.get('suggested_tp1_pct',0):.2f}% | TP2: {technical.get('suggested_tp2_pct',0):.2f}%
  Returns       : 1d={technical.get('return_1d',0):+.2f}% | 5d={technical.get('return_5d',0):+.2f}% | 20d={technical.get('return_20d',0):+.2f}%

  Key Levels    : {levels_text}

Per-timeframe breakdown:
{tf_text}
Signals (primary TF):
{sigs}

## MARKET INTELLIGENCE (NEWS + FUNDAMENTAL + MACRO)
{news_text}

---
INSTRUKSI: Analisis SEMUA data di atas secara menyeluruh.
Pikirkan langkah demi langkah (chain of thought).
Berikan keputusan LONG/SHORT/HOLD/AVOID dengan JSON valid.
Kamu adalah trader mandiri — putuskan dengan tegas dan berikan reasoning yang jelas.
"""
    return prompt


# ═══════════════════════════════════════════════════════════════
#  BUILD PROMPT — MANAJEMEN POSISI TERBUKA
# ═══════════════════════════════════════════════════════════════
def build_management_prompt(
    positions: list,
    prices: dict,
    technical_map: dict,
    market_ctx_map: dict,
    portfolio_ctx: dict,
) -> str:
    lines = []
    lines.append(f"# PORTFOLIO POSITION MANAGEMENT REQUEST")
    lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    lines.append(f"\n## PORTFOLIO OVERVIEW")
    lines.append(f"  Equity        : ${portfolio_ctx.get('equity',0):,.2f}")
    lines.append(f"  Total P&L     : {portfolio_ctx.get('total_pnl_pct',0):+.2f}%")
    lines.append(f"  Drawdown      : {portfolio_ctx.get('drawdown_pct',0):.2f}%")
    lines.append(f"  Open positions: {portfolio_ctx.get('open_positions',0)}")

    lines.append(f"\n## OPEN POSITIONS DETAIL")
    for p in positions:
        sym = p.symbol
        price = prices.get(sym, p.entry_price)
        pnl = p.unrealized_pnl(price)
        pnl_pct = p.unrealized_pnl_pct(price)
        if p.side == "LONG":
            move_pct = (price / p.entry_price - 1) * 100
            dist_sl = (price - p.stop_loss) / p.stop_loss * 100
            dist_tp1 = (p.take_profit_1 - price) / price * 100
        else:
            move_pct = (p.entry_price / price - 1) * 100
            dist_sl = (p.stop_loss - price) / p.stop_loss * 100
            dist_tp1 = (price - p.take_profit_1) / price * 100
        holding_h = 0.0
        try:
            from datetime import datetime as _dt
            entry_dt = _dt.fromisoformat(p.entry_time)
            holding_h = round((_dt.now() - entry_dt).total_seconds() / 3600, 1)
        except Exception:
            pass
        lines.append(
            f"  [{sym}] {p.side} @ ${p.entry_price:,.4f} → now ${price:,.4f} "
            f"({move_pct:+.2f}%) | P&L ${pnl:+,.2f} ({pnl_pct:+.2f}% margin) | "
            f"Lev {p.leverage:.0f}x | SL ${p.stop_loss:,.4f} (jarak {dist_sl:.2f}%) | "
            f"TP1 ${p.take_profit_1:,.4f} (jarak {dist_tp1:.2f}%) | "
            f"Liq ${p.liquidation_price:,.4f} | holding {holding_h}h | "
            f"partial_sold={p.partial_sold}"
        )
        tech = technical_map.get(sym, {})
        if tech:
            lines.append(
                f"    TA: score={tech.get('score',0):+.0f} | regime={tech.get('market_regime','?')} | "
                f"RSI={tech.get('primary_rsi',50):.1f} | ADX={tech.get('primary_adx',0):.1f} | "
                f"vol={tech.get('volume_ratio',1):.2f}x | conflict={tech.get('tf_conflict',False)}"
            )
            tf_dirs = tech.get("timeframe_bias", {})
            if tf_dirs:
                lines.append("    TF bias: " + ", ".join(f"{k}={v}" for k, v in tf_dirs.items()))
        ctx = market_ctx_map.get(sym, {})
        if ctx:
            sent = ctx.get("sentiment", {})
            fg = ctx.get("fear_greed", {})
            bc = ctx.get("binance_ctx", {})
            lines.append(
                f"    News sentiment: {sent.get('score',50):.0f}/100 ({sent.get('label','?')}) | "
                f"Fear&Greed: {fg.get('value',50)} ({fg.get('label','?')}) | "
                f"Funding: {bc.get('funding_pct',0):+.4f}%"
            )

    lines.append("""
---
INSTRUKSI: Analisis setiap posisi di atas. Putuskan dengan tegas tindakan terbaik
untuk memaksimalkan profit dan meminimalkan kerugian. Berikan JSON valid.
Hanya rekomendasikan CLOSE jika alasannya kuat dan berbasis data.
""")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  CALL PROVIDER — OpenAI-compatible (GPT, Kenari, Custom)
# ═══════════════════════════════════════════════════════════════
async def _call_openai_compatible(client, model: str, prompt: str, cfg: dict,
                                  label: str, system_prompt: str = None) -> Optional[dict]:
    sp = system_prompt or SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": sp},
        {"role": "user",   "content": prompt},
    ]
    kwargs = dict(
        model       = model,
        messages    = messages,
        max_tokens  = cfg.get("max_tokens", 2000),
        temperature = cfg.get("temperature", 0.05),
    )
    for attempt in range(2):
        try:
            # Coba dengan json_object format dulu
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        response_format={"type": "json_object"}, **kwargs),
                    timeout=45.0
                )
            except Exception:
                # Tidak support structured output → plain
                resp = await asyncio.wait_for(
                    client.chat.completions.create(**kwargs),
                    timeout=45.0
                )

            raw = (resp.choices[0].message.content or "").strip()
            if not raw:
                continue

            # Coba parse JSON — handle markdown code block dari model free
            # Pattern 1: ```json ... ``` atau ``` ... ```
            md = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
            if md:
                raw_json = md.group(1)
            else:
                # Pattern 2: JSON objek langsung di teks
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                raw_json = m.group() if m else None

            if not raw_json:
                console.print(f"  [dim]{label} {model}: no JSON in response[/dim]")
                break

            result = json.loads(raw_json)
            # Pastikan field penting ada — isi default jika tidak ada
            result.setdefault("action", "HOLD")
            result.setdefault("overall_score", 50)
            result.setdefault("confidence", 50)
            result.setdefault("confluence_score", 50)
            result.setdefault("risk_level", "MEDIUM")
            result["_source"] = f"{label}({model.split('/')[-1]})"
            short_model = model.split(":")[-2].split("/")[-1] if "/" in model else model.split(":")[0]
            console.print(f"  [green]✓ {label}[{short_model}][/green]")
            return result

        except json.JSONDecodeError as e:
            console.print(f"  [dim]{label} {model}: JSON parse error — {str(e)[:50]}[/dim]")
            break
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                if attempt == 1:
                    raise _KeyRateLimited()
                await asyncio.sleep(min(10, (attempt + 1) * 3))
            else:
                console.print(f"  [yellow]{label} {model}: {err[:80]}[/yellow]")
                break
    return None


# ═══════════════════════════════════════════════════════════════
#  CALL EACH AI — dengan fallback model otomatis
# ═══════════════════════════════════════════════════════════════
async def _call_gemini(prompt: str, system_prompt: str = None) -> Optional[dict]:
    """Gemini via google-genai SDK v2.x — multi-key rotasi & fallback model."""
    if not GEMINI_OK:
        return None
    sp = system_prompt or SYSTEM_PROMPT
    cfg = AI_MODELS["gemini"]
    models_to_try = [m for m in cfg.get("models", []) if m] or ["gemini-2.0-flash"]
    clients = _gemini_clients
    n = len(clients)

    full_prompt = sp + "\n\nOutput HANYA JSON valid, tidak ada teks lain di luar JSON.\n\n" + prompt

    for i in range(n):
        idx = (_rot["gemini"] + i) % n
        client = clients[idx]
        key_tail = _gemini_key_tails[idx] if idx < len(_gemini_key_tails) else ""
        rotate_key = False
        for model in models_to_try:
            for attempt in range(2):
                try:
                    response = await asyncio.to_thread(
                        client.models.generate_content,
                        model=model,
                        contents=full_prompt,
                        config=genai_types.GenerateContentConfig(
                            temperature=cfg.get("temperature", 0.05),
                            max_output_tokens=cfg.get("max_tokens", 3500),
                            response_mime_type="application/json",
                        ),
                    )
                    raw = (response.text or "").strip()
                    if not raw:
                        break
                    m = re.search(r'\{.*\}', raw, re.DOTALL)
                    if m:
                        result = json.loads(m.group())
                        result["_source"] = f"Gemini({model.split('/')[-1]})"
                        console.print(f"  [green]✓ Gemini [{model.split('/')[-1]}] "
                                      f"(key …{key_tail})[/green]")
                        _rot["gemini"] = idx   # sticky: mulai dari key ini
                        return result
                    break
                except Exception as e:
                    err = str(e)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err:
                        if attempt == 1:
                            console.print(f"  [yellow]Gemini key …{key_tail} rate limit → "
                                          f"ganti key[/yellow]")
                            rotate_key = True
                            break   # out of attempt loop (index i maju ke key berikut)
                        await asyncio.sleep(3)
                    elif "404" in err or "NOT_FOUND" in err:
                        console.print(f"  [dim]Model {model} tidak tersedia, skip[/dim]")
                        break
                    elif "400" in err and "INVALID_ARGUMENT" in err:
                        console.print(f"  [red]Gemini API key …{key_tail} error: "
                                      f"{err[:80]}[/red]")
                        break
                    else:
                        console.print(f"  [yellow]Gemini {model.split('/')[-1]}: "
                                      f"{err[:80]}[/yellow]")
                        break
            if rotate_key:
                break   # out of model loop → ganti key
    return None


async def _call_gpt(prompt: str, system_prompt: str = None) -> Optional[dict]:
    """GPT — ensemble opsional, multi-key rotasi."""
    if not GPT_OK:
        return None
    cfg = AI_MODELS["openai"]
    clients = openai_clients
    n = len(clients)
    for i in range(n):
        idx = (_rot["openai"] + i) % n
        for model in [cfg.get("model"), cfg.get("fallback")]:
            if not model:
                continue
            try:
                r = await _call_openai_compatible(clients[idx], model, prompt, cfg, "GPT", system_prompt)
            except _KeyRateLimited:
                break   # rotasi: i maju ke key berikut
            if r:
                _rot["openai"] = idx
                return r
        else:
            continue
    return None


async def _call_claude(prompt: str, system_prompt: str = None) -> Optional[dict]:
    """Claude 3.x — ensemble opsional, multi-key rotasi."""
    if not CLAUDE_OK:
        return None
    sp = system_prompt or SYSTEM_PROMPT
    cfg = AI_MODELS["anthropic"]
    clients = anthropic_clients
    n = len(clients)
    for i in range(n):
        idx = (_rot["anthropic"] + i) % n
        client = clients[idx]
        for model in [cfg.get("model"), cfg.get("fallback")]:
            if not model:
                continue
            try:
                msg = await client.messages.create(
                    model=model,
                    max_tokens=cfg.get("max_tokens", 3500),
                    system=sp + "\nOutput HANYA JSON valid, tidak ada teks lain.",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=cfg.get("temperature", 0.05),
                )
                raw = msg.content[0].text
                m = re.search(r'\{.*\}', raw, re.DOTALL)
                if m:
                    result = json.loads(m.group())
                    result["_source"] = f"Claude({model})"
                    console.print(f"  [green]✓ Claude ({model})[/green]")
                    _rot["anthropic"] = idx
                    return result
            except Exception as e:
                err = str(e)
                if "429" in err:
                    console.print(f"  [yellow]Claude key rate limit → ganti key[/yellow]")
                    break   # rotasi: i maju ke key berikut
                console.print(f"  [yellow]Claude {model}: {err[:80]}[/yellow]")
    return None


async def _call_one_kenari_model(
    client, model: str, prompt: str, cfg: dict, system_prompt: str = None
) -> Optional[dict]:
    """Panggil satu model Kenari, return dict atau None jika gagal."""
    try:
        result = await _call_openai_compatible(
            client, model, prompt, cfg, "Kenari", system_prompt
        )
        if result:
            result["_kenari_model"] = model
        return result
    except _KeyRateLimited:
        console.print(f"  [yellow]Kenari {model}: rate limit[/yellow]")
        return None
    except Exception as e:
        console.print(f"  [yellow]Kenari {model}: {str(e)[:60]}[/yellow]")
        return None


async def _call_kenari_primary(prompt: str, system_prompt: str = None) -> Optional[dict]:
    """
    Panggil model PRIMARY Kenari lebih dulu (KENARI_PRIMARY_MODEL).
    Jika berhasil → return langsung tanpa tunggu model lain.
    """
    if not KENARI_OK:
        return None
    client = _kenari_clients[0]   # pakai key pertama untuk primary
    cfg = AI_MODELS["kenari"]
    model = KENARI_PRIMARY_MODEL
    console.print(f"  [cyan]  ► Kenari PRIMARY [{model}]...[/cyan]")
    result = await _call_one_kenari_model(client, model, prompt, cfg, system_prompt)
    if result:
        console.print(f"  [bold green]  ✓ Kenari PRIMARY [{model}] berhasil[/bold green]")
    return result


async def _call_kenari_ensemble(
    prompt: str, system_prompt: str = None, skip_model: str = None
) -> list[dict]:
    """
    Panggil SEMUA model gratis Kenari secara PARALEL.
    skip_model: lewati model yang sudah dipanggil sebagai primary.
    Return: list semua hasil yang berhasil (bisa 0..N).
    """
    if not KENARI_OK:
        return []

    client = _kenari_clients[0]
    cfg    = AI_MODELS["kenari"]
    models_to_call = [
        m for m in KENARI_FREE_MODELS if m != skip_model
    ]

    if not models_to_call:
        return []

    console.print(
        f"  [cyan]  ► Kenari ENSEMBLE [{len(models_to_call)} models paralel]...[/cyan]"
    )

    tasks = [
        _call_one_kenari_model(client, m, prompt, cfg, system_prompt)
        for m in models_to_call
    ]
    raw = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for m, r in zip(models_to_call, raw):
        if isinstance(r, dict) and r:
            results.append(r)
        elif isinstance(r, Exception):
            console.print(f"  [dim]Kenari {m}: {str(r)[:50]}[/dim]")

    ok = len(results)
    console.print(
        f"  [{'green' if ok else 'yellow'}]  Kenari ensemble: "
        f"{ok}/{len(models_to_call)} models berhasil[/{'green' if ok else 'yellow'}]"
    )
    return results


async def _call_kenari(prompt: str, system_prompt: str = None) -> list[dict]:
    """
    Strategi Kenari dua-langkah:
      1. Primary model dipanggil dulu (cepat, langsung dapat sinyal awal).
      2. Sisa model gratis dipanggil paralel sebagai verifikasi/ensemble.
    Return: list semua hasil (primary + ensemble).
    """
    if not KENARI_OK:
        return []

    cfg = AI_MODELS["kenari"]
    all_results = []

    # Step 1: Primary
    primary_result = await _call_kenari_primary(prompt, system_prompt)
    if primary_result:
        all_results.append(primary_result)

    # Step 2: Ensemble (model selain primary, dipanggil paralel)
    ensemble_results = await _call_kenari_ensemble(
        prompt, system_prompt, skip_model=KENARI_PRIMARY_MODEL
    )
    all_results.extend(ensemble_results)

    n = len(all_results)
    console.print(
        f"  [bold cyan]  Kenari total: {n}/{len(KENARI_FREE_MODELS)} "
        f"model memberikan sinyal[/bold cyan]"
    )
    return all_results


async def _call_custom(prompt: str, system_prompt: str = None) -> Optional[dict]:
    """Custom OpenAI-compatible provider dari AI_BASE_URL + AI_API_KEY,
    mendukung beberapa key dipisah koma."""
    if not CUSTOM_OK:
        return None
    cfg = AI_MODELS["custom"]
    clients = _custom_clients
    n = len(clients)
    for i in range(n):
        idx = (_rot["custom"] + i) % n
        for model in [cfg.get("model"), cfg.get("fallback")]:
            if not model:
                continue
            try:
                r = await _call_openai_compatible(clients[idx], model, prompt, cfg, "Custom", system_prompt)
            except _KeyRateLimited:
                break   # rotasi: i maju ke key berikut
            if r:
                _rot["custom"] = idx
                return r
        else:
            continue
    return None


# ═══════════════════════════════════════════════════════════════
#  CALL ALL ENABLED PROVIDERS — paralel
# ═══════════════════════════════════════════════════════════════
_PRIMARY_CALLS = {
    "gemini":    _call_gemini,
    "openai":    _call_gpt,
    "anthropic": _call_claude,
    "kenari":    _call_kenari,
    "custom":    _call_custom,
}


async def _call_providers(prompt: str, system_prompt: str = None) -> list[dict]:
    """
    Panggil semua AI provider yang aktif.
    Kenari → returns list[dict] (multi-model ensemble).
    Lainnya → returns Optional[dict], dibungkus ke list.
    """
    flags   = _selected_providers()
    primary = (AI_PRIMARY_PROVIDER or "").lower()
    all_results: list[dict] = []

    # ── 1) Kenari PRIMARY: panggil dulu, collect semua hasilnya ──
    if primary == "kenari" and flags.get("kenari"):
        kenari_results = await _call_kenari(prompt, system_prompt)
        if kenari_results:
            console.print(
                f"  [bold green]★ KENARI PRIMARY — "
                f"{len(kenari_results)} model memberikan sinyal[/bold green]"
            )
            all_results.extend(kenari_results)
            # Jika ensemble cukup kuat (≥3 model setuju), return langsung
            action_votes: dict[str, int] = {}
            for r in kenari_results:
                a = r.get("action", "HOLD")
                action_votes[a] = action_votes.get(a, 0) + 1
            top_action, top_votes = max(action_votes.items(), key=lambda x: x[1])
            if top_votes >= 3 or len(kenari_results) >= len(KENARI_FREE_MODELS) // 2:
                return all_results   # sudah cukup konsensus
        else:
            console.print(f"  [yellow]  ⚠ Kenari gagal → fallback ke AI lain[/yellow]")

    # ── 2) Provider lain (non-kenari, non-primary) dipanggil paralel ──
    other_tasks = []
    other_names = []
    for name, enabled in flags.items():
        if not enabled or name == primary:
            continue
        if name == "kenari":
            other_tasks.append(_call_kenari(prompt, system_prompt))
        else:
            other_tasks.append(_PRIMARY_CALLS[name](prompt, system_prompt))
        other_names.append(name)

    if other_tasks:
        raw = await asyncio.gather(*other_tasks, return_exceptions=True)
        for name, r in zip(other_names, raw):
            if isinstance(r, list):
                all_results.extend([x for x in r if isinstance(x, dict) and x])
            elif isinstance(r, dict) and r:
                all_results.append(r)

    return all_results


# ═══════════════════════════════════════════════════════════════
#  ENSEMBLE — gabung hasil semua AI (ENTRY)
# ═══════════════════════════════════════════════════════════════
def _ensemble(results: list[dict]) -> dict:
    valid = [r for r in results if isinstance(r, dict) and r]
    if not valid:
        return {
            "action": "AVOID", "confidence": 0, "risk_level": "VERY_HIGH",
            "overall_score": 0, "confluence_score": 0,
            "verdict": "Semua AI gagal — tidak ada sinyal",
            "_sources": [], "_ai_count": 0,
        }

    def avg(key, default=50):
        vals = [float(r[key]) for r in valid if r.get(key) is not None]
        return round(sum(vals) / len(vals), 1) if vals else default

    # ── Voting dengan bobot ──
    vote_weights = {"Gemini": 2, "GPT": 1, "Claude": 1, "Kenari": 1, "Custom": 1}
    action_score = {}
    for r in valid:
        src = r.get("_source", "?")
        w   = next((v for k, v in vote_weights.items() if k in src), 1)
        a   = r.get("action", "HOLD")
        action_score[a] = action_score.get(a, 0) + w

    final_action = max(action_score, key=action_score.get)

    # Jika LONG dan SHORT keduanya dapat suara → HOLD (tidak ada konsensus)
    long_w  = action_score.get("LONG",  0)
    short_w = action_score.get("SHORT", 0)
    if long_w > 0 and short_w > 0:
        final_action = "HOLD"

    # ── Skor rata-rata ──
    final_conf       = avg("confidence", 50)
    final_score      = avg("overall_score", 50)
    final_tech       = avg("technical_score", 50)
    final_fund       = avg("fundamental_score", 50)
    final_sent       = avg("sentiment_score", 50)
    final_macro      = avg("macro_score", 50)
    final_confluence = avg("confluence_score", 0)
    sub_scores = {k: avg(k, 50) for k in
                  ("trend_score","momentum_score","structure_score",
                   "volume_score","context_score")}

    # Risk multiplier — rata-rata semua AI, dibatasi wajar
    rms = [float(r["suggested_risk_multiplier"]) for r in valid
           if r.get("suggested_risk_multiplier")]
    final_risk_mult = round(sum(rms) / len(rms), 2) if rms else 1.0
    final_risk_mult = max(0.25, min(1.5, final_risk_mult))

    # Turunkan confidence jika disagreement
    if len({r.get("action") for r in valid}) > 1:
        final_conf   = max(final_conf  - 20, 10)
        final_score  = max(final_score - 15, 10)

    # ── FILTER KETAT: confluence rendah → HOLD otomatis ──
    if final_confluence < 60 and final_action in ("LONG", "SHORT"):
        final_action = "HOLD"
        final_conf   = min(final_conf, 45)

    # ── Risk level paling konservatif ──
    risk_rank  = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "VERY_HIGH": 4}
    final_risk = max(
        (r.get("risk_level", "HIGH") for r in valid),
        key=lambda x: risk_rank.get(x, 2)
    )

    # VERY_HIGH risk → force AVOID
    if final_risk == "VERY_HIGH":
        final_action = "AVOID"

    # ── Price levels (rata-rata AI yang setuju dengan final_action) ──
    relevant = [r for r in valid if r.get("action") == final_action] or valid

    def avg_price(key):
        vals = [float(r[key]) for r in relevant
                if r.get(key) and float(r.get(key, 0)) > 0]
        return round(sum(vals) / len(vals), 6) if vals else None

    final_sl    = avg_price("stop_loss")
    final_tp1   = avg_price("take_profit_1")
    final_tp2   = avg_price("take_profit_2")
    final_tp3   = avg_price("take_profit_3")
    final_entry = avg_price("entry_price")

    # ── RRR validasi ──
    rrrs = [float(r["risk_reward_ratio"]) for r in relevant
            if r.get("risk_reward_ratio") and float(r.get("risk_reward_ratio", 0)) > 0]
    final_rrr = round(sum(rrrs) / len(rrrs), 2) if rrrs else 0

    # FILTER: RRR < MIN_RRR → HOLD
    if final_rrr < MIN_RRR and final_action in ("LONG", "SHORT"):
        final_action = "HOLD"
        final_conf   = min(final_conf, 40)

    # ── Leverage — paling konservatif dari semua AI, kap mengikuti config ──
    levs      = [float(r["leverage"]) for r in valid if r.get("leverage")]
    final_lev = min(levs) if levs else DEFAULT_LEVERAGE
    final_lev = min(final_lev, MAX_LEVERAGE)   # hard cap dari env (20x+)

    # ── Position size — terkecil, cap 1% (scalper) ──
    pos_sizes  = [float(r["position_size_pct"]) for r in valid if r.get("position_size_pct")]
    final_pos  = min(pos_sizes) if pos_sizes else 0.5
    final_pos  = min(final_pos, 1.0)

    # ── Holding period ──
    holds      = [r.get("holding_period", "swing_1-3hari") for r in valid]
    final_hold = Counter(holds).most_common(1)[0][0]

    # ── Analisis teks (dari AI dengan action == final_action) ──
    def first_text(key):
        for r in relevant:
            if r.get(key):
                return r[key]
        return ""

    # ── Risks & catalysts deduplicated ──
    risks = list(dict.fromkeys(r for v in valid for r in v.get("key_risks", [])))[:6]
    cats  = list(dict.fromkeys(c for v in valid for c in v.get("catalysts", [])))[:5]
    fails = list(dict.fromkeys(f for v in valid for f in v.get("failure_modes", [])))[:4]

    # ── Chain of thought gabungan ──
    cot_parts = []
    for r in valid:
        src = r.get("_source", "?")
        cot = r.get("chain_of_thought") or r.get("reasoning", "")
        if cot:
            cot_parts.append(f"[{src}]:\n{cot[:700]}")

    return {
        "action":            final_action,
        "confidence":        final_conf,
        "risk_level":        final_risk,
        "confluence_score":  final_confluence,
        "trend_score":       sub_scores["trend_score"],
        "momentum_score":    sub_scores["momentum_score"],
        "structure_score":   sub_scores["structure_score"],
        "volume_score":      sub_scores["volume_score"],
        "context_score":     sub_scores["context_score"],
        "suggested_risk_multiplier": final_risk_mult,
        "overall_score":     final_score,
        "technical_score":   final_tech,
        "fundamental_score": final_fund,
        "sentiment_score":   final_sent,
        "macro_score":       final_macro,
        "entry_price":       final_entry,
        "stop_loss":         final_sl,
        "take_profit_1":     final_tp1,
        "take_profit_2":     final_tp2,
        "take_profit_3":     final_tp3,
        "risk_reward_ratio": final_rrr,
        "position_size_pct": final_pos,
        "leverage":          final_lev,
        "holding_period":    final_hold,
        "funding_analysis":  first_text("funding_analysis"),
        "volume_analysis":   first_text("volume_analysis"),
        "support_resistance_analysis": first_text("support_resistance_analysis"),
        "key_risks":         risks,
        "failure_modes":     fails,
        "catalysts":         cats,
        "invalidation":      first_text("invalidation"),
        "chain_of_thought":  "\n\n".join(cot_parts),
        "trade_plan":        first_text("trade_plan"),
        "verdict":           first_text("verdict"),
        "action_votes":      action_score,
        "_sources":          [r.get("_source") for r in valid],
        "_ai_count":         len(valid),
        "_individual_results": valid,
    }


# ═══════════════════════════════════════════════════════════════
#  ENSEMBLE — manajemen posisi (gabung semua AI)
# ═══════════════════════════════════════════════════════════════
def _ensemble_management(results: list[dict], symbols: list[str]) -> dict:
    valid = [r for r in results if isinstance(r, dict) and r]

    def conservative_avg(key, default=1.0):
        vals = [float(r[key]) for r in valid if r.get(key) is not None]
        if not vals:
            return default
        # ambil rata-rata tapi tidak lebih dari 1.5 dan tidak di bawah 0.25
        v = sum(vals) / len(vals)
        return round(max(0.25, min(1.5, v)), 2)

    risk_mult = conservative_avg("recommended_risk_multiplier", 1.0)

    market_assessment = ""
    for r in valid:
        if r.get("market_assessment"):
            market_assessment = r["market_assessment"]
            break

    # ── Gabung keputusan per symbol (voting mayoritas + pilih paling konservatif) ──
    per_symbol = {}
    weight = {"Gemini": 2, "GPT": 1, "Claude": 1, "Kenari": 1, "Custom": 1}
    for r in valid:
        src = r.get("_source", "?")
        w   = next((v for k, v in weight.items() if k in src), 1)
        for rec in (r.get("positions") or []):
            sym = rec.get("symbol")
            if not sym:
                continue
            d = per_symbol.setdefault(sym, {"votes": {}, "recs": []})
            d["votes"][rec.get("action", "HOLD")] = d["votes"].get(rec.get("action", "HOLD"), 0) + w
            d["recs"].append(rec)

    final_positions = []
    for sym in symbols:
        d = per_symbol.get(sym)
        if not d:
            continue
        action = max(d["votes"], key=d["votes"].get)
        # Konflik CLOSE vs HOLD → pilih tindakan paling aman (konservatif)
        if d["votes"].get("CLOSE", 0) > 0 and action != "CLOSE":
            if d["votes"].get("CLOSE", 0) >= 1.0 and len(d["recs"]) > 1:
                pass  # tetap action mayoritas
        # Ambil rekomendasi dari AI yang setuju action terpilih
        chosen = [r for r in d["recs"] if r.get("action") == action] or d["recs"]
        rec = chosen[0]
        # SL baru: ambil yang paling mengunci profit (untuk LONG: tertinggi, SHORT: terendah)
        new_sls = [float(r["new_stop_loss"]) for r in chosen
                   if r.get("new_stop_loss")]
        new_tps = [float(r["new_take_profit"]) for r in chosen
                   if r.get("new_take_profit")]
        close_fracs = [float(r["close_fraction"]) for r in chosen
                       if r.get("close_fraction")]
        final_positions.append({
            "symbol":         sym,
            "action":         action,
            "reason":         rec.get("reason", ""),
            "urgency":        rec.get("urgency", "MEDIUM"),
            "new_stop_loss":  (max(new_sls) if new_sls else None),
            "new_take_profit":(min(new_tps) if new_tps else None),
            "close_fraction": (max(close_fracs) if close_fracs else 0.5),
            "_votes":         d["votes"],
        })

    return {
        "market_assessment":   market_assessment,
        "recommended_risk_multiplier": risk_mult,
        "positions":           final_positions,
        "_sources":            [r.get("_source") for r in valid],
        "_ai_count":           len(valid),
    }


# ═══════════════════════════════════════════════════════════════
#  FALLBACK TEKNIKAL (jika semua AI gagal)
# ═══════════════════════════════════════════════════════════════
def _technical_fallback(symbol: str, technical: dict, metrics: dict) -> dict:
    """Keputusan murni dari teknikal jika AI tidak tersedia."""
    score  = technical.get("score", 0)
    conf   = technical.get("confidence", 30)
    regime = technical.get("market_regime","RANGING")
    conflict = technical.get("tf_conflict", False)
    cur    = technical.get("current_price", 0)
    atr_p  = technical.get("atr_pct", 2.0)
    fr     = metrics.get("funding_rate", 0)

    if conflict or abs(score) < 25:
        action = "HOLD"
    elif score >= 40 and fr < 0.002:
        action = "LONG"
    elif score <= -40 and fr > -0.002:
        action = "SHORT"
    elif score >= 25:
        action = "LONG"
    elif score <= -25:
        action = "SHORT"
    else:
        action = "HOLD"

    # Volatilitas tinggi → hati-hati
    rvol = technical.get("realized_vol_annual_pct", 50)
    risk = "VERY_HIGH" if rvol>150 else ("HIGH" if rvol>80 else ("MEDIUM" if rvol>50 else "LOW"))
    if risk == "VERY_HIGH":
        action = "AVOID"

    # Scalper: SL tipis (0.3-1.5%), TP dekat, leverage 20x
    sl_p  = max(min(atr_p * 1.2, 1.5), 0.3)
    tp1_p = sl_p * 2.0
    tp2_p = sl_p * 3.2
    if action == "LONG":
        sl, tp1, tp2 = cur*(1-sl_p/100), cur*(1+tp1_p/100), cur*(1+tp2_p/100)
    elif action == "SHORT":
        sl, tp1, tp2 = cur*(1+sl_p/100), cur*(1-tp1_p/100), cur*(1-tp2_p/100)
    else:
        sl = tp1 = tp2 = cur

    overall = min(100, abs(score)*1.2 + 10)
    if action == "HOLD": overall = min(overall, 50)
    comp = min(100, abs(score) + 50) if action in ("LONG","SHORT") else 50

    return {
        "action":action,"confidence":min(conf*0.7,60),"risk_level":risk,
        "confluence_score": min(comp, 70), "trend_score": comp,
        "momentum_score": comp, "structure_score": comp,
        "volume_score": comp, "context_score": comp,
        "suggested_risk_multiplier": 1.0,
        "overall_score":round(overall,1),"technical_score":round(overall,1),
        "fundamental_score":50,"sentiment_score":50,"macro_score":50,
        "entry_price":cur,"stop_loss":round(sl,6),
        "take_profit_1":round(tp1,6),"take_profit_2":round(tp2,6),"take_profit_3":round(tp2*1.1,6),
        "risk_reward_ratio":round(tp1_p/sl_p,2),"position_size_pct":0.5,"leverage":20,
        "holding_period":"scalp_jam",
        "chain_of_thought":f"[Teknikal-Fallback] score={score:+.0f}, regime={regime}, conflict={conflict}",
        "key_risks":[f"Volatilitas {rvol:.0f}%/tahun", f"Regime: {regime}", "AI tidak tersedia"],
        "failure_modes":[],"catalysts":[],"trade_plan":"","verdict":f"Teknikal: {action} | Score {score:+.0f}",
        "funding_analysis":"","volume_analysis":"","support_resistance_analysis":"",
        "action_votes":{action:1},"_sources":["Teknikal-Fallback"],"_ai_count":1,
        "_individual_results":[],
    }


# ═══════════════════════════════════════════════════════════════
#  FUNGSI UTAMA — ANALISIS ENTRY (dipanggil dari trading_bot)
# ═══════════════════════════════════════════════════════════════
async def analyze_with_ai(
    symbol: str,
    technical: dict,
    metrics: dict,
    portfolio_ctx: dict,
    market_context: dict,
    order_book: dict = None,
    recent_trades: dict = None,
    chart_paths: dict = None,
) -> dict:
    """
    AI Trader mandiri — semua provider aktif dipanggil paralel,
    hasil digabungkan, return keputusan final.
    Fallback ke teknikal jika semua AI gagal.
    """
    prompt = build_prompt(
        symbol, technical, metrics, portfolio_ctx,
        market_context, order_book, recent_trades, chart_paths
    )

    console.print(f"  [cyan]🧠 Memanggil AI untuk {symbol}...[/cyan]")

    if not AI_AVAILABLE:
        console.print("  [yellow]Tidak ada AI tersedia → Teknikal Fallback[/yellow]")
        result = _technical_fallback(symbol, technical, metrics)
        result["symbol"] = symbol
        result["analyzed_at"] = datetime.now().isoformat()
        return result

    valid = await _call_providers(prompt)

    if not valid:
        console.print("  [red]Semua AI gagal → Teknikal Fallback[/red]")
        result = _technical_fallback(symbol, technical, metrics)
        result["symbol"] = symbol
        result["analyzed_at"] = datetime.now().isoformat()
        return result

    final = _ensemble(valid)
    final["symbol"]      = symbol
    final["analyzed_at"] = datetime.now().isoformat()

    action = final.get("action","?")
    score  = final.get("overall_score",0)
    conf   = final.get("confidence",0)
    srcs   = ", ".join(final.get("_sources",[]))
    votes  = final.get("action_votes",{})
    color  = {"LONG":"green","SHORT":"red","HOLD":"yellow","AVOID":"dim"}.get(action,"white")
    console.print(
        f"  [bold {color}]→ {action}[/bold {color}] | "
        f"Score:{score:.0f} | Conf:{conf:.0f}% | "
        f"Confluence:{final.get('confluence_score',0):.0f} | "
        f"Votes:{votes} | AI:{srcs}"
    )
    return final


# ═══════════════════════════════════════════════════════════════
#  FUNGSI UTAMA — MANAJEMEN POSISI TERBUKA
# ═══════════════════════════════════════════════════════════════
async def manage_positions_with_ai(
    positions: list,                # list Position dari portfolio.open_positions
    prices: dict,                   # {symbol: current_price}
    technical_map: dict,            # {symbol: hasil multi_timeframe_analysis}
    market_ctx_map: dict,           # {symbol: hasil get_full_market_context}
    portfolio_ctx: dict,
) -> dict:
    """
    AI sebagai Portfolio Manager — putuskan hold/close/partial/trail/TP
    untuk SEMUA posisi terbuka sekaligus (1 panggilan AI).
    Return dict dengan recommended_risk_multiplier + list keputusan per posisi.
    """
    if not positions:
        return {"market_assessment": "Tidak ada posisi terbuka.",
                "recommended_risk_multiplier": 1.0, "positions": [],
                "_sources": [], "_ai_count": 0}

    prompt = build_management_prompt(positions, prices, technical_map,
                                     market_ctx_map, portfolio_ctx)

    console.print(f"  [cyan]🧠 AI Portfolio Manager: {len(positions)} posisi...[/cyan]")

    if not AI_AVAILABLE:
        console.print("  [yellow]  AI tidak tersedia → biarkan posisi (aturan SL/TP auto aktif)[/yellow]")
        return {"market_assessment": "AI tidak tersedia.",
                "recommended_risk_multiplier": 1.0,
                "positions": [{"symbol": p.symbol, "action": "HOLD",
                               "reason": "AI tidak tersedia", "urgency": "LOW",
                               "new_stop_loss": None, "new_take_profit": None,
                               "close_fraction": 0.0} for p in positions],
                "_sources": [], "_ai_count": 0}

    valid = await _call_providers(prompt, MANAGEMENT_SYSTEM_PROMPT)

    if not valid:
        console.print("  [red]  AI manajemen gagal → HOLD semua posisi[/red]")
        return {"market_assessment": "AI manajemen gagal.",
                "recommended_risk_multiplier": 1.0,
                "positions": [{"symbol": p.symbol, "action": "HOLD",
                               "reason": "AI gagal", "urgency": "LOW",
                               "new_stop_loss": None, "new_take_profit": None,
                               "close_fraction": 0.0} for p in positions],
                "_sources": [], "_ai_count": 0}

    final = _ensemble_management(valid, [p.symbol for p in positions])
    for rec in final.get("positions", []):
        sym    = rec["symbol"]
        action = rec["action"]
        color  = {"CLOSE":"red","PARTIAL_CLOSE":"yellow","TRAIL_STOP":"cyan",
                  "MOVE_TP":"cyan","HOLD":"green","ADD":"magenta"}.get(action,"white")
        console.print(
            f"  [bold {color}]→ {sym}: {action}[/bold {color}] — {rec.get('reason','')[:90]}")
    console.print(f"  Risk multiplier disarankan: ×{final.get('recommended_risk_multiplier',1.0)}")
    return final
