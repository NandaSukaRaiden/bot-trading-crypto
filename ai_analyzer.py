"""
ai_analyzer.py — Gemini 2.0 Flash sebagai AI Trader Crypto Mandiri
Menggunakan google-genai SDK terbaru (v2.x) — bukan google-generativeai lama.
AI menerima: teknikal multi-TF + orderbook realtime + news tanpa login +
             fundamental CoinGecko + Fear&Greed + Makro global + chart paths
GPT-4o & Claude 3.5 sebagai ensemble opsional untuk akurasi lebih tinggi.
"""
import asyncio, json, re
from collections import Counter
from datetime import datetime
from typing import Optional

# ── google-genai SDK baru (v2.x) ─────────────────────────────
try:
    from google import genai as google_genai
    from google.genai import types as genai_types
    GENAI_NEW = True
except ImportError:
    google_genai = None
    genai_types  = None
    GENAI_NEW    = False

# ── GPT-4o (opsional ensemble) ────────────────────────────────
try:
    import openai
except ImportError:
    openai = None

# ── Claude 3.5 (opsional ensemble) ───────────────────────────
try:
    import anthropic
except ImportError:
    anthropic = None

from config import (
    GOOGLE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, AI_MODELS,
)
from rich.console import Console
console = Console()

# ── Init AI Clients ───────────────────────────────────────────
_gemini_client   = None
if GENAI_NEW and GOOGLE_API_KEY:
    try:
        _gemini_client = google_genai.Client(api_key=GOOGLE_API_KEY)
    except Exception as _e:
        console.print(f"[yellow]Gemini init error: {_e}[/yellow]")

openai_client    = None
if openai and OPENAI_API_KEY:
    try:
        openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        pass

anthropic_client = None
if anthropic and ANTHROPIC_API_KEY:
    try:
        anthropic_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    except Exception:
        pass

GEMINI_OK    = bool(_gemini_client)
GPT_OK       = bool(openai_client)
CLAUDE_OK    = bool(anthropic_client)
AI_AVAILABLE = GEMINI_OK or GPT_OK or CLAUDE_OK
AI_AVAILABLE = GEMINI_OK or GPT_OK or CLAUDE_OK

# ═══════════════════════════════════════════════════════════════
#  SYSTEM PROMPT — AI Trader Crypto Profesional Mandiri
# ═══════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """
Kamu adalah AI Crypto Futures Trader kelas dunia. Kamu telah trading secara profitable
selama 15 tahun melewati bull market 2017, bear market 2018, crash COVID 2020,
bull run 2021, bear market brutal 2022, dan recovery 2023-2024.
Kamu TIDAK pernah blown account karena kamu memprioritaskan SURVIVAL di atas segalanya.

═══════════════════════════════════════════════════════
FILOSOFI INTI (INTERNALIZED, BUKAN SEKEDAR ATURAN):
═══════════════════════════════════════════════════════
• "The best trade is often no trade." — Mayoritas waktu kamu MENUNGGU setup sempurna.
• "Cut losses short, let winners run." — SL dieksekusi tanpa emosi, TP dengan trailing.
• "Confluence is everything." — Kamu TIDAK masuk kecuali ≥3 faktor berbeda setuju.
• "Size kills." — Kamu TIDAK pernah over-leverage. Bertahan > 1000 trade lebih penting
  dari menang besar di 1 trade.
• "Trade what you see, not what you think." — Data objektif, bukan harapan.

═══════════════════════════════════════════════════════
SISTEM SCORING CONFLUENCE (WAJIB HITUNG SEBELUM ENTRY):
═══════════════════════════════════════════════════════
Kamu harus mengidentifikasi MINIMUM 3 dari 5 faktor berikut yang SEJALAN:

1. TREND (bobot 25%): 4h dan 1d harus searah. Jika berlawanan → HOLD.
2. MOMENTUM (bobot 20%): RSI + MACD histogram + EMA alignment searah.
3. STRUKTUR HARGA (bobot 20%): Entry di dekat support/resistance penting.
   Beli di support, jual di resistance — BUKAN di tengah-tengah.
4. VOLUME (bobot 20%): Volume breakout > 1.5x rata-rata mengkonfirmasi sinyal.
   Volume rendah = sinyal palsu, jangan masuk.
5. KONTEKS PASAR (bobot 15%): Fear&Greed, funding rate, berita, makro global
   tidak berlawanan dengan arah trade.

Jika skor confluence < 60% → HOLD atau AVOID.

═══════════════════════════════════════════════════════
ATURAN ENTRY YANG TIDAK BISA DILANGGAR:
═══════════════════════════════════════════════════════
1. RRR minimum 1:2.5 — Jika TP1 tidak bisa 2.5x jarak SL, SKIP trade ini.
2. SL WAJIB di bawah/atas level support/resistance VALID, BUKAN angka bulat acak.
   Gunakan ATR × 1.5 sebagai minimum jarak SL dari entry.
3. Funding rate:
   • > +0.10% → hindari LONG (long overcrowded, risiko squeeze)
   • < -0.10% → hindari SHORT (short overcrowded, risiko squeeze ke atas)
4. Fear & Greed:
   • > 80 (Extreme Greed) → TIDAK buka LONG baru kecuali ada breakout ATH baru
   • < 20 (Extreme Fear) → TIDAK buka SHORT baru kecuali ada breakdown support mayor
5. Berita negatif BERAT (hack > $10M, exchange collapse, regulatory ban negara besar)
   → AVOID semua trade 24 jam ke depan, volatilitas tidak bisa diprediksi.
6. TF conflict (4h bullish, 1d bearish atau sebaliknya) → HOLD, tunggu alignment.
7. Volume ratio < 0.7x rata-rata → SKIP, sinyal tidak valid tanpa konfirmasi volume.
8. Volatilitas harian > 8% → Leverage MAX 3x, perkecil sizing 50%.
9. Jangan buka posisi baru jika sudah ada 4+ posisi terbuka (fokus manage yang ada).
10. TIDAK pernah average down / menambah posisi yang sedang rugi.

═══════════════════════════════════════════════════════
STRATEGI MANAJEMEN POSISI AKTIF:
═══════════════════════════════════════════════════════
• TP1 (50% posisi): Target 2.5x jarak SL — ambil profit, geser SL ke breakeven
• TP2 (30% posisi): Target 4x jarak SL — trailing stop aktif setelah ini
• TP3 (20% posisi): Target 6x jarak SL atau sampai sinyal reversal kuat
• Jika harga menyentuh breakeven setelah TP1, TIDAK pernah rugi dari trade ini.

═══════════════════════════════════════════════════════
KAPAN BILANG HOLD (LEBIH SERING DARI LONG/SHORT):
═══════════════════════════════════════════════════════
• Confluence < 60%
• ADX < 20 (market ranging, tidak ada tren jelas)
• Harga di tengah range (bukan di support/resistance)
• Berita campur aduk (sebagian positif, sebagian negatif)
• Funding rate mendekati threshold
• Sudah ada posisi di coin ini
• Volatilitas terlalu tinggi untuk leverage yang layak

═══════════════════════════════════════════════════════
CARA BERPIKIR LANGKAH DEMI LANGKAH (CHAIN OF THOUGHT):
═══════════════════════════════════════════════════════
Tulis reasoning dengan urutan ini:
1. [TREN] Apa kata chart 1d dan 4h? Bullish/bearish/ranging?
2. [STRUKTUR] Di mana posisi harga sekarang vs support/resistance?
3. [MOMENTUM] RSI, MACD, EMA — apakah mengkonfirmasi atau divergen?
4. [VOLUME] Apakah volume mendukung pergerakan harga?
5. [KONTEKS] Fear&Greed, funding, berita — ada hal yang harus diwaspadai?
6. [CONFLUENCE SCORE] Hitung: berapa faktor yang setuju? (X/5)
7. [KEPUTUSAN] Berdasarkan di atas, keputusan adalah...
8. [TRADE PLAN] Entry di... SL di... karena... TP1 di... TP2 di... TP3 di...
9. [SIZING] Leverage berapa? Kenapa leverage itu aman?
10. [RISIKO UTAMA] Apa yang bisa membuat trade ini salah?

═══════════════════════════════════════════════════════
FORMAT OUTPUT (WAJIB JSON VALID, TIDAK ADA TEKS LAIN):
═══════════════════════════════════════════════════════
{
  "action": "LONG" | "SHORT" | "HOLD" | "AVOID",
  "confidence": 0-100,
  "risk_level": "LOW" | "MEDIUM" | "HIGH" | "VERY_HIGH",
  "confluence_score": 0-100,
  "entry_price": number,
  "stop_loss": number,
  "stop_loss_reason": "mengapa SL di level ini (support/ATR/dll)",
  "take_profit_1": number,
  "take_profit_2": number,
  "take_profit_3": number,
  "risk_reward_ratio": number,
  "position_size_pct": number,
  "leverage": number,
  "leverage_reason": "mengapa leverage ini aman",
  "holding_period": "intraday" | "swing_1-3hari" | "swing_3-7hari",
  "overall_score": 0-100,
  "technical_score": 0-100,
  "fundamental_score": 0-100,
  "sentiment_score": 0-100,
  "macro_score": 0-100,
  "chain_of_thought": "reasoning 10 langkah seperti di atas",
  "key_risks": ["risiko spesifik 1", "risiko 2", "risiko 3"],
  "invalidation": "kondisi yang akan membuat analisis ini salah",
  "catalysts": ["katalis positif konkret 1", "katalis 2"],
  "trade_plan": "entry X → SL Y (-Z%) → TP1 A (+B%) → TP2 C (+D%) → TP3 E (+F%)",
  "verdict": "keputusan final dalam 1 kalimat tegas"
}

PENTING:
- Jika confluence_score < 60 → action WAJIB "HOLD" atau "AVOID"
- Jika risk_level = "VERY_HIGH" → action WAJIB "AVOID"
- leverage TIDAK BOLEH lebih dari 10x untuk altcoin, 15x untuk BTC/ETH
- position_size_pct TIDAK BOLEH lebih dari 5% equity
- stop_loss untuk LONG WAJIB lebih kecil dari entry_price
- stop_loss untuk SHORT WAJIB lebih besar dari entry_price
- Lebih sering bilang HOLD daripada LONG/SHORT — itu tanda trader yang mature
"""

# ═══════════════════════════════════════════════════════════════
#  BUILD PROMPT — semua data ke AI
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
            tf_text += (f"  [{label}] score={r.get('score',0):+.0f} | "
                        f"dir={r.get('direction_bias','?')} | "
                        f"RSI={r.get('rsi',0):.1f} | ADX={r.get('adx',0):.1f} | "
                        f"ATR%={r.get('atr_pct',0):.2f}% | "
                        f"EMA9{'>'if r.get('ema_9',0)>r.get('ema_21',0) else '<'}EMA21\n")

    sr   = technical.get("support_resistance", {})
    sigs = "\n".join(f"  • {s}" for s in technical.get("signals_primary", [])[:20])

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

    # Chart paths
    chart_text = "tidak ada chart tersedia"
    if chart_paths:
        chart_text = "Chart tersedia: " + ", ".join(f"{tf}={p}" for tf,p in chart_paths.items())

    # Market context dari news_fetcher (Bloomberg Brief)
    news_text = market_context.get("text", "data berita tidak tersedia") if market_context else ""

    prompt = f"""
# CRYPTO FUTURES ANALYSIS REQUEST: {symbol}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC

## PORTFOLIO CONTEXT
  Equity        : ${portfolio_ctx.get('equity',0):,.2f}
  Available Margin: ${portfolio_ctx.get('available_margin',0):,.2f}
  Open Positions: {portfolio_ctx.get('open_positions',0)}
  Total P&L     : {portfolio_ctx.get('total_pnl_pct',0):+.2f}%
  Drawdown      : {portfolio_ctx.get('drawdown_pct',0):.2f}%
  Already holds : {portfolio_ctx.get('has_position',False)}

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
  Candle Patterns: {', '.join(technical.get('candlestick_patterns',[])) or 'none'}
  Support       : ${sr.get('support',0):,.6f}
  Resistance    : ${sr.get('resistance',0):,.6f}
  Suggested SL  : {technical.get('suggested_sl_pct',0):.2f}% | TP1: {technical.get('suggested_tp1_pct',0):.2f}% | TP2: {technical.get('suggested_tp2_pct',0):.2f}%
  Returns       : 1d={technical.get('return_1d',0):+.2f}% | 5d={technical.get('return_5d',0):+.2f}% | 20d={technical.get('return_20d',0):+.2f}%

Per-timeframe breakdown:
{tf_text}
Signals (primary TF):
{sigs}

## MARKET INTELLIGENCE (NEWS + FUNDAMENTAL + MACRO)
{news_text}

---
INSTRUKSI: Analisis semua data di atas secara menyeluruh.
Pikirkan langkah demi langkah (chain of thought).
Berikan keputusan LONG/SHORT/HOLD/AVOID dengan JSON valid.
Kamu adalah trader mandiri — putuskan dengan tegas dan berikan reasoning yang jelas.
"""
    return prompt

# ═══════════════════════════════════════════════════════════════
#  CALL EACH AI — dengan fallback otomatis
# ═══════════════════════════════════════════════════════════════
async def _call_gemini(prompt: str) -> Optional[dict]:
    """
    Gemini via google-genai SDK v2.x.
    Model priority: gemini-2.0-flash-lite → gemini-2.0-flash → gemini-1.5-flash
    Smart retry dengan exponential backoff untuk rate limit.
    """
    if not GEMINI_OK:
        return None

    cfg = AI_MODELS["gemini"]
    # Urutan: lite dulu (quota lebih besar), lalu flash, lalu fallback
    models_to_try = [
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        cfg.get("fallback", "gemini-1.5-flash-latest"),
    ]

    full_prompt = SYSTEM_PROMPT + "\n\nOutput HANYA JSON valid, tidak ada teks lain di luar JSON.\n\n" + prompt

    for model in models_to_try:
        for attempt in range(3):   # 3 retry per model
            try:
                response = await asyncio.to_thread(
                    _gemini_client.models.generate_content,
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
                    console.print(f"  [green]✓ Gemini [{model.split('/')[-1]}][/green]")
                    return result
                break  # respons ada tapi bukan JSON valid → skip model ini
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    # Rate limit — tunggu lalu retry atau skip ke model berikutnya
                    wait = (attempt + 1) * 10   # 10s, 20s, 30s
                    console.print(f"  [yellow]Gemini rate limit [{model.split('/')[-1]}], "
                                  f"retry dalam {wait}s...[/yellow]")
                    await asyncio.sleep(wait)
                    if attempt == 2:
                        console.print(f"  [dim]Skip {model} → coba model berikutnya[/dim]")
                elif "404" in err or "NOT_FOUND" in err:
                    console.print(f"  [dim]Model {model} tidak tersedia, skip[/dim]")
                    break  # langsung ke model berikutnya
                elif "400" in err and "INVALID_ARGUMENT" in err:
                    console.print(f"  [red]Gemini API key error: {err[:80]}[/red]")
                    return None  # key problem, stop semua
                else:
                    console.print(f"  [yellow]Gemini {model.split('/')[-1]}: {err[:80]}[/yellow]")
                    break
    return None

async def _call_gpt(prompt: str) -> Optional[dict]:
    """GPT-4o — ensemble opsional."""
    if not GPT_OK:
        return None
    cfg = AI_MODELS["openai"]
    for model in [cfg["model"], cfg["fallback"]]:
        try:
            resp = await openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=cfg["max_tokens"],
                temperature=cfg["temperature"],
                response_format={"type": "json_object"},
            )
            result = json.loads(resp.choices[0].message.content)
            result["_source"] = "GPT-4o"
            console.print(f"  [green]✓ GPT ({model})[/green]")
            return result
        except Exception as e:
            console.print(f"  [yellow]GPT {model}: {e}[/yellow]")
    return None

async def _call_claude(prompt: str) -> Optional[dict]:
    """Claude 3.5 Sonnet — ensemble opsional."""
    if not CLAUDE_OK:
        return None
    cfg = AI_MODELS["anthropic"]
    for model in [cfg["model"], cfg["fallback"]]:
        try:
            msg = await anthropic_client.messages.create(
                model=model,
                max_tokens=cfg["max_tokens"],
                system=SYSTEM_PROMPT + "\nOutput HANYA JSON valid, tidak ada teks lain.",
                messages=[{"role": "user", "content": prompt}],
                temperature=cfg["temperature"],
            )
            raw = msg.content[0].text
            m   = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                result = json.loads(m.group())
                result["_source"] = "Claude-3.5"
                console.print(f"  [green]✓ Claude ({model})[/green]")
                return result
        except Exception as e:
            console.print(f"  [yellow]Claude {model}: {e}[/yellow]")
    return None

# ═══════════════════════════════════════════════════════════════
#  ENSEMBLE — gabung hasil semua AI
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

    # ── Voting dengan bobot: Gemini 2x, GPT 1x, Claude 1x ──
    vote_weights = {"Gemini": 2, "GPT-4o": 1, "Claude-3.5": 1}
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
        final_action = "HOLD"  # disagreement = tidak masuk

    # ── Skor rata-rata ──
    final_conf      = avg("confidence", 50)
    final_score     = avg("overall_score", 50)
    final_tech      = avg("technical_score", 50)
    final_fund      = avg("fundamental_score", 50)
    final_sent      = avg("sentiment_score", 50)
    final_macro     = avg("macro_score", 50)
    final_confluence= avg("confluence_score", 0)

    # Turunkan confidence jika disagreement
    if len({r.get("action") for r in valid}) > 1:
        final_conf   = max(final_conf  - 20, 10)
        final_score  = max(final_score - 15, 10)

    # ── FILTER KETAT: confluence rendah → HOLD otomatis ──
    if final_confluence < 60 and final_action in ("LONG", "SHORT"):
        final_action = "HOLD"
        final_conf   = min(final_conf, 45)

    # ── Risk level paling konservatif ──
    risk_rank   = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "VERY_HIGH": 4}
    final_risk  = max(
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

    # FILTER: RRR < 2.0 → HOLD (tidak worth the risk)
    if final_rrr < 2.0 and final_action in ("LONG", "SHORT"):
        final_action = "HOLD"
        final_conf   = min(final_conf, 40)

    # ── Leverage — ambil yang paling konservatif dari semua AI ──
    levs      = [float(r["leverage"]) for r in valid if r.get("leverage")]
    final_lev = min(levs) if levs else 3.0
    final_lev = min(final_lev, 10.0)  # hard cap 10x

    # ── Position size — ambil terkecil ──
    pos_sizes  = [float(r["position_size_pct"]) for r in valid if r.get("position_size_pct")]
    final_pos  = min(pos_sizes) if pos_sizes else 2.0
    final_pos  = min(final_pos, 5.0)  # hard cap 5% equity

    # ── Holding period ──
    from collections import Counter
    holds      = [r.get("holding_period", "swing_1-3hari") for r in valid]
    final_hold = Counter(holds).most_common(1)[0][0]

    # ── Risks & catalysts deduplicated ──
    risks = list(dict.fromkeys(r for v in valid for r in v.get("key_risks", [])))[:5]
    cats  = list(dict.fromkeys(c for v in valid for c in v.get("catalysts", [])))[:4]
    invalidation = valid[0].get("invalidation", "")

    # ── Chain of thought gabungan ──
    cot_parts = []
    for r in valid:
        src = r.get("_source", "?")
        cot = r.get("chain_of_thought") or r.get("reasoning", "")
        if cot:
            cot_parts.append(f"[{src}]:\n{cot[:600]}")

    return {
        "action":            final_action,
        "confidence":        final_conf,
        "risk_level":        final_risk,
        "confluence_score":  final_confluence,
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
        "key_risks":         risks,
        "catalysts":         cats,
        "invalidation":      invalidation,
        "chain_of_thought":  "\n\n".join(cot_parts),
        "trade_plan":        valid[0].get("trade_plan", ""),
        "verdict":           valid[0].get("verdict", ""),
        "action_votes":      action_score,
        "_sources":          [r.get("_source") for r in valid],
        "_ai_count":         len(valid),
        "_individual_results": valid,
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

    sl_p  = max(atr_p * 1.5, 1.2)
    tp1_p = sl_p * 2.0
    tp2_p = sl_p * 3.5
    if action == "LONG":
        sl, tp1, tp2 = cur*(1-sl_p/100), cur*(1+tp1_p/100), cur*(1+tp2_p/100)
    elif action == "SHORT":
        sl, tp1, tp2 = cur*(1+sl_p/100), cur*(1-tp1_p/100), cur*(1-tp2_p/100)
    else:
        sl = tp1 = tp2 = cur

    overall = min(100, abs(score)*1.2 + 10)
    if action == "HOLD": overall = min(overall, 50)

    return {
        "action":action,"confidence":min(conf*0.7,60),"risk_level":risk,
        "overall_score":round(overall,1),"technical_score":round(overall,1),
        "fundamental_score":50,"sentiment_score":50,"macro_score":50,
        "entry_price":cur,"stop_loss":round(sl,6),
        "take_profit_1":round(tp1,6),"take_profit_2":round(tp2,6),"take_profit_3":round(tp2*1.1,6),
        "risk_reward_ratio":round(tp1_p/sl_p,2),"position_size_pct":2.0,"leverage":5,
        "holding_period":"swing_1-3hari",
        "chain_of_thought":f"[Teknikal-Fallback] score={score:+.0f}, regime={regime}, conflict={conflict}",
        "key_risks":[f"Volatilitas {rvol:.0f}%/tahun", f"Regime: {regime}", "AI tidak tersedia"],
        "catalysts":[],"trade_plan":"","verdict":f"Teknikal: {action} | Score {score:+.0f}",
        "action_votes":{action:1},"_sources":["Teknikal-Fallback"],"_ai_count":1,
        "_individual_results":[],
    }

# ═══════════════════════════════════════════════════════════════
#  FUNGSI UTAMA — dipanggil dari trading_bot
# ═══════════════════════════════════════════════════════════════
async def analyze_with_ai(
    symbol: str,
    technical: dict,
    metrics: dict,
    portfolio_ctx: dict,
    market_context: dict,     # dari news_fetcher.get_full_market_context
    order_book: dict = None,
    recent_trades: dict = None,
    chart_paths: dict = None,
) -> dict:
    """
    AI Trader mandiri — panggil Gemini + GPT + Claude paralel,
    gabungkan hasilnya, return keputusan final.
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

    # Jalankan semua AI paralel sekaligus
    ai_results = await asyncio.gather(
        _call_gemini(prompt),
        _call_gpt(prompt),
        _call_claude(prompt),
        return_exceptions=True
    )

    valid = [r for r in ai_results if isinstance(r, dict) and r]

    if not valid:
        console.print("  [red]Semua AI gagal → Teknikal Fallback[/red]")
        result = _technical_fallback(symbol, technical, metrics)
        result["symbol"] = symbol
        result["analyzed_at"] = datetime.now().isoformat()
        return result

    final = _ensemble(valid)
    final["symbol"]      = symbol
    final["analyzed_at"] = datetime.now().isoformat()

    # Log ringkas
    action = final.get("action","?")
    score  = final.get("overall_score",0)
    conf   = final.get("confidence",0)
    srcs   = ", ".join(final.get("_sources",[]))
    votes  = final.get("action_votes",{})
    color  = {"LONG":"green","SHORT":"red","HOLD":"yellow","AVOID":"dim"}.get(action,"white")
    console.print(
        f"  [bold {color}]→ {action}[/bold {color}] | "
        f"Score:{score:.0f} | Conf:{conf:.0f}% | "
        f"Votes:{votes} | AI:{srcs}"
    )
    return final
