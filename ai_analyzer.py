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
Kamu AI Crypto SCALPER profesional. Analisis CEPAT, keputusan TEGAS, eksekusi LANGSUNG.

GAYA TRADING: SCALPING 20x leverage
- Entry: 5m/15m, konfirmasi 1h
- Target: 0.5-2.5% profit per trade
- SL: 0.3-1.0% (TIPIS, sebelum likuidasi)
- Holding: MENIT sampai JAM (bukan hari)
- Risk: 0.5% equity per trade

ATURAN ENTRY (TIDAK BISA DILANGGAR):
1. RRR min 1:1.5 (TP1 minimal 1.5x jarak SL)
2. SL WAJIB di support/resistance valid (ATR × 1.2 dari entry)
3. Leverage 20-25x HANYA jika SL tipis (<1%)
4. Volume >1.5x rata-rata untuk breakout
5. TF conflict (5m vs 1h berlawanan) → HOLD
6. Confluence <60% → HOLD
7. Funding >+0.10% → hindari LONG | <-0.10% → hindari SHORT

💎 DEEP BUY PRIORITY:
Jika ada deep buy signal (drop >2% + oversold RSI<35 + volume spike):
- PRIORITAS TERTINGGI untuk LONG
- Size bisa 75% risk normal (chance bagus!)
- SL di recent low - ATR × 1.5

KAPAN HOLD:
- Confluence <60% | Volume rendah | Ranging market
- Harga di tengah range | Berita campur aduk
- Sudah ada posisi di coin ini

OUTPUT (JSON ONLY, NO TEXT):
{
  "action": "LONG|SHORT|HOLD",
  "confidence": 0-100,
  "risk_level": "LOW|MEDIUM|HIGH",
  "confluence_score": 0-100,
  "entry_price": number,
  "stop_loss": number,
  "stop_loss_reason": "why SL here",
  "take_profit_1": number,
  "take_profit_2": number,
  "risk_reward_ratio": number,
  "leverage": 20-25,
  "leverage_reason": "why safe",
  "holding_period": "scalp_menit|scalp_jam",
  "overall_score": 0-100,
  "verdict": "keputusan final 1 kalimat"
}

PENTING:
- confluence_score <60 → action WAJIB "HOLD"
- leverage × SL% harus <5% (agar SL menang sebelum likuidasi)
- stop_loss untuk LONG < entry_price, SHORT > entry_price
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
    market_context: dict,
    order_book: dict = None,
    recent_trades: dict = None,
    chart_paths: dict = None,
) -> str:
    """Build RINGKAS prompt untuk AI decision CEPAT."""
    
    price = metrics.get("price", 0)
    regime = technical.get("market_regime", "?")
    score = technical.get("score", 0)
    
    # Quick summary per TF
    tf_summary = []
    tf = technical.get("per_timeframe", {})
    for label in ["5m", "15m", "1h"]:  # Focus on key TFs only
        r = tf.get(label, {})
        if isinstance(r, dict):
            tf_summary.append(f"{label}: {r.get('direction_bias','?')} RSI={r.get('rsi',50):.0f} Vol={r.get('volume_ratio',1):.1f}x")
    
    # Support/Resistance
    sr = technical.get("support_resistance", {})
    sr_text = f"Sup1=${sr.get('support1',0):,.0f} Res1=${sr.get('resistance1',0):,.0f}"
    
    # News (only if important)
    news_text = ""
    if market_context:
        text = market_context.get("text", "")
        if len(text) > 50:  # ada berita
            news_text = f"\nNEWS: {text[:200]}..."
    
    # Position context
    has_pos = portfolio_ctx.get("has_position", False)
    pos_text = f"Has position: {has_pos}"
    if has_pos:
        pos_detail = portfolio_ctx.get("open_positions_detail", [])
        if pos_detail:
            p = pos_detail[0]
            pos_text += f" ({p.get('side')} @ ${p.get('entry_price',0):,.2f}, PnL={p.get('pnl_pct',0):+.1f}%)"
    
    prompt = f"""
# SCALP ANALYSIS: {symbol}

## MARKET DATA
Price: ${price:,.2f} | 24h: {metrics.get('change_24h_pct',0):+.1f}% | Funding: {metrics.get('funding_rate',0):+.4f}%
Regime: {regime} | Score: {score:+.0f} | ATR: {technical.get('atr_pct',0):.2f}%

## TIMEFRAMES
{chr(10).join(tf_summary)}

## LEVELS
{sr_text}

## PORTFOLIO
Equity: ${portfolio_ctx.get('equity',0):,.0f} | {pos_text}
{news_text}

DECIDE: LONG/SHORT/HOLD with tight SL (0.3-1.0%), leverage 20-25x, RRR ≥1.5
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
