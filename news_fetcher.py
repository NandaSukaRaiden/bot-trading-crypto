"""
news_fetcher.py — Agregator berita crypto Bloomberg-grade TANPA LOGIN
Sumber (semua publik, zero-auth):
  1. CryptoPanic API (free tier, tanpa login)
  2. RSS: CoinTelegraph, CoinDesk, Decrypt, TheBlock, BeInCrypto
  3. Google News RSS (keyword per coin)
  4. Reddit RSS (r/cryptocurrency, r/bitcoin, dll) — publik
  5. Fear & Greed Index (alternative.me) — gratis
  6. CoinGecko public API — fundamental + market data
  7. Binance public API — funding rate, OI, dominance
  8. Makro global: DXY, S&P500, Gold (via yfinance tanpa auth)
"""
import asyncio, re, json, time
import feedparser, aiohttp, requests
from datetime import datetime, timedelta, timezone
from cachetools import TTLCache
from rich.console import Console
from config import (
    CRYPTOPANIC_API_KEY, CRYPTOPANIC_BASE,
    FEAR_GREED_URL, COINGECKO_BASE, COINGECKO_API_KEY,
    COINGECKO_IDS, NEWS_MAX_ITEMS, NEWS_TTL_MIN,
    CRYPTO_NEWS_RSS, GOOGLE_NEWS_CRYPTO,
    BINANCE_FUNDING_URL, BINANCE_OI_URL,
)
console = Console()

# ── Cache ─────────────────────────────────────────────────────
_news_cache   = TTLCache(maxsize=200, ttl=NEWS_TTL_MIN * 60)
_fg_cache     = TTLCache(maxsize=1,   ttl=1800)    # 30 menit
_cg_cache     = TTLCache(maxsize=100, ttl=300)     # 5 menit
_macro_cache  = TTLCache(maxsize=10,  ttl=600)     # 10 menit

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ── Reddit RSS feeds (100% publik tanpa login) ────────────────
REDDIT_RSS = [
    "https://www.reddit.com/r/CryptoCurrency/hot/.rss?limit=15",
    "https://www.reddit.com/r/Bitcoin/hot/.rss?limit=10",
    "https://www.reddit.com/r/ethereum/hot/.rss?limit=10",
    "https://www.reddit.com/r/CryptoMarkets/hot/.rss?limit=10",
    "https://www.reddit.com/r/altcoin/hot/.rss?limit=8",
    "https://www.reddit.com/r/defi/hot/.rss?limit=8",
]

# ── Keyword bobot sentimen ─────────────────────────────────────
POS_WORDS = [
    "rally","surge","bullish","all-time high","ath","adoption","partnership",
    "upgrade","launch","mainnet","etf approved","institutional","accumulate",
    "buy","breakout","outperform","positive","growth","gains","pump","moon",
    "listing","integration","staking rewards","halving","supply crunch",
]
NEG_WORDS = [
    "hack","exploit","rug","scam","fraud","sec","lawsuit","ban","crash",
    "bearish","dump","sell","liquidation","bankruptcy","regulation","restrict",
    "ponzi","loss","decline","drop","plunge","warning","risk","bear","fear",
    "attack","vulnerability","shutdown","delist","probe","investigation",
]

# ═══════════════════════════════════════════════════════════════
#  1. FEAR & GREED INDEX — alternative.me (tanpa auth)
# ═══════════════════════════════════════════════════════════════
def fetch_fear_greed() -> dict:
    """Ambil Fear & Greed Index + tren 7 hari terakhir."""
    if "fg" in _fg_cache:
        return _fg_cache["fg"]
    try:
        r = requests.get(FEAR_GREED_URL, timeout=8, headers=HEADERS)
        data = r.json().get("data", [])
        if not data:
            raise ValueError("empty")
        latest = data[0]
        history = [{"date": d["timestamp"], "value": int(d["value"]),
                    "label": d["value_classification"]} for d in data[:7]]
        val = int(latest["value"])
        result = {
            "value":   val,
            "label":   latest["value_classification"],
            "history": history,
            "trend":   _fg_trend(history),
            "signal":  _fg_signal(val),
        }
        _fg_cache["fg"] = result
        return result
    except Exception as e:
        console.print(f"[dim]Fear&Greed error: {e}[/dim]")
        return {"value": 50, "label": "Neutral", "signal": "Market neutral", "trend": "stable"}

def _fg_trend(history: list) -> str:
    if len(history) < 3:
        return "stable"
    recent = [h["value"] for h in history[:3]]
    older  = [h["value"] for h in history[3:7]] if len(history) >= 7 else recent
    avg_r, avg_o = sum(recent)/len(recent), sum(older)/len(older)
    if avg_r > avg_o + 5: return "improving (greed increasing)"
    if avg_r < avg_o - 5: return "deteriorating (fear increasing)"
    return "stable"

def _fg_signal(val: int) -> str:
    if val <= 15: return "EXTREME FEAR — kapitulasi pasar, potensi LONG contrarian"
    if val <= 30: return "FEAR — sentimen negatif, potensi akumulasi bertahap"
    if val <= 45: return "FEAR RINGAN — waspada, konfirmasi teknikal diperlukan"
    if val <= 55: return "NEUTRAL — tidak ada sinyal kuat dari sentimen"
    if val <= 70: return "GREED — sentimen positif, tren bullish berlanjut"
    if val <= 85: return "GREED TINGGI — hati-hati overbought, pertimbangkan profit taking"
    return "EXTREME GREED — euforia, risiko reversal besar, hindari LONG baru"

# ═══════════════════════════════════════════════════════════════
#  2. COINGECKO — Fundamental crypto (publik, tanpa key)
# ═══════════════════════════════════════════════════════════════
def fetch_coingecko_fundamental(symbol: str) -> dict:
    """Ambil fundamental lengkap dari CoinGecko public API."""
    cg_id = COINGECKO_IDS.get(symbol)
    if not cg_id:
        return {}
    cache_key = f"cg_{cg_id}"
    if cache_key in _cg_cache:
        return _cg_cache[cache_key]
    try:
        base = COINGECKO_BASE
        headers = dict(HEADERS)
        if COINGECKO_API_KEY:
            headers["x-cg-pro-api-key"] = COINGECKO_API_KEY
        url = f"{base}/coins/{cg_id}?localization=false&tickers=false&market_data=true&community_data=true&developer_data=true"
        r = requests.get(url, headers=headers, timeout=12)
        if r.status_code != 200:
            return {}
        d = r.json()
        md = d.get("market_data", {})
        dd = d.get("developer_data", {})
        cd = d.get("community_data", {})
        def g(obj, *keys, default=None):
            for k in keys:
                if isinstance(obj, dict): obj = obj.get(k, default)
                else: return default
            return obj
        result = {
            "name":               d.get("name"),
            "symbol":             d.get("symbol","").upper(),
            "categories":         d.get("categories", [])[:5],
            "description":        (d.get("description",{}).get("en","") or "")[:500],
            "genesis_date":       d.get("genesis_date"),
            "hashing_algorithm":  d.get("hashing_algorithm"),
            # Market data
            "price_usd":          g(md,"current_price","usd"),
            "market_cap_usd":     g(md,"market_cap","usd"),
            "market_cap_rank":    md.get("market_cap_rank"),
            "fully_diluted_val":  g(md,"fully_diluted_valuation","usd"),
            "vol_24h":            g(md,"total_volume","usd"),
            "high_24h":           g(md,"high_24h","usd"),
            "low_24h":            g(md,"low_24h","usd"),
            "price_change_24h":   md.get("price_change_percentage_24h"),
            "price_change_7d":    md.get("price_change_percentage_7d"),
            "price_change_30d":   md.get("price_change_percentage_30d"),
            "ath":                g(md,"ath","usd"),
            "ath_change_pct":     g(md,"ath_change_percentage","usd"),
            "atl":                g(md,"atl","usd"),
            "circ_supply":        md.get("circulating_supply"),
            "total_supply":       md.get("total_supply"),
            "max_supply":         md.get("max_supply"),
            "supply_ratio":       round(md.get("circulating_supply",0) / md.get("total_supply",1) * 100, 1) if md.get("total_supply") else None,
            # Developer activity (proxy bisnis)
            "github_stars":       dd.get("stars"),
            "github_forks":       dd.get("forks"),
            "github_commits_4w":  dd.get("commit_count_4_weeks"),
            "github_contributors":dd.get("pull_request_contributors"),
            # Community
            "twitter_followers":  cd.get("twitter_followers"),
            "reddit_subscribers": cd.get("reddit_subscribers"),
            "reddit_active_24h":  cd.get("reddit_accounts_active_48h"),
            # Scores
            "coingecko_score":    d.get("coingecko_score"),
            "dev_score":          d.get("developer_score"),
            "community_score":    d.get("community_score"),
            "liquidity_score":    d.get("liquidity_score"),
            "public_interest_score": d.get("public_interest_score"),
        }
        _cg_cache[cache_key] = result
        return result
    except Exception as e:
        console.print(f"[dim]CoinGecko {symbol}: {e}[/dim]")
        return {}

# ═══════════════════════════════════════════════════════════════
#  3. BINANCE PUBLIC — Funding rate, OI, dominance (tanpa key)
# ═══════════════════════════════════════════════════════════════
def fetch_binance_market_context(symbol: str) -> dict:
    """Ambil funding rate + open interest dari Binance public API."""
    clean = symbol.replace("/","")   # BTC/USDT → BTCUSDT
    ctx = {}
    try:
        r = requests.get(BINANCE_FUNDING_URL, params={"symbol": clean}, timeout=6, headers=HEADERS)
        if r.status_code == 200:
            d = r.json()
            fr = float(d.get("lastFundingRate", 0))
            ctx["funding_rate"]       = fr
            ctx["funding_pct"]        = round(fr * 100, 4)
            ctx["next_funding_time"]  = d.get("nextFundingTime")
            ctx["mark_price"]         = float(d.get("markPrice", 0))
            ctx["index_price"]        = float(d.get("indexPrice", 0))
            ctx["funding_signal"] = _funding_signal(fr)
    except Exception:
        pass
    try:
        r2 = requests.get(BINANCE_OI_URL, params={"symbol": clean}, timeout=6, headers=HEADERS)
        if r2.status_code == 200:
            ctx["open_interest"] = float(r2.json().get("openInterest", 0))
    except Exception:
        pass
    return ctx

def _funding_signal(fr: float) -> str:
    if fr > 0.002:  return "FUNDING SANGAT TINGGI (+) → long overcrowded, risiko long squeeze"
    if fr > 0.0005: return "FUNDING POSITIF → lebih banyak long, hati-hati"
    if fr < -0.002: return "FUNDING SANGAT NEGATIF (-) → short overcrowded, potensi short squeeze ke atas"
    if fr < -0.0005:return "FUNDING NEGATIF → lebih banyak short, potensi squeeze"
    return "FUNDING NETRAL → tidak ada bias crowding"

# ═══════════════════════════════════════════════════════════════
#  4. MAKRO GLOBAL (tanpa auth, via yfinance public)
# ═══════════════════════════════════════════════════════════════
def fetch_macro_global() -> dict:
    """DXY, S&P500, Gold, Oil — indikator risk-on/off pasar global."""
    if "macro" in _macro_cache:
        return _macro_cache["macro"]
    import yfinance as yf
    tickers = {"DXY": "DX-Y.NYB", "SP500": "^GSPC", "Gold": "GC=F",
               "Oil": "CL=F", "VIX": "^VIX", "BTC_dom": None}
    macro = {}
    for name, yt in tickers.items():
        if yt is None: continue
        try:
            info = yf.Ticker(yt).info
            price  = info.get("regularMarketPrice") or info.get("ask") or 0
            change = info.get("regularMarketChangePercent") or 0
            macro[name] = {"price": round(float(price), 2),
                           "change_pct": round(float(change), 2)}
        except Exception:
            macro[name] = None
    macro["interpretation"] = _macro_interpret(macro)
    macro["fetched_at"] = datetime.now().isoformat()
    _macro_cache["macro"] = macro
    return macro

def _macro_interpret(macro: dict) -> str:
    parts = []
    dxy = macro.get("DXY") or {}
    sp  = macro.get("SP500") or {}
    vix = macro.get("VIX") or {}
    if dxy.get("change_pct", 0) > 0.5:
        parts.append("DXY menguat → tekanan jual crypto (risk-off)")
    elif dxy.get("change_pct", 0) < -0.5:
        parts.append("DXY melemah → kondusif untuk crypto (risk-on)")
    if sp.get("change_pct", 0) > 1:
        parts.append("S&P500 naik kuat → risk appetite tinggi, positif crypto")
    elif sp.get("change_pct", 0) < -1:
        parts.append("S&P500 turun → risk-off, potensi jual crypto")
    if vix.get("price", 20) > 30:
        parts.append(f"VIX={vix.get('price',0):.0f} (tinggi) → fear ekstrem di pasar global")
    elif vix.get("price", 20) < 15:
        parts.append(f"VIX={vix.get('price',0):.0f} (rendah) → kondisi pasar tenang")
    return " | ".join(parts) if parts else "Makro global dalam kondisi normal"

# ═══════════════════════════════════════════════════════════════
#  5. RSS NEWS — semua sumber crypto tanpa login
# ═══════════════════════════════════════════════════════════════
async def _fetch_rss(session: aiohttp.ClientSession, url: str) -> list:
    """Fetch satu RSS feed async."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                return []
            text = await resp.text(errors="replace")
        feed = feedparser.parse(text)
        articles = []
        for e in feed.entries[:12]:
            pub = getattr(e, "published", "") or getattr(e, "updated", "")
            try:
                from email.utils import parsedate_to_datetime
                pub_dt = parsedate_to_datetime(pub)
                if (datetime.now(pub_dt.tzinfo) - pub_dt).days > 3:
                    continue   # skip berita > 3 hari
            except Exception:
                pass
            summary = re.sub(r"<[^>]+>", " ", getattr(e, "summary", "") or "").strip()[:400]
            title   = re.sub(r"<[^>]+>", " ", getattr(e, "title", "") or "").strip()
            if not title:
                continue
            articles.append({
                "title":   title,
                "summary": summary,
                "link":    getattr(e, "link", ""),
                "pub":     pub[:30],
                "source":  (getattr(feed.feed, "title", "") or url.split("/")[2])[:40],
            })
        return articles
    except Exception:
        return []

async def _fetch_cryptopanic(coin_name: str) -> list:
    """CryptoPanic API — free tier tanpa login jika tidak ada key."""
    try:
        params = {
            "public": "true",
            "filter": "hot",
            "currencies": coin_name[:3].upper(),
            "kind": "news",
        }
        if CRYPTOPANIC_API_KEY:
            params["auth_token"] = CRYPTOPANIC_API_KEY
        async with aiohttp.ClientSession(headers=HEADERS) as session:
            async with session.get(CRYPTOPANIC_BASE, params=params,
                                   timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
        articles = []
        for item in (data.get("results") or [])[:15]:
            articles.append({
                "title":   item.get("title", ""),
                "summary": "",
                "link":    item.get("url", ""),
                "pub":     item.get("published_at", "")[:20],
                "source":  "CryptoPanic",
                "votes":   item.get("votes", {}),
            })
        return articles
    except Exception:
        return []

async def fetch_crypto_news_all(symbol: str) -> list[dict]:
    """
    Kumpulkan berita dari SEMUA sumber tanpa login:
    RSS (CoinTelegraph, CoinDesk, Decrypt, BeInCrypto, TheBlock) +
    Reddit RSS + Google News RSS + CryptoPanic public
    """
    cache_key = f"cnews_{symbol}"
    if cache_key in _news_cache:
        return _news_cache[cache_key]

    coin = symbol.split("/")[0].lower()    # BTC/USDT → btc
    coin_up = coin.upper()                 # BTC

    # Google News RSS per coin
    google_urls = [
        GOOGLE_NEWS_CRYPTO.format(query=f"{coin_up}+bitcoin" if coin=="btc" else coin_up),
        GOOGLE_NEWS_CRYPTO.format(query=f"{coin_up}+crypto+price"),
    ]

    all_urls = list(CRYPTO_NEWS_RSS) + list(REDDIT_RSS) + google_urls

    connector = aiohttp.TCPConnector(limit=20, ssl=False)
    async with aiohttp.ClientSession(headers=HEADERS, connector=connector) as session:
        rss_tasks = [_fetch_rss(session, url) for url in all_urls]
        rss_results = await asyncio.gather(*rss_tasks, return_exceptions=True)

    all_articles = []
    for r in rss_results:
        if isinstance(r, list):
            all_articles.extend(r)

    # CryptoPanic (terpisah, clientnya sendiri)
    cp_articles = await _fetch_cryptopanic(coin_up)
    all_articles.extend(cp_articles)

    # Filter relevan ke coin ini
    search_terms = [coin, coin_up, symbol.replace("/USDT","").lower()]
    # Coin name mapping
    coin_names = {
        "btc":"bitcoin","eth":"ethereum","bnb":"binance","sol":"solana",
        "xrp":"ripple","ada":"cardano","doge":"dogecoin","link":"chainlink",
        "dot":"polkadot","avax":"avalanche","matic":"polygon","ltc":"litecoin",
        "atom":"cosmos","near":"near protocol","apt":"aptos","arb":"arbitrum",
        "op":"optimism","sui":"sui","inj":"injective",
    }
    if coin in coin_names:
        search_terms.append(coin_names[coin])

    relevant = [a for a in all_articles
                if any(t in (a["title"]+" "+a["summary"]).lower() for t in search_terms)]

    # Jika terlalu sedikit, ambil semua berita crypto umum
    if len(relevant) < 5:
        relevant = all_articles[:20]

    # Deduplikasi
    seen, unique = set(), []
    for a in relevant:
        key = a["title"][:50].lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(a)

    unique = sorted(unique, key=lambda x: x.get("pub",""), reverse=True)
    unique = unique[:NEWS_MAX_ITEMS]
    _news_cache[cache_key] = unique
    return unique

# ═══════════════════════════════════════════════════════════════
#  6. SCORING SENTIMEN BERITA
# ═══════════════════════════════════════════════════════════════
def score_sentiment(articles: list) -> dict:
    """Hitung skor sentimen 0-100 dari headlines berita."""
    if not articles:
        return {"score": 50, "label": "NEUTRAL", "pos": 0, "neg": 0, "neu": 0}
    pos = neg = neu = 0
    for a in articles:
        text = (a.get("title","") + " " + a.get("summary","")).lower()
        p = sum(1 for w in POS_WORDS if w in text)
        n = sum(1 for w in NEG_WORDS if w in text)
        if p > n:   pos += 1
        elif n > p: neg += 1
        else:       neu += 1
    total = pos + neg + neu
    score = round(50 + (pos - neg) / max(total, 1) * 50, 1)
    score = max(0, min(100, score))
    label = ("VERY_BULLISH" if score>=75 else "BULLISH" if score>=60
             else "NEUTRAL" if score>=40 else "BEARISH" if score>=25 else "VERY_BEARISH")
    return {"score": score, "label": label, "pos": pos, "neg": neg, "neu": neu, "total": total}

# ═══════════════════════════════════════════════════════════════
#  7. FORMAT TEKS UNTUK AI PROMPT (Bloomberg Intelligence style)
# ═══════════════════════════════════════════════════════════════
def format_full_context_for_ai(
    symbol: str,
    articles: list,
    fundamental: dict,
    macro: dict,
    fear_greed: dict,
    binance_ctx: dict,
) -> str:
    """Buat ringkasan lengkap seperti Bloomberg Brief untuk AI."""
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"MARKET INTELLIGENCE BRIEF — {symbol}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    lines.append(f"{'='*60}")

    # ── Fear & Greed ──
    fg = fear_greed or {}
    lines.append(f"\n[MARKET SENTIMENT — CRYPTO FEAR & GREED INDEX]")
    lines.append(f"  Current: {fg.get('value',50)}/100 — {fg.get('label','Neutral')}")
    lines.append(f"  Signal:  {fg.get('signal','')}")
    lines.append(f"  7-Day Trend: {fg.get('trend','stable')}")

    # ── Makro Global ──
    lines.append(f"\n[GLOBAL MACRO CONTEXT]")
    for key in ["DXY","SP500","Gold","Oil","VIX"]:
        m = (macro or {}).get(key) or {}
        if m:
            arrow = "↑" if m.get("change_pct",0) >= 0 else "↓"
            lines.append(f"  {key:8}: ${m.get('price',0):,.2f}  {arrow}{abs(m.get('change_pct',0)):.2f}%")
    lines.append(f"  Interpretation: {(macro or {}).get('interpretation','N/A')}")

    # ── Binance Market Context ──
    bc = binance_ctx or {}
    if bc:
        lines.append(f"\n[BINANCE DERIVATIVES CONTEXT — {symbol}]")
        lines.append(f"  Funding Rate: {bc.get('funding_pct',0):+.4f}%  → {bc.get('funding_signal','')}")
        if bc.get("open_interest"):
            lines.append(f"  Open Interest: {bc.get('open_interest',0):,.0f} contracts")
        if bc.get("mark_price"):
            lines.append(f"  Mark Price: ${bc.get('mark_price',0):,.4f}  |  Index: ${bc.get('index_price',0):,.4f}")

    # ── CoinGecko Fundamental ──
    f = fundamental or {}
    if f:
        lines.append(f"\n[FUNDAMENTAL — CoinGecko]")
        lines.append(f"  Name       : {f.get('name','?')} ({f.get('symbol','?')})")
        lines.append(f"  Rank       : #{f.get('market_cap_rank','?')}")
        lines.append(f"  Market Cap : ${f.get('market_cap_usd',0):,.0f}")
        lines.append(f"  24h Vol    : ${f.get('vol_24h',0):,.0f}")
        lines.append(f"  Change 24h : {f.get('price_change_24h',0):+.2f}%  7d: {f.get('price_change_7d',0):+.2f}%  30d: {f.get('price_change_30d',0):+.2f}%")
        lines.append(f"  ATH        : ${f.get('ath',0):,.4f}  ({f.get('ath_change_pct',0):.1f}% dari ATH)")
        lines.append(f"  Supply     : {f.get('circ_supply',0):,.0f} / {f.get('total_supply',0) or 'unlimited'} ({f.get('supply_ratio','?')}% circulating)")
        if f.get("github_commits_4w") is not None:
            lines.append(f"  Dev Activity (4wk): {f.get('github_commits_4w',0)} commits | {f.get('github_contributors',0)} contributors")
        if f.get("coingecko_score"):
            lines.append(f"  CG Scores: overall={f.get('coingecko_score',0):.1f} dev={f.get('dev_score',0):.1f} community={f.get('community_score',0):.1f} liquidity={f.get('liquidity_score',0):.1f}")
        if f.get("description"):
            lines.append(f"  About: {f.get('description','')[:300]}")

    # ── Berita Terkini ──
    sentiment = score_sentiment(articles)
    lines.append(f"\n[LATEST NEWS & SENTIMENT — {symbol}]")
    lines.append(f"  Sentiment Score: {sentiment['score']:.0f}/100 ({sentiment['label']}) — "
                 f"{sentiment['pos']} bullish / {sentiment['neg']} bearish / {sentiment['neu']} neutral")
    lines.append(f"  Sources: {len(articles)} articles (CoinTelegraph, CoinDesk, Decrypt, Reddit, Google News, CryptoPanic)")
    for i, a in enumerate(articles[:15], 1):
        src  = a.get("source","?")[:25]
        title = a.get("title","")[:100]
        pub   = a.get("pub","")[:16]
        lines.append(f"  {i:2}. [{src}] {title} ({pub})")
        if a.get("summary"):
            lines.append(f"      → {a['summary'][:180]}")

    lines.append(f"{'='*60}")
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════
#  8. FUNGSI UTAMA — dipanggil dari trading_bot & ai_analyzer
# ═══════════════════════════════════════════════════════════════
async def get_full_market_context(symbol: str) -> dict:
    """
    Kumpulkan SEMUA konteks pasar untuk satu simbol secara paralel.
    Tidak memerlukan login apapun.
    Return: dict dengan semua data + teks siap masuk prompt AI.
    """
    # Jalankan semua fetch secara paralel
    news_task   = fetch_crypto_news_all(symbol)
    cg_task     = asyncio.to_thread(fetch_coingecko_fundamental, symbol)
    macro_task  = asyncio.to_thread(fetch_macro_global)
    fg_task     = asyncio.to_thread(fetch_fear_greed)
    bn_task     = asyncio.to_thread(fetch_binance_market_context, symbol)

    news, fundamental, macro, fear_greed, binance_ctx = await asyncio.gather(
        news_task, cg_task, macro_task, fg_task, bn_task,
        return_exceptions=True
    )
    if isinstance(news, Exception):       news       = []
    if isinstance(fundamental, Exception):fundamental = {}
    if isinstance(macro, Exception):      macro      = {}
    if isinstance(fear_greed, Exception): fear_greed = {"value":50,"label":"Neutral","signal":""}
    if isinstance(binance_ctx, Exception):binance_ctx = {}

    sentiment = score_sentiment(news)
    text = format_full_context_for_ai(symbol, news, fundamental, macro, fear_greed, binance_ctx)

    return {
        "symbol":      symbol,
        "articles":    news,
        "fundamental": fundamental,
        "macro":       macro,
        "fear_greed":  fear_greed,
        "binance_ctx": binance_ctx,
        "sentiment":   sentiment,
        "text":        text,          # ← siap masuk prompt AI
        "fetched_at":  datetime.now().isoformat(),
    }

def get_full_market_context_sync(symbol: str) -> dict:
    """Versi synchronous untuk penggunaan di luar async context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                return pool.submit(asyncio.run, get_full_market_context(symbol)).result(timeout=30)
        return loop.run_until_complete(get_full_market_context(symbol))
    except RuntimeError:
        return asyncio.run(get_full_market_context(symbol))
