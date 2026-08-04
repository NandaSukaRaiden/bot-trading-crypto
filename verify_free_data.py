"""
verify_free_data.py
Verifikasi semua sumber data gratis berjalan tanpa API key berbayar.
"""
from dotenv import load_dotenv; load_dotenv()
import asyncio, time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()
results = []

def check(name, source, fn, *args):
    import pandas as pd
    try:
        start = time.time()
        data  = fn(*args)
        ms    = int((time.time()-start)*1000)
        ok    = (not data.empty) if isinstance(data, pd.DataFrame) else bool(data)
        val   = ""
        if isinstance(data, pd.DataFrame):
            val = f"{len(data)} bars"
        elif isinstance(data, dict):
            val = str(list(data.items())[:1])[:50]
        elif isinstance(data, list):
            val = f"{len(data)} items"
        elif isinstance(data, (int,float)):
            val = str(round(data,4))
        results.append((name, source, "✅ OK" if ok else "❌ EMPTY", f"{ms}ms", val[:45]))
        color = "green" if ok else "red"
        console.print(f"  [{color}]{'OK' if ok else 'FAIL'}[/{color}] {name} ({ms}ms)" + (f" — {val}" if val else ""))
        return data
    except Exception as e:
        results.append((name, source, "❌ ERROR", "?", str(e)[:45]))
        console.print(f"  [red]FAIL[/red] {name}: {str(e)[:60]}")
        return None


async def main():
    console.print(Panel(
        "[bold cyan]🔍 VERIFIKASI DATA GRATIS — SEMUA SUMBER[/bold cyan]\n"
        "Semua data ini gratis tanpa perlu bayar apapun.\n"
        "Satu-satunya yang berbayar: API AI (Gemini/GPT/Claude)",
        border_style="cyan"
    ))

    # ── 1. Binance Public REST ─────────────────────────────────
    console.print("\n[bold]1. Binance Public REST (100% gratis, tanpa key)[/bold]")
    from data_fetcher import (
        fetch_ticker_binance, fetch_order_book, fetch_recent_trades,
        fetch_funding_rate, fetch_open_interest, fetch_ohlcv
    )
    check("BTC/USDT ticker",    "Binance REST", fetch_ticker_binance, "BTC/USDT")
    check("ETH/USDT ticker",    "Binance REST", fetch_ticker_binance, "ETH/USDT")
    check("SOL/USDT ticker",    "Binance REST", fetch_ticker_binance, "SOL/USDT")
    check("Order book BTC",     "Binance REST", fetch_order_book, "BTC/USDT")
    check("Recent trades BTC",  "Binance REST", fetch_recent_trades, "BTC/USDT")
    check("Funding rate BTC",   "Binance REST", fetch_funding_rate, "BTC/USDT")
    check("Open interest BTC",  "Binance REST", fetch_open_interest, "BTC/USDT")
    check("OHLCV 1h BTC",       "Binance REST", fetch_ohlcv, "BTC/USDT","1h",50)
    check("OHLCV 4h ETH",       "Binance REST", fetch_ohlcv, "ETH/USDT","4h",50)
    check("OHLCV 1d SOL",       "Binance REST", fetch_ohlcv, "SOL/USDT","1d",30)

    # ── 2. Fear & Greed ────────────────────────────────────────
    console.print("\n[bold]2. Fear & Greed Index (alternative.me, gratis)[/bold]")
    from news_fetcher import fetch_fear_greed
    fg = check("Fear & Greed",  "alternative.me", fetch_fear_greed)
    if fg:
        console.print(f"     → Value: {fg.get('value')}/100 ({fg.get('label')}) | Signal: {fg.get('signal','')[:60]}")

    # ── 3. CoinGecko fundamental ───────────────────────────────
    console.print("\n[bold]3. CoinGecko API (free tier, tanpa key)[/bold]")
    from news_fetcher import fetch_coingecko_fundamental
    cg = check("CoinGecko BTC fundamental", "CoinGecko", fetch_coingecko_fundamental, "BTC/USDT")
    if cg and cg.get("name"):
        console.print(f"     → {cg['name']} | Rank #{cg.get('market_cap_rank')} | Cap ${cg.get('market_cap_usd',0)/1e9:.1f}B")
        tw = cg.get('twitter_followers') or 0
        cm = cg.get('github_commits_4w') or 0
        console.print(f"     → Dev commits 4w: {cm} | Twitter: {tw:,}")

    # ── 4. Makro global ────────────────────────────────────────
    console.print("\n[bold]4. Makro Global via Yahoo Finance (gratis)[/bold]")
    from news_fetcher import fetch_macro_global
    macro = check("DXY/VIX/SP500",  "Yahoo Finance", fetch_macro_global)
    if macro:
        for k in ["DXY","SP500","VIX","Gold","Oil"]:
            d = macro.get(k) or {}
            if d.get("price"):
                console.print(f"     → {k}: ${d['price']:,.2f} ({d.get('change_pct',0):+.2f}%)")
        console.print(f"     → Note: {macro.get('interpretation','')[:70]}")

    # ── 5. RSS News (tanpa login, tanpa key) ──────────────────
    console.print("\n[bold]5. RSS Berita Crypto (CoinTelegraph/CoinDesk/Reddit/Google, gratis)[/bold]")
    from news_fetcher import fetch_crypto_news_all, score_sentiment
    articles = await fetch_crypto_news_all("BTC/USDT")
    sent = score_sentiment(articles)
    check_done = bool(articles)
    color = "green" if check_done else "red"
    console.print(f"  [{color}]{'OK' if check_done else 'FAIL'}[/{color}] RSS news ({len(articles)} artikel)")
    if articles:
        console.print(f"     → Sentimen: {sent['score']:.0f}/100 ({sent['label']})")
        for a in articles[:3]:
            console.print(f"     → [{a.get('source','?')[:20]}] {a.get('title','')[:65]}")
    results.append(("RSS News (15+ sumber)", "RSS Publik",
                    "✅ OK" if check_done else "❌ FAIL", "~", f"{len(articles)} artikel"))

    # ── 6. Binance Derivatives context ────────────────────────
    console.print("\n[bold]6. Binance Derivatives (gratis, tanpa key)[/bold]")
    from news_fetcher import fetch_binance_market_context
    bn = check("Funding+OI BTC", "Binance REST", fetch_binance_market_context, "BTC/USDT")
    if bn:
        console.print(f"     → Funding: {bn.get('funding_pct',0):+.4f}% | OI: {bn.get('open_interest',0):,.0f} | Signal: {bn.get('funding_signal','')[:50]}")

    # ── 7. Technical Analysis ──────────────────────────────────
    console.print("\n[bold]7. Technical Analysis (lokal, gratis)[/bold]")
    from technical_analyzer import multi_timeframe_analysis
    df1h = fetch_ohlcv("BTC/USDT","1h",100)
    df4h = fetch_ohlcv("BTC/USDT","4h",80)
    df1d = fetch_ohlcv("BTC/USDT","1d",50)
    tech = multi_timeframe_analysis({"1h":df1h,"4h":df4h,"1d":df1d}, 0.0001)
    console.print(f"  [green]OK[/green] Multi-TF Analysis")
    console.print(f"     → Score: {tech['score']:+.1f} | Dir: {tech['direction']} | Regime: {tech['market_regime']}")
    console.print(f"     → RSI: {tech['primary_rsi']:.1f} | ADX: {tech['primary_adx']:.1f} | VWAP: ${tech.get('vwap',0):,.2f}")
    console.print(f"     → CMF: {tech.get('cmf',0):.3f} | OBV rising: {tech.get('obv_rising')} | BB BW: {tech.get('bb_bw',0):.2f}%")
    console.print(f"     → PoC: ${tech.get('vp_poc',0):,.4f} | VAH: ${tech.get('vp_vah',0):,.4f} | VAL: ${tech.get('vp_val',0):,.4f}")
    console.print(f"     → SL saran: -{tech['suggested_sl_pct']:.2f}% | TP1: +{tech['suggested_tp1_pct']:.2f}% | RRR: 1:{tech['suggested_tp1_pct']/tech['suggested_sl_pct']:.1f}")
    results.append(("Technical Analysis (RSI+MACD+BB+ADX+VWAP+Ichimoku+VProfile)",
                    "Lokal (Python)", "✅ OK", "-", f"Score={tech['score']:+.1f}"))

    # ── Summary ───────────────────────────────────────────────
    console.print("\n")
    table = Table(title="📊 Ringkasan Sumber Data", style="cyan", show_lines=True)
    table.add_column("Data",    width=40)
    table.add_column("Sumber",  width=18)
    table.add_column("Status",  width=10)
    table.add_column("Waktu",   width=8)
    table.add_column("Sample",  width=40)
    for row in results:
        c = "green" if "OK" in row[2] else "red"
        table.add_row(*[f"[{c}]{row[0]}[/{c}]", row[1], row[2], row[3], row[4]])
    console.print(table)

    ok  = sum(1 for r in results if "OK" in r[2])
    err = sum(1 for r in results if "ERROR" in r[2] or "FAIL" in r[2])

    console.print(Panel(
        f"[bold green]✅ {ok} sumber data GRATIS berjalan[/bold green]\n"
        + (f"[red]❌ {err} gagal[/red]\n" if err else "")
        + "\n[bold]Biaya yang perlu dibayar:[/bold]\n"
        "  • [yellow]Hanya API AI (Gemini/GPT/Claude)[/yellow]\n"
        "  • [green]Semua data market: GRATIS 100%[/green]\n"
        "  • [green]Binance data (harga, OB, trades, funding, OI): GRATIS[/green]\n"
        "  • [green]CoinGecko fundamental: GRATIS[/green]\n"
        "  • [green]Fear & Greed, RSS news, makro: GRATIS[/green]\n"
        "  • [green]Technical analysis (VWAP, Ichimoku, dll): GRATIS[/green]",
        border_style="green"
    ))


if __name__ == "__main__":
    asyncio.run(main())
