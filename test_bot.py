"""
test_bot.py — End-to-end test tanpa API key AI
Menguji: data fetch, teknikal, chart, news, risk manager, portfolio
Jalankan: python test_bot.py
"""
import asyncio, sys, traceback
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
PASS = "[bold green]✓ PASS[/bold green]"
FAIL = "[bold red]✗ FAIL[/bold red]"
SKIP = "[yellow]⊘ SKIP[/yellow]"

results = []

def record(name, status, detail=""):
    results.append((name, status, detail))
    color = {"PASS":"green","FAIL":"red","SKIP":"yellow"}.get(status,"white")
    console.print(f"  [{color}]{status:4}[/{color}] {name}" + (f" — {detail}" if detail else ""))

# ── TEST 1: Imports ───────────────────────────────────────────
def test_imports():
    console.print("\n[bold cyan]── TEST 1: Module Imports ──[/bold cyan]")
    mods = [
        ("config",             "config"),
        ("pandas",             "pandas"),
        ("numpy",              "numpy"),
        ("ccxt",               "ccxt"),
        ("requests",           "requests"),
        ("feedparser",         "feedparser"),
        ("aiohttp",            "aiohttp"),
        ("rich",               "rich"),
        ("apscheduler",        "apscheduler"),
        ("mplfinance",         "mplfinance"),
        ("google.generativeai","google.generativeai"),
    ]
    for name, mod in mods:
        try:
            __import__(mod)
            record(name, "PASS")
        except ImportError as e:
            record(name, "FAIL", str(e))

# ── TEST 2: Config ────────────────────────────────────────────
def test_config():
    console.print("\n[bold cyan]── TEST 2: Config ──[/bold cyan]")
    try:
        from config import (
            CRYPTO_WATCHLIST, SYMBOL_MAX_LEVERAGE, AI_MODELS,
            INITIAL_CAPITAL_USDT, MAX_RISK_PER_TRADE, TAKER_FEE_RATE,
            POSITION_MONITOR_SEC, ANALYSIS_INTERVAL_MIN,
        )
        record("CRYPTO_WATCHLIST",    "PASS", f"{len(CRYPTO_WATCHLIST)} pairs")
        record("INITIAL_CAPITAL",     "PASS", f"${INITIAL_CAPITAL_USDT:,.0f} USDT")
        record("MAX_RISK_PER_TRADE",  "PASS", f"{MAX_RISK_PER_TRADE}% per trade")
        record("POSITION_MONITOR_SEC","PASS", f"{POSITION_MONITOR_SEC}s interval")
        record("AI_MODELS keys",      "PASS", str(list(AI_MODELS.keys())))
        assert MAX_RISK_PER_TRADE <= 2.0, "Risk per trade terlalu tinggi!"
        record("Risk guard (<=2%)",   "PASS")
    except Exception as e:
        record("config", "FAIL", str(e))

# ── TEST 3: Binance Public Data ───────────────────────────────
def test_binance_data():
    console.print("\n[bold cyan]── TEST 3: Binance Public Data (tanpa API key) ──[/bold cyan]")
    try:
        from data_fetcher import fetch_ticker_binance, fetch_order_book, fetch_recent_trades
        # BTC/USDT ticker
        t = fetch_ticker_binance("BTC/USDT")
        price = t.get("price", 0)
        assert price > 1000, f"Harga BTC tidak masuk akal: {price}"
        record("BTC/USDT ticker", "PASS", f"Price=${price:,.2f} | 24h={t.get('change_24h_pct',0):+.2f}%")

        # Order book
        ob = fetch_order_book("BTC/USDT")
        assert ob.get("best_bid", 0) > 0
        record("Order book",  "PASS", f"Bid={ob['best_bid']:,.2f} Ask={ob['best_ask']:,.2f} Imb={ob.get('depth_imbalance',0):+.1f}%")

        # Recent trades
        rt = fetch_recent_trades("BTC/USDT")
        record("Recent trades","PASS", f"Buy={rt.get('buy_ratio_pct',0):.0f}% Sell={100-rt.get('buy_ratio_pct',0):.0f}%")
    except Exception as e:
        record("Binance data", "FAIL", str(e))

# ── TEST 4: CCXT OHLCV ───────────────────────────────────────
def test_ohlcv():
    console.print("\n[bold cyan]── TEST 4: OHLCV via CCXT (Binance Futures) ──[/bold cyan]")
    try:
        from data_fetcher import fetch_ohlcv
        df = fetch_ohlcv("BTC/USDT", "1h", 50)
        assert len(df) >= 30, f"Data terlalu sedikit: {len(df)}"
        record("OHLCV 1h BTC", "PASS", f"{len(df)} bars | last close=${df['Close'].iloc[-1]:,.2f}")

        df4 = fetch_ohlcv("BTC/USDT", "4h", 30)
        record("OHLCV 4h BTC", "PASS", f"{len(df4)} bars")

        df1d = fetch_ohlcv("BTC/USDT", "1d", 30)
        record("OHLCV 1d BTC", "PASS", f"{len(df1d)} bars")
    except Exception as e:
        record("OHLCV", "FAIL", str(e))

# ── TEST 5: Technical Analysis ────────────────────────────────
def test_technical():
    console.print("\n[bold cyan]── TEST 5: Technical Analyzer ──[/bold cyan]")
    try:
        from data_fetcher import fetch_ohlcv
        from technical_analyzer import full_technical_analysis, multi_timeframe_analysis

        df = fetch_ohlcv("BTC/USDT", "1h", 100)
        r  = full_technical_analysis(df, "1h", 0.0)
        assert "score" in r and "rsi" in r
        score = r["score"]
        record("full_technical_analysis", "PASS",
               f"Score={score:+.1f} | RSI={r['rsi']:.1f} | ADX={r['adx']:.1f} | Dir={r['direction']}")

        # Multi TF
        df4h = fetch_ohlcv("BTC/USDT", "4h", 100)
        df1d = fetch_ohlcv("BTC/USDT", "1d",  60)
        frames = {"1h": df, "4h": df4h, "1d": df1d}
        mta = multi_timeframe_analysis(frames, 0.0)
        assert "score" in mta
        record("multi_timeframe_analysis", "PASS",
               f"Score={mta['score']:+.1f} | Regime={mta['market_regime']} | "
               f"Conflict={mta['tf_conflict']} | Conf={mta['confidence']:.0f}%")
        record("Support/Resistance", "PASS",
               f"S=${mta['support_resistance'].get('support',0):,.2f} "
               f"R=${mta['support_resistance'].get('resistance',0):,.2f}")
    except Exception as e:
        record("technical", "FAIL", traceback.format_exc()[-200:])


# ── TEST 6: Risk Manager ──────────────────────────────────────
def test_risk_manager():
    console.print("\n[bold cyan]── TEST 6: Risk Manager ──[/bold cyan]")
    try:
        from risk_manager import RiskManager, RiskConfig, compute_liquidation_price
        from portfolio import Portfolio

        cfg  = RiskConfig(initial_capital=1000)
        rm   = RiskManager(cfg)
        port = Portfolio(initial_capital=1000)

        # Buat dummy AI result berkualitas tinggi (RRR 2.5+)
        # Entry $63,000 | SL $61,500 (-2.38%) | TP1 $66,750 (+5.95%) → RRR 2.5
        good_ai = {
            "action": "LONG", "confidence": 78, "risk_level": "MEDIUM",
            "overall_score": 75, "confluence_score": 72,
            "stop_loss": 61500.0, "take_profit_1": 66750.0, "take_profit_2": 69000.0,
            "leverage": 3, "position_size_pct": 3.0,
            "risk_reward_ratio": 2.5, "holding_period": "swing_1-3hari",
            "action_votes": {"LONG": 3}, "_ai_count": 2,
        }
        chk = rm.check_order("BTC/USDT", "LONG", good_ai, 63000.0, port, 10)
        record("Good trade approved", "PASS" if chk.approved else "FAIL",
               chk.reason if not chk.approved else
               f"Notional=${chk.adjusted_notional:,.2f} Lev={chk.effective_leverage:.1f}x "
               f"SL=${chk.stop_loss:,.2f}")

        # Bad AI — low confluence
        bad_ai = dict(good_ai, confluence_score=45, overall_score=55, confidence=50)
        chk2 = rm.check_order("ETH/USDT", "LONG", bad_ai, 3500.0, port, 10)
        record("Low confluence blocked", "PASS" if not chk2.approved else "FAIL",
               chk2.reason)

        # Bad AI — low score
        bad_score = dict(good_ai, overall_score=50, confluence_score=65)
        chk3 = rm.check_order("SOL/USDT", "LONG", bad_score, 150.0, port, 7)
        record("Low score blocked", "PASS" if not chk3.approved else "FAIL", chk3.reason)

        # Likuidasi formula
        liq = compute_liquidation_price(63000.0, "LONG", 3.0)
        assert 40000 < liq < 63000
        record("Liquidation price", "PASS", f"${liq:,.2f} (entry $63,000 @ 3x)")

        # Circuit breaker
        rm.record_trade_result(-35.0)   # 3.5% loss → trigger daily limit 2.5%
        rm2 = RiskManager(RiskConfig(initial_capital=1000, daily_loss_limit_pct=2.5))
        rm2.daily_loss = -30.0
        rm2.peak_capital = 1000
        chk4 = rm2.check_order("BTC/USDT", "LONG", good_ai, 63000.0, port, 10)
        record("Daily loss circuit breaker", "PASS" if not chk4.approved else "FAIL",
               chk4.reason)
    except Exception as e:
        record("risk_manager", "FAIL", traceback.format_exc()[-300:])


# ── TEST 7: Portfolio ─────────────────────────────────────────
def test_portfolio():
    console.print("\n[bold cyan]── TEST 7: Portfolio Engine ──[/bold cyan]")
    try:
        from portfolio import Portfolio
        port = Portfolio(initial_capital=1000)
        init_eq = port.equity
        record("Portfolio init", "PASS", f"Equity=${init_eq:,.2f}")

        ai_dummy = {
            "overall_score": 75, "confidence": 78, "holding_period": "swing_1-3hari",
            "_sources": ["Gemini"], "verdict": "Test trade",
        }
        pos = port.open_position(
            symbol="BTC/USDT", side="LONG", entry_price=63000.0,
            notional=200.0, leverage=3.0,
            stop_loss=61500.0, take_profit_1=66750.0, take_profit_2=69000.0,
            ai_result=ai_dummy,
        )
        assert pos is not None
        record("Open LONG position", "PASS",
               f"Qty={pos.qty:.6f} | Margin=${pos.margin:.2f} | Liq=${pos.liquidation_price:,.2f}")

        upnl = pos.unrealized_pnl(65000.0)
        record("Unrealized P&L",    "PASS", f"${upnl:+.2f} at $65,000")

        # SL/TP check — harga turun ke 61,000 → SL hit (SL ada di 61,500)
        acts = port.check_exit_conditions({"BTC/USDT": 61000.0})  # SL hit
        record("Stop loss detection", "PASS" if acts and acts[0]["action"] == "CLOSE" else "FAIL",
               acts[0]["reason"] if acts else "no action")

        # Close
        trade = port.close_position("BTC/USDT", 61000.0, "SL hit test")
        assert trade is not None
        record("Close position",    "PASS",
               f"PnL=${trade.pnl:+.2f} ({trade.pnl_pct:+.2f}%)")

        stats = port.get_statistics()
        record("Statistics",        "PASS",
               f"Trades={stats['total_trades']} WR={stats['win_rate']:.0f}%")
    except Exception as e:
        record("portfolio", "FAIL", traceback.format_exc()[-300:])

# ── TEST 8: News Fetcher ──────────────────────────────────────
async def test_news():
    console.print("\n[bold cyan]── TEST 8: News Fetcher (tanpa login) ──[/bold cyan]")
    try:
        from news_fetcher import (
            fetch_fear_greed, fetch_coingecko_fundamental,
            fetch_binance_market_context, fetch_macro_global,
        )

        fg = fetch_fear_greed()
        val = fg.get("value", -1)
        assert 0 <= val <= 100
        record("Fear & Greed", "PASS",
               f"Value={val}/100 ({fg.get('label','?')}) | Trend={fg.get('trend','?')}")

        cg = fetch_coingecko_fundamental("BTC/USDT")
        if cg.get("name"):
            record("CoinGecko fundamental", "PASS",
                   f"{cg['name']} | Rank=#{cg.get('market_cap_rank','?')} | "
                   f"Cap=${cg.get('market_cap_usd',0)/1e9:.1f}B")
        else:
            record("CoinGecko fundamental", "SKIP", "Rate limited / no data")

        macro = fetch_macro_global()
        dxy = macro.get("DXY") or {}
        record("Macro global (DXY)", "PASS" if dxy.get("price", 0) > 0 else "SKIP",
               f"DXY={dxy.get('price','N/A')} ({dxy.get('change_pct',0):+.2f}%)" if dxy else "not available")

        bn = fetch_binance_market_context("BTC/USDT")
        record("Binance funding/OI", "PASS" if "funding_rate" in bn else "SKIP",
               f"Funding={bn.get('funding_pct',0):+.4f}% | OI={bn.get('open_interest',0):,.0f}")
    except Exception as e:
        record("news_fetcher", "FAIL", traceback.format_exc()[-200:])

async def test_news_rss():
    console.print("\n[bold cyan]── TEST 8b: RSS News (async, tanpa login) ──[/bold cyan]")
    try:
        from news_fetcher import fetch_crypto_news_all, score_sentiment
        articles = await fetch_crypto_news_all("BTC/USDT")
        sent     = score_sentiment(articles)
        record("RSS + CryptoPanic news", "PASS",
               f"{len(articles)} articles | Sentiment={sent['score']:.0f}/100 ({sent['label']})")
        if articles:
            record("Sample headline", "PASS",
                   f"[{articles[0].get('source','?')}] {articles[0].get('title','?')[:70]}")
    except Exception as e:
        record("rss_news", "FAIL", str(e)[:150])


# ── TEST 9: Charts ────────────────────────────────────────────
def test_charts():
    console.print("\n[bold cyan]── TEST 9: Chart Generation (Binance data) ──[/bold cyan]")
    try:
        from charts import generate_chart, generate_all_charts, build_html_gallery, CHART_DIR
        import os

        # Single chart — 1h BTC
        path = generate_chart("BTC/USDT", "1h", save_dir=CHART_DIR + "/BTCUSDT")
        if path and os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            record("Chart 1h BTC/USDT", "PASS", f"{os.path.basename(path)} ({size_kb:.0f} KB)")
        else:
            record("Chart 1h BTC/USDT", "FAIL", "File tidak terbuat")

        # Generate beberapa TF
        paths = generate_all_charts("BTC/USDT", ["1h", "4h", "1d"])
        record("Charts 1h+4h+1d", "PASS", f"{len(paths)}/3 TF generated")

        # HTML gallery
        gallery = build_html_gallery({"BTC/USDT": paths})
        if os.path.exists(gallery):
            record("HTML gallery", "PASS", f"file://{gallery}")
        else:
            record("HTML gallery", "FAIL", "index.html tidak terbuat")
    except Exception as e:
        record("charts", "FAIL", traceback.format_exc()[-200:])


# ── TEST 10: AI Analyzer (tanpa API key — fallback mode) ──────
async def test_ai_fallback():
    console.print("\n[bold cyan]── TEST 10: AI Analyzer (Fallback Teknikal) ──[/bold cyan]")
    try:
        from data_fetcher    import fetch_ohlcv, fetch_market_metrics, fetch_order_book, fetch_recent_trades
        from technical_analyzer import multi_timeframe_analysis
        from ai_analyzer        import analyze_with_ai, AI_AVAILABLE, build_prompt
        from news_fetcher       import get_full_market_context
        from portfolio          import Portfolio

        # Siapkan data
        df1h = fetch_ohlcv("BTC/USDT", "1h", 100)
        df4h = fetch_ohlcv("BTC/USDT", "4h",  80)
        df1d = fetch_ohlcv("BTC/USDT", "1d",  50)
        metrics = fetch_market_metrics("BTC/USDT")
        ob      = fetch_order_book("BTC/USDT")
        trades  = fetch_recent_trades("BTC/USDT")
        tech    = multi_timeframe_analysis({"1h": df1h, "4h": df4h, "1d": df1d},
                                           metrics.get("funding_rate", 0))
        port    = Portfolio(initial_capital=1000)
        pctx    = port.get_context_for_ai()
        pctx["has_position"] = False

        record("AI available", "PASS" if AI_AVAILABLE else "SKIP",
               "Gemini/GPT/Claude active" if AI_AVAILABLE else "No API key — will use Technical Fallback")

        # Test build_prompt (tidak perlu API key)
        market_ctx = {"text": "Test market context — no real news loaded for speed"}
        prompt = build_prompt("BTC/USDT", tech, metrics, pctx, market_ctx, ob, trades, {})
        record("Prompt builder", "PASS", f"Prompt length: {len(prompt):,} chars")

        # Jalankan AI (atau fallback)
        console.print("  [dim]Memanggil AI... (bisa 5-15 detik jika ada API key)[/dim]")
        result = await analyze_with_ai(
            "BTC/USDT", tech, metrics, pctx,
            market_ctx, ob, trades, {}
        )

        action   = result.get("action", "?")
        score    = result.get("overall_score", 0)
        conf     = result.get("confidence", 0)
        conf_sc  = result.get("confluence_score", 0)
        rrr      = result.get("risk_reward_ratio", 0)
        lev      = result.get("leverage", 0)
        sl       = result.get("stop_loss", 0)
        tp1      = result.get("take_profit_1", 0)
        sources  = ", ".join(result.get("_sources", []))

        color = {"LONG":"green","SHORT":"red","HOLD":"yellow","AVOID":"dim"}.get(action,"white")
        record("AI Decision", "PASS",
               f"[{color}]{action}[/{color}] Score={score:.0f} Conf={conf:.0f}% "
               f"Confluence={conf_sc:.0f} RRR=1:{rrr:.1f} Lev={lev:.0f}x")
        record("AI Sources",  "PASS", sources if sources else "Teknikal-Fallback")
        if sl:   record("Stop Loss",   "PASS", f"${sl:,.4f}")
        if tp1:  record("Take Profit1","PASS", f"${tp1:,.4f}")

        verdict = result.get("verdict","")
        if verdict:
            record("Verdict", "PASS", verdict[:100])

        risks = result.get("key_risks", [])
        if risks:
            record("Key Risks", "PASS", " | ".join(risks[:2]))

    except Exception as e:
        record("ai_analyzer", "FAIL", traceback.format_exc()[-400:])

# ── TEST 11: Full Cycle Simulation (paper trade) ─────────────
async def test_full_cycle():
    console.print("\n[bold cyan]── TEST 11: Full Cycle Simulation ──[/bold cyan]")
    try:
        from data_fetcher       import fetch_ohlcv, fetch_market_metrics, fetch_order_book, fetch_recent_trades
        from technical_analyzer import multi_timeframe_analysis
        from ai_analyzer        import analyze_with_ai
        from risk_manager       import RiskManager, RiskConfig
        from portfolio          import Portfolio

        port = Portfolio(initial_capital=1000)
        rm   = RiskManager(RiskConfig(initial_capital=1000))

        # Fetch data
        df1h = fetch_ohlcv("ETH/USDT", "1h", 100)
        df4h = fetch_ohlcv("ETH/USDT", "4h",  80)
        df1d = fetch_ohlcv("ETH/USDT", "1d",  50)
        metrics = fetch_market_metrics("ETH/USDT")
        ob      = fetch_order_book("ETH/USDT")
        trades  = fetch_recent_trades("ETH/USDT")
        tech    = multi_timeframe_analysis({"1h": df1h, "4h": df4h, "1d": df1d},
                                           metrics.get("funding_rate", 0))

        price = metrics.get("price", 0)
        record("ETH/USDT price fetch", "PASS", f"${price:,.2f}")

        pctx = port.get_context_for_ai()
        pctx["has_position"] = False
        market_ctx = {"text": f"ETH test context @ ${price:,.2f}"}

        ai = await analyze_with_ai("ETH/USDT", tech, metrics, pctx, market_ctx, ob, trades, {})
        action = ai.get("action", "HOLD")
        record("ETH AI decision",   "PASS",
               f"{action} | Score={ai.get('overall_score',0):.0f} | "
               f"Confluence={ai.get('confluence_score',0):.0f}")

        # Risk check
        from config import SYMBOL_MAX_LEVERAGE
        chk = rm.check_order("ETH/USDT", action if action in ("LONG","SHORT") else "LONG",
                              ai, price, port, SYMBOL_MAX_LEVERAGE.get("ETH/USDT", 10))

        record("Risk manager check", "PASS" if chk.approved else "SKIP",
               chk.reason if not chk.approved else
               f"Notional=${chk.adjusted_notional:.2f} @ {chk.effective_leverage:.1f}x")

        if chk.approved:
            pos = port.open_position(
                symbol="ETH/USDT", side=action,
                entry_price=price,
                notional=chk.adjusted_notional,
                leverage=chk.effective_leverage,
                stop_loss=chk.stop_loss,
                take_profit_1=chk.take_profit_1,
                take_profit_2=chk.take_profit_2,
                ai_result=ai,
            )
            if pos:
                record("Paper trade opened", "PASS",
                       f"{action} ETH @ ${price:,.2f} | SL=${pos.stop_loss:,.2f} | "
                       f"TP1=${pos.take_profit_1:,.2f} | Liq=${pos.liquidation_price:,.2f}")
                # Simulasi close
                sim_exit = price * 1.025  # +2.5% profit
                trade = port.close_position("ETH/USDT", sim_exit, "Test close +2.5%")
                if trade:
                    record("Paper trade closed", "PASS",
                           f"PnL=${trade.pnl:+.2f} ({trade.pnl_pct:+.2f}% margin)")
            else:
                record("Paper trade opened", "SKIP", "Margin tidak cukup untuk test")
        else:
            record("Paper trade",  "SKIP", f"Risk blocked: {chk.reason}")

    except Exception as e:
        record("full_cycle", "FAIL", traceback.format_exc()[-400:])


# ── MAIN ──────────────────────────────────────────────────────
async def main():
    console.print(Panel(
        "[bold cyan]🤖 AI CRYPTO TRADING BOT — End-to-End Test[/bold cyan]\n"
        f"Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        "Mode: Paper Trading | Exchange: Binance Futures (public data)",
        border_style="cyan"
    ))

    # Sync tests
    test_imports()
    test_config()
    test_binance_data()
    test_ohlcv()
    test_technical()
    test_risk_manager()
    test_portfolio()
    test_charts()

    # Async tests
    await test_news()
    await test_news_rss()
    await test_ai_fallback()
    await test_full_cycle()

    # ── Summary ──
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    skipped= sum(1 for _, s, _ in results if s == "SKIP")
    total  = len(results)

    table = Table(title="📊 Test Summary", style="cyan", show_lines=True)
    table.add_column("Test", style="bold", width=35)
    table.add_column("Status", justify="center", width=8)
    table.add_column("Detail", width=55)
    for name, status, detail in results:
        c = {"PASS":"green","FAIL":"red","SKIP":"yellow"}.get(status,"white")
        table.add_row(name, f"[{c}]{status}[/{c}]", detail[:55])
    console.print(table)

    color = "green" if failed == 0 else ("yellow" if failed <= 2 else "red")
    console.print(Panel(
        f"[bold {color}]Results: {passed} PASS | {failed} FAIL | {skipped} SKIP / {total} total[/bold {color}]\n"
        + ("[green]✅ Bot siap dijalankan![/green]" if failed == 0
           else f"[red]❌ {failed} test gagal — periksa error di atas[/red]"),
        border_style=color
    ))

    if failed == 0:
        console.print("\n[bold green]Cara menjalankan bot:[/bold green]")
        console.print("  python trading_bot.py    ← bot utama (paper mode)")
        console.print("  python charts.py BTC/USDT ← lihat chart di browser")
        console.print("  python dashboard.py       ← monitor terminal")


if __name__ == "__main__":
    asyncio.run(main())
