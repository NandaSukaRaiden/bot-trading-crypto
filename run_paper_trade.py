"""
run_paper_trade.py — Satu siklus analisis + eksekusi BTC/USDT.

TRADING_MODE=paper → simulasi lokal
TRADING_MODE=live  → order nyata ke Binance Futures Testnet
"""
import asyncio, json, traceback
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from data_fetcher import (
    fetch_ticker_binance, fetch_ohlcv, fetch_order_book,
    fetch_recent_trades, fetch_funding_rate, fetch_open_interest,
)
from technical_analyzer import multi_timeframe_analysis
from news_fetcher import get_full_market_context
from ai_analyzer import analyze_with_ai, AI_AVAILABLE, get_active_providers
from smart_trigger import should_call_ai_for_entry, get_trigger_stats
from risk_manager import RiskManager, RiskConfig
from portfolio import Portfolio
from testnet_executor import (
    is_live_mode, set_leverage, place_market_order,
    place_stop_loss, place_take_profit, get_account_balance,
)
from config import (
    CRYPTO_WATCHLIST, SYMBOL_MAX_LEVERAGE,
    TRADING_STYLE, DEFAULT_LEVERAGE, MAX_LEVERAGE,
    USE_TESTNET, USE_SMART_TRIGGER,
)

console = Console()

PAIRS     = CRYPTO_WATCHLIST   # ["BTC/USDT"]
RISK_PCT  = 0.5                # % equity per trade
TFS       = ["1m", "5m", "15m", "1h", "4h", "1d"]


async def main():
    # ── 0. Tentukan modal ─────────────────────────────────────
    capital = 1000.0
    if is_live_mode():
        bal  = await get_account_balance()
        free = bal.get("USDT", {}).get("free", 0)
        if free > 0:
            capital = free

    # ── Banner ─────────────────────────────────────────────────
    mode_str = "[bold red]TESTNET LIVE[/bold red]" if is_live_mode() else "[bold yellow]PAPER[/bold yellow]"
    net_str  = "testnet.binancefuture.com" if USE_TESTNET else "binance.com"
    ai_str   = " + ".join(get_active_providers()) if AI_AVAILABLE else "TEKNIKAL FALLBACK"

    console.print(Panel(
        f"[bold cyan]BOT TRADING — {TRADING_STYLE.upper()} BTC/USDT[/bold cyan]\n"
        f"Mode       : {mode_str}\n"
        f"Exchange   : {net_str}\n"
        f"Modal      : ${capital:,.2f} USDT\n"
        f"Risk/trade : {RISK_PCT}% | Leverage: {DEFAULT_LEVERAGE:.0f}x–{MAX_LEVERAGE:.0f}x\n"
        f"AI Ensemble: [green]{ai_str}[/green]\n"
        f"Waktu      : {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
        border_style="cyan"
    ))

    # ── 1. Init portfolio + risk manager ──────────────────────
    port = Portfolio(initial_capital=capital)
    if not is_live_mode():
        port.balance = capital
        port.open_positions.clear()
        port.closed_trades.clear()
        port._save_state()

    rm = RiskManager(RiskConfig(
        initial_capital         = capital,
        max_risk_per_trade_pct  = RISK_PCT,
        max_open_positions      = 3,
        default_leverage        = DEFAULT_LEVERAGE,
        max_leverage            = MAX_LEVERAGE,
        min_rrr                 = 1.5,
        min_score_to_trade      = 50,   # lebih rendah agar fallback teknikal bisa masuk
        min_confidence          = 25.0,  # turunkan untuk testing (AI rate limit)
        min_confluence_score    = 50,
        max_risk_level_to_trade = "HIGH",
        daily_loss_limit_pct    = 2.0,
        loss_cooldown_minutes   = 5,
    ))

    # ── 2. Analisis AI ─────────────────────────────────────────
    console.print(f"\n[bold]STEP 1 — FETCH + AI ANALYSIS ({len(PAIRS)} pair)[/bold]")

    tbl = Table(title="AI Analysis — BTC Testnet", style="cyan", show_lines=True)
    for col, w, j in [
        ("Pair",10,"left"),("Price",13,"right"),("24h%",8,"right"),
        ("Score",7,"right"),("Conf",7,"right"),("Confluence",11,"right"),
        ("Regime",12,"left"),("Funding",10,"right"),("Models",7,"right"),
        ("Action",9,"center"),
    ]:
        tbl.add_column(col, width=w, justify=j)

    candidates = []

    for pair in PAIRS:
        try:
            console.print(f"\n[bold cyan]── {pair} ──[/bold cyan]")

            t   = fetch_ticker_binance(pair)
            ob  = fetch_order_book(pair)
            rt  = fetch_recent_trades(pair)
            fr  = fetch_funding_rate(pair)
            oi  = fetch_open_interest(pair)
            price = t.get("price", 0)
            console.print(f"  Price: ${price:,.2f} | 24h: {t.get('change_24h_pct',0):+.2f}%")

            frames = {}
            for tf in TFS:
                try:
                    frames[tf] = fetch_ohlcv(pair, tf, 200)
                except Exception as e:
                    console.print(f"    [dim]{tf}: {e}[/dim]")

            tech = multi_timeframe_analysis(frames, fr)
            console.print(
                f"  TA: score={tech.get('score',0):+.0f} | "
                f"regime={tech.get('market_regime','?')} | "
                f"conflict={tech.get('tf_conflict',False)}"
            )

            metrics = {
                "symbol": pair, "price": price,
                "high_24h": t.get("high_24h", 0),
                "low_24h": t.get("low_24h", 0),
                "change_24h_pct": t.get("change_24h_pct", 0),
                "volume_24h": t.get("volume_24h", 0),
                "funding_rate": fr,
                "open_interest": oi,
            }

            console.print(f"  Fetching news...")
            market_ctx = await get_full_market_context(pair)

            pctx = port.get_context_for_ai()
            pctx["has_position"]          = pair in port.open_positions
            pctx["risk_per_trade_pct"]    = RISK_PCT
            pctx["max_positions"]         = 3
            pctx["open_positions_detail"] = []

            # ── SMART TRIGGER CHECK ──────────────────────────
            call_ai = True
            trigger_reason = "SINGLE_RUN"
            
            if USE_SMART_TRIGGER:
                call_ai, trigger_reason = should_call_ai_for_entry(
                    pair, tech, metrics, market_ctx
                )
                
                if not call_ai:
                    console.print(f"  [dim cyan]⚡ SKIP AI: {trigger_reason}[/dim cyan]")
                    # Pakai teknikal pure
                    ai = {
                        "action": "HOLD",
                        "overall_score": tech.get("score", 0),
                        "confidence": 20,
                        "confluence_score": 50,
                        "verdict": f"Teknikal: {trigger_reason}",
                        "skip_ai": True,
                        "_ai_count": 0,
                        "action_votes": {"HOLD": 1},
                    }
                else:
                    console.print(f"  [bold green]🚨 TRIGGER: {trigger_reason}[/bold green]")
                    console.print(f"  Calling AI ensemble...")
                    ai = await analyze_with_ai(
                        pair, tech, metrics, pctx, market_ctx, ob, rt, {}
                    )
            else:
                console.print(f"  Calling AI ensemble...")
                ai = await analyze_with_ai(
                    pair, tech, metrics, pctx, market_ctx, ob, rt, {}
                )

            action    = ai.get("action", "HOLD")
            n_models  = ai.get("_ai_count", 0)
            votes     = ai.get("action_votes", {})
            ac        = {"LONG":"green","SHORT":"red","HOLD":"yellow","AVOID":"dim"}.get(action,"white")
            cc        = "green" if t.get("change_24h_pct", 0) >= 0 else "red"

            tbl.add_row(
                pair,
                f"${price:,.2f}",
                f"[{cc}]{t.get('change_24h_pct',0):+.2f}%[/{cc}]",
                f"{ai.get('overall_score',0):.0f}",
                f"{ai.get('confidence',0):.0f}%",
                f"{ai.get('confluence_score',0):.0f}",
                tech.get("market_regime", "?"),
                f"{fr*100:+.4f}%",
                str(n_models),
                f"[{ac}]{action}[/{ac}]",
            )

            console.print(
                f"  [{ac}]{action}[/{ac}] score={ai.get('overall_score',0):.0f} "
                f"conf={ai.get('confidence',0):.0f}% "
                f"confluence={ai.get('confluence_score',0):.0f} "
                f"votes={votes} models={n_models}"
            )
            if ai.get("verdict"):
                console.print(f"  Verdict: {ai['verdict'][:120]}")

            if action in ("LONG", "SHORT"):
                candidates.append({
                    "pair": pair, "signal": action,
                    "price": price, "ai": ai, "tech": tech,
                })

        except Exception as e:
            console.print(f"  [red]Error {pair}: {e}[/red]")
            console.print(f"  [dim]{traceback.format_exc()[-400:]}[/dim]")

    console.print(tbl)

    # ── 3. Risk check + eksekusi ───────────────────────────────
    console.print(f"\n[bold]STEP 2 — RISK CHECK + EKSEKUSI ({len(candidates)} kandidat)[/bold]")
    candidates.sort(key=lambda x: x["ai"].get("overall_score", 0), reverse=True)

    executed = 0
    for c in candidates:
        pair   = c["pair"]
        signal = c["signal"]
        price  = c["price"]
        ai     = c["ai"]

        console.print(f"\n  [{('green' if signal=='LONG' else 'red')}]{signal}[/] {pair} @ ${price:,.2f}")

        rm.set_ai_risk_multiplier(ai.get("suggested_risk_multiplier", 1.0))
        chk = rm.check_order(pair, signal, ai, price, port,
                             SYMBOL_MAX_LEVERAGE.get(pair, MAX_LEVERAGE))

        if not chk.approved:
            console.print(f"    [red]BLOCKED: {chk.reason}[/red]")
            continue

        for w in chk.warnings:
            console.print(f"    [dim]{w}[/dim]")

        eff_lev     = int(chk.effective_leverage)
        entry_price = price

        # Kirim ke Binance testnet jika live
        if is_live_mode():
            await set_leverage(pair, eff_lev)
            order = await place_market_order(pair, signal, chk.adjusted_notional, price)
            if order is None:
                console.print(f"    [red]Market order gagal, skip.[/red]")
                continue
            entry_price = order["fill_price"]
            qty         = order["qty"]
            await place_stop_loss(pair, signal, chk.stop_loss, qty)
            await place_take_profit(pair, signal, chk.take_profit_1, round(qty * 0.5, 6))

        pos = port.open_position(
            symbol        = pair,
            side          = signal,
            entry_price   = entry_price,
            notional      = chk.adjusted_notional,
            leverage      = chk.effective_leverage,
            stop_loss     = chk.stop_loss,
            take_profit_1 = chk.take_profit_1,
            take_profit_2 = chk.take_profit_2,
            ai_result     = ai,
        )

        if pos:
            executed += 1
            tag   = "TESTNET" if is_live_mode() else "PAPER"
            sl_d  = abs(entry_price - pos.stop_loss) / entry_price * 100
            liq_d = abs(entry_price - pos.liquidation_price) / entry_price * 100
            console.print(
                f"\n    [bold green]ORDER DIBUKA [{tag}][/bold green]\n"
                f"    {signal} {pair} @ ${entry_price:,.4f} | {eff_lev}x\n"
                f"    Notional : ${pos.notional:,.2f} | Margin: ${pos.margin:,.2f}\n"
                f"    SL       : ${pos.stop_loss:,.4f} (-{sl_d:.2f}%)\n"
                f"    TP1      : ${pos.take_profit_1:,.4f}\n"
                f"    TP2      : ${pos.take_profit_2:,.4f}\n"
                f"    Liq      : ${pos.liquidation_price:,.4f} ({liq_d:.1f}% dari entry)\n"
                f"    RRR      : 1:{ai.get('risk_reward_ratio',0):.1f}\n"
                f"    Votes    : {ai.get('action_votes',{})}\n"
                f"    Models   : {', '.join(ai.get('_sources',[]))}\n"
                f"    Verdict  : {(ai.get('verdict') or '')[:150]}"
            )

    # ── 4. Portfolio summary ───────────────────────────────────
    console.print(f"\n[bold]STEP 3 — PORTFOLIO[/bold]")
    cur_prices = {}
    for pair in PAIRS:
        if pair in port.open_positions:
            try:
                cur_prices[pair] = fetch_ticker_binance(pair).get("price", 0)
            except Exception:
                pass
    port.print_portfolio(cur_prices)

    # ── 5. Summary panel ──────────────────────────────────────
    mode_note = (
        "[bold red]TESTNET LIVE — order nyata dikirim ke testnet.binancefuture.com[/bold red]"
        if is_live_mode() else
        "[yellow]PAPER — simulasi lokal, tidak ada order nyata[/yellow]"
    )
    
    # Trigger stats
    trigger_stats = get_trigger_stats()
    trigger_note = ""
    if USE_SMART_TRIGGER and trigger_stats.get("trigger_count", 0) + trigger_stats.get("skip_count", 0) > 0:
        trigger_note = (f"\n\n[bold cyan]Smart Trigger Stats:[/bold cyan]\n"
                       f"  AI Calls    : {trigger_stats['trigger_count']}\n"
                       f"  Skipped     : {trigger_stats['skip_count']}\n"
                       f"  API Reduction: [green]{trigger_stats['reduction_pct']:.0f}%[/green]")
    
    console.print(Panel(
        f"[bold green]RINGKASAN[/bold green]\n\n"
        f"  Mode      : {mode_note}\n"
        f"  Analisis  : {len(PAIRS)} pair\n"
        f"  Kandidat  : {len(candidates)}\n"
        f"  Dieksekusi: {executed}\n"
        f"  Modal     : ${capital:,.2f}\n"
        f"  Tersisa   : ${port.available_margin:,.2f}\n"
        f"  Equity    : ${port.equity:,.2f}\n\n"
        f"  AI        : [cyan]{ai_str}[/cyan]"
        f"{trigger_note}\n\n"
        f"[bold]Loop otomatis:[/bold]\n"
        f"  python trading_bot.py",
        border_style="green"
    ))

    # ── 6. Simpan log ─────────────────────────────────────────
    log = {
        "timestamp":      datetime.utcnow().isoformat(),
        "mode":           "testnet_live" if is_live_mode() else "paper",
        "capital":        capital,
        "style":          TRADING_STYLE,
        "ai_providers":   get_active_providers(),
        "executed":       executed,
        "positions": [
            {
                "pair":     sym,
                "side":     p.side,
                "entry":    p.entry_price,
                "notional": p.notional,
                "margin":   p.margin,
                "leverage": p.leverage,
                "sl":       p.stop_loss,
                "tp1":      p.take_profit_1,
                "liq":      p.liquidation_price,
            }
            for sym, p in port.open_positions.items()
        ],
        "cash_remaining": port.balance,
    }
    with open("paper_trade_log.json", "w") as f:
        json.dump(log, f, indent=2)
    console.print("[dim]Log disimpan ke paper_trade_log.json[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
