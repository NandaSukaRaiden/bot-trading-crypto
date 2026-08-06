"""
trading_bot.py — Bot Crypto Futures Mandiri
Loop: fetch data → charts → news → AI analisis → risk check → eksekusi
AI bertindak seperti trader profesional tanpa intervensi manusia.
Mode : live + USE_TESTNET=true → order nyata ke Binance Futures Testnet
       paper                   → simulasi lokal (tidak kirim ke exchange)
"""
import asyncio
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from config import (
    CRYPTO_WATCHLIST, ANALYSIS_INTERVAL_MIN, TRADING_MODE,
    SIGNAL_LEVELS, COLORS, SYMBOL_MAX_LEVERAGE, MAX_POSITIONS,
    POSITION_MONITOR_SEC, TRADING_STYLE, AI_ANALYZE_ALL_PAIRS,
    AI_MANAGE_POSITIONS, AI_MANAGE_MIN_MOVE_PCT,
    PRIMARY_TIMEFRAME, TREND_TIMEFRAME, CONTEXT_TIMEFRAME,
    MAX_RISK_PER_TRADE, MAX_LEVERAGE, DEFAULT_LEVERAGE,
    USE_TESTNET, EXCHANGE_API_KEY, EXCHANGE_API_SECRET,
)
from data_fetcher import (
    fetch_all_market_data_async, fetch_current_prices, get_market_status,
    fetch_ohlcv, fetch_funding_rate,
)
from technical_analyzer import multi_timeframe_analysis
from news_fetcher import get_full_market_context
from ai_analyzer import (
    analyze_with_ai, manage_positions_with_ai,
    AI_AVAILABLE, get_active_providers,
)
from charts import generate_all_charts, build_html_gallery, start_chart_server
from risk_manager import RiskManager, RiskConfig
from portfolio import Portfolio
from notifier import send_telegram_notification
from testnet_executor import (
    is_live_mode, set_leverage, place_market_order,
    place_stop_loss, place_take_profit,
    close_position_market, cancel_all_orders,
    get_account_balance,
)

console = Console()


def signal_label(score: float) -> str:
    for lbl, (lo, hi) in SIGNAL_LEVELS.items():
        if lo <= score <= hi: return lbl
    return "NEUTRAL"


class TradingBot:
    def __init__(self):
        self.portfolio    = Portfolio()
        self.risk_mgr     = RiskManager(RiskConfig())
        self.scheduler    = AsyncIOScheduler(timezone="UTC")
        self.running      = False
        self.cycle        = 0
        self.last_results = {}
        self.risk_mgr.current_capital = self.portfolio.equity
        self.risk_mgr.peak_capital    = max(self.portfolio.equity,
                                            self.portfolio.initial_capital)

    # ══════════════════════════════════════════════════════════
    #  SIKLUS UTAMA
    # ══════════════════════════════════════════════════════════
    async def run_cycle(self):
        self.cycle += 1
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        console.print(Panel(
            f"[bold cyan]🤖 SIKLUS #{self.cycle} — {now} UTC[/bold cyan]\n"
            f"{get_market_status()}\n"
            f"Equity:${self.portfolio.equity:,.2f} | "
            f"Margin used:${self.portfolio.used_margin:,.2f} | "
            f"Positions:{len(self.portfolio.open_positions)}",
            border_style="cyan"
        ))

        if self.risk_mgr.circuit_breaker:
            console.print(f"[bold red]🚨 CIRCUIT BREAKER: {self.risk_mgr.cb_reason}[/bold red]")
            return

        # 1. Monitor posisi aktif (SL/TP safety net)
        await self._monitor_positions()

        # 1b. AI mengelola posisi terbuka (close/partial/trail/TP)
        if AI_MANAGE_POSITIONS:
            await self._manage_positions_with_ai()

        # 2. Fetch market data semua pair
        symbols = self._watchlist()
        console.print(f"\n[cyan]📡 Fetching {len(symbols)} pairs...[/cyan]")
        market_data = await fetch_all_market_data_async(symbols)
        console.print(f"[green]✓ {len(market_data)} pairs fetched[/green]")

        # 3. Analisis per pair (teknikal + news + AI)
        candidates = []
        for symbol, data in market_data.items():
            result = await self._analyze_symbol(symbol, data)
            if result:
                candidates.append(result)

        if not candidates:
            console.print("[dim]Tidak ada kandidat valid.[/dim]")
            return

        candidates.sort(key=lambda x: x["ai"].get("overall_score",0), reverse=True)
        self._print_summary(candidates)

        # 4. Eksekusi keputusan AI
        await self._execute(candidates)

        # 5. Update
        self.risk_mgr.current_capital = self.portfolio.equity
        await self._show_portfolio_live(market_data)
        console.print(f"\n[cyan]⏰ Berikutnya: {ANALYSIS_INTERVAL_MIN} menit.[/cyan]\n")

    # ══════════════════════════════════════════════════════════
    #  ANALISIS SATU SIMBOL — AI mandiri dengan filter ketat
    # ══════════════════════════════════════════════════════════
    async def _analyze_symbol(self, symbol: str, data: dict) -> Optional[dict]:
        try:
            console.print(f"\n[bold]📊 {symbol}[/bold]")
            frames  = data["dataframes"]
            metrics = data["metrics"]
            ob      = data.get("order_book", {})
            trades  = data.get("trades", {})

            # Teknikal multi-TF
            technical = multi_timeframe_analysis(frames, metrics.get("funding_rate", 0))
            if "error" in technical:
                console.print(f"  [dim]Skip: {technical['error']}[/dim]")
                return None

            has_position = symbol in self.portfolio.open_positions

            # Pre-filter OPSIONAL (matikan agar AI memutuskan SEMUA pair)
            if not AI_ANALYZE_ALL_PAIRS:
                # Pre-filter: skip yang terlalu lemah kecuali sudah ada posisi
                tech_score = abs(technical.get("score", 0))
                if tech_score < 20 and not has_position:
                    console.print(f"  [dim]Skip pre-filter: skor {technical['score']:+.0f}[/dim]")
                    return None
                # TF conflict + tidak ada posisi = skip (hemat API)
                if technical.get("tf_conflict") and not has_position:
                    console.print(f"  [dim]Skip: TF conflict ({PRIMARY_TIMEFRAME} vs {TREND_TIMEFRAME} berlawanan)[/dim]")
                    return None

            # Generate charts
            console.print(f"  [dim]📈 Charts...[/dim]")
            chart_paths = await asyncio.to_thread(generate_all_charts, symbol)

            # Fetch news + fundamental + makro
            console.print(f"  [dim]📰 News & fundamentals...[/dim]")
            market_ctx = await get_full_market_context(symbol)

            # Konteks portfolio untuk AI
            pctx = self.portfolio.get_context_for_ai()
            pctx["has_position"] = has_position
            pctx["risk_per_trade_pct"] = self.risk_mgr.cfg.max_risk_per_trade_pct
            pctx["max_positions"]      = self.risk_mgr.cfg.max_open_positions
            pctx["open_positions_detail"] = [
                {
                    "symbol":      p.symbol,
                    "side":        p.side,
                    "entry_price": p.entry_price,
                    "stop_loss":   p.stop_loss,
                    "leverage":    p.leverage,
                    "pnl_pct":     round(p.unrealized_pnl_pct(
                        metrics.get("price", p.entry_price)), 2),
                }
                for p in self.portfolio.open_positions.values()
            ]

            # Panggil AI (Gemini primary + GPT + Claude paralel)
            ai_result = await analyze_with_ai(
                symbol, technical, metrics, pctx,
                market_ctx, ob, trades, chart_paths
            )
            self.last_results[symbol] = ai_result

            # Log confluence
            conf_score = ai_result.get("confluence_score", 0)
            conf_color = "green" if conf_score >= 70 else ("yellow" if conf_score >= 60 else "red")
            console.print(f"  Confluence: [{conf_color}]{conf_score:.0f}/100[/{conf_color}] | "
                          f"RRR: 1:{ai_result.get('risk_reward_ratio', 0):.1f} | "
                          f"Lev: {ai_result.get('leverage', 0):.0f}x")

            return {
                "symbol":     symbol,
                "technical":  technical,
                "metrics":    metrics,
                "ai":         ai_result,
                "market_ctx": market_ctx,
                "price":      technical.get("current_price", metrics.get("price", 0)),
            }
        except Exception as e:
            console.print(f"  [red]Error {symbol}: {e}[/red]")
            return None

    # ══════════════════════════════════════════════════════════
    #  EKSEKUSI — AI sebagai decision maker
    # ══════════════════════════════════════════════════════════
    async def _execute(self, candidates: list):
        console.print("\n[bold yellow]⚡ EKSEKUSI KEPUTUSAN AI[/bold yellow]")
        for item in candidates:
            symbol = item["symbol"]
            ai     = item["ai"]
            price  = item["price"]
            action = ai.get("action","HOLD")
            has_pos = symbol in self.portfolio.open_positions

            if action in ("LONG","SHORT") and not has_pos:
                await self._open(symbol, action, price, ai)
            elif action == "AVOID" and has_pos:
                await self._close(symbol, price, "AI: AVOID signal — kondisi berbahaya")
            elif action in ("LONG","SHORT") and has_pos:
                pos_side = self.portfolio.open_positions[symbol].side
                if action != pos_side:
                    await self._close(symbol, price, f"AI reversal: {pos_side}→{action}")

    async def _open(self, symbol: str, side: str, price: float, ai: dict):
        max_lev = SYMBOL_MAX_LEVERAGE.get(symbol, DEFAULT_LEVERAGE)
        self.risk_mgr.set_ai_risk_multiplier(ai.get("suggested_risk_multiplier", 1.0))
        chk = self.risk_mgr.check_order(symbol, side, ai, price, self.portfolio, max_lev)
        if not chk.approved:
            console.print(f"  [red]✗ BLOCKED {side} {symbol}: {chk.reason}[/red]")
            return
        for w in chk.warnings:
            console.print(f"  [dim]  {w}[/dim]")

        eff_lev = int(chk.effective_leverage)

        # ── Live testnet: kirim order ke exchange ──────────────
        if is_live_mode():
            # 1. Set leverage
            await set_leverage(symbol, eff_lev)

            # 2. Place MARKET order
            order = await place_market_order(symbol, side, chk.adjusted_notional, price)
            if order is None:
                console.print(f"  [red]✗ Market order gagal untuk {symbol}[/red]")
                return

            fill_price = order["fill_price"]
            qty        = order["qty"]

            # 3. Pasang SL + TP di exchange
            sl_qty  = qty                   # SL: tutup seluruh posisi
            tp1_qty = round(qty * 0.5, 6)  # TP1: 50% posisi
            await place_stop_loss(symbol, side, chk.stop_loss, sl_qty)
            await place_take_profit(symbol, side, chk.take_profit_1, tp1_qty)

            # Gunakan fill price sebagai entry price
            entry_price = fill_price
        else:
            entry_price = price

        # ── Catat di portfolio lokal (semua mode) ──────────────
        pos = self.portfolio.open_position(
            symbol=symbol, side=side, entry_price=entry_price,
            notional=chk.adjusted_notional, leverage=chk.effective_leverage,
            stop_loss=chk.stop_loss, take_profit_1=chk.take_profit_1,
            take_profit_2=chk.take_profit_2, ai_result=ai,
        )
        if pos:
            mode_tag = "🔴 TESTNET LIVE" if is_live_mode() else "📄 PAPER"
            icon = "🟢" if side == "LONG" else "🔴"
            console.print(
                f"\n  [bold]{mode_tag} {icon} ORDER DIBUKA: {side} {symbol}[/bold]\n"
                f"  Entry  : ${entry_price:,.4f} @ {eff_lev}x\n"
                f"  Notional: ${pos.notional:,.2f} | Margin: ${pos.margin:,.2f}\n"
                f"  SL     : ${pos.stop_loss:,.4f}\n"
                f"  TP1    : ${pos.take_profit_1:,.4f}\n"
                f"  Liq    : ${pos.liquidation_price:,.4f}\n"
                f"  AI     : {', '.join(ai.get('_sources',[]))}\n"
                f"  Verdict: {(ai.get('verdict') or '')[:150]}"
            )
            await send_telegram_notification(
                f"{icon} {side} {symbol} {'[TESTNET]' if USE_TESTNET else ''}\n"
                f"Entry: ${entry_price:,.4f} | {eff_lev}x\n"
                f"Size: ${pos.notional:,.2f}\n"
                f"SL: ${pos.stop_loss:,.4f} | TP1: ${pos.take_profit_1:,.4f}\n"
                f"Liq: ${pos.liquidation_price:,.4f}\n"
                f"Score: {ai.get('overall_score',0):.0f}/100\n"
                f"AI({ai.get('_ai_count',1)} models): {', '.join(ai.get('_sources',[]))}\n"
                f"{ai.get('verdict','')[:200]}"
            )

    async def _close(self, symbol: str, price: float, reason: str):
        pos = self.portfolio.open_positions.get(symbol)

        # ── Live testnet: kirim close order ke exchange ────────
        if is_live_mode() and pos:
            result = await close_position_market(symbol, pos.side, pos.qty, reason)
            if result and not result.get("paper"):
                price = result["fill_price"] or price  # gunakan fill price nyata

        trade = self.portfolio.close_position(symbol, price, reason)
        if trade:
            self.risk_mgr.record_trade_result(trade.pnl)
            c = "green" if trade.pnl >= 0 else "red"
            console.print(
                f"\n  [bold {c}]🔒 CLOSE {trade.side} {symbol} "
                f"{'[TESTNET]' if USE_TESTNET else ''}[/bold {c}]\n"
                f"  Exit  : ${price:,.4f}\n"
                f"  P&L   : ${trade.pnl:+,.2f} ({trade.pnl_pct:+.2f}%)\n"
                f"  Reason: {reason}"
            )
            await send_telegram_notification(
                f"🔒 CLOSE {trade.side} {symbol} {'[TESTNET]' if USE_TESTNET else ''}\n"
                f"${price:,.4f} | P&L: ${trade.pnl:+,.2f} ({trade.pnl_pct:+.2f}%)\n"
                f"{reason}"
            )

    # ══════════════════════════════════════════════════════════
    #  MONITOR POSISI AKTIF (SL/TP auto)
    # ══════════════════════════════════════════════════════════
    async def _monitor_positions(self):
        if not self.portfolio.open_positions:
            return
        syms = list(self.portfolio.open_positions.keys())
        console.print(f"\n[cyan]🔍 Monitoring {len(syms)} posisi...[/cyan]")
        prices = await asyncio.to_thread(fetch_current_prices, syms)
        for act in self.portfolio.check_exit_conditions(prices):
            sym    = act["symbol"]
            action = act["action"]
            price  = act["price"]
            reason = act["reason"]
            if action == "CLOSE":
                console.print(f"  [yellow]{reason}[/yellow]")
                await self._close(sym, price, reason)
            elif action == "UPDATE_SL":
                console.print(f"  [cyan]{reason}[/cyan]")

    # ══════════════════════════════════════════════════════════
    #  AI MANAJEMEN POSISI TERBUKA — keputusan keluar dari AI
    # ══════════════════════════════════════════════════════════
    async def _manage_positions_with_ai(self):
        """
        AI Portfolio Manager: analisis semua posisi terbuka sekaligus dan
        putuskan HOLD / CLOSE / PARTIAL / TRAIL / geser TP berdasarkan data
        teknikal + berita + sentimen terbaru.
        Safety: SL hanya boleh digeser mengunci profit, TP hanya searah profit.
        """
        if not self.portfolio.open_positions:
            return
        if not AI_AVAILABLE:
            return   # tanpa AI, aturan SL/TP otomatis tetap jalan

        positions = list(self.portfolio.open_positions.values())

        # Hanya kelola posisi yang sudah bergerak cukup jauh (hemat API)
        active = []
        prices = await asyncio.to_thread(
            fetch_current_prices, [p.symbol for p in positions])
        for p in positions:
            price = prices.get(p.symbol, p.entry_price)
            if p.side == "LONG":
                move = (price / p.entry_price - 1) * 100
            else:
                move = (p.entry_price / price - 1) * 100
            if abs(move) >= AI_MANAGE_MIN_MOVE_PCT:
                active.append(p)
        if not active:
            return

        console.print(f"\n[cyan]🧠 AI mengelola {len(active)} posisi...[/cyan]")
        technical_map = {}
        market_ctx_map = {}
        for p in active:
            try:
                df1m = fetch_ohlcv(p.symbol, "1m", 200)
                df5m = fetch_ohlcv(p.symbol, "5m", 200)
                df15m = fetch_ohlcv(p.symbol, "15m", 150)
                df1h = fetch_ohlcv(p.symbol, "1h", 150)
                fr   = fetch_funding_rate(p.symbol)
                technical_map[p.symbol] = multi_timeframe_analysis(
                    {"1m": df1m, "5m": df5m, "15m": df15m, "1h": df1h}, fr)
            except Exception as e:
                console.print(f"  [dim]TA {p.symbol}: {e}[/dim]")
            try:
                market_ctx_map[p.symbol] = await get_full_market_context(p.symbol)
            except Exception:
                market_ctx_map[p.symbol] = {}

        pctx = self.portfolio.get_context_for_ai()
        result = await manage_positions_with_ai(
            active, prices, technical_map, market_ctx_map, pctx)

        # Terapkan rekomendasi risk multiplier untuk trade berikutnya
        rm = result.get("recommended_risk_multiplier")
        if rm:
            self.risk_mgr.set_ai_risk_multiplier(rm)

        # Terapkan keputusan per posisi (dengan safety check)
        for rec in result.get("positions", []):
            symbol = rec.get("symbol")
            pos = self.portfolio.open_positions.get(symbol)
            if not pos:
                continue
            action = rec.get("action", "HOLD")
            price  = prices.get(symbol, pos.entry_price)
            reason = (rec.get("reason") or "")[:150]

            if action == "CLOSE":
                await self._close(symbol, price, f"AI MANAGE: {reason}")
            elif action == "PARTIAL_CLOSE":
                try:
                    frac = min(max(float(rec.get("close_fraction", 0.5)), 0.1), 0.9)
                except (TypeError, ValueError):
                    frac = 0.5
                trade = self.portfolio.close_position(
                    symbol, price, f"AI MANAGE partial {frac*100:.0f}%: {reason}",
                    partial=frac)
                if trade:
                    self.risk_mgr.record_trade_result(trade.pnl)
                    await send_telegram_notification(
                        f"✂️ PARTIAL {pos.side} {symbol}\n"
                        f"${price:,.4f} | P&L:{trade.pnl:+,.2f}$ ({trade.pnl_pct:+.2f}%)\n"
                        f"{reason}")
            elif action in ("TRAIL_STOP", "MOVE_TP"):
                changed = False
                new_sl = rec.get("new_stop_loss")
                if new_sl:
                    new_sl = float(new_sl)
                    if pos.side == "LONG" and new_sl > pos.stop_loss:
                        pos.stop_loss = new_sl
                        changed = True
                    elif pos.side == "SHORT" and new_sl < pos.stop_loss:
                        pos.stop_loss = new_sl
                        changed = True
                new_tp = rec.get("new_take_profit")
                if new_tp:
                    new_tp = float(new_tp)
                    if pos.side == "LONG" and new_tp > pos.take_profit_1:
                        pos.take_profit_1 = new_tp
                        changed = True
                    elif pos.side == "SHORT" and new_tp < pos.take_profit_1:
                        pos.take_profit_1 = new_tp
                        changed = True
                if changed:
                    self.portfolio._save_state()
                    console.print(f"  [cyan]{symbol}: {action} → {reason}[/cyan]")

    # ══════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════
    def _watchlist(self) -> list[str]:
        p = list(self.portfolio.open_positions.keys())
        for s in CRYPTO_WATCHLIST:
            if s not in p: p.append(s)
        return p

    def _print_summary(self, candidates: list):
        table = Table(title="📊 AI Analysis Results", style="cyan", show_lines=True)
        cols = [("Symbol","left"),("Price","right"),("Action","center"),
                ("Score","right"),("Conf","right"),("Risk","center"),
                ("Fund","right"),("Sent","right"),("RRR","right"),
                ("Leverage","right"),("AI Sources","left"),("Verdict","left")]
        for col, just in cols:
            table.add_column(col, justify=just)
        for item in candidates[:15]:
            ai  = item["ai"]
            action = ai.get("action","HOLD")
            score  = ai.get("overall_score",0)
            lbl    = signal_label(score)
            color  = COLORS.get(lbl,"white")
            risk   = ai.get("risk_level","?")
            rc     = {"LOW":"green","MEDIUM":"yellow","HIGH":"red","VERY_HIGH":"bold red"}.get(risk,"white")
            table.add_row(
                item["symbol"],
                f"${item['price']:,.4f}",
                f"[{color}]{action}[/{color}]",
                f"[{color}]{score:.0f}[/{color}]",
                f"{ai.get('confidence',0):.0f}%",
                f"[{rc}]{risk}[/{rc}]",
                f"{ai.get('fundamental_score',50):.0f}",
                f"{ai.get('sentiment_score',50):.0f}",
                f"1:{ai.get('risk_reward_ratio',0):.1f}",
                f"{ai.get('leverage',5):.0f}x",
                ", ".join(ai.get("_sources",[]))[:20],
                (ai.get("verdict") or "")[:30],
            )
        console.print(table)

    async def _show_portfolio_live(self, market_data: dict):
        prices = {s: d["metrics"].get("price",0)
                  for s, d in market_data.items() if d}
        self.portfolio.print_portfolio(prices)

    # ══════════════════════════════════════════════════════════
    #  START / STOP
    # ══════════════════════════════════════════════════════════
    async def start(self, open_charts: bool = True):
        self.running = True
        if AI_AVAILABLE:
            ai_str = get_active_providers()
        else:
            ai_str = ["Teknikal-Only (isi AI key di .env)"]

        style_txt = "SCALPER" if TRADING_STYLE == "scalper" else "SWING"
        style_desc = ("timeframe 1m/5m/15m, leverage 20x+, holding menit-jam"
                      if TRADING_STYLE == "scalper"
                      else "timeframe 1h/4h/1d, leverage rendah, holding 1-7 hari")

        console.print(Panel(
            f"[bold green]🚀 AI CRYPTO FUTURES BOT STARTED[/bold green]\n"
            f"Exchange  : Binance USDT-M Futures "
            f"({'[bold red]TESTNET[/bold red]' if USE_TESTNET else '[bold yellow]LIVE[/bold yellow]'})\n"
            f"Mode      : [yellow]{TRADING_MODE.upper()}[/yellow] | "
            f"Style     : [bold]{style_txt}[/bold] | "
            f"Exec      : {'🔴 Real orders → testnet.binancefuture.com' if is_live_mode() else '📄 Paper simulation'}\n"
            f"AI        : {' + '.join(ai_str)}\n"
            f"Timeframes: {PRIMARY_TIMEFRAME}(entry) + {TREND_TIMEFRAME}(tren) + {CONTEXT_TIMEFRAME}(konteks)\n"
            f"Style desc: {style_desc}\n"
            f"Pairs     : {CRYPTO_WATCHLIST} (BTC only)\n"
            f"Risk/trade: {MAX_RISK_PER_TRADE}% equity | Leverage: {DEFAULT_LEVERAGE:.0f}x–{MAX_LEVERAGE:.0f}x | Min RRR: 1:{self.risk_mgr.cfg.min_rrr}\n"
            f"Equity    : ${self.portfolio.equity:,.2f}\n"
            f"Monitor   : SL/TP setiap {POSITION_MONITOR_SEC}s | Analisis setiap {ANALYSIS_INTERVAL_MIN}m",
            border_style="green"
        ))

        if open_charts:
            start_chart_server(open_browser=False)

        # ── Cek koneksi testnet sebelum mulai ──
        if is_live_mode():
            bal = await get_account_balance()
            usdt = bal.get("USDT", {})
            if not bal.get("error"):
                console.print(
                    f"[green]✅ Testnet OK — "
                    f"USDT: ${usdt.get('total',0):,.2f} total / "
                    f"${usdt.get('free',0):,.2f} available[/green]"
                )
            else:
                console.print(f"[red]⚠ Testnet connection issue: {bal.get('error')}[/red]")
                console.print("[yellow]Bot tetap jalan — order akan gagal tanpa API secret[/yellow]")

        # Jalankan siklus pertama
        await self.run_cycle()

        # ── Dua loop paralel: analisis + fast monitor ──
        self.scheduler.add_job(
            self.run_cycle, "interval", minutes=ANALYSIS_INTERVAL_MIN,
            id="main_cycle", max_instances=1, coalesce=True,
        )
        self.scheduler.add_job(
            self._fast_position_monitor, "interval", seconds=POSITION_MONITOR_SEC,
            id="fast_monitor", max_instances=1, coalesce=True,
        )
        self.scheduler.start()

        try:
            while self.running:
                await asyncio.sleep(60)
        except (KeyboardInterrupt, SystemExit):
            await self.stop()

    async def _fast_position_monitor(self):
        """
        Loop cepat — cek SL/TP setiap 30 detik TANPA memanggil AI.
        Ini memastikan stop loss dieksekusi tepat waktu.
        """
        if not self.portfolio.open_positions:
            return
        try:
            syms   = list(self.portfolio.open_positions.keys())
            prices = await asyncio.to_thread(fetch_current_prices, syms)
            for act in self.portfolio.check_exit_conditions(prices):
                sym    = act["symbol"]
                action = act["action"]
                price  = act["price"]
                reason = act["reason"]
                pnl_pct= act.get("pnl_pct", 0)
                if action == "CLOSE":
                    console.print(f"  [bold yellow]⚡ FAST MONITOR: {reason} | {pnl_pct:+.1f}%[/bold yellow]")
                    await self._close(sym, price, reason)
                elif action == "UPDATE_SL":
                    console.print(f"  [cyan]⚡ FAST MONITOR: {reason}[/cyan]")
        except Exception as e:
            console.print(f"[dim]Fast monitor error: {e}[/dim]")

    async def stop(self):
        self.running = False
        if self.scheduler.running:
            self.scheduler.shutdown()
        self.portfolio.print_portfolio()
        stats = self.portfolio.get_statistics()
        console.print(Panel(
            f"[bold red]🛑 BOT BERHENTI[/bold red]\n"
            f"Trades: {stats.get('total_trades',0)} | "
            f"WinRate: {stats.get('win_rate',0):.1f}% | "
            f"P&L: ${stats.get('total_pnl',0):+,.2f} | "
            f"PF: {stats.get('profit_factor',0):.2f}",
            border_style="red"
        ))


if __name__ == "__main__":
    bot = TradingBot()
    asyncio.run(bot.start())
