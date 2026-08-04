"""
dashboard.py — Terminal dashboard realtime crypto futures bot
Tampilkan: posisi aktif, P&L live, risk status, sinyal terbaru, chart server link
Jalankan: python dashboard.py
"""
import asyncio
from datetime import datetime
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich.panel import Panel

from portfolio import Portfolio
from risk_manager import RiskManager, RiskConfig
from data_fetcher import fetch_current_prices, get_market_status
from config import CHART_SERVER_PORT

console = Console()


def _build(portfolio: Portfolio, risk: RiskManager, prices: dict, last_results: dict) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="hdr",   size=5),
        Layout(name="mid"),
        Layout(name="foot",  size=4),
    )
    layout["mid"].split_row(Layout(name="pos", ratio=3), Layout(name="right", ratio=2))
    layout["right"].split_column(Layout(name="risk"), Layout(name="signals"))

    # ── Header ──
    rs  = risk.get_status()
    cb  = f"[bold red]🚨 CB: {rs['cb_reason']}[/bold red]" if rs["circuit_breaker"] else "[green]✅ Normal[/green]"
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    eq  = portfolio.equity
    pnl = portfolio.total_pnl
    pc  = "green" if pnl >= 0 else "red"
    layout["hdr"].update(Panel(
        f"[bold cyan]🤖 AI CRYPTO FUTURES BOT[/bold cyan]  |  {get_market_status()}  |  {now} UTC\n"
        f"Equity: [bold]${eq:,.2f}[/bold]  |  "
        f"P&L: [{pc}]{pnl:+,.2f}$ ({portfolio.total_pnl_pct:+.2f}%)[/{pc}]  |  "
        f"Risk: {cb}  |  Charts: http://localhost:{CHART_SERVER_PORT}",
        border_style="cyan",
    ))

    # ── Posisi Aktif ──
    tbl = Table(title="💼 Open Positions", style="cyan", expand=True, show_lines=True)
    for col, j in [("Symbol","l"),("Side","l"),("Entry","r"),("Mark","r"),
                   ("Qty","r"),("Notional","r"),("Margin","r"),
                   ("uPnL $","r"),("uPnL%","r"),("Lev","r"),("SL","r"),("Liq","r"),("Holding","r")]:
        tbl.add_column(col, justify={"l":"left","r":"right"}[j])

    for sym, pos in portfolio.open_positions.items():
        cp = prices.get(sym, pos.entry_price)
        up = pos.unrealized_pnl(cp)
        uc = "green" if up >= 0 else "red"
        upc = pos.unrealized_pnl_pct(cp)
        from datetime import datetime as dt
        try:
            hrs = (dt.now() - dt.fromisoformat(pos.entry_time)).total_seconds() / 3600
            hold_str = f"{hrs:.1f}h"
        except Exception:
            hold_str = "?"
        tbl.add_row(
            sym, pos.side,
            f"${pos.entry_price:,.4f}", f"${cp:,.4f}",
            f"{pos.qty:.4f}", f"${pos.notional:,.1f}",
            f"${pos.margin:,.1f}",
            f"[{uc}]{up:+,.2f}[/{uc}]", f"[{uc}]{upc:+.1f}%[/{uc}]",
            f"{pos.leverage:.0f}x",
            f"${pos.stop_loss:,.4f}", f"${pos.liquidation_price:,.4f}",
            hold_str,
        )
    if not portfolio.open_positions:
        tbl.add_row(*["—"]*13)
    layout["pos"].update(tbl)

    # ── Risk Status ──
    dd_c = "green" if rs["drawdown_pct"] < 5 else ("yellow" if rs["drawdown_pct"] < 10 else "red")
    risk_txt = (
        f"[bold]📊 Portfolio[/bold]\n"
        f"  Balance       : ${portfolio.balance:,.2f}\n"
        f"  Margin Used   : ${portfolio.used_margin:,.2f}\n"
        f"  Available     : ${portfolio.available_margin:,.2f}\n"
        f"  Drawdown      : [{dd_c}]{rs['drawdown_pct']:.2f}%[/{dd_c}]\n"
        f"  Daily Loss    : {rs['daily_loss_pct']:.2f}% / 3.0%\n"
        f"  Weekly Loss   : {rs['weekly_loss_pct']:.2f}% / 7.0%\n"
        f"\n[bold]🛡️ Limits[/bold]\n"
        f"  Max Drawdown  : 15%\n"
        f"  Positions     : {len(portfolio.open_positions)} / 5\n"
        f"  Circuit Breaker: {'ON' if rs['circuit_breaker'] else 'OFF'}"
    )
    layout["risk"].update(Panel(risk_txt, title="Risk", border_style="yellow"))

    # ── Last AI Signals ──
    stats = portfolio.get_statistics()
    sig_txt = (
        f"[bold]📈 Performance[/bold]\n"
        f"  Trades: {stats.get('total_trades',0)} | "
        f"Win: {stats.get('win_rate',0):.1f}%\n"
        f"  P&L: ${stats.get('total_pnl',0):+,.2f}\n"
        f"  PF: {stats.get('profit_factor',0):.2f} | "
        f"Best: {stats.get('best_pct',0):+.1f}%\n"
        f"  Avg Win: +{stats.get('avg_win_pct',0):.1f}% | "
        f"Avg Loss: {stats.get('avg_loss_pct',0):.1f}%\n"
        f"\n[bold]🤖 AI Sources[/bold]\n"
        f"  Primary  : Gemini 1.5 Pro\n"
        f"  Ensemble : GPT-4o + Claude 3.5\n"
        f"  News     : CryptoPanic + RSS (no login)\n"
        f"  Charts   : http://localhost:{CHART_SERVER_PORT}"
    )
    layout["signals"].update(Panel(sig_txt, title="Stats & Info", border_style="blue"))

    # ── Footer ──
    latest = ""
    if last_results:
        parts = []
        for sym, r in list(last_results.items())[:5]:
            a = r.get("action","?")
            s = r.get("overall_score",0)
            c = {"LONG":"green","SHORT":"red","HOLD":"yellow","AVOID":"dim"}.get(a,"white")
            parts.append(f"[{c}]{sym}:{a}({s:.0f})[/{c}]")
        latest = "Last signals: " + "  ".join(parts)
    layout["foot"].update(Panel(
        f"{latest}\n[dim]Ctrl+C to exit  |  Refresh: 30s  |  "
        f"{datetime.utcnow().strftime('%H:%M:%S')} UTC[/dim]",
        border_style="dim",
    ))
    return layout


async def run_dashboard():
    portfolio = Portfolio()
    risk      = RiskManager(RiskConfig())
    risk.current_capital = portfolio.equity
    prices    = {}
    last_results = {}

    with Live(console=console, refresh_per_second=0.5, screen=True) as live:
        try:
            while True:
                if portfolio.open_positions:
                    try:
                        prices = fetch_current_prices(list(portfolio.open_positions.keys()))
                    except Exception:
                        pass
                live.update(_build(portfolio, risk, prices, last_results))
                await asyncio.sleep(30)
        except KeyboardInterrupt:
            pass
    console.print("[cyan]Dashboard closed.[/cyan]")


if __name__ == "__main__":
    asyncio.run(run_dashboard())
