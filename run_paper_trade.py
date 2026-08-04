"""
run_paper_trade.py — Eksekusi percobaan paper trading SATU siklus penuh.
Gunakan Teknikal Fallback (tanpa AI key) untuk demo eksekusi.
Saat Gemini quota tersedia, ganti ke AI mode otomatis.
"""
import asyncio, json
from datetime import datetime
from dotenv import load_dotenv; load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from data_fetcher import (
    fetch_ticker_binance, fetch_ohlcv, fetch_order_book,
    fetch_recent_trades, fetch_funding_rate, fetch_open_interest
)
from technical_analyzer import multi_timeframe_analysis
from risk_manager import RiskManager, RiskConfig
from portfolio import Portfolio
from config import SYMBOL_MAX_LEVERAGE, TAKER_FEE_RATE

console = Console()

# ── Konfigurasi percobaan ─────────────────────────────────────
PAIRS        = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
CAPITAL      = 1000.0   # $1000 USDT paper
RISK_PCT     = 1.0      # 1% per trade
MIN_SCORE    = 5        # sangat rendah hanya untuk demo eksekusi
MIN_ADX      = 10       # minimal ada sedikit tren
MAX_LEVERAGE = 5        # max 5x untuk percobaan konservatif


async def main():
    console.print(Panel(
        "[bold cyan]🤖 PAPER TRADING — SIKLUS PERCOBAAN[/bold cyan]\n"
        f"Modal    : ${CAPITAL:,.0f} USDT\n"
        f"Risk/trade: {RISK_PCT}% | Max Leverage: {MAX_LEVERAGE}x\n"
        f"Waktu    : {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
        f"Mode     : Technical Analysis (Gemini AI akan aktif saat quota tersedia)",
        border_style="cyan"
    ))

    # Init portfolio & risk manager baru untuk percobaan
    port = Portfolio(initial_capital=CAPITAL)
    port.balance = CAPITAL              # reset bersih
    port.open_positions.clear()
    port.closed_trades.clear()

    risk_cfg = RiskConfig(
        initial_capital        = CAPITAL,
        max_risk_per_trade_pct = RISK_PCT,
        max_open_positions     = 3,
        default_leverage       = 3,
        max_leverage           = MAX_LEVERAGE,
        min_rrr                = 2.0,
        min_score_to_trade     = MIN_SCORE,
        min_confidence         = 40.0,
        min_confluence_score   = 0,      # skip confluence check untuk demo
        max_risk_level_to_trade= "HIGH",
        daily_loss_limit_pct   = 5.0,
        loss_cooldown_minutes  = 0,
    )
    rm = RiskManager(risk_cfg)

    # ── STEP 1: Fetch & Analisis Semua Pair ───────────────────
    console.print("\n[bold]STEP 1: Fetch data & analisis teknikal[/bold]")

    analysis_table = Table(title="📊 Hasil Analisis Teknikal", style="cyan", show_lines=True)
    analysis_table.add_column("Pair",      width=12)
    analysis_table.add_column("Price",     justify="right", width=12)
    analysis_table.add_column("24h%",      justify="right", width=8)
    analysis_table.add_column("Score",     justify="right", width=8)
    analysis_table.add_column("Regime",    width=14)
    analysis_table.add_column("RSI",       justify="right", width=6)
    analysis_table.add_column("ADX",       justify="right", width=6)
    analysis_table.add_column("Funding",   justify="right", width=10)
    analysis_table.add_column("OB Imbal",  justify="right", width=10)
    analysis_table.add_column("Tape",      justify="right", width=10)
    analysis_table.add_column("Signal",    width=10)

    candidates = []

    for pair in PAIRS:
        try:
            # Fetch data
            t   = fetch_ticker_binance(pair)
            ob  = fetch_order_book(pair)
            rt  = fetch_recent_trades(pair)
            fr  = fetch_funding_rate(pair)
            oi  = fetch_open_interest(pair)

            price   = t.get("price", 0)
            chg24   = t.get("change_24h_pct", 0)
            imb     = ob.get("depth_imbalance", 0)
            buy_pct = rt.get("buy_ratio_pct", 50)

            # OHLCV multi-TF
            df1h = fetch_ohlcv(pair, "1h", 100)
            df4h = fetch_ohlcv(pair, "4h",  80)
            df1d = fetch_ohlcv(pair, "1d",  50)
            tech = multi_timeframe_analysis({"1h": df1h, "4h": df4h, "1d": df1d}, fr)

            score    = tech.get("score", 0)
            regime   = tech.get("market_regime", "?")
            rsi      = tech.get("primary_rsi", 50)
            adx      = tech.get("primary_adx", 0)
            conflict = tech.get("tf_conflict", False)
            sl_pct   = tech.get("suggested_sl_pct", 2.0)
            tp1_pct  = tech.get("suggested_tp1_pct", 4.0)
            tp2_pct  = tech.get("suggested_tp2_pct", 6.0)
            sr       = tech.get("support_resistance", {})

            # Tentukan sinyal
            if conflict:
                signal = "AVOID"
                signal_color = "dim"
            elif abs(score) < 5:
                signal = "HOLD"
                signal_color = "yellow"
            elif score >= 5 and fr < 0.002:
                signal = "LONG"
                signal_color = "green"
            elif score <= -5:
                signal = "SHORT"
                signal_color = "red"
            else:
                signal = "HOLD"
                signal_color = "yellow"

            # Hitung SL/TP
            if signal == "LONG":
                sl  = round(price * (1 - sl_pct / 100), 6)
                tp1 = round(price * (1 + tp1_pct / 100), 6)
                tp2 = round(price * (1 + tp2_pct / 100), 6)
            elif signal == "SHORT":
                sl  = round(price * (1 + sl_pct / 100), 6)
                tp1 = round(price * (1 - tp1_pct / 100), 6)
                tp2 = round(price * (1 - tp2_pct / 100), 6)
            else:
                sl = tp1 = tp2 = price

            chg_color = "green" if chg24 >= 0 else "red"
            sc_color  = "green" if score > 15 else ("red" if score < -15 else "yellow")

            analysis_table.add_row(
                pair,
                f"${price:,.4f}",
                f"[{chg_color}]{chg24:+.2f}%[/{chg_color}]",
                f"[{sc_color}]{score:+.1f}[/{sc_color}]",
                regime,
                f"{rsi:.1f}",
                f"{adx:.1f}",
                f"{fr*100:+.4f}%",
                f"{imb:+.1f}%",
                f"{buy_pct:.0f}%B/{100-buy_pct:.0f}%S",
                f"[{signal_color}]{signal}[/{signal_color}]",
            )

            if signal in ("LONG", "SHORT"):
                candidates.append({
                    "pair": pair, "signal": signal, "price": price,
                    "score": abs(score), "adx": adx, "rsi": rsi,
                    "sl": sl, "tp1": tp1, "tp2": tp2,
                    "sl_pct": sl_pct, "tp1_pct": tp1_pct,
                    "fr": fr, "imb": imb, "buy_pct": buy_pct,
                    "tech": tech,
                })

        except Exception as e:
            console.print(f"  [red]Error {pair}: {e}[/red]")

    console.print(analysis_table)

    # ── STEP 2: Terapkan Risk Filter & Eksekusi ───────────────
    console.print(f"\n[bold]STEP 2: Risk filter & eksekusi paper trade[/bold]")
    console.print(f"  Kandidat: {len(candidates)} pair memiliki sinyal")

    # Sort by absolute score DESC
    candidates.sort(key=lambda x: x["score"], reverse=True)

    executed = 0
    for c in candidates:
        pair   = c["pair"]
        signal = c["signal"]
        price  = c["price"]

        console.print(f"\n[bold]  → Proses {signal} {pair} @ ${price:,.4f}[/bold]")

        # Buat AI result dummy dari teknikal
        ai_result = {
            "action":            signal,
            "confidence":        min(80, 40 + c["score"]),
            "risk_level":        "MEDIUM" if c["adx"] > 20 else "HIGH",
            "overall_score":     min(100, MIN_SCORE + c["score"]),
            "confluence_score":  70 if not c["tech"].get("tf_conflict") else 30,
            "stop_loss":         c["sl"],
            "take_profit_1":     c["tp1"],
            "take_profit_2":     c["tp2"],
            "leverage":          3,
            "position_size_pct": RISK_PCT,
            "risk_reward_ratio": round(c["tp1_pct"] / c["sl_pct"], 2),
            "holding_period":    "swing_1-3hari",
            "action_votes":      {signal: 2},
            "_ai_count":         2,
            "verdict":           f"Teknikal {signal}: score={c['score']:+.0f}, ADX={c['adx']:.1f}",
        }

        max_lev = SYMBOL_MAX_LEVERAGE.get(pair, 5)
        chk = rm.check_order(pair, signal, ai_result, price, port, max_lev)

        if not chk.approved:
            console.print(f"    [red]✗ BLOCKED: {chk.reason}[/red]")
            continue

        for w in chk.warnings:
            console.print(f"    [dim]{w}[/dim]")

        pos = port.open_position(
            symbol        = pair,
            side          = signal,
            entry_price   = price,
            notional      = chk.adjusted_notional,
            leverage      = chk.effective_leverage,
            stop_loss     = chk.stop_loss,
            take_profit_1 = chk.take_profit_1,
            take_profit_2 = chk.take_profit_2,
            ai_result     = ai_result,
        )

        if pos:
            executed += 1
            liq_dist = abs(price - pos.liquidation_price) / price * 100
            console.print(
                f"    [bold green]✅ ORDER DIBUKA[/bold green]\n"
                f"    {signal} {pair} @ ${price:,.4f}\n"
                f"    Notional : ${pos.notional:,.2f} USDT\n"
                f"    Margin   : ${pos.margin:,.2f} USDT @ {pos.leverage:.0f}x\n"
                f"    Stop Loss: ${pos.stop_loss:,.4f} (-{c['sl_pct']:.2f}%)\n"
                f"    Target 1 : ${pos.take_profit_1:,.4f} (+{c['tp1_pct']:.2f}%)\n"
                f"    Target 2 : ${pos.take_profit_2:,.4f}\n"
                f"    Liq Price: ${pos.liquidation_price:,.4f} ({liq_dist:.1f}% dari entry)\n"
                f"    RRR      : 1:{ai_result['risk_reward_ratio']:.1f}"
            )

    # ── STEP 3: Ringkasan Portfolio ───────────────────────────
    console.print(f"\n[bold]STEP 3: Status portfolio setelah eksekusi[/bold]")

    # Ambil harga terkini untuk unrealized P&L
    cur_prices = {}
    for pair in [p for p in PAIRS if p in port.open_positions]:
        try:
            t = fetch_ticker_binance(pair)
            cur_prices[pair] = t.get("price", 0)
        except Exception:
            pass

    port.print_portfolio(cur_prices)

    # ── STEP 4: Ringkasan Eksekusi ────────────────────────────
    console.print(Panel(
        f"[bold green]📋 RINGKASAN EKSEKUSI PERCOBAAN[/bold green]\n\n"
        f"  Pair dianalisis  : {len(PAIRS)}\n"
        f"  Sinyal teknikal  : {len(candidates)}\n"
        f"  Order dieksekusi : {executed}\n"
        f"  Modal awal       : ${CAPITAL:,.2f}\n"
        f"  Kas tersisa      : ${port.available_margin:,.2f}\n"
        f"  Margin dipakai   : ${port.used_margin:,.2f}\n"
        f"  Equity sekarang  : ${port.equity:,.2f}\n\n"
        f"[yellow]Mode: PAPER TRADING (tidak ada uang nyata)[/yellow]\n"
        f"[cyan]Saat Gemini quota aktif: jalankan python trading_bot.py[/cyan]\n\n"
        f"[bold]Cara monitor posisi ini:[/bold]\n"
        f"  python dashboard.py    ← lihat P&L realtime\n"
        f"  python charts.py BTC/USDT ← lihat chart",
        border_style="green"
    ))

    # Simpan log
    log = {
        "timestamp":  datetime.utcnow().isoformat(),
        "capital":    CAPITAL,
        "executed":   executed,
        "positions":  [
            {
                "pair":    sym,
                "side":    pos.side,
                "entry":   pos.entry_price,
                "notional":pos.notional,
                "margin":  pos.margin,
                "leverage":pos.leverage,
                "sl":      pos.stop_loss,
                "tp1":     pos.take_profit_1,
                "liq":     pos.liquidation_price,
            }
            for sym, pos in port.open_positions.items()
        ],
        "cash_remaining": port.balance,
    }
    with open("paper_trade_log.json", "w") as f:
        json.dump(log, f, indent=2)
    console.print("[dim]Log disimpan ke paper_trade_log.json[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
