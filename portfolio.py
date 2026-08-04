"""
portfolio.py
Futures paper trading engine — LONG & SHORT dengan leverage, margin, likuidasi.
Mendukung: SL, TP1 (partial), TP2, trailing stop, fee realistic, persistensi.
"""
import json
import os
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
from rich.console import Console
from rich.table import Table

from config import INITIAL_CAPITAL_USDT, TRADING_MODE, TAKER_FEE_RATE
from risk_manager import compute_liquidation_price

console = Console()
PORTFOLIO_FILE = "portfolio_state.json"

TRAIL_TRIGGER_PCT = 6.0   # mulai trailing setelah profit > 6%
TRAIL_STOP_PCT    = 3.0   # jarak trailing stop 3%


@dataclass
class Position:
    symbol: str
    side: str                 # LONG | SHORT
    entry_price: float
    qty: float
    notional: float
    margin: float
    leverage: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    liquidation_price: float
    entry_time: str
    holding_period: str
    ai_score: float
    ai_confidence: float
    ai_sources: list
    notes: str = ""
    partial_sold: bool = False

    def unrealized_pnl(self, price: float) -> float:
        if self.side == "LONG":
            return (price - self.entry_price) * self.qty
        return (self.entry_price - price) * self.qty

    def unrealized_pnl_pct(self, price: float) -> float:
        return (self.unrealized_pnl(price) / self.margin) * 100


@dataclass
class Trade:
    symbol: str
    side: str                 # LONG | SHORT
    action: str               # OPEN | CLOSE | PARTIAL_CLOSE
    price: float
    qty: float
    notional: float
    pnl: float
    pnl_pct: float
    time: str
    reason: str
    ai_score: float
    leverage: float
    holding_hours: float = 0


class Portfolio:
    def __init__(self, initial_capital: float = INITIAL_CAPITAL_USDT, mode: str = TRADING_MODE):
        self.mode            = mode
        self.initial_capital = initial_capital
        self.balance         = initial_capital
        self.open_positions: dict[str, Position] = {}
        self.closed_trades: list[Trade] = []
        self._load_state()

    # ── PROPERTI ─────────────────────────────────────────────
    @property
    def used_margin(self) -> float:
        return sum(p.margin for p in self.open_positions.values())

    @property
    def available_margin(self) -> float:
        return max(0.0, self.balance - self.used_margin)

    def unrealized_pnl_total(self, current_prices: dict = None) -> float:
        current_prices = current_prices or {}
        total = 0.0
        for symbol, pos in self.open_positions.items():
            price = current_prices.get(symbol, pos.entry_price)
            total += pos.unrealized_pnl(price)
        return total

    @property
    def equity(self) -> float:
        return self.balance + self.unrealized_pnl_total()

    @property
    def total_pnl(self) -> float:
        return self.equity - self.initial_capital

    @property
    def total_pnl_pct(self) -> float:
        return (self.equity / self.initial_capital - 1) * 100

    @property
    def drawdown_pct(self) -> float:
        peak = max(self.initial_capital, self.balance + self.unrealized_pnl_total())
        if peak <= 0:
            return 0
        return max(0.0, (peak - self.equity) / peak * 100)

    # ── EXECUTE OPEN ────────────────────────────────────────
    def open_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        notional: float,
        leverage: float,
        stop_loss: float,
        take_profit_1: float,
        take_profit_2: float,
        ai_result: dict,
    ) -> Optional[Position]:
        """Buka posisi LONG/SHORT (paper)."""
        margin  = notional / leverage
        open_fee = notional * TAKER_FEE_RATE

        if margin + open_fee > self.balance:
            margin = max(self.balance - open_fee, 0)
            notional = margin * leverage
            if notional < 10:
                console.print(f"[yellow]Skip {symbol}: margin tidak cukup[/yellow]")
                return None

        qty = notional / entry_price
        liq_price = compute_liquidation_price(entry_price, side, leverage)

        pos = Position(
            symbol=symbol, side=side, entry_price=entry_price, qty=qty,
            notional=notional, margin=margin, leverage=leverage,
            stop_loss=stop_loss, take_profit_1=take_profit_1,
            take_profit_2=take_profit_2, liquidation_price=liq_price,
            entry_time=datetime.now().isoformat(),
            holding_period=ai_result.get("holding_period", "swing_1-3hari"),
            ai_score=ai_result.get("overall_score", 0),
            ai_confidence=ai_result.get("confidence", 0),
            ai_sources=ai_result.get("_sources", []),
            notes=ai_result.get("verdict", ""),
        )
        self.open_positions[symbol] = pos
        self.balance -= margin + open_fee

        mode = "[PAPER]" if self.mode == "paper" else "[LIVE]"
        arrow = "🟢" if side == "LONG" else "🔴"
        console.print(
            f"\n[bold green]{mode} {arrow} {side} {symbol}[/bold green]\n"
            f"  Entry: ${entry_price:,.4f}\n"
            f"  Qty: {qty:.6f} | Notional: ${notional:,.2f}\n"
            f"  Margin: ${margin:,.2f} @ {leverage:.1f}x\n"
            f"  SL: ${stop_loss:,.4f} | TP1: ${take_profit_1:,.4f} | TP2: ${take_profit_2:,.4f}\n"
            f"  Liq: ${liq_price:,.4f}\n"
            f"  Score: {ai_result.get('overall_score',0):.0f} | Fee: ${open_fee:.2f}\n"
        )
        self._save_state()
        return pos

    # ── EXECUTE CLOSE ───────────────────────────────────────
    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str = "AI Signal",
        partial: float = 1.0,
    ) -> Optional[Trade]:
        """Tutup posisi (sebagian atau seluruhnya)."""
        if symbol not in self.open_positions:
            return None

        pos = self.open_positions[symbol]
        close_qty = pos.qty * partial
        close_notional = close_qty * exit_price
        close_fee = close_notional * TAKER_FEE_RATE

        if pos.side == "LONG":
            price_pnl = (exit_price - pos.entry_price) * close_qty
        else:
            price_pnl = (pos.entry_price - exit_price) * close_qty

        pnl = price_pnl - close_fee
        pnl_pct = (pnl / (pos.margin * partial)) * 100 if pos.margin * partial > 0 else 0

        entry_dt = datetime.fromisoformat(pos.entry_time)
        holding_hours = (datetime.now() - entry_dt).total_seconds() / 3600

        trade = Trade(
            symbol=symbol, side=pos.side,
            action="CLOSE" if partial >= 0.999 else "PARTIAL_CLOSE",
            price=exit_price, qty=close_qty, notional=close_notional,
            pnl=pnl, pnl_pct=round(pnl_pct, 2),
            time=datetime.now().isoformat(), reason=reason,
            ai_score=pos.ai_score, leverage=pos.leverage,
            holding_hours=round(holding_hours, 1),
        )
        self.closed_trades.append(trade)

        # Akuntansi yang benar:
        # balance += realized_pnl (harga) - close_fee + margin yang dilepas
        margin_release = pos.margin * partial
        self.balance += pnl + margin_release

        pnl_color = "green" if pnl >= 0 else "red"
        arrow = "🟢" if pos.side == "LONG" else "🔴"
        console.print(
            f"\n[bold {pnl_color}]{arrow} CLOSE {pos.side} {symbol} ({'partial' if partial<1 else 'full'})[/bold {pnl_color}]\n"
            f"  Exit: ${exit_price:,.4f}\n"
            f"  P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}% margin)\n"
            f"  Alasan: {reason}\n"
        )

        if partial >= 0.999:
            del self.open_positions[symbol]
        else:
            pos.qty -= close_qty
            pos.notional -= close_notional
            pos.margin -= margin_release
            pos.partial_sold = True
            if pos.side == "LONG":
                pos.stop_loss = max(pos.stop_loss, pos.entry_price * 1.001)  # SL ke breakeven+
            else:
                pos.stop_loss = min(pos.stop_loss, pos.entry_price * 0.999)

        self._save_state()
        return trade

    # ── EXIT CONDITIONS (SL/TP/trailing) ─────────────────────
    def check_exit_conditions(self, current_prices: dict) -> list[dict]:
        actions = []
        for symbol, pos in list(self.open_positions.items()):
            price = current_prices.get(symbol)
            if not price:
                continue

            pnl_pct = pos.unrealized_pnl_pct(price)

            if pos.side == "LONG":
                # STOP LOSS
                if price <= pos.stop_loss:
                    actions.append({
                        "symbol": symbol, "action": "CLOSE", "price": price,
                        "reason": f"🛑 STOP LOSS HIT ({pos.side})", "partial": 1.0, "pnl_pct": pnl_pct})
                # TP1 → partial 50%
                elif price >= pos.take_profit_1 and not pos.partial_sold:
                    actions.append({
                        "symbol": symbol, "action": "CLOSE", "price": price,
                        "reason": f"🎯 TP1 HIT — take profit 50% ({pos.side})", "partial": 0.5, "pnl_pct": pnl_pct})
                # TP2 → close semua
                elif price >= pos.take_profit_2 and pos.partial_sold:
                    actions.append({
                        "symbol": symbol, "action": "CLOSE", "price": price,
                        "reason": f"🏁 TP2 HIT — tutup semua ({pos.side})", "partial": 1.0, "pnl_pct": pnl_pct})
                # Trailing stop
                elif (price / pos.entry_price - 1) * 100 >= TRAIL_TRIGGER_PCT:
                    new_sl = price * (1 - TRAIL_STOP_PCT / 100)
                    if new_sl > pos.stop_loss:
                        pos.stop_loss = new_sl
                        actions.append({
                            "symbol": symbol, "action": "UPDATE_SL", "price": price,
                            "reason": f"📈 Trailing SL naik ke ${new_sl:,.4f}", "new_sl": new_sl, "pnl_pct": pnl_pct})
            else:  # SHORT
                if price >= pos.stop_loss:
                    actions.append({
                        "symbol": symbol, "action": "CLOSE", "price": price,
                        "reason": f"🛑 STOP LOSS HIT ({pos.side})", "partial": 1.0, "pnl_pct": pnl_pct})
                elif price <= pos.take_profit_1 and not pos.partial_sold:
                    actions.append({
                        "symbol": symbol, "action": "CLOSE", "price": price,
                        "reason": f"🎯 TP1 HIT — take profit 50% ({pos.side})", "partial": 0.5, "pnl_pct": pnl_pct})
                elif price <= pos.take_profit_2 and pos.partial_sold:
                    actions.append({
                        "symbol": symbol, "action": "CLOSE", "price": price,
                        "reason": f"🏁 TP2 HIT — tutup semua ({pos.side})", "partial": 1.0, "pnl_pct": pnl_pct})
                elif (pos.entry_price / price - 1) * 100 >= TRAIL_TRIGGER_PCT:
                    new_sl = price * (1 + TRAIL_STOP_PCT / 100)
                    if new_sl < pos.stop_loss:
                        pos.stop_loss = new_sl
                        actions.append({
                            "symbol": symbol, "action": "UPDATE_SL", "price": price,
                            "reason": f"📈 Trailing SL turun ke ${new_sl:,.4f}", "new_sl": new_sl, "pnl_pct": pnl_pct})

        return actions

    # ── STATISTIK ────────────────────────────────────────────
    def get_statistics(self) -> dict:
        closes = [t for t in self.closed_trades if t.action in ("CLOSE", "PARTIAL_CLOSE")]
        if not closes:
            return {"total_trades": 0, "win_rate": 0, "total_pnl": 0, "profit_factor": 0,
                    "avg_win_pct": 0, "avg_loss_pct": 0, "best_pct": 0, "worst_pct": 0}

        wins   = [t for t in closes if t.pnl > 0]
        loses  = [t for t in closes if t.pnl < 0]
        gross_p = sum(t.pnl for t in wins)
        gross_l = abs(sum(t.pnl for t in loses))

        return {
            "total_trades": len(closes),
            "win_trades":   len(wins),
            "loss_trades":  len(loses),
            "win_rate":     round(len(wins) / len(closes) * 100, 1),
            "total_pnl":    round(sum(t.pnl for t in closes), 2),
            "profit_factor": round(gross_p / gross_l, 2) if gross_l > 0 else float("inf"),
            "avg_win_pct":  round(sum(t.pnl_pct for t in wins) / len(wins), 2) if wins else 0,
            "avg_loss_pct": round(sum(t.pnl_pct for t in loses) / len(loses), 2) if loses else 0,
            "best_pct":     round(max((t.pnl_pct for t in closes), default=0), 2),
            "worst_pct":    round(min((t.pnl_pct for t in closes), default=0), 2),
        }

    # ── DISPLAY ──────────────────────────────────────────────
    def print_portfolio(self, current_prices: dict = None):
        current_prices = current_prices or {}
        eq = self.equity
        pnl = self.total_pnl
        pnl_pct = self.total_pnl_pct
        stats = self.get_statistics()
        color = "green" if pnl >= 0 else "red"

        console.print(f"\n[bold cyan]═══ PORTFOLIO SUMMARY ═══[/bold cyan]")
        console.print(f"  Equity        : ${eq:>14,.2f}")
        console.print(f"  Balance       : ${self.balance:>14,.2f}")
        console.print(f"  Margin Dipakai: ${self.used_margin:>14,.2f} | Tersedia: ${self.available_margin:,.2f}")
        console.print(f"  P&L Total     : [{color}]${pnl:>+14,.2f} ({pnl_pct:+.2f}%)[/{color}]")
        console.print(f"  Drawdown      : {self.drawdown_pct:.2f}%")
        console.print(f"  Win Rate      : {stats.get('win_rate',0):.1f}% ({stats.get('win_trades',0)}W/{stats.get('loss_trades',0)}L)")
        console.print(f"  Profit Factor : {stats.get('profit_factor',0):.2f}")

        if self.open_positions:
            table = Table(title="\nPosisi Aktif", style="cyan")
            for col in ["Symbol", "Side", "Entry", "Harga", "Qty", "Margin", "P&L", "P&L%", "Lev", "SL", "Liq"]:
                table.add_column(col, justify="right" if col not in ("Symbol", "Side") else "left")
            for symbol, pos in self.open_positions.items():
                cp = current_prices.get(symbol, pos.entry_price)
                upnl = pos.unrealized_pnl(cp)
                upnl_pct = pos.unrealized_pnl_pct(cp)
                c = "green" if upnl >= 0 else "red"
                table.add_row(
                    symbol, pos.side,
                    f"${pos.entry_price:,.4f}", f"${cp:,.4f}", f"{pos.qty:.6f}",
                    f"${pos.margin:,.2f}",
                    f"[{c}]${upnl:+,.2f}[/{c}]", f"[{c}]{upnl_pct:+.1f}%[/{c}]",
                    f"{pos.leverage:.0f}x", f"${pos.stop_loss:,.4f}", f"${pos.liquidation_price:,.4f}")
            console.print(table)

    # ── PERSISTENCE ──────────────────────────────────────────
    def _save_state(self):
        state = {
            "mode": self.mode,
            "initial_capital": self.initial_capital,
            "balance": self.balance,
            "open_positions": {k: asdict(v) for k, v in self.open_positions.items()},
            "closed_trades": [asdict(t) for t in self.closed_trades[-500:]],
            "saved_at": datetime.now().isoformat(),
        }
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def _load_state(self):
        if not os.path.exists(PORTFOLIO_FILE):
            return
        try:
            with open(PORTFOLIO_FILE) as f:
                state = json.load(f)
            self.balance = float(state.get("balance", self.initial_capital))
            for k, v in state.get("open_positions", {}).items():
                self.open_positions[k] = Position(**v)
            for t in state.get("closed_trades", []):
                self.closed_trades.append(Trade(**t))
            console.print(f"[cyan]📂 Portfolio dimuat dari {PORTFOLIO_FILE}[/cyan]")
        except Exception as e:
            console.print(f"[yellow]Warning load portfolio: {e}[/yellow]")

    def get_context_for_ai(self) -> dict:
        return {
            "equity":           round(self.equity, 2),
            "available_margin": round(self.available_margin, 2),
            "open_positions":   len(self.open_positions),
            "total_pnl_pct":    round(self.total_pnl_pct, 2),
            "drawdown_pct":     round(self.drawdown_pct, 2),
            "mode":             self.mode,
        }
