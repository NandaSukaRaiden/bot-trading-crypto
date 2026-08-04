"""
backtest.py
Uji strategi LONG/SHORT + leverage pada data historis crypto.
Jalankan: python backtest.py --symbol BTC/USDT --timeframe 1h --bars 500

Simulasi realistis: SL/TP/liq di-check terhadap high/low tiap bar, fee, margin,
dan likuidasi. Scoring memakai MULTI-TIMEFRAME (1h/4h/1d) — sama dengan bot live.
"""
import argparse
import math
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from data_fetcher import fetch_ohlcv
from technical_analyzer import full_technical_analysis
from risk_manager import compute_liquidation_price
from config import TAKER_FEE_RATE

console = Console()

TF_WEIGHTS = {"1h": 0.50, "4h": 0.30, "1d": 0.20}

# Multipliers agar tiap timeframe punya jumlah bar sebanding
TF_BAR_MULT = {"1h": 1, "4h": 4, "1d": 24}


def _build_score_series(symbol: str, timeframe: str, bars: int) -> dict:
    """
    Pre-komputasi skor teknikal per timeframe (kumulatif, tanpa lookahead).
    Returns {tf: pd.Series(score, index=timestamp)} — index disinkronkan ke
    timestamp timeframe utama.
    """
    primary_tf = timeframe
    tfs = [primary_tf]
    if primary_tf == "1h":
        tfs += ["4h", "1d"]
    elif primary_tf == "4h":
        tfs += ["1d"]
    elif primary_tf == "30m":
        tfs += ["1h", "4h"]

    series = {}
    primary = None
    for tf in tfs:
        n = max(bars // TF_BAR_MULT.get(tf, 1), 100)
        df = fetch_ohlcv(symbol, timeframe=tf, limit=n)
        if df is None or len(df) < 60:
            continue
        if primary is None:
            primary = df
        scores = []
        ts = []
        for i in range(60, len(df)):
            r = full_technical_analysis(df.iloc[:i], timeframe_label=tf)
            if "error" in r:
                scores.append(0.0)
            else:
                scores.append(r["score"])
            ts.append(df.index[i])
        s = pd.Series(scores, index=ts)
        series[tf] = s

    if primary is None or not series:
        return {}

    # Sinkronkan semua ke index timeframe utama (asof: pakai nilai terbaru <= t)
    aligned = {}
    for tf, s in series.items():
        if tf == primary_tf:
            aligned[tf] = s
        else:
            aligned[tf] = s.reindex(primary.index, method="ffill").fillna(0.0)
    return aligned


def _multi_tf_score(series: dict, ts) -> float:
    """Gabungkan skor multi-TF + penalti konflik, sama seperti bot live."""
    values = {}
    for tf, s in series.items():
        if ts in s.index:
            values[tf] = s.loc[ts]
    if not values:
        return 0.0

    w_total = sum(TF_WEIGHTS.get(tf, 0.3) for tf in values)
    if w_total <= 0:
        return 0.0
    combined = sum(values[tf] * TF_WEIGHTS.get(tf, 0.3) for tf in values) / w_total

    # Penalti konflik: timeframe utama vs tren berlawanan
    bias = {}
    for tf, v in values.items():
        bias[tf] = "LONG" if v > 0 else ("SHORT" if v < 0 else "NEUTRAL")
    primary_bias = bias.get("1h") or next(iter(bias.values()))
    trend_bias   = bias.get("4h") or bias.get("1d") or "NEUTRAL"
    if primary_bias in ("LONG", "SHORT") and trend_bias in ("LONG", "SHORT") \
            and primary_bias != trend_bias:
        combined *= 0.45
    return max(-100, min(100, combined))


class CryptoBacktester:
    def __init__(
        self,
        initial_capital: float = 1000,
        risk_pct: float        = 1.5,
        leverage: float        = 5.0,
        long_threshold: float  = 38,
        short_threshold: float = -38,
        min_rrr: float         = 2.0,
        min_hold_bars: int     = 3,
        trend_filter: str      = "none",  # none | soft | strict
    ):
        self.capital          = initial_capital
        self.init_capital     = initial_capital
        self.risk_pct         = risk_pct
        self.leverage         = leverage
        self.long_threshold   = long_threshold
        self.short_threshold  = short_threshold
        self.min_rrr          = min_rrr
        self.min_hold_bars    = min_hold_bars
        self.trend_filter     = trend_filter
        self.trades           = []
        self.equity_curve     = []

    def _trend_allows(self, side: str, tf_bias: dict, score: float) -> bool:
        """Filter tren: LONG hanya jika 4h/1d tidak SHORT (dst)."""
        trend = tf_bias.get("4h") or tf_bias.get("1d") or "NEUTRAL"
        if self.trend_filter == "none":
            return True
        if self.trend_filter == "soft":
            if trend == "NEUTRAL":
                return True
            return trend == side
        if self.trend_filter == "strict":
            daily = tf_bias.get("1d", "NEUTRAL")
            if trend == "NEUTRAL" and daily == "NEUTRAL":
                return abs(score) >= self.long_threshold + 8
            if daily == "NEUTRAL":
                return trend == side and abs(score) >= self.long_threshold + 5
            return trend == side and daily == side
        return True

    def run(self, symbol: str, timeframe: str = "1h", bars: int = 500) -> dict:
        console.print(f"\n[cyan]Backtest: {symbol} ({timeframe}) — {bars} bar (multi-TF, trend_filter={self.trend_filter})[/cyan]")
        series = _build_score_series(symbol, timeframe, bars)
        if not series:
            return {"error": "Data tidak cukup", "total_trades": 0}

        primary_tf = timeframe
        df_idx = list(series[primary_tf].index)

        position = None
        equity   = self.init_capital

        for i in range(1, len(df_idx)):
            ts  = df_idx[i]
            score = float(series[primary_tf].loc[ts])
            tf_bias = {}
            for tf, s in series.items():
                v = float(s.loc[ts]) if ts in s.index else 0.0
                tf_bias[tf] = "LONG" if v > 0 else ("SHORT" if v < 0 else "NEUTRAL")
            combo = _multi_tf_score(series, ts)

            bar_idx = ts
            bar = None
            # butuh df asli untuk O/H/L/C; ambil dari cache di loop pertama:
            if not hasattr(self, "_raw_cache"):
                self._raw_cache = fetch_ohlcv(symbol, timeframe=primary_tf, limit=max(len(df_idx), 100))
            df = self._raw_cache
            if bar_idx in df.index:
                bar = df.loc[bar_idx]

            if bar is None:
                continue
            o, h, l, c = bar["Open"], bar["High"], bar["Low"], bar["Close"]
            atr_pct = 2.0

            if position is None:
                # ── Buka posisi ──
                side = None
                if combo >= self.long_threshold:
                    side = "LONG"
                elif combo <= self.short_threshold:
                    side = "SHORT"
                if side and not self._trend_allows(side, tf_bias, combo):
                    side = None

                if side:
                    # Jarak SL berbasis kekuatan skor (skor kuat → SL lebih ketat)
                    strength = abs(combo)
                    sl_pct = max(1.0, min(3.5, 2.8 - strength / 40))
                    sl_dist = sl_pct / 100
                    notional = min((equity * self.risk_pct / 100) / sl_dist, equity * 0.5)
                    margin   = notional / self.leverage
                    if margin > equity * 0.5:
                        margin = equity * 0.5
                        notional = margin * self.leverage
                    if notional >= 10:
                        sl = c * (1 - sl_dist) if side == "LONG" else c * (1 + sl_dist)
                        tp = c * (1 + sl_dist * self.min_rrr) if side == "LONG" else c * (1 - sl_dist * self.min_rrr)
                        liq = compute_liquidation_price(c, side, self.leverage)
                        position = {
                            "side": side, "entry": c, "sl": sl, "tp": tp,
                            "liq": liq, "notional": notional, "margin": margin,
                            "qty": notional / c, "i": i, "date": bar_idx,
                        }
            else:
                # ── Monitor posisi ──
                exit_price, exit_reason = None, None
                holding_bars = i - position["i"]
                if position["side"] == "LONG":
                    if l <= position["sl"]:
                        exit_price, exit_reason = position["sl"], "SL"
                    elif h >= position["tp"]:
                        exit_price, exit_reason = position["tp"], "TP"
                    elif l <= position["liq"]:
                        exit_price, exit_reason = position["liq"], "LIQUIDATED"
                    elif holding_bars >= self.min_hold_bars and combo <= self.short_threshold:
                        exit_price, exit_reason = c, "REVERSAL"
                else:  # SHORT
                    if h >= position["sl"]:
                        exit_price, exit_reason = position["sl"], "SL"
                    elif l <= position["tp"]:
                        exit_price, exit_reason = position["tp"], "TP"
                    elif h >= position["liq"]:
                        exit_price, exit_reason = position["liq"], "LIQUIDATED"
                    elif holding_bars >= self.min_hold_bars and combo >= self.long_threshold:
                        exit_price, exit_reason = c, "REVERSAL"

                if exit_reason is None and i == len(df_idx) - 1:
                    exit_price, exit_reason = c, "END"

                if exit_reason:
                    pos = position
                    if pos["side"] == "LONG":
                        price_pnl = (exit_price - pos["entry"]) * pos["qty"]
                    else:
                        price_pnl = (pos["entry"] - exit_price) * pos["qty"]

                    fee     = pos["notional"] * TAKER_FEE_RATE + exit_price * pos["qty"] * TAKER_FEE_RATE
                    pnl     = price_pnl - fee
                    pnl_pct = (pnl / pos["margin"]) * 100

                    equity += pnl
                    holding = i - pos["i"]

                    self.trades.append({
                        "symbol": symbol, "side": pos["side"],
                        "entry_date": str(pos["date"])[:16],
                        "exit_date":  str(bar_idx)[:16],
                        "entry": pos["entry"], "exit": exit_price,
                        "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 2),
                        "holding": holding, "reason": exit_reason,
                    })
                    position = None

            self.equity_curve.append(equity)

        return self._calculate_metrics()

    def _calculate_metrics(self) -> dict:
        if not self.trades:
            return {"total_trades": 0, "win_rate": 0, "total_return_pct": 0,
                    "final_capital": self.init_capital, "max_drawdown_pct": 0,
                    "profit_factor": 0, "avg_win_pct": 0, "avg_loss_pct": 0,
                    "sharpe": 0, "long_trades": 0, "short_trades": 0,
                    "liq_count": 0, "best_trade_pct": 0, "worst_trade_pct": 0}

        wins  = [t for t in self.trades if t["pnl"] > 0]
        loses = [t for t in self.trades if t["pnl"] < 0]
        gross_p = sum(t["pnl"] for t in wins)
        gross_l = abs(sum(t["pnl"] for t in loses))

        eq = pd.Series(self.equity_curve)
        peak = eq.cummax()
        dd   = ((eq - peak) / peak * 100)
        max_dd = abs(dd.min())

        final = eq.iloc[-1]
        total_return = (final / self.init_capital - 1) * 100

        longs  = [t for t in self.trades if t["side"] == "LONG"]
        shorts = [t for t in self.trades if t["side"] == "SHORT"]
        liq_count = sum(1 for t in self.trades if t["reason"] == "LIQUIDATED")

        if len(eq) > 1:
            rets = eq.pct_change().dropna()
            sharpe = (rets.mean() / rets.std() * math.sqrt(len(rets))) if rets.std() > 0 else 0
        else:
            sharpe = 0

        return {
            "total_trades": len(self.trades),
            "win_rate":     round(len(wins) / len(self.trades) * 100, 1),
            "total_return_pct": round(total_return, 2),
            "final_capital": round(final, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "profit_factor": round(gross_p / gross_l, 2) if gross_l > 0 else 999,
            "avg_win_pct":   round(sum(t["pnl_pct"] for t in wins) / len(wins), 2) if wins else 0,
            "avg_loss_pct":  round(sum(t["pnl_pct"] for t in loses) / len(loses), 2) if loses else 0,
            "sharpe":        round(sharpe, 2),
            "long_trades":   len(longs),
            "short_trades":  len(shorts),
            "liq_count":     liq_count,
            "best_trade_pct": round(max(t["pnl_pct"] for t in self.trades), 2),
            "worst_trade_pct": round(min(t["pnl_pct"] for t in self.trades), 2),
        }

    def print_results(self, metrics: dict):
        if "error" in metrics:
            console.print(f"[red]{metrics['error']}[/red]")
            return

        color = "green" if metrics["total_return_pct"] > 0 else "red"
        console.print(Panel(
            f"[bold]📊 HASIL BACKTEST[/bold]\n\n"
            f"Total Trade     : {metrics['total_trades']} (LONG {metrics['long_trades']} / SHORT {metrics['short_trades']})\n"
            f"Win Rate        : {metrics['win_rate']:.1f}%\n"
            f"[{color}]Total Return    : {metrics['total_return_pct']:+.2f}%[/{color}]\n"
            f"Final Capital   : ${metrics['final_capital']:,.2f}\n"
            f"Max Drawdown    : -{metrics['max_drawdown_pct']:.2f}%\n"
            f"Profit Factor   : {metrics['profit_factor']:.2f}\n"
            f"Avg Win         : +{metrics['avg_win_pct']:.2f}% | Avg Loss: {metrics['avg_loss_pct']:.2f}%\n"
            f"Sharpe (est.)   : {metrics['sharpe']:.2f}\n"
            f"Likuidasi       : {metrics['liq_count']} (harusnya 0)\n"
            f"Best / Worst    : +{metrics['best_trade_pct']:.2f}% / {metrics['worst_trade_pct']:.2f}%",
            border_style=color
        ))

        if self.trades:
            table = Table(title="Trade History", style="cyan")
            for col in ["Entry", "Exit", "Side", "Entry$", "Exit$", "P&L%", "Bar", "Reason"]:
                table.add_column(col, justify="right" if col not in ("Side", "Reason", "Entry", "Exit") else "left")
            for t in self.trades[-25:]:
                c = "green" if t["pnl"] > 0 else "red"
                table.add_row(
                    t["entry_date"], t["exit_date"], t["side"],
                    f"{t['entry']:.4f}", f"{t['exit']:.4f}",
                    f"[{c}]{t['pnl_pct']:+.2f}%[/{c}]",
                    str(t["holding"]), t["reason"])
            console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest crypto futures bot")
    parser.add_argument("--symbol",     default="BTC/USDT")
    parser.add_argument("--timeframe",  default="1h")
    parser.add_argument("--bars",       default=500, type=int)
    parser.add_argument("--capital",    default=1000, type=float)
    parser.add_argument("--risk",       default=1.5, type=float)
    parser.add_argument("--leverage",   default=5.0, type=float)
    parser.add_argument("--long-threshold",  default=38, type=float)
    parser.add_argument("--short-threshold", default=-38, type=float)
    parser.add_argument("--min-hold",   default=3, type=int)
    parser.add_argument("--trend-filter", default="none", choices=["none", "soft", "strict"])
    args = parser.parse_args()

    bt = CryptoBacktester(
        initial_capital=args.capital, risk_pct=args.risk, leverage=args.leverage,
        long_threshold=args.long_threshold, short_threshold=args.short_threshold,
        min_hold_bars=args.min_hold, trend_filter=args.trend_filter,
    )
    result = bt.run(args.symbol, args.timeframe, args.bars)
    bt.print_results(result)
