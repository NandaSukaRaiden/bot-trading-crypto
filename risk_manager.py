"""
risk_manager.py
GUARDIAN — filter keamanan untuk SEMUA order (LONG/SHORT) dengan leverage.

Formula kunci:
  notional  = risk_amount / sl_distance        (risiko dikunci, bukan margin)
  margin    = notional / effective_leverage
  leverage dibatasi oleh jarak SL agar SL HIT SEBELUM LIQUIDASI:
      sl_distance * leverage <= liquidation_safety (0.6)

Circuit breaker, daily/weekly loss limit, drawdown protection.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from rich.console import Console

from config import (
    INITIAL_CAPITAL_USDT, MAX_RISK_PER_TRADE, DEFAULT_LEVERAGE,
    MAX_LEVERAGE, MIN_AI_SCORE_TO_TRADE, SYMBOL_MAX_LEVERAGE, TAKER_FEE_RATE,
    MIN_RRR, MIN_SL_DISTANCE_PCT, DAILY_LOSS_LIMIT_PCT,
    WEEKLY_LOSS_LIMIT_PCT, MAX_DRAWDOWN_PCT,
)

console = Console()


@dataclass
class RiskConfig:
    initial_capital: float         = INITIAL_CAPITAL_USDT
    max_risk_per_trade_pct: float  = MAX_RISK_PER_TRADE       # 1% equity per trade
    max_portfolio_exposure_pct: float = 50.0   # max 50% modal terekspos
    max_single_notional_pct: float = 30.0      # max 1 posisi = 30% equity
    max_open_positions: int        = 4         # fokus, tidak scatter

    default_leverage: float        = DEFAULT_LEVERAGE          # 10x default (scalper)
    max_leverage: float            = MAX_LEVERAGE              # 20x hard cap
    min_rrr: float                 = MIN_RRR                   # scalper: 1.5
    liquidation_safety: float      = 0.50      # SL harus hit sebelum 50% margin habis
    min_sl_distance: float         = MIN_SL_DISTANCE_PCT / 100  # SL min (scalper 0.1%)

    # Loss limits (dari env — proteksi utama scalper)
    daily_loss_limit_pct: float    = DAILY_LOSS_LIMIT_PCT      # 2% per hari
    weekly_loss_limit_pct: float   = WEEKLY_LOSS_LIMIT_PCT     # 5% per minggu
    max_drawdown_pct: float        = MAX_DRAWDOWN_PCT          # 10% circuit breaker

    # AI quality filters
    min_score_to_trade: float      = MIN_AI_SCORE_TO_TRADE    # 65
    min_confidence: float          = 55.0      # confidence minimum
    min_confluence_score: float    = 55.0      # filter confluence AI
    max_risk_level_to_trade: str   = "HIGH"    # izinkan HIGH untuk testnet scalper

    # Cooldown
    loss_cooldown_minutes: int     = 5         # 5 menit cooldown (testnet)

    # Order minimum
    min_order_notional: float      = 15.0      # minimum $15

    # SHORT lebih ketat (squeeze risk)
    short_score_bonus: float       = 8.0       # SHORT butuh skor lebih tinggi
    short_notional_factor: float   = 0.7       # notional SHORT lebih kecil
    short_confidence_extra: float  = 8.0

    max_leverage_blacklist: dict   = field(default_factory=dict)


@dataclass
class RiskCheckResult:
    approved: bool
    reason: str
    adjusted_notional: float = 0.0
    margin_required: float   = 0.0
    effective_leverage: float = 0.0
    stop_loss: float         = 0.0
    take_profit_1: float     = 0.0
    take_profit_2: float     = 0.0
    liquidation_price: float = 0.0
    warnings: list           = field(default_factory=list)


def compute_liquidation_price(entry: float, side: str, leverage: float, mmr: float = 0.005) -> float:
    """Harga likuidasi (isolated margin, dengan maintenance margin)."""
    loss_frac = (1 - mmr) / leverage
    if side == "LONG":
        return entry * (1 - loss_frac)
    return entry * (1 + loss_frac)


class RiskManager:
    def __init__(self, config: Optional[RiskConfig] = None):
        self.cfg          = config or RiskConfig()
        self.peak_capital = self.cfg.initial_capital
        self.current_capital = self.cfg.initial_capital
        self.daily_loss   = 0.0
        self.weekly_loss  = 0.0
        self.total_loss   = 0.0
        self.trade_history = []
        self.last_loss_time = None
        self.circuit_breaker = False
        self.cb_reason     = ""
        self._day_start    = datetime.now().date()
        self._week_start   = datetime.now().date() - timedelta(days=datetime.now().weekday())
        # Multiplier risiko yang disarankan AI (0.25 = sangat hati-hati,
        # 1.0 = normal, 1.5 = pasar sangat jelas). Dipakai untuk sizing.
        self.risk_multiplier = 1.0

    def set_ai_risk_multiplier(self, multiplier: float):
        """Atur multiplier risiko dari rekomendasi AI (dibatasi aman)."""
        try:
            m = float(multiplier)
        except (TypeError, ValueError):
            m = 1.0
        self.risk_multiplier = max(0.25, min(1.5, m))

    def _check_reset(self):
        today = datetime.now().date()
        if today != self._day_start:
            self.daily_loss = 0.0
            self._day_start = today
        week_start = today - timedelta(days=today.weekday())
        if week_start != self._week_start:
            self.weekly_loss = 0.0
            self._week_start = week_start

    # ─────────────────────────────────────────────
    #  CHECK ORDER (LONG/SHORT)
    # ─────────────────────────────────────────────
    def check_order(
        self,
        symbol: str,
        side: str,               # "LONG" | "SHORT"
        ai_result: dict,
        current_price: float,
        portfolio,
        symbol_max_leverage: Optional[float] = None,
    ) -> RiskCheckResult:
        """
        Validasi order LONG/SHORT dengan leverage-aware position sizing.
        portfolio harus punya: equity, available_margin, open_positions.
        """
        self._check_reset()
        warnings = []
        cfg = self.cfg

        # ── 1. Circuit breaker ──
        if self.circuit_breaker:
            return RiskCheckResult(approved=False, reason=f"CIRCUIT BREAKER: {self.cb_reason}")

        # ── 2. Daily loss limit ──
        daily_pct = abs(self.daily_loss) / self.peak_capital * 100 if self.peak_capital else 0
        if daily_pct >= cfg.daily_loss_limit_pct:
            self._activate_circuit_breaker(f"Daily loss {daily_pct:.1f}%")
            return RiskCheckResult(approved=False, reason=f"Daily loss limit {daily_pct:.1f}%")
        # ── 3. Weekly loss limit ──
        weekly_pct = abs(self.weekly_loss) / self.peak_capital * 100 if self.peak_capital else 0
        if weekly_pct >= cfg.weekly_loss_limit_pct:
            self._activate_circuit_breaker(f"Weekly loss {weekly_pct:.1f}%")
            return RiskCheckResult(approved=False, reason=f"Weekly loss limit {weekly_pct:.1f}%")
        # ── 4. Max drawdown ──
        dd = (self.peak_capital - self.current_capital) / self.peak_capital * 100 if self.peak_capital else 0
        if dd >= cfg.max_drawdown_pct:
            self._activate_circuit_breaker(f"Max drawdown {dd:.1f}%")
            return RiskCheckResult(approved=False, reason=f"Max drawdown {dd:.1f}%")

        # ── 5. Loss cooldown ──
        if self.last_loss_time:
            elapsed = (datetime.now() - self.last_loss_time).total_seconds() / 60
            if elapsed < cfg.loss_cooldown_minutes:
                return RiskCheckResult(
                    approved=False,
                    reason=f"Cooldown setelah loss: tunggu {cfg.loss_cooldown_minutes - elapsed:.0f}m lagi")

        # ── 6. Skor minimum (SHORT lebih ketat) ──
        required_score = cfg.min_score_to_trade
        if side == "SHORT":
            required_score += cfg.short_score_bonus
        score = ai_result.get("overall_score", 0)
        if score < required_score:
            return RiskCheckResult(
                approved=False,
                reason=f"Skor {score:.0f} < min {required_score:.0f} untuk {side}")

        # ── 7. Confidence minimum ──
        required_conf = cfg.min_confidence
        if side == "SHORT":
            required_conf += cfg.short_confidence_extra
        conf = ai_result.get("confidence", 0)
        if conf < required_conf:
            return RiskCheckResult(
                approved=False,
                reason=f"Confidence {conf:.0f}% < {required_conf:.0f}% untuk {side}")

        # ── 8. Risk level — lebih ketat: max MEDIUM ──
        risk_level = ai_result.get("risk_level", "HIGH")
        allowed = {"LOW", "MEDIUM"}
        if cfg.max_risk_level_to_trade == "HIGH":
            allowed.add("HIGH")
        if risk_level not in allowed:
            return RiskCheckResult(approved=False, reason=f"Risk level {risk_level} tidak diizinkan (max {cfg.max_risk_level_to_trade})")

        # ── 8b. Confluence score — filter kualitas sinyal ──
        confluence = ai_result.get("confluence_score", 100)  # default 100 jika tidak ada (fallback)
        if confluence < cfg.min_confluence_score:
            return RiskCheckResult(
                approved=False,
                reason=f"Confluence score {confluence:.0f} < {cfg.min_confluence_score:.0f} — setup tidak cukup kuat"
            )

        # ── 9. Voting consensus ──
        votes = ai_result.get("action_votes", {})
        ai_count = ai_result.get("_ai_count", 1)
        side_votes = votes.get(side, 0)
        if ai_count >= 3 and side_votes < 2:
            return RiskCheckResult(
                approved=False,
                reason=f"Konsensus {side} tidak cukup ({side_votes}/{ai_count})")

        # ── 10. Max positions ──
        if len(portfolio.open_positions) >= cfg.max_open_positions:
            return RiskCheckResult(approved=False, reason=f"Max posisi {cfg.max_open_positions} tercapai")
        # ── 11. Sudah ada posisi? ──
        if symbol in portfolio.open_positions:
            return RiskCheckResult(approved=False, reason=f"Sudah ada posisi {symbol}")

        # ── 12. Equity & margin tersedia ──
        equity = portfolio.equity
        if equity <= 0:
            return RiskCheckResult(approved=False, reason="Equity tidak tersedia")
        available_margin = portfolio.available_margin
        if available_margin <= 0:
            return RiskCheckResult(approved=False, reason="Margin tidak tersedia")

        # ── 13. Entry / SL / TP dasar ──
        sl_price = ai_result.get("stop_loss")
        if not sl_price:
            sl_pct = ai_result.get("sl_pct", 2.0)
            sl_price = current_price * (1 - sl_pct/100) if side == "LONG" else current_price * (1 + sl_pct/100)

        sl_distance = abs(current_price - sl_price) / current_price
        min_sl = cfg.min_sl_distance
        if sl_distance <= min_sl:   # SL terlalu dekat (scalper: < 0.1%)
            sl_price = current_price * (1 - min_sl * 2) if side == "LONG" else current_price * (1 + min_sl * 2)
            sl_distance = min_sl * 2
            warnings.append(f"SL terlalu dekat → digeser ke {sl_distance*100:.2f}%")

        # ── 14. Risk sizing (dengan multiplier risiko dari AI) ──
        risk_amount = equity * (cfg.max_risk_per_trade_pct / 100) * self.risk_multiplier
        notional    = risk_amount / sl_distance
        max_notional = equity * (cfg.max_single_notional_pct / 100)
        if side == "SHORT":
            max_notional *= cfg.short_notional_factor
        notional = min(notional, max_notional)

        # ── 15. Leverage aman ──
        symbol_max_lev = symbol_max_leverage or cfg.max_leverage
        safe_lev_by_sl = cfg.liquidation_safety / sl_distance
        requested_lev  = ai_result.get("leverage") or cfg.default_leverage
        effective_lev  = min(requested_lev, symbol_max_lev, cfg.max_leverage, safe_lev_by_sl)
        if effective_lev < 1:
            effective_lev = 1

        margin_required = notional / effective_lev

        # ── 16. Margin cap ──
        margin_budget = equity * (cfg.max_portfolio_exposure_pct / 100) - portfolio.used_margin
        margin_budget = max(0.0, min(margin_budget, available_margin * 0.98))
        if margin_required > margin_budget:
            notional = margin_budget * effective_lev
            margin_required = notional / effective_lev
            warnings.append(f"Margin dibatasi → notional dikurangi ke ${notional:,.2f}")

        if notional < cfg.min_order_notional:
            return RiskCheckResult(approved=False, reason=f"Notional < min ${cfg.min_order_notional}")

        # ── 17. RRR validation — minimum 2.5 ──
        tp1_price = ai_result.get("take_profit_1")
        if not tp1_price:
            tp1_price = (current_price * (1 + sl_distance * 2.5) if side == "LONG"
                         else current_price * (1 - sl_distance * 2.5))
        reward = abs(tp1_price - current_price) / current_price
        rrr = reward / sl_distance if sl_distance > 0 else 0
        if rrr < cfg.min_rrr:
            # Coba dengan TP2
            tp2_try = ai_result.get("take_profit_2")
            if tp2_try:
                reward2 = abs(float(tp2_try) - current_price) / current_price
                rrr2 = reward2 / sl_distance
                if rrr2 >= cfg.min_rrr:
                    tp1_price = float(tp2_try)
                    rrr = rrr2
                    warnings.append(f"TP digeser ke TP2 untuk memenuhi RRR {cfg.min_rrr}")
                else:
                    return RiskCheckResult(
                        approved=False,
                        reason=f"RRR {rrr:.2f} < minimum {cfg.min_rrr} — trade tidak worth the risk"
                    )
            else:
                return RiskCheckResult(
                    approved=False,
                    reason=f"RRR {rrr:.2f} < minimum {cfg.min_rrr} — trade tidak worth the risk"
                )

        tp2_price = ai_result.get("take_profit_2")
        if not tp2_price:
            tp2_price = current_price * (1 + sl_distance * 3.2) if side == "LONG" else current_price * (1 - sl_distance * 3.2)

        # ── 18. Liquidasi safety ──
        liq_price = compute_liquidation_price(current_price, side, effective_lev)
        liq_dist = abs(current_price - liq_price) / current_price
        if sl_distance >= liq_dist * 0.9:
            # SL terlalu dekat dengan liquidasi — turunkan leverage
            new_lev = (cfg.liquidation_safety / sl_distance)
            new_lev = min(new_lev, effective_lev)
            if new_lev >= 1:
                effective_lev = new_lev
                liq_price = compute_liquidation_price(current_price, side, effective_lev)
                margin_required = notional / effective_lev
                warnings.append(f"Leverage diturunkan ke {effective_lev:.1f}x demi keamanan likuidasi")
            else:
                return RiskCheckResult(approved=False, reason="SL terlalu jauh — tidak aman untuk leverage apapun")

        # ── 19. Fee impact check ──
        fee_cost = notional * TAKER_FEE_RATE * 2
        if fee_cost > risk_amount * 0.25:
            warnings.append(f"⚠️ Biaya fee ({fee_cost:.2f}$) cukup besar vs risk ({risk_amount:.2f}$)")

        position_pct = (notional / equity) * 100
        warnings.append(f"✅ Notional: ${notional:,.2f} ({position_pct:.1f}% equity)")
        warnings.append(f"✅ Margin: ${margin_required:,.2f} @ {effective_lev:.1f}x")
        warnings.append(f"✅ SL: {current_price:,.4f} → {sl_price:,.4f} (-{sl_distance*100:.2f}%)")
        warnings.append(f"✅ TP1: ${tp1_price:,.4f} | RRR 1:{rrr:.2f}")
        warnings.append(f"✅ Liq: ${liq_price:,.4f} (SL sebelum likuidasi ✓)")

        return RiskCheckResult(
            approved=True,
            reason=f"Semua risk check PASSED ({side} @ {effective_lev:.1f}x)",
            adjusted_notional=notional,
            margin_required=margin_required,
            effective_leverage=effective_lev,
            stop_loss=sl_price,
            take_profit_1=tp1_price,
            take_profit_2=tp2_price,
            liquidation_price=liq_price,
            warnings=warnings,
        )

    # ─────────────────────────────────────────────
    #  UPDATE STATE
    # ─────────────────────────────────────────────
    def record_trade_result(self, realized_pnl: float):
        """Catat hasil trade ke equity tracker."""
        self.current_capital += realized_pnl
        if realized_pnl < 0:
            self.daily_loss  += realized_pnl
            self.weekly_loss += realized_pnl
            self.total_loss  += realized_pnl
            self.last_loss_time = datetime.now()
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        self.trade_history.append({
            "pnl": realized_pnl,
            "time": datetime.now().isoformat(),
        })

    def _activate_circuit_breaker(self, reason: str):
        self.circuit_breaker = True
        self.cb_reason = reason
        console.print(f"\n[bold red]🚨 CIRCUIT BREAKER: {reason}[/bold red]\n")

    def reset_circuit_breaker(self):
        self.circuit_breaker = False
        self.cb_reason = ""
        console.print("[green]✅ Circuit breaker direset.[/green]")

    def get_status(self) -> dict:
        self._check_reset()
        dd = (self.peak_capital - self.current_capital) / self.peak_capital * 100 if self.peak_capital else 0
        return {
            "current_capital": self.current_capital,
            "peak_capital":    self.peak_capital,
            "daily_loss":      self.daily_loss,
            "weekly_loss":     self.weekly_loss,
            "drawdown_pct":    round(dd, 2),
            "daily_loss_pct":  round(abs(self.daily_loss) / self.peak_capital * 100, 2) if self.peak_capital else 0,
            "weekly_loss_pct": round(abs(self.weekly_loss) / self.peak_capital * 100, 2) if self.peak_capital else 0,
            "circuit_breaker": self.circuit_breaker,
            "cb_reason":       self.cb_reason,
        }
