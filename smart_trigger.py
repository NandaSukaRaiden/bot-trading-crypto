"""
smart_trigger.py — Smart AI Trigger System

Sistem filter PINTAR yang hanya panggil AI kalau kondisi PENTING terjadi.
Hemat 90-99% API calls dengan tetap dapat decision quality tinggi.

FILOSOFI:
- Bot hitung indikator lokal (gratis, cepat)
- Filter kondisi: breakout, volatility spike, volume anomaly, divergence
- HANYA panggil AI kalau kondisi "layak analisis" terpenuhi
- AI fokus pada: berita/fundamental, konfirmasi entry, risk adjustment

TRIGGER CONDITIONS (AI dipanggil kalau salah satu TRUE):
1. BREAKOUT       → harga tembus resistance/support mayor + volume
2. VOLATILITY     → ATR spike >50% dari MA(14)
3. VOLUME ANOMALY → volume >2x rata-rata
4. DIVERGENCE     → RSI/MACD diverge kuat dengan harga
5. CONFLUENCE     → 3+ indikator align (high probability setup)
6. NEWS IMPACT    → keyword penting di berita (hack/ETF/Fed/ban)
7. POSITION MOVE  → posisi profit/loss >2% (trailing/close decision)
8. FINAL CONFIRM  → sebelum open posisi baru (safety check)

SKIP AI (pakai teknikal pure) kalau:
- Ranging market (no breakout, volume low)
- Score teknikal <50
- Tidak ada sinyal confluence
- Posisi flat (<0.5% movement)
"""
from typing import Optional
from datetime import datetime
from rich.console import Console

console = Console()


class SmartTrigger:
    """Filter pintar untuk decide kapan panggil AI."""
    
    def __init__(self):
        self.last_trigger_reason = None
        self.trigger_count = 0
        self.skip_count = 0
        
    def should_call_ai_for_entry(
        self,
        symbol: str,
        technical: dict,
        metrics: dict,
        market_context: dict = None,
    ) -> tuple[bool, str]:
        """
        Decide apakah perlu panggil AI untuk analisis ENTRY.
        
        Returns:
            (should_call: bool, reason: str)
        """
        reasons = []
        
        # ── 0. DEEP BUY CHECK (PRIORITAS TERTINGGI!) ─────────
        if self._check_deep_buy_opportunity(technical, metrics):
            reasons.append("🚨DEEP_BUY")
        
        # ── 1. BREAKOUT CHECK ─────────────────────────────────
        if self._check_breakout(technical, metrics):
            reasons.append("BREAKOUT")
        
        # ── 2. VOLATILITY SPIKE ───────────────────────────────
        if self._check_volatility_spike(technical):
            reasons.append("VOLATILITY_SPIKE")
        
        # ── 3. VOLUME ANOMALY ─────────────────────────────────
        if self._check_volume_anomaly(technical, metrics):
            reasons.append("VOLUME_ANOMALY")
        
        # ── 4. DIVERGENCE ─────────────────────────────────────
        if self._check_divergence(technical):
            reasons.append("DIVERGENCE")
        
        # ── 5. HIGH CONFLUENCE ────────────────────────────────
        if self._check_confluence(technical):
            reasons.append("HIGH_CONFLUENCE")
        
        # ── 6. NEWS IMPACT ────────────────────────────────────
        if self._check_news_impact(market_context):
            reasons.append("NEWS_IMPACT")
        
        # ── Decision ──────────────────────────────────────────
        if reasons:
            self.trigger_count += 1
            self.last_trigger_reason = " + ".join(reasons)
            return True, self.last_trigger_reason
        
        # Skip AI — pakai teknikal pure
        self.skip_count += 1
        skip_reason = self._get_skip_reason(technical, metrics)
        return False, skip_reason
    
    def should_call_ai_for_position(
        self,
        position: dict,
        current_price: float,
        technical: dict,
    ) -> tuple[bool, str]:
        """
        Decide apakah perlu panggil AI untuk MANAGE posisi terbuka.
        
        Returns:
            (should_call: bool, reason: str)
        """
        entry = position.get("entry_price", current_price)
        side = position.get("side", "LONG")
        
        # Hitung movement
        if side == "LONG":
            move_pct = (current_price / entry - 1) * 100
        else:
            move_pct = (entry / current_price - 1) * 100
        
        # ── Trigger: posisi bergerak signifikan ──────────────
        if abs(move_pct) >= 2.0:
            self.trigger_count += 1
            reason = f"POSITION_MOVE_{move_pct:+.1f}%"
            return True, reason
        
        # ── Trigger: trend reversal signal ───────────────────
        regime = technical.get("market_regime", "")
        if side == "LONG" and "DOWN" in regime.upper():
            self.trigger_count += 1
            return True, "TREND_REVERSAL_BEARISH"
        if side == "SHORT" and "UP" in regime.upper():
            self.trigger_count += 1
            return True, "TREND_REVERSAL_BULLISH"
        
        # ── Trigger: volatility spike (risk adjustment) ──────
        if self._check_volatility_spike(technical):
            self.trigger_count += 1
            return True, "VOLATILITY_SPIKE"
        
        # Skip AI — posisi flat, no action needed
        self.skip_count += 1
        return False, f"FLAT_POSITION_{move_pct:+.1f}%"
    
    def should_call_ai_final_confirm(self) -> tuple[bool, str]:
        """
        ALWAYS panggil AI untuk final confirmation sebelum open posisi.
        Ini safety check terakhir sebelum eksekusi order nyata.
        """
        self.trigger_count += 1
        return True, "FINAL_CONFIRMATION"
    
    def get_stats(self) -> dict:
        """Statistik trigger vs skip untuk monitoring."""
        total = self.trigger_count + self.skip_count
        if total == 0:
            return {
                "trigger_count": 0,
                "skip_count": 0,
                "trigger_pct": 0,
                "reduction_pct": 0,
            }
        return {
            "trigger_count": self.trigger_count,
            "skip_count": self.skip_count,
            "trigger_pct": round(self.trigger_count / total * 100, 1),
            "reduction_pct": round(self.skip_count / total * 100, 1),
            "last_reason": self.last_trigger_reason,
        }
    
    # ═══════════════════════════════════════════════════════════
    #  INTERNAL CHECKS
    # ═══════════════════════════════════════════════════════════
    
    def _check_deep_buy_opportunity(self, technical: dict, metrics: dict) -> bool:
        """
        Deep Buy = sudden drop + reversal signal (oversold + volume spike).
        Perfect untuk catch bottom sebelum rebound.
        """
        price = metrics.get("price", 0)
        
        # 1. Check recent sharp drop (return -2% atau lebih dalam 5 bar)
        tf = technical.get("per_timeframe", {})
        for label, r in tf.items():
            if not isinstance(r, dict):
                continue
            
            ret_5 = r.get("return_5", 0)
            ret_1 = r.get("return_1", 0)
            
            # Sharp drop tapi mulai rebound
            if ret_5 < -2.0 and ret_1 > -0.5:  # turun 2%+ lalu mulai naik
                # 2. Check oversold condition
                rsi = r.get("rsi", 50)
                rsi7 = r.get("rsi7", 50)
                stoch_k = r.get("stoch_k", 50)
                
                if rsi < 35 or rsi7 < 35 or stoch_k < 25:  # oversold
                    # 3. Check volume spike (confirm reversal)
                    vol_ratio = technical.get("volume_ratio", 1.0)
                    if vol_ratio > 1.3:  # volume naik
                        console.print(f"    [bold green]💎 DEEP BUY OPPORTUNITY DETECTED![/bold green]")
                        console.print(f"      Drop: {ret_5:.1f}% | RSI: {rsi:.0f} | Vol: {vol_ratio:.1f}x")
                        return True
        
        # 4. Check wick rejection (strong support)
        signals = technical.get("signals_primary", [])
        for sig in signals:
            if "hammer" in sig.lower() or "doji" in sig.lower() or "engulfing" in sig.lower():
                vol_ratio = technical.get("volume_ratio", 1.0)
                if vol_ratio > 1.2:
                    return True
        
        return False
    
    def _check_breakout(self, technical: dict, metrics: dict) -> bool:
        """Breakout = harga tembus level penting + volume tinggi."""
        price = metrics.get("price", 0)
        sr = technical.get("support_resistance", {})
        vol_ratio = technical.get("volume_ratio", 1.0)
        
        # Butuh volume >1.5x untuk breakout valid
        if vol_ratio < 1.5:
            return False
        
        # Cek apakah harga dekat resistance/support (±0.5%)
        for key in ["resistance1", "resistance2", "support1", "support2"]:
            level = sr.get(key)
            if level:
                dist_pct = abs(price - level) / level * 100
                if dist_pct < 0.5:  # dalam 0.5% dari level
                    return True
        
        # Cek pivot point
        pivot = sr.get("pivot")
        if pivot:
            dist_pct = abs(price - pivot) / pivot * 100
            if dist_pct < 0.3:
                return True
        
        return False
    
    def _check_volatility_spike(self, technical: dict) -> bool:
        """Volatility spike = ATR naik >50% dari biasanya."""
        atr_pct = technical.get("atr_pct", 0)
        
        # Untuk scalping, ATR normal ~0.3-1.0%
        # Spike = >1.5%
        if atr_pct > 1.5:
            return True
        
        # Cek apakah ATR naik tajam vs timeframe lain
        tf = technical.get("per_timeframe", {})
        atrs = [r.get("atr_pct", 0) for r in tf.values() if isinstance(r, dict)]
        if atrs:
            avg_atr = sum(atrs) / len(atrs)
            if atr_pct > avg_atr * 1.5:  # 50% lebih tinggi dari rata-rata
                return True
        
        return False
    
    def _check_volume_anomaly(self, technical: dict, metrics: dict) -> bool:
        """Volume anomaly = volume >2x rata-rata."""
        vol_ratio = technical.get("volume_ratio", 1.0)
        
        # Volume >2x rata-rata = anomaly (bisa breakout atau news)
        if vol_ratio >= 2.0:
            return True
        
        return False
    
    def _check_divergence(self, technical: dict) -> bool:
        """Divergence = RSI/MACD diverge dengan harga."""
        # Cek apakah ada signal "divergence" dari technical analyzer
        signals = technical.get("signals_primary", [])
        for sig in signals:
            if "divergence" in sig.lower() or "divergen" in sig.lower():
                return True
        
        # Cek manual: RSI trending opposite price
        tf = technical.get("per_timeframe", {})
        for label, r in tf.items():
            if not isinstance(r, dict):
                continue
            
            # Harga naik tapi RSI turun = bearish divergence
            ret_5 = r.get("return_5", 0)
            rsi = r.get("rsi", 50)
            rsi7 = r.get("rsi7", 50)
            
            if ret_5 > 1.0 and rsi < 45 and rsi7 < rsi:
                return True  # bearish div
            if ret_5 < -1.0 and rsi > 55 and rsi7 > rsi:
                return True  # bullish div
        
        return False
    
    def _check_confluence(self, technical: dict) -> bool:
        """High confluence = 3+ indikator align."""
        score = technical.get("score", 0)
        confidence = technical.get("confidence", 0)
        
        # Score tinggi + confidence tinggi = confluence
        if abs(score) >= 60 and confidence >= 70:
            return True
        
        # Cek timeframe alignment
        tf_bias = technical.get("timeframe_bias", {})
        if len(tf_bias) >= 3:
            # Hitung berapa yang align
            bullish = sum(1 for b in tf_bias.values() if b in ["bullish", "up"])
            bearish = sum(1 for b in tf_bias.values() if b in ["bearish", "down"])
            
            # 3+ timeframe agree = confluence
            if bullish >= 3 or bearish >= 3:
                return True
        
        return False
    
    def _check_news_impact(self, market_context: dict = None) -> bool:
        """News impact = keyword penting di berita."""
        if not market_context:
            return False
        
        text = market_context.get("text", "").lower()
        
        # Keywords penting yang butuh AI analysis
        important_keywords = [
            "hack", "hacked", "exploit", "stolen",
            "etf", "approval", "sec",
            "fed", "fomc", "interest rate", "powell",
            "ban", "regulation", "lawsuit",
            "halted", "crashed", "outage",
            "upgrade", "hardfork", "merge",
            "bankruptcy", "collapse", "insolvent",
        ]
        
        for keyword in important_keywords:
            if keyword in text:
                return True
        
        return False
    
    def _get_skip_reason(self, technical: dict, metrics: dict) -> str:
        """Explain why we skip AI."""
        score = technical.get("score", 0)
        regime = technical.get("market_regime", "?")
        vol_ratio = technical.get("volume_ratio", 1.0)
        
        reasons = []
        
        if abs(score) < 50:
            reasons.append("LOW_SCORE")
        
        if "RANGING" in regime.upper():
            reasons.append("RANGING")
        
        if vol_ratio < 0.8:
            reasons.append("LOW_VOLUME")
        
        if not reasons:
            reasons.append("NO_TRIGGER")
        
        return "_".join(reasons)


# ── Global instance ───────────────────────────────────────────
_trigger = SmartTrigger()


def should_call_ai_for_entry(
    symbol: str,
    technical: dict,
    metrics: dict,
    market_context: dict = None,
) -> tuple[bool, str]:
    """Wrapper untuk easy import."""
    return _trigger.should_call_ai_for_entry(symbol, technical, metrics, market_context)


def should_call_ai_for_position(
    position: dict,
    current_price: float,
    technical: dict,
) -> tuple[bool, str]:
    """Wrapper untuk easy import."""
    return _trigger.should_call_ai_for_position(position, current_price, technical)


def get_trigger_stats() -> dict:
    """Get statistics."""
    return _trigger.get_stats()


# ── Testing ───────────────────────────────────────────────────
if __name__ == "__main__":
    console.print("[bold cyan]Smart Trigger System — Test Mode[/bold cyan]\n")
    
    # Test case 1: Breakout + volume
    tech1 = {
        "score": 75,
        "volume_ratio": 2.5,
        "atr_pct": 0.8,
        "support_resistance": {"resistance1": 65000},
        "market_regime": "TRENDING_UP",
    }
    metrics1 = {"price": 65025}
    
    should_call, reason = should_call_ai_for_entry("BTC/USDT", tech1, metrics1)
    console.print(f"Test 1 (Breakout): {should_call} — {reason}")
    
    # Test case 2: Ranging + low volume
    tech2 = {
        "score": 35,
        "volume_ratio": 0.6,
        "atr_pct": 0.5,
        "market_regime": "RANGING",
    }
    metrics2 = {"price": 64500}
    
    should_call, reason = should_call_ai_for_entry("BTC/USDT", tech2, metrics2)
    console.print(f"Test 2 (Ranging): {should_call} — {reason}")
    
    # Test case 3: Volatility spike
    tech3 = {
        "score": 60,
        "volume_ratio": 1.2,
        "atr_pct": 2.5,  # spike!
        "market_regime": "VOLATILE",
    }
    metrics3 = {"price": 64800}
    
    should_call, reason = should_call_ai_for_entry("BTC/USDT", tech3, metrics3)
    console.print(f"Test 3 (Volatility): {should_call} — {reason}")
    
    # Stats
    console.print(f"\n[bold]Stats:[/bold] {get_trigger_stats()}")
