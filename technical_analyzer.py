"""
technical_analyzer.py — Professional Grade Technical Analysis
Indikator: RSI, MACD, Bollinger, ADX, Stochastic, VWAP, Ichimoku,
           Volume Profile, EMA Stack, OBV, CMF, ATR, Divergence
Skor: -100 (strong short) hingga +100 (strong long)
Multi-TF: 1h (50%) + 4h (30%) + 1d (20%) dengan konfirmasi wajib
"""
import numpy as np
import pandas as pd
from config import TA_CONFIG


# ═══════════════════════════════════════════════════════════════
#  INDIKATOR DASAR
# ═══════════════════════════════════════════════════════════════
def rsi(s: pd.Series, p: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p-1, min_periods=p).mean()
    l = (-d.clip(upper=0)).ewm(com=p-1, min_periods=p).mean()
    return (100 - 100 / (1 + g / l.replace(0, np.nan))).fillna(50)


def macd(s: pd.Series, fast=12, slow=26, sig=9):
    f = s.ewm(span=fast, adjust=False).mean()
    sl= s.ewm(span=slow, adjust=False).mean()
    m = f - sl
    sg= m.ewm(span=sig, adjust=False).mean()
    return m, sg, m - sg


def bollinger(s: pd.Series, p=20, std=2):
    mid = s.rolling(p).mean()
    sd  = s.rolling(p).std()
    up  = mid + std * sd
    dn  = mid - std * sd
    pb  = (s - dn) / (up - dn).replace(0, np.nan)
    bw  = (up - dn) / mid.replace(0, np.nan) * 100   # bandwidth %
    return up, mid, dn, pb.fillna(0.5), bw.fillna(0)


def stochastic(df: pd.DataFrame, k=14, d=3):
    lo = df["Low"].rolling(k).min()
    hi = df["High"].rolling(k).max()
    K  = 100 * (df["Close"] - lo) / (hi - lo).replace(0, np.nan)
    D  = K.rolling(d).mean()
    return K.fillna(50), D.fillna(50)


def atr(df: pd.DataFrame, p=14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"]  - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(com=p-1, min_periods=p).mean()


def adx(df: pd.DataFrame, p=14):
    hd = df["High"].diff()
    ld = df["Low"].diff()
    pdm = np.where((hd > ld) & (hd > 0), hd, 0.0)
    mdm = np.where((ld > hd) & (ld > 0), ld,  0.0)
    a   = atr(df, p)
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(com=p-1, min_periods=p).mean() / a
    mdi = 100 * pd.Series(mdm, index=df.index).ewm(com=p-1, min_periods=p).mean() / a
    dx  = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    ADX = dx.ewm(com=p-1, min_periods=p).mean()
    return ADX.fillna(0), pdi.fillna(0), mdi.fillna(0)


def obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["Close"].diff()).fillna(0)
    return (sign * df["Volume"]).cumsum()


def cmf(df: pd.DataFrame, p=20) -> pd.Series:
    """Chaikin Money Flow — tekanan beli/jual dengan volume."""
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) \
          / (df["High"] - df["Low"]).replace(0, np.nan)
    mfv = clv * df["Volume"]
    return mfv.rolling(p).sum() / df["Volume"].rolling(p).sum().replace(0, np.nan)


def vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP — Volume Weighted Average Price (level harga fair value)."""
    tp  = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_vol = df["Volume"].cumsum()
    cum_tpv = (tp * df["Volume"]).cumsum()
    return (cum_tpv / cum_vol.replace(0, np.nan)).ffill().bfill()


def ichimoku(df: pd.DataFrame):
    """
    Ichimoku Kinko Hyo — sistem trading lengkap:
    Tenkan (9), Kijun (26), Senkou A & B, Chikou
    """
    def midpoint(h, l, p): return (h.rolling(p).max() + l.rolling(p).min()) / 2
    tenkan  = midpoint(df["High"], df["Low"], 9)
    kijun   = midpoint(df["High"], df["Low"], 26)
    senk_a  = ((tenkan + kijun) / 2).shift(26)
    senk_b  = midpoint(df["High"], df["Low"], 52).shift(26)
    chikou  = df["Close"].shift(-26)
    return tenkan, kijun, senk_a, senk_b, chikou


def volume_profile(df: pd.DataFrame, bins=20) -> dict:
    """
    Volume Profile — identifikasi Point of Control (PoC) dan Value Area.
    PoC = level harga dengan volume tertinggi (magnet price).
    """
    if len(df) < 10:
        return {"poc": df["Close"].iloc[-1], "vah": df["High"].iloc[-1],
                "val": df["Low"].iloc[-1]}
    price_range = np.linspace(df["Low"].min(), df["High"].max(), bins + 1)
    vol_per_bin = []
    for i in range(bins):
        lo, hi = price_range[i], price_range[i+1]
        mask   = (df["Close"] >= lo) & (df["Close"] < hi)
        vol_per_bin.append(df.loc[mask, "Volume"].sum())
    vol_arr  = np.array(vol_per_bin)
    poc_idx  = vol_arr.argmax()
    poc      = (price_range[poc_idx] + price_range[poc_idx+1]) / 2
    # Value Area = 70% total volume
    total_vol = vol_arr.sum()
    va_vol    = 0
    sorted_idx = np.argsort(vol_arr)[::-1]
    va_bins   = []
    for idx in sorted_idx:
        va_vol += vol_arr[idx]
        va_bins.append(idx)
        if va_vol >= total_vol * 0.7:
            break
    vah = max((price_range[i] + price_range[i+1])/2 for i in va_bins)
    val = min((price_range[i] + price_range[i+1])/2 for i in va_bins)
    return {"poc": round(poc, 6), "vah": round(vah, 6), "val": round(val, 6)}


def realized_vol(s: pd.Series, p=20) -> pd.Series:
    lr = np.log(s / s.shift(1))
    return (lr.rolling(p).std() * np.sqrt(365) * 100).fillna(0)


# ═══════════════════════════════════════════════════════════════
#  SUPPORT / RESISTANCE — Pivot + Swing High/Low + Volume Profile
# ═══════════════════════════════════════════════════════════════
def detect_sr(df: pd.DataFrame) -> dict:
    cur = df["Close"].iloc[-1]
    recent = df.tail(100)

    # Swing pivot
    hi = recent["High"]
    lo = recent["Low"]
    swing_hi = hi[hi == hi.rolling(5, center=True).max()].tolist()
    swing_lo = lo[lo == lo.rolling(5, center=True).min()].tolist()

    res_levels = sorted([r for r in swing_hi if r > cur])
    sup_levels = sorted([s for s in swing_lo if s < cur], reverse=True)

    r1 = res_levels[0]  if res_levels else cur * 1.03
    s1 = sup_levels[0]  if sup_levels else cur * 0.97
    r2 = res_levels[1]  if len(res_levels) > 1 else r1 * 1.02
    s2 = sup_levels[1]  if len(sup_levels) > 1 else s1 * 0.98

    # Classic pivot
    pv = df.tail(1)
    pivot = (pv["High"].iloc[0] + pv["Low"].iloc[0] + pv["Close"].iloc[0]) / 3

    return {
        "resistance1": round(r1, 6),
        "resistance2": round(r2, 6),
        "support1":    round(s1, 6),
        "support2":    round(s2, 6),
        "pivot":       round(pivot, 6),
        "near_resistance": abs(cur - r1) / cur < 0.012,
        "near_support":    abs(cur - s1) / cur < 0.012,
        "above_pivot":     cur > pivot,
    }


# ═══════════════════════════════════════════════════════════════
#  CANDLESTICK PATTERNS — Profesional
# ═══════════════════════════════════════════════════════════════
def candle_patterns(df: pd.DataFrame) -> list[str]:
    if len(df) < 4:
        return []
    p = []
    c, o, h, l = df["Close"], df["Open"], df["High"], df["Low"]
    body    = (c - o).abs()
    rng     = (h - l).replace(0, np.nan)
    up      = c > o  # bullish candle

    lb = body.iloc[-1]
    lr = rng.iloc[-1] if not pd.isna(rng.iloc[-1]) else 0.001
    ll = (min(o.iloc[-1], c.iloc[-1]) - l.iloc[-1])
    lu = (h.iloc[-1] - max(o.iloc[-1], c.iloc[-1]))

    if lr > 0:
        # Hammer / Inverted Hammer
        if ll > 2 * lb and lu < lb * 0.5:
            p.append("Hammer (bullish reversal)")
        if lu > 2 * lb and ll < lb * 0.5 and not up.iloc[-1]:
            p.append("Shooting Star (bearish reversal)")
        # Doji types
        if lb / lr < 0.05:
            p.append("Doji (indecision)")
        elif lb / lr < 0.1:
            p.append("Spinning Top (indecision)")
        # Marubozu (strong candle, no wick)
        if ll < lr * 0.05 and lu < lr * 0.05:
            p.append("Bullish Marubozu (strong bull)" if up.iloc[-1]
                     else "Bearish Marubozu (strong bear)")

    # 2-candle patterns
    if len(df) >= 2:
        # Engulfing
        if (not up.iloc[-2] and up.iloc[-1]
                and o.iloc[-1] <= c.iloc[-2]
                and c.iloc[-1] >= o.iloc[-2]):
            p.append("Bullish Engulfing (strong reversal up)")
        if (up.iloc[-2] and not up.iloc[-1]
                and o.iloc[-1] >= c.iloc[-2]
                and c.iloc[-1] <= o.iloc[-2]):
            p.append("Bearish Engulfing (strong reversal down)")
        # Harami
        if (not up.iloc[-2] and up.iloc[-1]
                and o.iloc[-1] > c.iloc[-2]
                and c.iloc[-1] < o.iloc[-2]):
            p.append("Bullish Harami")
        # Tweezer bottom / top
        if abs(l.iloc[-1] - l.iloc[-2]) / max(l.iloc[-1], 0.001) < 0.002:
            p.append("Tweezer Bottom (support hold)")
        if abs(h.iloc[-1] - h.iloc[-2]) / max(h.iloc[-1], 0.001) < 0.002:
            p.append("Tweezer Top (resistance hold)")

    # 3-candle patterns
    if len(df) >= 3:
        # Morning Star
        if (not up.iloc[-3]
                and body.iloc[-2] < rng.iloc[-2] * 0.3
                and up.iloc[-1]
                and c.iloc[-1] > (o.iloc[-3] + c.iloc[-3]) / 2):
            p.append("Morning Star (strong bullish reversal)")
        # Evening Star
        if (up.iloc[-3]
                and body.iloc[-2] < rng.iloc[-2] * 0.3
                and not up.iloc[-1]
                and c.iloc[-1] < (o.iloc[-3] + c.iloc[-3]) / 2):
            p.append("Evening Star (strong bearish reversal)")
        # Three White Soldiers
        if all(up.iloc[-3:]) and all(body.iloc[-3:] > rng.iloc[-3:] * 0.6):
            p.append("Three White Soldiers (strong bull continuation)")
        # Three Black Crows
        if all(~up.iloc[-3:]) and all(body.iloc[-3:] > rng.iloc[-3:] * 0.6):
            p.append("Three Black Crows (strong bear continuation)")

    return p


# ═══════════════════════════════════════════════════════════════
#  RSI DIVERGENCE — Bullish/Bearish klasik & tersembunyi
# ═══════════════════════════════════════════════════════════════
def rsi_divergence(df: pd.DataFrame, rsi_s: pd.Series, lb=30) -> list[str]:
    if len(df) < lb + 3:
        return []
    dv  = []
    w   = df.tail(lb)
    wr  = rsi_s.tail(lb)

    # Bearish: higher high price, lower high RSI
    ph1_idx = w["Close"].idxmax()
    if ph1_idx != w.index[-1]:
        tail_close = w.loc[ph1_idx:]["Close"]
        tail_rsi   = wr.loc[ph1_idx:]
        if len(tail_close) > 1:
            ph2_idx = tail_close.idxmax()
            if (tail_close.loc[ph2_idx] >= w.loc[ph1_idx, "Close"]
                    and wr.get(ph2_idx, 100) < wr.get(ph1_idx, 0)):
                dv.append("🔴 Bearish RSI Divergence — potensi reversal turun")

    # Bullish: lower low price, higher low RSI
    pl1_idx = w["Close"].idxmin()
    if pl1_idx != w.index[-1]:
        tail_close = w.loc[pl1_idx:]["Close"]
        tail_rsi   = wr.loc[pl1_idx:]
        if len(tail_close) > 1:
            pl2_idx = tail_close.idxmin()
            if (tail_close.loc[pl2_idx] <= w.loc[pl1_idx, "Close"]
                    and wr.get(pl2_idx, 0) > wr.get(pl1_idx, 100)):
                dv.append("🟢 Bullish RSI Divergence — potensi reversal naik")
    return dv


# ═══════════════════════════════════════════════════════════════
#  CORE SCORING — Satu Timeframe, Skor -100..+100
# ═══════════════════════════════════════════════════════════════
def _score_one_tf(df: pd.DataFrame, cfg: dict, funding_rate: float = 0.0) -> dict:
    """
    Hitung semua indikator + skor arah untuk SATU timeframe.
    Metode scoring berbasis KONFIRMASI GANDA — sinyal harus dikonfirmasi
    oleh setidaknya 2 indikator berbeda kategori.
    """
    if len(df) < 35:
        return {"error": "data kurang", "score": 0}

    c   = df["Close"]
    cur = c.iloc[-1]
    v   = df["Volume"]

    # ── Hitung semua indikator ──
    rsi14     = rsi(c, 14)
    rsi7      = rsi(c, 7)    # fast RSI untuk konfirmasi
    ml, sg, mh = macd(c)
    bb_u, bb_m, bb_d, pb, bw = bollinger(c, 20, 2)
    stk, std  = stochastic(df)
    atr14     = atr(df, 14)
    ADX, PDI, MDI = adx(df, 14)
    OBV       = obv(df)
    CMF       = cmf(df)
    VWAP_val  = vwap(df)
    ten, kij, sa, sb, chi = ichimoku(df)
    vpro      = volume_profile(df)
    rvol      = realized_vol(c)
    vol_sma   = v.rolling(cfg["volume_sma_period"]).mean()

    # EMAs
    ema9  = c.ewm(span=9,   adjust=False).mean()
    ema21 = c.ewm(span=21,  adjust=False).mean()
    ema50 = c.ewm(span=50,  adjust=False).mean()
    ema200= c.ewm(span=200, adjust=False).mean()
    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()

    # Current values
    v_rsi    = rsi14.iloc[-1]
    v_rsi7   = rsi7.iloc[-1]
    v_ml     = ml.iloc[-1];  v_sg = sg.iloc[-1]; v_mh = mh.iloc[-1]
    v_mh_prev= mh.iloc[-2] if len(mh) > 1 else 0
    v_pb     = pb.iloc[-1]
    v_bw     = bw.iloc[-1]   # bandwidth — squeeze jika rendah
    v_stk    = stk.iloc[-1]; v_std = std.iloc[-1]
    v_atr    = atr14.iloc[-1]
    v_atr_pct= (v_atr / cur * 100) if cur > 0 else 2
    v_adx    = ADX.iloc[-1]; v_pdi = PDI.iloc[-1]; v_mdi = MDI.iloc[-1]
    v_obv    = OBV.iloc[-1]
    v_obv20  = OBV.iloc[-20] if len(OBV) > 20 else v_obv
    v_cmf    = CMF.iloc[-1] if not pd.isna(CMF.iloc[-1]) else 0
    v_vwap   = VWAP_val.iloc[-1] if not pd.isna(VWAP_val.iloc[-1]) else cur
    v_ema9   = ema9.iloc[-1]; v_ema21 = ema21.iloc[-1]
    v_ema50  = ema50.iloc[-1]; v_ema200= ema200.iloc[-1]
    v_sma20  = sma20.iloc[-1]; v_sma50 = sma50.iloc[-1]
    v_vol    = v.iloc[-1]
    v_vol_sma= vol_sma.iloc[-1] if not pd.isna(vol_sma.iloc[-1]) else v_vol
    v_rvol   = rvol.iloc[-1] if not pd.isna(rvol.iloc[-1]) else 50
    poc      = vpro["poc"]
    vah      = vpro["vah"]
    val      = vpro["val"]

    # Ichimoku current
    v_ten = ten.iloc[-1] if not pd.isna(ten.iloc[-1]) else cur
    v_kij = kij.iloc[-1] if not pd.isna(kij.iloc[-1]) else cur
    v_sa  = sa.iloc[-1]  if not pd.isna(sa.iloc[-1])  else None
    v_sb  = sb.iloc[-1]  if not pd.isna(sb.iloc[-1])  else None

    score   = 0
    signals = []
    vol_ratio = v_vol / v_vol_sma if v_vol_sma > 0 else 1

    # ════════════════════════════════════════════════════════
    # KATEGORI 1: TREND (bobot besar — 30 poin max)
    # ════════════════════════════════════════════════════════

    # EMA Stack — "Golden Order" = bull, "Death Order" = bear
    ema_stack_bull = cur > v_ema9 > v_ema21 > v_ema50
    ema_stack_bear = cur < v_ema9 < v_ema21 < v_ema50
    if ema_stack_bull:
        score += 18
        signals.append("✅ EMA Stack BULLISH (9>21>50, price above all)")
    elif ema_stack_bear:
        score -= 18
        signals.append("🔴 EMA Stack BEARISH (9<21<50, price below all)")
    elif cur > v_ema21 > v_ema50:
        score += 10
        signals.append("EMA partial bullish (above 21 & 50)")
    elif cur < v_ema21 < v_ema50:
        score -= 10
        signals.append("EMA partial bearish (below 21 & 50)")

    # EMA200 — trend jangka panjang
    if cur > v_ema200:
        score += 8
        signals.append("✅ Di atas EMA200 (bull trend jangka panjang)")
    else:
        score -= 8
        signals.append("🔴 Di bawah EMA200 (bear trend jangka panjang)")

    # ADX — kekuatan tren (bobot sangat besar)
    if v_adx > 30:
        if v_pdi > v_mdi:
            score += 15
            signals.append(f"✅ ADX={v_adx:.1f} TREN NAIK SANGAT KUAT (+DI={v_pdi:.1f})")
        else:
            score -= 15
            signals.append(f"🔴 ADX={v_adx:.1f} TREN TURUN SANGAT KUAT (-DI={v_mdi:.1f})")
    elif v_adx > 20:
        if v_pdi > v_mdi:
            score += 8
            signals.append(f"ADX={v_adx:.1f} tren naik moderat")
        else:
            score -= 8
            signals.append(f"ADX={v_adx:.1f} tren turun moderat")
    else:
        signals.append(f"ADX={v_adx:.1f} — market RANGING (hindari entry tren)")

    # ════════════════════════════════════════════════════════
    # KATEGORI 2: MOMENTUM (25 poin max)
    # ════════════════════════════════════════════════════════

    # RSI — dengan konfirmasi RSI7
    if v_rsi < 28:
        score += 16
        signals.append(f"✅ RSI={v_rsi:.1f} OVERSOLD EKSTREM — potensi bounce kuat")
    elif v_rsi < 38:
        score += 9
        signals.append(f"RSI={v_rsi:.1f} oversold — potensi rebound")
    elif v_rsi > 72:
        score -= 14
        signals.append(f"🔴 RSI={v_rsi:.1f} OVERBOUGHT EKSTREM — potensi koreksi")
    elif v_rsi > 62:
        score -= 7
        signals.append(f"RSI={v_rsi:.1f} overbought")
    else:
        signals.append(f"RSI={v_rsi:.1f} netral ({v_rsi:.0f})")

    # RSI7 sebagai konfirmasi arah
    if v_rsi7 < 30 and v_rsi < 40:
        score += 5
        signals.append("RSI7 konfirmasi oversold")
    elif v_rsi7 > 70 and v_rsi > 60:
        score -= 5
        signals.append("RSI7 konfirmasi overbought")

    # MACD histogram — momentum perubahan
    macd_accel = "⬆️" if v_mh > v_mh_prev else "⬇️"
    if v_mh > 0 and v_mh > v_mh_prev:
        score += 12
        signals.append(f"✅ MACD histogram {macd_accel} bullish MENGUAT")
    elif v_mh > 0:
        score += 5
        signals.append(f"MACD histogram bullish melemah {macd_accel}")
    elif v_mh < 0 and v_mh < v_mh_prev:
        score -= 12
        signals.append(f"🔴 MACD histogram {macd_accel} bearish MENGUAT")
    elif v_mh < 0:
        score -= 5
        signals.append(f"MACD histogram bearish melemah {macd_accel}")

    # Stochastic — crossover
    if v_stk < 20 and v_stk > v_std:
        score += 8
        signals.append(f"✅ Stoch={v_stk:.1f} oversold + cross UP — sinyal beli kuat")
    elif v_stk > 80 and v_stk < v_std:
        score -= 8
        signals.append(f"🔴 Stoch={v_stk:.1f} overbought + cross DOWN — sinyal jual kuat")

    # ════════════════════════════════════════════════════════
    # KATEGORI 3: VOLUME & FLOW (20 poin max)
    # ════════════════════════════════════════════════════════

    # Volume ratio — konfirmasi pergerakan harga
    if vol_ratio > 2.5:
        vol_bonus = 10 if score > 0 else -10
        score += vol_bonus
        signals.append(f"✅ Volume {vol_ratio:.1f}x — KONFIRMASI SANGAT KUAT")
    elif vol_ratio > 1.5:
        vol_bonus = 5 if score > 0 else -5
        score += vol_bonus
        signals.append(f"Volume {vol_ratio:.1f}x — konfirmasi arah")
    elif vol_ratio < 0.6:
        score = int(score * 0.7)   # kurangi 30% skor karena volume lemah
        signals.append(f"⚠️ Volume rendah {vol_ratio:.1f}x — sinyal tidak terkonfirmasi")

    # OBV trend
    obv_trend = v_obv > v_obv20
    if obv_trend and score > 0:
        score += 8
        signals.append("✅ OBV naik — akumulasi (smart money beli)")
    elif not obv_trend and score < 0:
        score -= 8
        signals.append("🔴 OBV turun — distribusi (smart money jual)")
    elif obv_trend and score < 0:
        signals.append("⚠️ OBV diverge — OBV naik tapi harga turun (potensi reversal)")
    elif not obv_trend and score > 0:
        signals.append("⚠️ OBV diverge — OBV turun tapi harga naik (warning)")

    # CMF — Chaikin Money Flow
    if v_cmf > 0.15:
        score += 7
        signals.append(f"✅ CMF={v_cmf:.3f} — tekanan beli institusional kuat")
    elif v_cmf < -0.15:
        score -= 7
        signals.append(f"🔴 CMF={v_cmf:.3f} — tekanan jual institusional kuat")
    elif v_cmf > 0.05:
        score += 3
        signals.append(f"CMF={v_cmf:.3f} mildly bullish")
    elif v_cmf < -0.05:
        score -= 3
        signals.append(f"CMF={v_cmf:.3f} mildly bearish")

    # ════════════════════════════════════════════════════════
    # KATEGORI 4: STRUKTUR HARGA (15 poin max)
    # ════════════════════════════════════════════════════════

    # VWAP — price vs fair value
    vwap_dist_pct = (cur - v_vwap) / v_vwap * 100
    if cur > v_vwap * 1.005:
        score += 6
        signals.append(f"✅ Harga di atas VWAP ${v_vwap:,.4f} (+{vwap_dist_pct:.2f}%)")
    elif cur < v_vwap * 0.995:
        score -= 6
        signals.append(f"🔴 Harga di bawah VWAP ${v_vwap:,.4f} ({vwap_dist_pct:.2f}%)")

    # Volume Profile PoC
    poc_dist = (cur - poc) / poc * 100
    if abs(poc_dist) < 0.5:
        signals.append(f"⚠️ Harga dekat PoC ${poc:,.4f} — area konsolidasi")
    elif cur > vah:
        score += 6
        signals.append(f"✅ Harga di atas Value Area High ${vah:,.4f} — bullish breakout")
    elif cur < val:
        score -= 6
        signals.append(f"🔴 Harga di bawah Value Area Low ${val:,.4f} — bearish breakdown")

    # Bollinger — mean reversion + squeeze
    if v_pb < 0.15:
        score += 8
        signals.append(f"✅ BB %B={v_pb:.2f} — dekat lower band (oversold area)")
    elif v_pb > 0.85:
        score -= 7
        signals.append(f"BB %B={v_pb:.2f} — dekat upper band (overbought area)")
    if v_bw < 5:   # squeeze = eksplosif selanjutnya
        signals.append(f"⚡ BB Squeeze (BW={v_bw:.1f}%) — volatilitas akan meledak segera")

    # Ichimoku
    if v_sa and v_sb:
        kumo_top    = max(v_sa, v_sb)
        kumo_bottom = min(v_sa, v_sb)
        if cur > kumo_top:
            score += 10
            signals.append(f"✅ Ichimoku: harga di atas Cloud (bullish territory)")
        elif cur < kumo_bottom:
            score -= 10
            signals.append(f"🔴 Ichimoku: harga di bawah Cloud (bearish territory)")
        else:
            signals.append(f"⚠️ Ichimoku: harga di dalam Cloud (konsolidasi, hindari entry)")

    if cur > v_ten and cur > v_kij:
        score += 5
        signals.append("Ichimoku: di atas Tenkan & Kijun (bull)")
    elif cur < v_ten and cur < v_kij:
        score -= 5
        signals.append("Ichimoku: di bawah Tenkan & Kijun (bear)")

    # ════════════════════════════════════════════════════════
    # KATEGORI 5: KONTEKS (10 poin max)
    # ════════════════════════════════════════════════════════

    # Support/Resistance
    sr = detect_sr(df)
    if sr["near_support"] and score >= 0:
        score += 5
        signals.append(f"✅ Dekat support ${sr['support1']:,.4f} — potensi bounce")
    if sr["near_resistance"] and score < 0:
        score -= 5
        signals.append(f"🔴 Dekat resistance ${sr['resistance1']:,.4f} — potensi rejection")

    # Momentum harga
    ret_1  = (cur / c.iloc[-2] - 1) * 100  if len(c) >= 2  else 0
    ret_5  = (cur / c.iloc[-6] - 1) * 100  if len(c) >= 6  else 0
    ret_20 = (cur / c.iloc[-21] - 1) * 100 if len(c) >= 21 else 0

    if ret_20 > 8:
        score += 5
        signals.append(f"Return 20-bar: {ret_20:+.1f}% — momentum bullish kuat")
    elif ret_20 < -8:
        score -= 5
        signals.append(f"Return 20-bar: {ret_20:+.1f}% — momentum bearish kuat")

    # Funding rate
    if funding_rate > 0.001:
        score -= 5
        signals.append(f"⚠️ Funding {funding_rate*100:+.4f}% — long overcrowded, risiko squeeze")
    elif funding_rate < -0.001:
        score += 5
        signals.append(f"✅ Funding {funding_rate*100:+.4f}% — short crowded, potensi short squeeze")

    # Candlestick patterns
    patterns = candle_patterns(df)
    for p in patterns:
        if any(x in p for x in ["bullish", "Bullish", "Morning", "Soldiers", "Hammer"]):
            score += 6
        elif any(x in p for x in ["bearish", "Bearish", "Evening", "Crows", "Shooting"]):
            score -= 6
    if patterns:
        signals.append(f"Candle: {', '.join(patterns)}")

    # RSI Divergence
    divs = rsi_divergence(df, rsi14)
    for d in divs:
        score += 10 if "Bullish" in d else -10
        signals.append(d)

    # Clamp
    score = max(-100, min(100, score))

    # Direction
    direction = ("STRONG_LONG" if score >= 60 else
                 "LONG"        if score >= 25 else
                 "STRONG_SHORT"if score <= -60 else
                 "SHORT"       if score <= -25 else "NEUTRAL")

    # ATR SL/TP
    sl_pct  = round(max(1.2, min(5.0, v_atr_pct * 1.5)), 2)
    tp1_pct = round(sl_pct * 2.5, 2)
    tp2_pct = round(sl_pct * 4.0, 2)

    return {
        "score":         round(score, 1),
        "direction":     direction,
        "direction_bias":"LONG" if score > 0 else ("SHORT" if score < 0 else "NEUTRAL"),
        "signals":       signals,
        "current_price": round(cur, 6),
        "rsi":           round(v_rsi, 2),
        "rsi7":          round(v_rsi7, 2),
        "macd_hist":     round(v_mh, 6),
        "bb_pb":         round(v_pb, 3),
        "bb_bw":         round(v_bw, 2),
        "stoch_k":       round(v_stk, 2),
        "stoch_d":       round(v_std, 2),
        "adx":           round(v_adx, 2),
        "plus_di":       round(v_pdi, 2),
        "minus_di":      round(v_mdi, 2),
        "atr":           round(v_atr, 6),
        "atr_pct":       round(v_atr_pct, 3),
        "ema9":          round(v_ema9, 6),
        "ema21":         round(v_ema21, 6),
        "ema50":         round(v_ema50, 6),
        "ema200":        round(v_ema200, 6),
        "vwap":          round(v_vwap, 6),
        "cmf":           round(v_cmf, 4),
        "obv_rising":    bool(obv_trend),
        "volume_ratio":  round(vol_ratio, 2),
        "realized_vol":  round(v_rvol, 1),
        "vp_poc":        poc,
        "vp_vah":        vah,
        "vp_val":        val,
        "ichi_tenkan":   round(v_ten, 6),
        "ichi_kijun":    round(v_kij, 6),
        "support_resistance": sr,
        "candlestick_patterns": patterns,
        "rsi_divergences": divs,
        "suggested_sl_pct":  sl_pct,
        "suggested_tp1_pct": tp1_pct,
        "suggested_tp2_pct": tp2_pct,
        "return_1":      round(ret_1, 2),
        "return_5":      round(ret_5, 2),
        "return_20":     round(ret_20, 2),
        "funding_signal": f"{funding_rate*100:+.4f}%",
    }


# ─── alias agar kode lama tetap jalan ─────────────────────────
def full_technical_analysis(df, timeframe_label="1h", funding_rate=0.0):
    r = _score_one_tf(df, TA_CONFIG, funding_rate)
    if "error" not in r:
        r["timeframe"] = timeframe_label
    return r


# ═══════════════════════════════════════════════════════════════
#  MULTI-TIMEFRAME — gabung 1h + 4h + 1d
# ═══════════════════════════════════════════════════════════════
def multi_timeframe_analysis(dataframes: dict, funding_rate: float = 0.0) -> dict:
    """
    Gabungkan analisis 3 timeframe dengan bobot berbeda:
      1h  → 50%  (keputusan entry)
      4h  → 30%  (konfirmasi tren)
      1d  → 20%  (konteks makro)

    Aturan konflik: jika 1h dan 4h berlawanan → skor dikurangi 55%
    """
    weights = {"1h": 0.50, "4h": 0.30, "1d": 0.20}
    results = {}
    combined = 0.0
    w_total  = 0.0

    for tf, df in dataframes.items():
        if df is None or len(df) < 35:
            continue
        r = _score_one_tf(df, TA_CONFIG, funding_rate)
        if "error" in r:
            continue
        r["timeframe"] = tf
        results[tf] = r
        w = weights.get(tf, 0.25)
        combined += r["score"] * w
        w_total  += w

    if not results:
        return {"error": "Tidak ada TF valid", "score": 0, "direction": "NEUTRAL",
                "tf_conflict": False, "market_regime": "UNKNOWN",
                "suggested_sl_pct": 2.0, "suggested_tp1_pct": 4.0,
                "suggested_tp2_pct": 6.0, "support_resistance": {},
                "candlestick_patterns": [], "signals_primary": []}

    if w_total > 0:
        combined /= w_total
    combined = max(-100, min(100, combined))

    # ── Cek konflik 1h vs 4h ──
    bias_1h = results.get("1h", {}).get("direction_bias", "NEUTRAL")
    bias_4h = results.get("4h", {}).get("direction_bias", "NEUTRAL")
    tf_conflict = (bias_1h in ("LONG","SHORT")
                   and bias_4h in ("LONG","SHORT")
                   and bias_1h != bias_4h)
    if tf_conflict:
        combined *= 0.45   # hukum berat jika konflik

    # ── Agreement bonus ──
    biases = [r["direction_bias"] for r in results.values()]
    if biases.count("LONG")  == len(biases):
        combined = min(combined * 1.15, 100)
    elif biases.count("SHORT") == len(biases):
        combined = max(combined * 1.15, -100)

    # ── Direction & regime ──
    direction = ("STRONG_LONG"  if combined >= 60 else
                 "LONG"         if combined >= 25 else
                 "STRONG_SHORT" if combined <= -60 else
                 "SHORT"        if combined <= -25 else "NEUTRAL")

    regime = ("TRENDING_UP"   if combined > 30 else
              "TRENDING_DOWN" if combined < -30 else "RANGING")

    confidence = min(100, abs(combined) * 0.85 + 15)
    if tf_conflict:
        confidence *= 0.6

    # ── Primary TF untuk SL/TP/signals ──
    primary = results.get("1h") or results.get("4h") or next(iter(results.values()))

    return {
        "score":             round(combined, 1),
        "direction":         direction,
        "confidence":        round(confidence, 1),
        "market_regime":     regime,
        "tf_conflict":       tf_conflict,
        "timeframe_bias":    {tf: r["direction_bias"] for tf, r in results.items()},
        "per_timeframe":     results,

        # Dari primary TF
        "current_price":     primary.get("current_price", 0),
        "primary_rsi":       primary.get("rsi", 50),
        "primary_adx":       primary.get("adx", 0),
        "atr_pct":           primary.get("atr_pct", 2.0),
        "volume_ratio":      primary.get("volume_ratio", 1),
        "realized_vol_annual_pct": primary.get("realized_vol", 0),
        "signals_primary":   primary.get("signals", []),
        "candlestick_patterns": primary.get("candlestick_patterns", []),
        "support_resistance":primary.get("support_resistance", {}),
        "return_1d":         primary.get("return_1", 0),
        "return_5d":         primary.get("return_5", 0),
        "return_20d":        primary.get("return_20", 0),
        "funding_signal":    primary.get("funding_signal", "0%"),

        # SL/TP dari ATR primary
        "suggested_sl_pct":  primary.get("suggested_sl_pct", 2.0),
        "suggested_tp1_pct": primary.get("suggested_tp1_pct", 4.0),
        "suggested_tp2_pct": primary.get("suggested_tp2_pct", 6.0),

        # Indikator tambahan untuk AI prompt
        "vwap":              primary.get("vwap", 0),
        "cmf":               primary.get("cmf", 0),
        "obv_rising":        primary.get("obv_rising", False),
        "vp_poc":            primary.get("vp_poc", 0),
        "vp_vah":            primary.get("vp_vah", 0),
        "vp_val":            primary.get("vp_val", 0),
        "bb_pb":             primary.get("bb_pb", 0.5),
        "bb_bw":             primary.get("bb_bw", 0),
        "ema9":              primary.get("ema9", 0),
        "ema21":             primary.get("ema21", 0),
        "ema50":             primary.get("ema50", 0),
        "ema200":            primary.get("ema200", 0),
        "ichi_tenkan":       primary.get("ichi_tenkan", 0),
        "ichi_kijun":        primary.get("ichi_kijun", 0),
    }
