"""
get_market_data.py - Ambil data market realtime untuk analisis manual
"""
from dotenv import load_dotenv; load_dotenv()
from data_fetcher import (
    fetch_ticker_binance, fetch_order_book, fetch_recent_trades,
    fetch_funding_rate, fetch_open_interest, fetch_ohlcv
)
from technical_analyzer import multi_timeframe_analysis
from news_fetcher import fetch_fear_greed, fetch_coingecko_fundamental, fetch_macro_global
import json

pairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']

print("=" * 70)
print("MARKET DATA REALTIME — BINANCE FUTURES")
print("=" * 70)

# Fear & Greed
fg = fetch_fear_greed()
print(f"\nFEAR & GREED INDEX: {fg['value']}/100 ({fg['label']})")
print(f"Signal: {fg.get('signal','')}")
print(f"Trend:  {fg.get('trend','')}")

# Macro
macro = fetch_macro_global()
dxy = macro.get('DXY') or {}
sp  = macro.get('SP500') or {}
vix = macro.get('VIX') or {}
print(f"\nMACRO GLOBAL:")
print(f"  DXY  : {dxy.get('price','N/A')} ({dxy.get('change_pct',0):+.2f}%)")
print(f"  SP500: {sp.get('price','N/A')} ({sp.get('change_pct',0):+.2f}%)")
print(f"  VIX  : {vix.get('price','N/A')}")
print(f"  Note : {macro.get('interpretation','')}")

print("\n" + "=" * 70)
print("PER-PAIR ANALYSIS")
print("=" * 70)

all_data = {}
for pair in pairs:
    print(f"\n--- {pair} ---")
    try:
        t  = fetch_ticker_binance(pair)
        ob = fetch_order_book(pair, depth=20)
        rt = fetch_recent_trades(pair)
        fr = fetch_funding_rate(pair)
        oi = fetch_open_interest(pair)

        price   = t.get('price', 0)
        chg24   = t.get('change_24h_pct', 0)
        vol24   = t.get('volume_24h', 0)
        imb     = ob.get('depth_imbalance', 0)
        buy_pct = rt.get('buy_ratio_pct', 50)

        print(f"  Price      : ${price:,.4f}")
        print(f"  24h Change : {chg24:+.2f}%")
        print(f"  24h Volume : ${vol24/1e6:.0f}M USDT")
        print(f"  Funding    : {fr*100:+.5f}%")
        print(f"  Open Int   : {oi:,.0f}")
        print(f"  OB Imbal   : {imb:+.1f}% ({'BUY pressure' if imb>5 else 'SELL pressure' if imb<-5 else 'Neutral'})")
        print(f"  Tape Buy%  : {buy_pct:.0f}% buy / {100-buy_pct:.0f}% sell")

        # Technical multi-TF
        df1h = fetch_ohlcv(pair, '1h', 100)
        df4h = fetch_ohlcv(pair, '4h', 80)
        df1d = fetch_ohlcv(pair, '1d', 50)
        tech = multi_timeframe_analysis({'1h': df1h, '4h': df4h, '1d': df1d}, fr)

        score   = tech.get('score', 0)
        regime  = tech.get('market_regime', '?')
        conf    = tech.get('confidence', 0)
        rsi     = tech.get('primary_rsi', 50)
        adx     = tech.get('primary_adx', 0)
        conflict= tech.get('tf_conflict', False)
        sl_pct  = tech.get('suggested_sl_pct', 0)
        tp1_pct = tech.get('suggested_tp1_pct', 0)
        sr      = tech.get('support_resistance', {})
        cur     = tech.get('current_price', price)

        print(f"  Tech Score : {score:+.1f}/100 ({regime})")
        print(f"  RSI/ADX    : RSI={rsi:.1f} | ADX={adx:.1f}")
        print(f"  Confidence : {conf:.0f}%")
        print(f"  TF Conflict: {conflict}")
        print(f"  Support    : ${sr.get('support',0):,.4f}")
        print(f"  Resistance : ${sr.get('resistance',0):,.4f}")
        print(f"  Suggest SL : -{sl_pct:.2f}% (${cur*(1-sl_pct/100):,.4f})")
        print(f"  Suggest TP1: +{tp1_pct:.2f}% (${cur*(1+tp1_pct/100):,.4f})")

        # Candles
        patterns = tech.get('candlestick_patterns', [])
        if patterns:
            print(f"  Candles    : {', '.join(patterns)}")

        # TF breakdown
        tf_data = tech.get('per_timeframe', {})
        for tf, r in tf_data.items():
            if isinstance(r, dict):
                print(f"  [{tf:3}]      : score={r.get('score',0):+.0f} | {r.get('direction_bias','?')} | RSI={r.get('rsi',50):.1f}")

        # Simpan untuk prompt
        all_data[pair] = {
            'price': price, 'change_24h': chg24, 'volume_24h_m': round(vol24/1e6, 1),
            'funding_pct': round(fr*100, 5), 'open_interest': round(oi, 0),
            'ob_imbalance': round(imb, 1), 'buy_ratio_pct': buy_pct,
            'tech_score': score, 'regime': regime, 'confidence': round(conf, 1),
            'rsi': round(rsi, 1), 'adx': round(adx, 1), 'tf_conflict': conflict,
            'support': round(sr.get('support',0), 4),
            'resistance': round(sr.get('resistance',0), 4),
            'suggested_sl_pct': sl_pct, 'suggested_tp1_pct': tp1_pct,
            'candle_patterns': patterns,
            'tf_breakdown': {tf: {'score': r.get('score',0), 'bias': r.get('direction_bias','?'), 'rsi': round(r.get('rsi',50),1)}
                             for tf, r in tf_data.items() if isinstance(r, dict)},
        }
    except Exception as e:
        print(f"  ERROR: {e}")

# Simpan ke file untuk dibaca
with open('market_snapshot.json', 'w') as f:
    json.dump({
        'fear_greed': fg,
        'macro': {'dxy': dxy, 'sp500': sp, 'vix': vix, 'note': macro.get('interpretation','')},
        'pairs': all_data,
    }, f, indent=2)

print("\n" + "=" * 70)
print("Data disimpan ke market_snapshot.json")
print("=" * 70)
