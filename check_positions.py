from dotenv import load_dotenv; load_dotenv()
from data_fetcher import fetch_ticker_binance

sol_now   = fetch_ticker_binance('SOL/USDT').get('price', 0)
eth_now   = fetch_ticker_binance('ETH/USDT').get('price', 0)
sol_entry = 72.99
eth_entry = 1860.16

# SHORT: profit jika harga TURUN, rugi jika harga NAIK
sol_pnl_pct = (sol_entry - sol_now) / sol_entry * 100
eth_pnl_pct = (eth_entry - eth_now) / eth_entry * 100
sol_pnl_usd = (sol_entry - sol_now) * 2.877106
eth_pnl_usd = (eth_entry - eth_now) * 0.104981
total       = sol_pnl_usd + eth_pnl_usd

print("=" * 55)
print("  CEK POSISI PAPER TRADING REALTIME")
print("=" * 55)
print(f"  SOL SHORT @ ${sol_entry}")
print(f"    Harga sekarang : ${sol_now:.4f}")
print(f"    P&L            : {sol_pnl_pct:+.2f}%  (${sol_pnl_usd:+.2f})")
print(f"    Stop Loss      : $73.8659  → {'⚠️ HAMPIR HIT' if sol_now > 73.5 else '✅ Aman'}")
print(f"    Take Profit    : $70.1872  → {'🎯 HAMPIR HIT' if sol_now < 71.0 else '⏳ Menunggu'}")
print()
print(f"  ETH SHORT @ ${eth_entry}")
print(f"    Harga sekarang : ${eth_now:.4f}")
print(f"    P&L            : {eth_pnl_pct:+.2f}%  (${eth_pnl_usd:+.2f})")
print(f"    Stop Loss      : $1882.48  → {'⚠️ HAMPIR HIT' if eth_now > 1870 else '✅ Aman'}")
print(f"    Take Profit    : $1815.52  → {'🎯 HAMPIR HIT' if eth_now < 1830 else '⏳ Menunggu'}")
print()
print(f"  TOTAL P&L        : ${total:+.2f} USDT")
print(f"  Status           : {'PROFIT' if total > 0 else 'LOSS (unrealized, posisi masih terbuka)'}")
print("=" * 55)
