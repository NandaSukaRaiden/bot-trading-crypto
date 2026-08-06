"""Quick script untuk close posisi BTC di testnet"""
import asyncio
from testnet_executor import close_position_market, get_open_positions

async def main():
    positions = await get_open_positions()
    if not positions:
        print("Tidak ada posisi terbuka")
        return
    
    for p in positions:
        symbol = p.get("symbol", "")
        qty = abs(float(p.get("positionAmt", 0)))
        side = "LONG" if float(p.get("positionAmt", 0)) > 0 else "SHORT"
        
        print(f"\nClosing {side} {symbol} qty={qty}")
        result = await close_position_market(
            symbol=symbol.replace("USDT", "/USDT"),
            side=side,
            qty=qty,
            reason="Manual close untuk re-test"
        )
        if result:
            print(f"✅ Position closed @ ${result['fill_price']}")
        else:
            print("❌ Failed to close")

if __name__ == "__main__":
    asyncio.run(main())
