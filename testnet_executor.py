"""
testnet_executor.py — Eksekutor order ke Binance Futures Demo Trading

Binance Futures Testnet: https://testnet.binancefuture.com
Daftar API key: login dengan GitHub di URL di atas → API Management

CATATAN: ccxt set_sandbox_mode() sudah deprecated.
Kita pakai requests langsung ke testnet endpoint untuk reliability.
"""
import asyncio, hmac, hashlib, time
import requests as _requests
from typing import Optional
from rich.console import Console
from config import (
    EXCHANGE_API_KEY, EXCHANGE_API_SECRET,
    USE_TESTNET, TRADING_MODE, TAKER_FEE_RATE,
)

console = Console()

TESTNET_BASE = "https://testnet.binancefuture.com"
LIVE_BASE    = "https://fapi.binance.com"

def _base() -> str:
    return TESTNET_BASE if USE_TESTNET else LIVE_BASE

def _sign(params: dict) -> str:
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return hmac.new(
        EXCHANGE_API_SECRET.encode(),
        query.encode(),
        hashlib.sha256
    ).hexdigest()

def _headers() -> dict:
    return {"X-MBX-APIKEY": EXCHANGE_API_KEY}

def _ts() -> int:
    return int(time.time() * 1000)

def _get(path: str, params: dict = None) -> dict:
    """Authenticated GET."""
    p = dict(params or {})
    p["timestamp"]  = _ts()
    p["recvWindow"] = 10000
    p["signature"]  = _sign(p)
    r = _requests.get(f"{_base()}{path}", params=p,
                      headers=_headers(), timeout=10)
    r.raise_for_status()
    return r.json()

def _post(path: str, params: dict = None) -> dict:
    """Authenticated POST."""
    p = dict(params or {})
    p["timestamp"]  = _ts()
    p["recvWindow"] = 10000
    p["signature"]  = _sign(p)
    r = _requests.post(f"{_base()}{path}", params=p,
                       headers=_headers(), timeout=10)
    try:
        r.raise_for_status()
    except Exception as e:
        # Print response body untuk debug
        console.print(f"[red]API Error: {e}[/red]")
        console.print(f"[dim]Response: {r.text[:500]}[/dim]")
        raise
    return r.json()

def _delete(path: str, params: dict = None) -> dict:
    """Authenticated DELETE."""
    p = dict(params or {})
    p["timestamp"]  = _ts()
    p["recvWindow"] = 10000
    p["signature"]  = _sign(p)
    r = _requests.delete(f"{_base()}{path}", params=p,
                         headers=_headers(), timeout=10)
    r.raise_for_status()
    return r.json()


def is_live_mode() -> bool:
    """True jika bot eksekusi order nyata."""
    return (
        TRADING_MODE == "live"
        and bool(EXCHANGE_API_KEY)
        and bool(EXCHANGE_API_SECRET)
    )


# ═══════════════════════════════════════════════════════════════
#  ACCOUNT
# ═══════════════════════════════════════════════════════════════
async def get_account_balance() -> dict:
    """Ambil saldo USDT."""
    if not is_live_mode():
        return {"USDT": {"total": 0, "free": 0}, "paper": True}
    try:
        data = await asyncio.to_thread(_get, "/fapi/v2/balance")
        for asset in data:
            if asset.get("asset") == "USDT":
                return {
                    "USDT": {
                        "total": float(asset.get("balance", 0)),
                        "free":  float(asset.get("availableBalance", 0)),
                    },
                    "paper": False,
                }
        return {"USDT": {"total": 0, "free": 0}, "paper": False}
    except Exception as e:
        console.print(f"[red]Fetch balance error: {e}[/red]")
        return {"USDT": {"total": 0, "free": 0}, "error": str(e)}


async def get_open_positions() -> list[dict]:
    """Ambil semua posisi terbuka (size != 0)."""
    if not is_live_mode():
        return []
    try:
        data = await asyncio.to_thread(_get, "/fapi/v2/positionRisk")
        return [p for p in data if float(p.get("positionAmt", 0)) != 0]
    except Exception as e:
        console.print(f"[yellow]Fetch positions gagal: {e}[/yellow]")
        return []


# ═══════════════════════════════════════════════════════════════
#  ORDER EXECUTION
# ═══════════════════════════════════════════════════════════════
async def set_leverage(symbol: str, leverage: int) -> bool:
    """Set leverage untuk simbol."""
    if not is_live_mode():
        return True
    try:
        clean = symbol.replace("/", "")
        await asyncio.to_thread(_post, "/fapi/v1/leverage", {
            "symbol":   clean,
            "leverage": int(leverage),
        })
        console.print(f"  [cyan]Leverage {symbol} → {leverage}x ✓[/cyan]")
        return True
    except Exception as e:
        console.print(f"  [yellow]Set leverage gagal ({symbol}): {e}[/yellow]")
        return False


async def place_market_order(
    symbol: str,
    side: str,           # "LONG" atau "SHORT"
    notional: float,     # nilai dalam USDT
    current_price: float,
) -> Optional[dict]:
    """
    Place MARKET order.
    Return {order_id, fill_price, qty} atau None jika gagal.
    """
    if not is_live_mode():
        qty = round(notional / max(current_price, 1), 6)
        return {
            "order_id": "PAPER", "fill_price": current_price,
            "qty": qty, "side": side, "paper": True,
        }
    try:
        clean     = symbol.replace("/", "")
        direction = "BUY" if side == "LONG" else "SELL"
        qty       = round(notional / max(current_price, 1), 3)

        console.print(
            f"  [cyan]▶ MARKET {direction} {symbol} "
            f"qty={qty:.6f} (~${notional:,.2f})[/cyan]"
        )
        data = await asyncio.to_thread(_post, "/fapi/v1/order", {
            "symbol":     clean,
            "side":       direction,
            "type":       "MARKET",
            "quantity":   qty,
        })
        # Binance testnet kadang return avgPrice=0 untuk market order
        # Fallback ke price field atau current_price
        fill = float(data.get("avgPrice") or data.get("price") or 0)
        if fill <= 0:
            fill = current_price   # pakai harga saat order sebagai fallback
        filled_qty = float(data.get("executedQty") or data.get("origQty") or qty)
        if filled_qty <= 0:
            filled_qty = qty
        oid = data.get("orderId")
        console.print(
            f"  [bold green]✅ TERISI: {direction} {symbol} "
            f"@ ${fill:,.4f} qty={filled_qty:.6f} id={oid}[/bold green]"
        )
        return {
            "order_id": oid, "fill_price": fill,
            "qty": filled_qty, "side": side, "paper": False,
        }
    except Exception as e:
        console.print(f"  [red]Market order gagal ({symbol}): {e}[/red]")
        return None


async def place_stop_loss(
    symbol: str, side: str, sl_price: float, qty: float
) -> Optional[str]:
    """Pasang STOP_MARKET SL order menggunakan stopLoss conditional order."""
    if not is_live_mode():
        return "PAPER_SL"
    try:
        clean      = symbol.replace("/", "")
        close_side = "SELL" if side == "LONG" else "BUY"
        #  Binance Futures conditional order: gunakan STOP_MARKET dengan workingType=MARK_PRICE
        data = await asyncio.to_thread(_post, "/fapi/v1/order", {
            "symbol":       clean,
            "side":         close_side,
            "type":         "STOP_MARKET",
            "stopPrice":    round(sl_price, 2),
            "quantity":     qty,
            "reduceOnly":   "true",
            "workingType":  "CONTRACT_PRICE",  # trigger berdasarkan last price
        })
        oid = data.get("orderId")
        console.print(f"  [cyan]SL @ ${sl_price:,.4f} ✓ id={oid}[/cyan]")
        return str(oid)
    except Exception as e:
        # Jika masih gagal, ignore — market order tetap bisa manual close
        console.print(f"  [dim yellow]SL order skip (akan manual monitor): {str(e)[:80]}[/dim yellow]")
        return None


async def place_take_profit(
    symbol: str, side: str, tp_price: float, qty: float
) -> Optional[str]:
    """Pasang TAKE_PROFIT_MARKET TP order."""
    if not is_live_mode():
        return "PAPER_TP"
    try:
        clean      = symbol.replace("/", "")
        close_side = "SELL" if side == "LONG" else "BUY"
        # Bulatkan qty ke 3 desimal untuk BTC (sesuai Binance futures precision)
        qty_rounded = round(qty, 3)
        data = await asyncio.to_thread(_post, "/fapi/v1/order", {
            "symbol":         clean,
            "side":           close_side,
            "type":           "TAKE_PROFIT_MARKET",
            "stopPrice":      round(tp_price, 2),
            "quantity":       qty_rounded,
            "reduceOnly":     "true",
            "workingType":    "CONTRACT_PRICE",
        })
        oid = data.get("orderId")
        console.print(f"  [cyan]TP @ ${tp_price:,.4f} ✓ id={oid}[/cyan]")
        return str(oid)
    except Exception as e:
        console.print(f"  [dim yellow]TP order skip (akan manual monitor): {str(e)[:80]}[/dim yellow]")
        return None


async def cancel_all_orders(symbol: str) -> bool:
    """Cancel semua open order sebelum close."""
    if not is_live_mode():
        return True
    try:
        clean = symbol.replace("/", "")
        await asyncio.to_thread(_delete, "/fapi/v1/allOpenOrders", {
            "symbol": clean,
        })
        console.print(f"  [dim]Open orders {symbol} dibatalkan ✓[/dim]")
        return True
    except Exception as e:
        console.print(f"  [yellow]Cancel orders gagal: {e}[/yellow]")
        return False


async def close_position_market(
    symbol: str, side: str, qty: float, reason: str = ""
) -> Optional[dict]:
    """Tutup posisi via MARKET order."""
    if not is_live_mode():
        return {"fill_price": 0, "qty": qty, "paper": True}
    try:
        await cancel_all_orders(symbol)
        clean      = symbol.replace("/", "")
        close_side = "SELL" if side == "LONG" else "BUY"
        console.print(
            f"  [yellow]▶ CLOSE {close_side} {symbol} "
            f"qty={qty:.6f} [{reason[:60]}][/yellow]"
        )
        data = await asyncio.to_thread(_post, "/fapi/v1/order", {
            "symbol":     clean,
            "side":       close_side,
            "type":       "MARKET",
            "quantity":   qty,
            "reduceOnly": "true",
        })
        fill       = float(data.get("avgPrice") or data.get("price") or 0)
        filled_qty = float(data.get("executedQty") or data.get("origQty") or qty)
        if filled_qty <= 0:
            filled_qty = qty
        console.print(f"  [green]✅ CLOSE terisi @ ${fill:,.4f}[/green]")
        return {"fill_price": fill, "qty": filled_qty, "paper": False}
    except Exception as e:
        console.print(f"  [red]Close order gagal: {e}[/red]")
        return None


# ── Connection test ───────────────────────────────────────────
if __name__ == "__main__":
    async def test():
        mode = "DEMO TRADING" if USE_TESTNET else "LIVE"
        console.print(f"[bold cyan]═══ Binance Futures {mode} Test ═══[/bold cyan]")
        console.print(f"URL        : {_base()}")
        console.print(f"Live exec  : {is_live_mode()}")
        console.print(
            f"API Key    : {EXCHANGE_API_KEY[:16]}..."
            if EXCHANGE_API_KEY else "API Key    : (kosong)"
        )
        console.print(f"API Secret : {'***set***' if EXCHANGE_API_SECRET else '(kosong)'}")

        if not is_live_mode():
            console.print(
                "\n[yellow]Paper mode aktif.[/yellow]\n"
                "Set TRADING_MODE=live + isi EXCHANGE_API_KEY/SECRET di .env\n"
                "untuk koneksi ke testnet."
            )
            return

        console.print("\n[cyan]Menghubungkan...[/cyan]")
        bal = await get_account_balance()
        usdt = bal.get("USDT", {})

        if bal.get("error"):
            console.print(f"\n[red]✗ Gagal: {bal['error']}[/red]")
            return

        console.print(f"\n[bold green]✅ {mode} terhubung![/bold green]")
        console.print(f"  USDT Balance : ${usdt.get('total', 0):,.2f}")
        console.print(f"  Available    : ${usdt.get('free', 0):,.2f}")

        positions = await get_open_positions()
        console.print(f"  Open positions: {len(positions)}")
        for p in positions:
            amt = float(p.get("positionAmt", 0))
            side_str = "LONG" if amt > 0 else "SHORT"
            console.print(
                f"    {p.get('symbol')} {side_str} "
                f"qty={abs(amt)} "
                f"entry=${float(p.get('entryPrice', 0)):,.4f} "
                f"pnl=${float(p.get('unRealizedProfit', 0)):+,.2f}"
            )

        console.print("\n[bold green]Siap trading![/bold green]")

    asyncio.run(test())
