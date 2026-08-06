"""
Direct test ke Binance Futures Testnet pakai requests raw
untuk verify API key + secret valid.
"""
import hmac, hashlib, time, requests
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY    = os.getenv("EXCHANGE_API_KEY", "")
API_SECRET = os.getenv("EXCHANGE_API_SECRET", "")

BASE = "https://testnet.binancefuture.com"

def sign(params: dict, secret: str) -> str:
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()

print(f"API Key    : {API_KEY[:16]}...")
print(f"API Secret : {'set' if API_SECRET else 'KOSONG'}")
print(f"Base URL   : {BASE}")
print()

# Test 1: public endpoint (tidak perlu auth)
print("Test 1 — Public ping...")
r = requests.get(f"{BASE}/fapi/v1/ping", timeout=10)
print(f"  Status: {r.status_code} → {'OK' if r.status_code == 200 else r.text}")

# Test 2: public server time
print("Test 2 — Server time...")
r = requests.get(f"{BASE}/fapi/v1/time", timeout=10)
if r.status_code == 200:
    st = r.json().get("serverTime", 0)
    lt = int(time.time() * 1000)
    print(f"  Server time : {st}")
    print(f"  Local time  : {lt}")
    print(f"  Diff (ms)   : {st - lt}")
else:
    print(f"  Error: {r.text}")

# Test 3: private account endpoint (butuh auth)
print("Test 3 — Account balance (auth)...")
ts = int(time.time() * 1000)
params = {"timestamp": ts, "recvWindow": 10000}
params["signature"] = sign(params, API_SECRET)
headers = {"X-MBX-APIKEY": API_KEY}
r = requests.get(f"{BASE}/fapi/v2/balance", params=params, headers=headers, timeout=10)
print(f"  Status: {r.status_code}")
if r.status_code == 200:
    data = r.json()
    for asset in data:
        if asset.get("asset") == "USDT":
            print(f"  USDT Balance  : {asset.get('balance')}")
            print(f"  Available     : {asset.get('availableBalance')}")
            print(f"\n  [OK] Testnet terhubung dan API key valid!")
else:
    print(f"  Response: {r.text}")
    if "-2008" in r.text:
        print("\n  [ERROR] API key tidak dikenali testnet.")
        print("  Pastikan key dibuat di https://testnet.binancefuture.com")
        print("  bukan dari binance.com biasa.")
    elif "-1021" in r.text:
        print("\n  [ERROR] Timestamp error — sinkronkan jam PC.")
