"""
charts.py — Candlestick chart crypto semua timeframe dari Binance
Timeframes: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M
Data source: Binance public REST API (tanpa auth)
Output: PNG per TF + HTML gallery browser di port 8008
"""
import os, threading, webbrowser
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
import requests
import mplfinance as mpf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from cachetools import TTLCache
from rich.console import Console

from config import CHART_DIR, CHART_SERVER_PORT, BINANCE_KLINES_URL

console = Console()

_chart_cache = TTLCache(maxsize=500, ttl=300)

# ── Binance interval → limit bars ─────────────────────────────
TF_CONFIG = {
    "1m":  {"label": "1 Minute",  "limit": 300},
    "5m":  {"label": "5 Minutes", "limit": 300},
    "15m": {"label": "15 Minutes","limit": 300},
    "30m": {"label": "30 Minutes","limit": 300},
    "1h":  {"label": "1 Hour",    "limit": 300},
    "4h":  {"label": "4 Hours",   "limit": 300},
    "1d":  {"label": "Daily",     "limit": 365},
    "1w":  {"label": "Weekly",    "limit": 104},
    "1M":  {"label": "Monthly",   "limit":  48},
}

CHART_STYLE = mpf.make_mpf_style(
    base_mpf_style="charles",
    marketcolors=mpf.make_marketcolors(
        up="#00e676", down="#ff1744",
        edge="inherit", wick="inherit",
        volume={"up": "#00e676", "down": "#ff1744"},
    ),
    gridstyle=":", gridcolor="#1e1e2e",
    facecolor="#0d0d1a", figcolor="#0d0d1a", edgecolor="#2a2a4a",
    rc={
        "axes.labelcolor": "#aaa", "axes.titlecolor": "#fff",
        "xtick.color": "#888",     "ytick.color": "#888",
        "text.color": "#ccc",      "font.size": 8,
    },
)

HEADERS = {"User-Agent": "Mozilla/5.0 TradingBot/2.0"}


# ═══════════════════════════════════════════════════════════════
#  FETCH OHLCV dari Binance REST public
# ═══════════════════════════════════════════════════════════════
def _fetch_binance_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame | None:
    """Ambil klines dari Binance Futures public endpoint — tanpa auth."""
    key = f"ck_{symbol}_{interval}_{limit}"
    if key in _chart_cache:
        return _chart_cache[key]

    clean = symbol.replace("/", "")   # BTC/USDT → BTCUSDT
    try:
        r = requests.get(
            BINANCE_KLINES_URL,
            params={"symbol": clean, "interval": interval, "limit": limit},
            headers=HEADERS, timeout=10,
        )
        if r.status_code != 200:
            return None
        raw = r.json()
        if not raw:
            return None
        df = pd.DataFrame(raw, columns=[
            "ts","Open","High","Low","Close","Volume",
            "close_time","qv","trades","tbbv","tbqv","ignore"
        ])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df.set_index("ts", inplace=True)
        for col in ["Open","High","Low","Close","Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["Open","High","Low","Close","Volume"]].dropna()
        if len(df) < 5:
            return None
        _chart_cache[key] = df
        return df
    except Exception as e:
        console.print(f"[dim]Binance klines {symbol}[{interval}]: {e}[/dim]")
        return None


# ═══════════════════════════════════════════════════════════════
#  INDIKATOR OVERLAY
# ═══════════════════════════════════════════════════════════════
def _indicators(df: pd.DataFrame) -> dict:
    c = df["Close"]
    n = len(df)

    ema9  = c.ewm(span=9,   adjust=False).mean()
    ema21 = c.ewm(span=21,  adjust=False).mean()
    sma50 = c.rolling(50).mean()  if n >= 50  else pd.Series(dtype=float)
    sma200= c.rolling(200).mean() if n >= 200 else pd.Series(dtype=float)

    bb_mid   = c.rolling(20).mean()
    bb_upper = bb_mid + 2 * c.rolling(20).std()
    bb_lower = bb_mid - 2 * c.rolling(20).std()

    # RSI
    d = c.diff()
    gain = d.clip(lower=0).ewm(com=13, min_periods=14).mean()
    loss = (-d.clip(upper=0)).ewm(com=13, min_periods=14).mean()
    rsi  = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(50)

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    msig  = macd.ewm(span=9, adjust=False).mean()
    mhist = macd - msig

    return dict(
        ema9=ema9, ema21=ema21, sma50=sma50, sma200=sma200,
        bb_upper=bb_upper, bb_lower=bb_lower,
        rsi=rsi, macd=macd, msig=msig, mhist=mhist,
    )


# ═══════════════════════════════════════════════════════════════
#  GENERATE SATU CHART
# ═══════════════════════════════════════════════════════════════
def generate_chart(symbol: str, timeframe: str, save_dir: str) -> str | None:
    cfg = TF_CONFIG.get(timeframe)
    if not cfg:
        return None

    df = _fetch_binance_klines(symbol, timeframe, cfg["limit"])
    if df is None or len(df) < 10:
        return None

    ind = _indicators(df)

    # ── addplots ──
    ap = []
    c0 = {"color":"#e040fb","width":1.0}
    c1 = {"color":"#00bcd4","width":1.0}
    c2 = {"color":"#2196f3","width":1.1,"linestyle":"--"}
    c3 = {"color":"#ff9800","width":1.1,"linestyle":"--"}
    cbb= {"color":"#546e7a","width":0.7,"linestyle":"--"}

    def ap0(s, **kw):
        if s is not None and not s.dropna().empty:
            ap.append(mpf.make_addplot(s.reindex(df.index), panel=0, **kw))

    ap0(ind["ema9"],    **c0)
    ap0(ind["ema21"],   **c1)
    ap0(ind["sma50"],   **c2)
    ap0(ind["sma200"],  **c3)
    ap0(ind["bb_upper"],**cbb)
    ap0(ind["bb_lower"],**cbb)

    # RSI panel
    rsi = ind["rsi"].reindex(df.index)
    if not rsi.dropna().empty:
        ap.append(mpf.make_addplot(rsi, panel=2, color="#ce93d8", width=1.2,
                                   ylabel="RSI", y_on_right=False))
        ap.append(mpf.make_addplot(pd.Series(70,index=df.index), panel=2,
                                   color="#ff5252", width=0.5, linestyle="--"))
        ap.append(mpf.make_addplot(pd.Series(30,index=df.index), panel=2,
                                   color="#69f0ae", width=0.5, linestyle="--"))

    # MACD panel
    mh = ind["mhist"].reindex(df.index)
    ml = ind["macd"].reindex(df.index)
    ms = ind["msig"].reindex(df.index)
    if not ml.dropna().empty:
        ap.append(mpf.make_addplot(mh.clip(lower=0), panel=3, type="bar",
                                   color="#00e676", alpha=0.7, ylabel="MACD"))
        ap.append(mpf.make_addplot(mh.clip(upper=0), panel=3, type="bar",
                                   color="#ff1744", alpha=0.7))
        ap.append(mpf.make_addplot(ml, panel=3, color="#42a5f5", width=1.0))
        ap.append(mpf.make_addplot(ms, panel=3, color="#ffa726", width=1.0))

    # ── Title ──
    cur = df["Close"].iloc[-1]
    chg = (cur / df["Close"].iloc[-2] - 1) * 100 if len(df) >= 2 else 0
    arrow = "▲" if chg >= 0 else "▼"
    clean_sym = symbol.replace("/","")
    title = f"{symbol} — {cfg['label']} | ${cur:,.4f} {arrow}{abs(chg):.2f}% | {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC"

    os.makedirs(save_dir, exist_ok=True)
    fpath = os.path.join(save_dir, f"{clean_sym}_{timeframe}.png")

    try:
        fig, axes = mpf.plot(
            df, type="candle", style=CHART_STYLE,
            title=f"\n{title}", volume=True,
            addplot=ap if ap else None,
            panel_ratios=(5, 1.2, 1.5, 1.5),
            figsize=(16, 10), returnfig=True,
            warn_too_much_data=99999, tight_layout=True,
        )
        # Legend
        ax0 = axes[0]
        handles = [
            mpatches.Patch(color="#e040fb", label="EMA9"),
            mpatches.Patch(color="#00bcd4", label="EMA21"),
            mpatches.Patch(color="#2196f3", label="SMA50"),
            mpatches.Patch(color="#ff9800", label="SMA200"),
            mpatches.Patch(color="#546e7a", label="BB(20)"),
        ]
        ax0.legend(handles=handles, loc="upper left", fontsize=7,
                   framealpha=0.3, facecolor="#0d0d1a", labelcolor="white")

        fig.savefig(fpath, dpi=110, bbox_inches="tight",
                    facecolor="#0d0d1a", edgecolor="none")
        plt.close(fig)
        return fpath
    except Exception as e:
        console.print(f"[dim]Chart error {symbol}[{timeframe}]: {e}[/dim]")
        try: plt.close("all")
        except Exception: pass
        return None


# ═══════════════════════════════════════════════════════════════
#  GENERATE SEMUA TF — satu simbol
# ═══════════════════════════════════════════════════════════════
def generate_all_charts(symbol: str, timeframes: list[str] = None) -> dict[str, str]:
    """Generate chart semua timeframe untuk satu simbol. Return {tf: filepath}."""
    tfs = timeframes or list(TF_CONFIG.keys())
    clean = symbol.replace("/","")
    save_dir = os.path.join(CHART_DIR, clean)
    os.makedirs(save_dir, exist_ok=True)
    results = {}
    console.print(f"  [cyan]📈 Charts {symbol} [{', '.join(tfs)}]...[/cyan]")

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(generate_chart, symbol, tf, save_dir): tf for tf in tfs}
        for fut, tf in futures.items():
            try:
                p = fut.result(timeout=30)
                if p:
                    results[tf] = p
            except Exception as e:
                console.print(f"  [dim]✗ {tf}: {e}[/dim]")

    done = len(results)
    console.print(f"  [green]✓ {done}/{len(tfs)} charts generated[/green]")
    return results


# ═══════════════════════════════════════════════════════════════
#  HTML GALLERY (dark theme, modal zoom, filter TF)
# ═══════════════════════════════════════════════════════════════
def build_html_gallery(chart_results: dict) -> str:
    """Build interactive HTML gallery. chart_results = {symbol: {tf: path}}"""
    os.makedirs(CHART_DIR, exist_ok=True)
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    tf_labels = {"1m":"1m","5m":"5m","15m":"15m","30m":"30m",
                 "1h":"1H","4h":"4H","1d":"Daily","1w":"Weekly","1M":"Monthly"}

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Crypto Bot — Chart Gallery</title>
<style>
:root{{--bg:#0d0d1a;--card:#13132b;--border:#1e1e3a;--txt:#ddd;--acc:#00e676;--red:#ff1744;--blue:#2196f3}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--txt);font-family:'Segoe UI',sans-serif;font-size:14px}}
header{{background:var(--card);border-bottom:1px solid var(--border);padding:14px 22px;display:flex;align-items:center;gap:16px}}
header h1{{font-size:1.25rem;color:var(--acc)}}header span{{color:#666;font-size:.8rem}}
.bar{{background:var(--card);border-bottom:1px solid var(--border);padding:10px 22px;display:flex;gap:10px;flex-wrap:wrap;align-items:center}}
.btn{{background:var(--border);border:1px solid #2e2e5e;color:var(--txt);padding:5px 13px;border-radius:5px;cursor:pointer;font-size:.78rem;transition:.15s}}
.btn:hover,.btn.active{{background:var(--blue);border-color:var(--blue)}}
input.search{{padding:5px 12px;background:var(--border);border:1px solid #2e2e5e;color:var(--txt);border-radius:5px;font-size:.78rem;width:180px}}
.sec{{margin:18px 22px}}
.sec-hdr{{display:flex;align-items:center;gap:10px;padding-bottom:8px;margin-bottom:12px;border-bottom:1px solid var(--border)}}
.sym{{font-size:1.05rem;font-weight:700;color:var(--acc)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));gap:14px}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:8px;overflow:hidden;transition:.2s}}
.card:hover{{transform:translateY(-2px);box-shadow:0 4px 18px rgba(0,230,118,.12)}}
.card .lbl{{padding:7px 12px;font-size:.72rem;color:#888;background:#0a0a18;border-bottom:1px solid var(--border)}}
.card img{{width:100%;display:block;cursor:zoom-in}}
.modal{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.88);z-index:999;align-items:center;justify-content:center}}
.modal.on{{display:flex}}.modal img{{max-width:95vw;max-height:92vh;border-radius:6px}}
.cls{{position:absolute;top:14px;right:18px;font-size:1.8rem;color:#fff;cursor:pointer;z-index:1000}}
</style></head><body>
<header><h1>📊 AI Crypto Trading Bot — Charts</h1>
<span>Updated: {now}</span><span style="margin-left:auto">{len(chart_results)} pairs</span></header>
<div class="bar"><span style="color:#666;font-size:.75rem">Timeframe:</span>
"""
    for tf, lbl in tf_labels.items():
        html += f'<button class="btn" onclick="fTF(\'{tf}\')">{lbl}</button>\n'
    html += '<button class="btn active" onclick="fTF(\'all\')">All</button>\n'
    html += '<input class="search" type="text" placeholder="Search symbol..." oninput="fSym(this.value)">\n'
    html += '<a href="/index.html" style="margin-left:auto;color:#42a5f5;font-size:.78rem">🔄 Refresh</a></div>\n'

    for symbol, tf_paths in sorted(chart_results.items()):
        if not tf_paths: continue
        clean = symbol.replace("/","")
        html += f'<div class="sec" data-sym="{clean.lower()}">\n'
        html += f'<div class="sec-hdr"><span class="sym">{symbol}</span>'
        html += f'<span style="color:#555;font-size:.75rem">{len(tf_paths)} charts</span></div>\n'
        html += '<div class="grid">\n'
        for tf in ["1m","5m","15m","30m","1h","4h","1d","1w","1M"]:
            p = tf_paths.get(tf)
            lbl = tf_labels.get(tf, tf)
            if p and os.path.exists(p):
                rel = os.path.relpath(p, CHART_DIR).replace("\\","/")
                html += f'<div class="card" data-tf="{tf}">'
                html += f'<div class="lbl">⏱ {lbl} — {symbol}</div>'
                html += f'<img src="{rel}" loading="lazy" onclick="openM(this.src)"></div>\n'
        html += '</div></div>\n'

    html += """<div class="modal" id="M" onclick="closeM()">
<span class="cls" onclick="closeM()">✕</span>
<img id="MI" src="" alt="chart">
</div>
<script>
function openM(s){document.getElementById('MI').src=s;document.getElementById('M').classList.add('on')}
function closeM(){document.getElementById('M').classList.remove('on')}
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeM()});
function fTF(tf){
  document.querySelectorAll('.btn').forEach(b=>b.classList.remove('active'));
  event.target.classList.add('active');
  document.querySelectorAll('.card').forEach(c=>{
    c.style.display=(tf==='all'||c.dataset.tf===tf)?'':'none';
  });
}
function fSym(v){
  v=v.toLowerCase();
  document.querySelectorAll('.sec').forEach(s=>{
    s.style.display=(!v||s.dataset.sym.includes(v))?'':'none';
  });
}
</script></body></html>"""

    path = os.path.join(CHART_DIR, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


# ═══════════════════════════════════════════════════════════════
#  LOCAL HTTP SERVER
# ═══════════════════════════════════════════════════════════════
_srv = None
_thr = None

def start_chart_server(open_browser: bool = True) -> str:
    global _srv, _thr
    if _thr and _thr.is_alive():
        return f"http://localhost:{CHART_SERVER_PORT}"
    os.makedirs(CHART_DIR, exist_ok=True)

    class H(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=CHART_DIR, **kw)
        def log_message(self, *a): pass

    try:
        _srv = HTTPServer(("localhost", CHART_SERVER_PORT), H)
        _thr = threading.Thread(target=_srv.serve_forever, daemon=True)
        _thr.start()
        url = f"http://localhost:{CHART_SERVER_PORT}"
        console.print(f"[green]✓ Chart server: {url}[/green]")
        if open_browser:
            webbrowser.open(f"{url}/index.html")
        return url
    except Exception as e:
        console.print(f"[red]Chart server gagal: {e}[/red]")
        return ""

def stop_chart_server():
    global _srv, _thr
    if _srv: _srv.shutdown(); _srv = None
    _thr = None


# ── CLI: python charts.py BTC/USDT ────────────────────────────
if __name__ == "__main__":
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "BTC/USDT"
    tfs = sys.argv[2].split(",") if len(sys.argv) > 2 else list(TF_CONFIG.keys())
    console.print(f"[bold cyan]Generating charts {sym} — {tfs}[/bold cyan]")
    res = generate_all_charts(sym, tfs)
    gal = build_html_gallery({sym: res})
    console.print(f"[green]Gallery: {gal}[/green]")
    start_chart_server(open_browser=True)
    console.print("[dim]Ctrl+C untuk keluar[/dim]")
    try:
        import time
        while True: time.sleep(1)
    except KeyboardInterrupt:
        pass
