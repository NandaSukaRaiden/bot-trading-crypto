@echo off
chcp 65001 >nul
echo ============================================================
echo  AI CRYPTO FUTURES BOT — Setup Otomatis
echo ============================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python tidak ditemukan. Install Python 3.10+ dari python.org
    pause & exit /b 1
)

echo [1/4] Membuat virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo.
echo [2/4] Install dependencies...
pip install --upgrade pip --quiet
pip install -r requirements.txt

echo.
echo [3/4] Membuat .env dari template...
if not exist .env (
    copy .env.example .env
    echo File .env dibuat. ISI GOOGLE_API_KEY terlebih dahulu!
) else (
    echo .env sudah ada.
)

echo.
echo [4/4] Verifikasi imports...
python -c "import ccxt, pandas, numpy, mplfinance, google.generativeai, rich, apscheduler; print('Semua module OK!')"

echo.
echo ============================================================
echo  SETUP SELESAI!
echo ============================================================
echo.
echo LANGKAH SELANJUTNYA:
echo.
echo 1. Edit .env dan isi minimal:
echo    GOOGLE_API_KEY=AIzaSy...  ^(dari aistudio.google.com — GRATIS^)
echo.
echo 2. Jalankan bot (paper mode):
echo    venv\Scripts\activate
echo    python trading_bot.py
echo.
echo 3. Lihat chart di browser:
echo    python charts.py BTC/USDT
echo    Buka: http://localhost:8008
echo.
echo 4. Monitor dashboard di terminal lain:
echo    python dashboard.py
echo ============================================================
pause
