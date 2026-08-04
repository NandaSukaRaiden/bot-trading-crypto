@echo off
chcp 65001 >nul
echo.
echo ============================================================
echo  GIT PUSH — AI Trading Bot
echo ============================================================
echo.

:: Cek apakah ada perubahan
git status --short > nul 2>&1
if errorlevel 1 (
    echo [ERROR] Bukan git repository.
    pause
    exit /b 1
)

:: Tampilkan file yang berubah
echo File yang akan di-commit:
git status --short
echo.

:: Minta pesan commit
set /p MSG="Pesan commit (Enter = 'update'): "
if "%MSG%"=="" set MSG=update

:: Add semua (kecuali yang di .gitignore)
git add .

:: Commit
git commit -m "%MSG%"
if errorlevel 1 (
    echo Tidak ada perubahan untuk di-commit.
    pause
    exit /b 0
)

:: Push
echo.
echo Pushing ke GitHub...
git push
if errorlevel 1 (
    echo [ERROR] Push gagal. Cek koneksi atau credentials.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  BERHASIL! Kode sudah di-push ke GitHub.
echo ============================================================
echo  Repo: https://github.com/NandaSukaRaiden/bot-trading-crypto
echo ============================================================
pause
