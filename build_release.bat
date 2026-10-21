@echo off
setlocal
cd /d "%~dp0"

echo Building Gesture Manager release...
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" build_release.py %*
) else (
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3.13 build_release.py %*
    ) else (
        python build_release.py %*
    )
)
set "GM_BUILD_EXIT=%ERRORLEVEL%"

if not "%GM_BUILD_EXIT%"=="0" echo Build failed with exit code %GM_BUILD_EXIT%.
pause
exit /b %GM_BUILD_EXIT%
