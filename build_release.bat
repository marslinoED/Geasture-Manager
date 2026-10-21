@echo off
setlocal
cd /d "%~dp0"

echo Building Gesture Manager release...
if not exist ".venv\Scripts\python.exe" (
    echo Creating the Python 3.13 build environment...
    where py >nul 2>nul
    if errorlevel 1 (
        echo Python Launcher was not found. Install 64-bit Python 3.13, then run this script again.
        set "GM_BUILD_EXIT=1"
        goto done
    )
    py -3.13 -m venv .venv
    if errorlevel 1 (
        echo Could not create .venv. Confirm that 64-bit Python 3.13 is installed.
        set "GM_BUILD_EXIT=1"
        goto done
    )
)

echo Installing application and build dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools
if errorlevel 1 (
    set "GM_BUILD_EXIT=%ERRORLEVEL%"
    goto done
)
".venv\Scripts\python.exe" -m pip install --editable ".[build]"
if errorlevel 1 (
    set "GM_BUILD_EXIT=%ERRORLEVEL%"
    goto done
)

echo Building the portable release...
".venv\Scripts\python.exe" build_release.py %*
set "GM_BUILD_EXIT=%ERRORLEVEL%"

:done
if not "%GM_BUILD_EXIT%"=="0" echo Build failed with exit code %GM_BUILD_EXIT%.
pause
exit /b %GM_BUILD_EXIT%
