@echo off
REM ============================================================================
REM OneNote Whiteboard Scanner - Light Installer Build
REM ============================================================================
REM Creates a small installer that requires Python to be pre-installed
REM Output: ~100 MB installer
REM ============================================================================

echo.
echo ============================================================================
echo BUILDING LIGHT INSTALLER
echo ============================================================================
echo.
echo This will create a small installer that requires users to have Python 3.9+
echo installed on their system. Python packages will auto-install on first run.
echo.
echo Estimated build time: 5 minutes
echo Estimated installer size: 100 MB
echo.
pause

REM Check Node.js
echo [1/5] Checking Node.js...
where node >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Node.js not found! Please install Node.js 18+ from https://nodejs.org
    pause
    exit /b 1
)
echo   - Node.js found

REM Navigate to desktop app
echo.
echo [2/5] Preparing desktop app...
cd /d "%~dp0..\desktop-app"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: desktop-app folder not found!
    pause
    exit /b 1
)

REM Install dependencies if needed
if not exist "node_modules" (
    echo   - Installing npm dependencies...
    call npm install
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: npm install failed!
        pause
        exit /b 1
    )
) else (
    echo   - Dependencies already installed
)

REM Build React app
echo.
echo [3/5] Building React frontend...
call npm run build
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: React build failed!
    pause
    exit /b 1
)

REM Ensure backend files are included
echo.
echo [4/5] Preparing Python backend...
if not exist "..\local-ai-backend\requirements.txt" (
    echo ERROR: Backend requirements.txt not found!
    pause
    exit /b 1
)
echo   - Backend files ready

REM Build installer with Electron Builder
echo.
echo [5/5] Building installer with Electron Builder...
call npm run build:win
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Electron Builder failed!
    pause
    exit /b 1
)

REM Move output to exporter/dist
echo.
echo [FINAL] Moving installer to exporter/dist...
if not exist "..\exporter\dist" mkdir "..\exporter\dist"
move /y "dist\*.exe" "..\exporter\dist\" >nul 2>&1
move /y "dist\*.yml" "..\exporter\dist\" >nul 2>&1

echo.
echo ============================================================================
echo BUILD COMPLETE!
echo ============================================================================
echo.
echo Installer created in: exporter\dist\
dir /b "..\exporter\dist\*.exe"
echo.
echo INSTALLER REQUIREMENTS:
echo   - User must have Python 3.9+ installed
echo   - Internet connection on first run (to install Python packages)
echo   - ~500 MB free disk space for Python dependencies
echo.
echo NEXT STEPS:
echo   1. Test installer on a clean Windows machine
echo   2. Verify Python auto-installation works
echo   3. Check that all features work correctly
echo.
echo ============================================================================
pause
