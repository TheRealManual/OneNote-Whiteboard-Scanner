@echo off
REM ============================================================
REM OneNote Whiteboard Scanner - Development Start
REM ============================================================

echo.
echo ============================================================
echo   OneNote Whiteboard Scanner - Development Mode
echo ============================================================
echo.

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Create log directory
if not exist "%SCRIPT_DIR%logs" mkdir "%SCRIPT_DIR%logs"

REM Kill any existing backend processes
echo [0/2] Cleaning up existing backend processes...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5000 ^| findstr LISTENING') do (
    echo Killing process on port 5000: %%a
    taskkill /F /PID %%a >nul 2>&1
)
taskkill /F /IM python.exe /FI "COMMANDLINE eq *app.py*" >nul 2>&1
timeout /t 1 /nobreak > nul

REM Start backend in new window with title
echo [1/2] Starting backend server...
start "Backend Server" cmd /c "cd /d "%SCRIPT_DIR%local-ai-backend" && python app.py 2>&1 || pause"

REM Wait for backend to start
echo Waiting for backend to start...
timeout /t 5 /nobreak > nul

REM Start frontend (logs will show in current window)
echo [2/2] Starting desktop app...
echo.
echo ============================================================
echo   Frontend logs below:
echo ============================================================
echo.
cd /d "%SCRIPT_DIR%desktop-app"
call npm start

REM Kill backend when frontend closes
echo.
echo ============================================================
echo Stopping backend server...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Backend Server*" >nul 2>&1
taskkill /F /IM python.exe /FI "COMMANDLINE eq *app.py*" >nul 2>&1

echo.
echo Application closed.
echo.
