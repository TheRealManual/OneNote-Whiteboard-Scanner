@echo off
REM ============================================================
REM OneNote Whiteboard Scanner - Production Build Script
REM ============================================================
REM Creates a portable package ready to distribute
REM No installer needed - users just extract and run!
REM ============================================================

echo.
echo ============================================================
echo Building Portable Package...
echo ============================================================
echo.
echo This will create a ZIP file that users can extract and run.
echo No installation or admin rights needed!
echo.

call "%~dp0build-portable.bat"

REM Check for PyInstaller
echo [CHECKING] PyInstaller availability... >> "%LOGFILE%"
python -c "import PyInstaller" 2>NUL
if errorlevel 1 (
    echo [INFO] PyInstaller not found. Installing...
    echo [INFO] PyInstaller not found. Installing... >> "%LOGFILE%"
    echo Installing PyInstaller...
    python -m pip install pyinstaller >> "%LOGFILE%" 2>&1
    if errorlevel 1 (
        echo [ERROR] Failed to install PyInstaller >> "%LOGFILE%"
        echo [ERROR] Failed to install PyInstaller
        echo.
        echo See log: %LOGFILE%
        pause
        exit /b 1
    )
    echo [OK] PyInstaller installed >> "%LOGFILE%"
    echo [OK] PyInstaller installed
) else (
    echo [OK] PyInstaller already installed >> "%LOGFILE%"
    echo [OK] PyInstaller found
)

REM ============================================================
REM STEP 1: Build Backend Executable (5-15 minutes)
REM ============================================================
set /a CURRENT_STEP+=1
echo.
echo ============================================================ >> "%LOGFILE%"
echo STEP !CURRENT_STEP!/%TOTAL_STEPS%: Building Backend Executable >> "%LOGFILE%"
echo Start time: %TIME% >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"
echo.
echo [!CURRENT_STEP!/%TOTAL_STEPS%] Building Backend Executable with PyInstaller
echo -------------------------------------------------------
echo [INFO] This creates a standalone backend.exe with all dependencies
echo [INFO] First build: 5-15 minutes (downloads PyTorch, OpenCV, etc.)
echo [INFO] Subsequent builds: 2-5 minutes (cached)
echo [STATUS] Starting backend build...
echo.

cd /d "%~dp0..\local-ai-backend"
echo [DEBUG] Changed to directory: %CD% >> "%LOGFILE%"

REM Clean previous builds
if exist "dist\backend" (
    echo [STATUS] Cleaning previous backend build...
    echo [STATUS] Cleaning previous backend build... >> "%LOGFILE%"
    rmdir /s /q "dist\backend" 2>NUL
    echo [OK] Previous build cleaned >> "%LOGFILE%"
)
if exist "build" (
    echo [STATUS] Cleaning build cache... >> "%LOGFILE%"
    rmdir /s /q "build" 2>NUL
    echo [OK] Build cache cleaned >> "%LOGFILE%"
)

REM Build with PyInstaller
echo [STATUS] Running PyInstaller... >> "%LOGFILE%"
echo [STATUS] Running PyInstaller (this will take several minutes)...
echo [STATUS] Watch for real-time progress below...
echo.
echo ---- PyInstaller Output (LIVE) ----
echo.

REM Run PyInstaller with live output using PowerShell
powershell -Command "& {python -m PyInstaller backend.spec --clean --noconfirm 2>&1 | ForEach-Object { Write-Host $_; Add-Content -Path '%LOGFILE%' -Value $_ }}"

if errorlevel 1 (
    echo.
    echo [ERROR] Backend build failed! >> "%LOGFILE%"
    echo [ERROR] Backend build failed!
    echo [ERROR] Check log for details: %LOGFILE%
    echo.
    echo Last 50 lines of log:
    powershell -Command "Get-Content '%LOGFILE%' -Tail 50"
    pause
    exit /b 1
)

if not exist "dist\backend\backend.exe" (
    echo.
    echo [ERROR] backend.exe was not created! >> "%LOGFILE%"
    echo [ERROR] backend.exe was not created!
    echo [ERROR] PyInstaller completed but no executable found
    echo.
    pause
    exit /b 1
)

echo.
echo [OK] Backend executable created >> "%LOGFILE%"
echo [OK] Backend executable created
for %%F in ("dist\backend\backend.exe") do (
    set "SIZE=%%~zF"
    set /a SIZE_MB=!SIZE! / 1048576
    echo [INFO] Size: !SIZE_MB! MB ^(%%~zF bytes^) >> "%LOGFILE%"
    echo       Size: !SIZE_MB! MB
)
echo End time: %TIME% >> "%LOGFILE%"
echo.

echo [========================================] Step !CURRENT_STEP!/%TOTAL_STEPS% Complete
echo.

echo [========================================] Step !CURRENT_STEP!/%TOTAL_STEPS% Complete
echo.

REM ============================================================
REM STEP 2: Build Frontend (30-60 seconds)
REM ============================================================
set /a CURRENT_STEP+=1
echo ============================================================ >> "%LOGFILE%"
echo STEP !CURRENT_STEP!/%TOTAL_STEPS%: Building Frontend >> "%LOGFILE%"
echo Start time: %TIME% >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"
echo.
echo [!CURRENT_STEP!/%TOTAL_STEPS%] Building Frontend with Webpack
echo -------------------------------------------------------
echo [INFO] Bundling React app for production
echo [STATUS] Running webpack...
echo.

cd /d "%~dp0"
echo [DEBUG] Changed to directory: %CD% >> "%LOGFILE%"

echo ---- Webpack Output (LIVE) ----
echo.

REM Run webpack with live output
powershell -Command "& {npm run build 2>&1 | ForEach-Object { Write-Host $_; Add-Content -Path '%LOGFILE%' -Value $_ }}"

if errorlevel 1 (
    echo.
    echo [ERROR] Frontend build failed! >> "%LOGFILE%"
    echo [ERROR] Frontend build failed!
    echo [ERROR] Check log for details: %LOGFILE%
    echo.
    echo Last 50 lines of log:
    powershell -Command "Get-Content '%LOGFILE%' -Tail 50"
    pause
    exit /b 1
)

echo.
echo [OK] Frontend built successfully >> "%LOGFILE%"
echo [OK] Frontend built successfully
echo End time: %TIME% >> "%LOGFILE%"
echo.

echo [========================================] Step !CURRENT_STEP!/%TOTAL_STEPS% Complete
echo.

echo [========================================] Step !CURRENT_STEP!/%TOTAL_STEPS% Complete
echo.

REM ============================================================
REM STEP 3: Create Installer (2-5 minutes)
REM ============================================================
set /a CURRENT_STEP+=1
echo ============================================================ >> "%LOGFILE%"
echo STEP !CURRENT_STEP!/%TOTAL_STEPS%: Creating Windows Installer >> "%LOGFILE%"
echo Start time: %TIME% >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"
echo.
echo [!CURRENT_STEP!/%TOTAL_STEPS%] Creating Windows Installer
echo -------------------------------------------------------
echo [INFO] Building NSIS installer with electron-builder
echo [STATUS] Bundling backend.exe and frontend...
echo [STATUS] Creating installer package...
echo.

REM Disable code signing
set CSC_IDENTITY_AUTO_DISCOVERY=false
echo [DEBUG] Code signing disabled >> "%LOGFILE%"

echo ---- Electron Builder Output (LIVE) ----
echo.

REM Run electron-builder with live output
powershell -Command "& {npm run build:win 2>&1 | ForEach-Object { Write-Host $_; Add-Content -Path '%LOGFILE%' -Value $_ }}"

if errorlevel 1 (
    echo.
    echo [ERROR] Installer creation failed! >> "%LOGFILE%"
    echo [ERROR] Installer creation failed!
    echo [ERROR] Check log for details: %LOGFILE%
    echo.
    echo Last 50 lines of log:
    powershell -Command "Get-Content '%LOGFILE%' -Tail 50"
    pause
    exit /b 1
)

echo.
echo [OK] Installer created successfully >> "%LOGFILE%"
echo [OK] Installer created successfully
echo End time: %TIME% >> "%LOGFILE%"
echo.

echo [========================================] Step !CURRENT_STEP!/%TOTAL_STEPS% Complete
echo.
echo [========================================] Step !CURRENT_STEP!/%TOTAL_STEPS% Complete
echo.

REM ============================================================
REM BUILD COMPLETE
REM ============================================================
echo ============================================================ >> "%LOGFILE%"
echo BUILD COMPLETED SUCCESSFULLY >> "%LOGFILE%"
echo End time: %TIME% >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"

echo.
echo ============================================================
echo BUILD COMPLETE!
echo ============================================================
echo.
echo Installer created at:
echo   desktop-app\dist\OneNote Whiteboard Scanner Setup *.exe
echo.

REM List installer files
for %%F in ("dist\*.exe") do (
    set "SIZE=%%~zF"
    set /a SIZE_MB=!SIZE! / 1048576
    echo   %%~nxF ^(!SIZE_MB! MB^)
    echo Installer: %%~nxF ^(!SIZE_MB! MB^) >> "%LOGFILE%"
)

echo.
echo This installer includes:
echo   [X] Standalone backend (no Python required)
echo   [X] All dependencies bundled
echo   [X] Ready for distribution
echo.
echo Target machines need:
echo   [X] Windows 10/11 (64-bit)
echo   [X] ~2 GB disk space
echo   [X] Internet connection (for OneNote API)
echo.
echo Target machines DO NOT need:
echo   [ ] Python installation
echo   [ ] pip or package managers
echo   [ ] Visual C++ redistributables
echo.
echo Build log saved to: %LOGFILE%
echo.
echo ============================================================
echo.
pause
