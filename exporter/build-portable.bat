@echo off
REM ============================================================
REM OneNote Whiteboard Scanner - Portable Package Builder
REM ============================================================
REM Creates a portable folder with everything needed to run
REM ============================================================

setlocal EnableDelayedExpansion

echo.
echo ============================================================
echo OneNote Whiteboard Scanner - Portable Package Builder
echo ============================================================
echo.

REM Create portable package directory
set "PACKAGE_DIR=%~dp0portable-package"
set "APP_DIR=%PACKAGE_DIR%\OneNote-Whiteboard-Scanner"

echo [1/4] Creating package directory structure...
if exist "%PACKAGE_DIR%" (
    echo Cleaning old package...
    rmdir /s /q "%PACKAGE_DIR%"
)
mkdir "%APP_DIR%"
mkdir "%APP_DIR%\backend"
mkdir "%APP_DIR%\app"
echo [OK] Directory structure created
echo.

REM Build backend if needed
echo [2/4] Checking backend executable...
if not exist "%~dp0local-ai-backend\dist\backend\backend.exe" (
    echo Backend not built yet. Building now...
    echo This will take 5-15 minutes...
    echo.
    
    cd /d "%~dp0local-ai-backend"
    
    python -c "import PyInstaller" 2>NUL
    if errorlevel 1 (
        echo Installing PyInstaller...
        python -m pip install pyinstaller
    )
    
    python -m PyInstaller backend.spec --clean --noconfirm
    
    if errorlevel 1 (
        echo [ERROR] Backend build failed!
        pause
        exit /b 1
    )
    
    cd /d "%~dp0"
)

echo [OK] Backend executable found
echo.

REM Copy backend files
echo [3/4] Copying backend files...
xcopy /E /I /Y "%~dp0local-ai-backend\dist\backend\*" "%APP_DIR%\backend\" >NUL
echo [OK] Backend files copied
echo.

REM Build frontend
echo [4/4] Building and copying frontend...
cd /d "%~dp0desktop-app"
call npm run build >NUL 2>&1

REM Copy frontend files
xcopy /Y "electron-main.js" "%APP_DIR%\app\" >NUL
xcopy /Y "preload.js" "%APP_DIR%\app\" >NUL
xcopy /Y "package.json" "%APP_DIR%\app\" >NUL
xcopy /E /I /Y "renderer" "%APP_DIR%\app\renderer\" >NUL
xcopy /E /I /Y "assets" "%APP_DIR%\app\assets\" >NUL
xcopy /E /I /Y "node_modules" "%APP_DIR%\app\node_modules\" >NUL

echo [OK] Frontend files copied
echo.

REM Create run script
echo Creating run script...
(
echo @echo off
echo REM OneNote Whiteboard Scanner - Startup Script
echo.
echo echo.
echo echo ============================================================
echo echo OneNote Whiteboard Scanner
echo echo ============================================================
echo echo.
echo echo Starting application...
echo echo.
echo.
echo REM Create VBScript to run backend completely hidden
echo echo Set WshShell = CreateObject("WScript.Shell"^) ^> "%%TEMP%%\run_backend.vbs"
echo echo WshShell.Run "cmd /c cd /d ""%%~dp0backend"" ^&^& backend.exe", 0, False ^>^> "%%TEMP%%\run_backend.vbs"
echo.
echo REM Run backend hidden
echo cscript //nologo "%%TEMP%%\run_backend.vbs"
echo.
echo REM Wait for backend to start
echo timeout /t 3 /nobreak ^>nul
echo.
echo REM Start Electron app
echo cd /d "%%~dp0app"
echo npx electron .
echo.
echo REM Cleanup - Kill backend and remove temp files
echo taskkill /F /IM backend.exe ^>nul 2^>^&1
echo del "%%TEMP%%\run_backend.vbs" ^>nul 2^>^&1
) > "%APP_DIR%\Run OneNote Scanner.bat"

echo [OK] Run script created
echo.

REM Create README
(
echo # OneNote Whiteboard Scanner - Portable Edition
echo.
echo ## How to Run
echo.
echo 1. Double-click "Run OneNote Scanner.bat"
echo 2. The backend will start automatically
echo 3. The app window will open
echo 4. Login with your Microsoft account
echo 5. Start scanning whiteboards!
echo.
echo ## Requirements
echo.
echo - Windows 10/11 (64-bit^)
echo - No Python installation needed
echo - No admin rights needed
echo - ~2 GB disk space
echo.
echo ## Files
echo.
echo - `backend\` - Standalone backend with all AI models
echo - `app\` - Electron desktop application
echo - `Run OneNote Scanner.bat` - Startup script
echo.
echo ## Troubleshooting
echo.
echo If the app doesn't start:
echo 1. Make sure no other instance is running
echo 2. Check Windows Firewall isn't blocking it
echo 3. Run "Run OneNote Scanner.bat" again
echo.
echo To fully close the app, close the window and check Task Manager
echo for any remaining "backend.exe" processes.
) > "%APP_DIR%\README.txt"

echo [OK] README created
echo.

REM Create ZIP package
echo Creating ZIP package...
cd /d "%PACKAGE_DIR%"
powershell -Command "Compress-Archive -Path 'OneNote-Whiteboard-Scanner' -DestinationPath '../OneNote-Whiteboard-Scanner-Portable.zip' -Force"

if exist "%~dp0OneNote-Whiteboard-Scanner-Portable.zip" (
    echo.
    echo ============================================================
    echo PACKAGE COMPLETE!
    echo ============================================================
    echo.
    echo Portable folder: %APP_DIR%
    echo ZIP package: %~dp0OneNote-Whiteboard-Scanner-Portable.zip
    echo.
    for %%F in ("%~dp0OneNote-Whiteboard-Scanner-Portable.zip") do (
        set "SIZE=%%~zF"
        set /a SIZE_MB=!SIZE! / 1048576
        echo ZIP size: !SIZE_MB! MB
    )
    echo.
    echo To distribute:
    echo   1. Send the ZIP file to users
    echo   2. Users extract the ZIP
    echo   3. Users run "Run OneNote Scanner.bat"
    echo.
    echo No installation needed!
    echo.
) else (
    echo [ERROR] Failed to create ZIP package
)

pause
