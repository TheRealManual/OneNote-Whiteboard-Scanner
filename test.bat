@echo off
setlocal enabledelayedexpansion

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"

:menu
cls
echo.
echo ================================================================
echo   OneNote Whiteboard Scanner - Comprehensive Test Suite
echo ================================================================
echo.
echo   Select a test to run:
echo.
echo   1. Run ALL tests
echo   2. Test API Endpoints
echo   3. Test OneNote Integration
echo   4. Test Image Processing
echo.
echo   0. Exit
echo.
echo ================================================================
echo.
echo   Note: Tests will start and stop their own backend server
echo.

set /p choice="Enter your choice (0-4): "

if "%choice%"=="0" goto :eof
if "%choice%"=="1" goto run_all
if "%choice%"=="2" goto test_api
if "%choice%"=="3" goto test_onenote
if "%choice%"=="4" goto test_image

echo Invalid choice. Please try again.
timeout /t 2 >nul
goto menu

:run_all
cls
echo.
echo ================================================================
echo   Running ALL Tests
echo ================================================================
echo.

set total_passed=0
set total_failed=0

REM Run API tests
echo.
echo [1/3] Running API Endpoint Tests...
echo ----------------------------------------------------------------
cd /d "%SCRIPT_DIR%tests"
python test_api_endpoints.py
if %ERRORLEVEL% EQU 0 (
    set /a total_passed+=1
    echo   ✓ API tests PASSED
) else (
    set /a total_failed+=1
    echo   ✗ API tests FAILED
)

REM Run OneNote tests
echo.
echo [2/3] Running OneNote Integration Tests...
echo ----------------------------------------------------------------
cd /d "%SCRIPT_DIR%tests"
python test_onenote_integration.py
if %ERRORLEVEL% EQU 0 (
    set /a total_passed+=1
    echo   ✓ OneNote tests PASSED
) else (
    set /a total_failed+=1
    echo   ✗ OneNote tests FAILED
)

REM Run Image Processing tests
echo.
echo [3/3] Running Image Processing Tests...
echo ----------------------------------------------------------------
cd /d "%SCRIPT_DIR%tests"
python test_image_processing.py
if %ERRORLEVEL% EQU 0 (
    set /a total_passed+=1
    echo   ✓ Image Processing tests PASSED
) else (
    set /a total_failed+=1
    echo   ✗ Image Processing tests FAILED
)

REM Final summary
echo.
echo ================================================================
echo   FINAL RESULTS: !total_passed! test suites passed, !total_failed! failed
echo ================================================================
echo.
pause
goto menu

:test_api
cls
echo.
echo ================================================================
echo   Testing API Endpoints
echo ================================================================
echo.

cd /d "%SCRIPT_DIR%tests"
python test_api_endpoints.py

echo.
pause
goto menu

:test_onenote
cls
echo.
echo ================================================================
echo   Testing OneNote Integration
echo ================================================================
echo.

cd /d "%SCRIPT_DIR%tests"
python test_onenote_integration.py

echo.
pause
goto menu

:test_image
cls
echo.
echo ================================================================
echo   Testing Image Processing
echo ================================================================
echo.

cd /d "%SCRIPT_DIR%tests"
python test_image_processing.py

echo.
pause
goto menu
