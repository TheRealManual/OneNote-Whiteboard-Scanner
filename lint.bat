@echo off
REM ============================================================
REM OneNote Whiteboard Scanner - Code Linting
REM ============================================================

echo.
echo ================================================================
echo   Python Code Linter
echo ================================================================
echo.

set /p choice="Lint [1] Backend only, [2] Tests only, [3] All Python files: "

if "%choice%"=="1" goto lint_backend
if "%choice%"=="2" goto lint_tests
if "%choice%"=="3" goto lint_all

echo Invalid choice.
goto :eof

:lint_backend
echo.
echo Linting backend code...
echo ----------------------------------------------------------------
cd local-ai-backend
flake8 *.py ai/
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✓ No issues found in backend!
) else (
    echo.
    echo ✗ Issues found - please review above
)
cd ..
pause
goto :eof

:lint_tests
echo.
echo Linting test files...
echo ----------------------------------------------------------------
flake8 tests/
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✓ No issues found in tests!
) else (
    echo.
    echo ✗ Issues found - please review above
)
pause
goto :eof

:lint_all
echo.
echo Linting all Python files...
echo ----------------------------------------------------------------
echo.
echo [1/2] Backend code...
cd local-ai-backend
flake8 *.py ai/
set backend_result=%ERRORLEVEL%
cd ..

echo.
echo [2/2] Test files...
flake8 tests/
set tests_result=%ERRORLEVEL%

echo.
echo ================================================================
if %backend_result% EQU 0 if %tests_result% EQU 0 (
    echo ✓ All checks passed!
) else (
    echo ✗ Issues found - please review above
)
echo ================================================================
echo.
pause
