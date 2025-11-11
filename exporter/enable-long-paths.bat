@echo off
REM ============================================================================
REM Enable Long Path Support in Windows
REM ============================================================================
REM This removes the 260 character path length limit in Windows
REM Requires Administrator privileges
REM ============================================================================

echo.
echo ============================================================================
echo ENABLE LONG PATH SUPPORT
echo ============================================================================
echo.
echo This will enable support for paths longer than 260 characters in Windows.
echo This is required for building with large Python dependencies.
echo.
echo REQUIRES: Administrator privileges
echo.
pause

REM Check for admin rights
net session >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: This script must be run as Administrator!
    echo.
    echo Right-click this file and select "Run as administrator"
    echo.
    pause
    exit /b 1
)

echo.
echo Enabling long path support in Windows Registry...
echo.

REM Enable long paths in Windows 10/11
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================================
    echo SUCCESS! Long path support enabled.
    echo ============================================================================
    echo.
    echo IMPORTANT: You may need to restart your computer for changes to take effect.
    echo.
    echo After restart, you will be able to use paths longer than 260 characters.
    echo.
) else (
    echo.
    echo ERROR: Failed to enable long path support.
    echo Please check that you have administrator privileges.
    echo.
)

pause
