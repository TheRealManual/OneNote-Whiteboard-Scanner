@echo off
REM Add Windows Defender exclusion for build folder
REM Must be run as Administrator

echo ============================================================================
echo ADDING WINDOWS DEFENDER EXCLUSION
echo ============================================================================
echo.
echo This will add the desktop-app/dist folder to Windows Defender exclusions
echo to prevent it from interfering with the build process.
echo.
echo You must run this as Administrator!
echo.
pause

echo Adding exclusion...
powershell -Command "Add-MpPreference -ExclusionPath '%~dp0..\desktop-app\dist'"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo SUCCESS: Exclusion added successfully!
    echo You can now build without rcedit errors.
) else (
    echo.
    echo ERROR: Failed to add exclusion.
    echo Make sure you're running this as Administrator.
    echo.
    echo Right-click this file and select "Run as administrator"
)

echo.
pause
