echo off
set LOCALHOST=%COMPUTERNAME%
set KILL_CMD="C:\PROGRA~1\ANSYSI~1\v242\fluent/ntbin/win64/winkill.exe"

start "tell.exe" /B "C:\PROGRA~1\ANSYSI~1\v242\fluent\ntbin\win64\tell.exe" BME-BANER-01 63424 CLEANUP_EXITING
timeout /t 1
"C:\PROGRA~1\ANSYSI~1\v242\fluent\ntbin\win64\kill.exe" tell.exe
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 10916) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 21884) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 25692) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 25916) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 26116) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 25008)
del "C:\Users\Research\Desktop\CBT_pred\cleanup-fluent-BME-BANER-01-26116.bat"
