echo off
set LOCALHOST=%COMPUTERNAME%
set KILL_CMD="C:\PROGRA~1\ANSYSI~1\v242\fluent/ntbin/win64/winkill.exe"

start "tell.exe" /B "C:\PROGRA~1\ANSYSI~1\v242\fluent\ntbin\win64\tell.exe" BME-BANER-01 57207 CLEANUP_EXITING
timeout /t 1
"C:\PROGRA~1\ANSYSI~1\v242\fluent\ntbin\win64\kill.exe" tell.exe
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 13916) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 20760) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 22988) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 9376) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 15244) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 2860)
del "C:\Users\Research\Desktop\CBT_pred\cleanup-fluent-BME-BANER-01-15244.bat"
