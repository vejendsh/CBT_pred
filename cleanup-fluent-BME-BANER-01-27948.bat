echo off
set LOCALHOST=%COMPUTERNAME%
set KILL_CMD="C:\PROGRA~1\ANSYSI~1\v242\fluent/ntbin/win64/winkill.exe"

start "tell.exe" /B "C:\PROGRA~1\ANSYSI~1\v242\fluent\ntbin\win64\tell.exe" BME-BANER-01 51300 CLEANUP_EXITING
timeout /t 1
"C:\PROGRA~1\ANSYSI~1\v242\fluent\ntbin\win64\kill.exe" tell.exe
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 28052) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 27944) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 18976) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 22124) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 27948) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 28196)
del "C:\Users\Research\Desktop\CBT_pred\cleanup-fluent-BME-BANER-01-27948.bat"
