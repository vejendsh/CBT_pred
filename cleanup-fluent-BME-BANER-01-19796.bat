echo off
set LOCALHOST=%COMPUTERNAME%
set KILL_CMD="C:\PROGRA~1\ANSYSI~1\v242\fluent/ntbin/win64/winkill.exe"

start "tell.exe" /B "C:\PROGRA~1\ANSYSI~1\v242\fluent\ntbin\win64\tell.exe" BME-BANER-01 51406 CLEANUP_EXITING
timeout /t 1
"C:\PROGRA~1\ANSYSI~1\v242\fluent\ntbin\win64\kill.exe" tell.exe
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 24484) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 1456) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 19888) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 21776) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 19796) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 27068)
del "C:\Users\Research\Desktop\CBT_pred\cleanup-fluent-BME-BANER-01-19796.bat"
