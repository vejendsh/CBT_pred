echo off
set LOCALHOST=%COMPUTERNAME%
set KILL_CMD="C:\PROGRA~1\ANSYSI~1\v242\fluent/ntbin/win64/winkill.exe"

start "tell.exe" /B "C:\PROGRA~1\ANSYSI~1\v242\fluent\ntbin\win64\tell.exe" BME-BANER-01 57408 CLEANUP_EXITING
timeout /t 1
"C:\PROGRA~1\ANSYSI~1\v242\fluent\ntbin\win64\kill.exe" tell.exe
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 21888) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 21812) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 18652) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 23928) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 4584) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 24224)
del "C:\Users\Research\Desktop\CBT_pred\src\numerical_solution\cleanup-fluent-BME-BANER-01-4584.bat"
