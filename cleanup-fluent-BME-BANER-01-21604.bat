echo off
set LOCALHOST=%COMPUTERNAME%
set KILL_CMD="C:\PROGRA~1\ANSYSI~1\v242\fluent/ntbin/win64/winkill.exe"

start "tell.exe" /B "C:\PROGRA~1\ANSYSI~1\v242\fluent\ntbin\win64\tell.exe" BME-BANER-01 58447 CLEANUP_EXITING
timeout /t 1
"C:\PROGRA~1\ANSYSI~1\v242\fluent\ntbin\win64\kill.exe" tell.exe
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 26392) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 928) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 10956) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 25888) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 21604) 
if /i "%LOCALHOST%"=="BME-BANER-01" (%KILL_CMD% 17316)
del "C:\Users\Research\Desktop\CBT_pred\cleanup-fluent-BME-BANER-01-21604.bat"
