@echo off
rem NegativePluribus web table: stop the server (the PID is written by server.py into webapp\server.pid).
rem Only a process whose command line is webapp\server.py is stopped, never another python.
cd /d "%~dp0"
if not exist server.pid (
  echo No server.pid: the server does not seem to be running.
  exit /b 0
)
set /p SRVPID=<server.pid
powershell -NoProfile -Command "$p = Get-CimInstance Win32_Process -Filter 'ProcessId=%SRVPID%'; if ($p -and $p.CommandLine -match 'webapp.server\.py') { Stop-Process -Id %SRVPID% -Force; 'Server stopped.' } else { 'Process %SRVPID% is not the web server (already stopped).' }"
del server.pid >nul 2>&1
