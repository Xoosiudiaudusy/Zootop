@echo off
rem NegativePluribus web table: start the server if it is not running, then open the browser.
rem ASCII only on purpose: cmd.exe misreads batch files with UTF-8 text after "chcp 65001".
cd /d "%~dp0.."
set "PORT=8777"
if not "%NP_WEB_PORT%"=="" set "PORT=%NP_WEB_PORT%"
call :listening
if not errorlevel 1 goto open
echo Starting the server on port %PORT% ...
set "NP_WEB_NO_BROWSER=1"
start "NegativePluribus table" /min cmd /c "python webapp\server.py 1>webapp\server_out.log 2>webapp\server_err.log"
set /a TRIES=0
:wait
ping -n 2 127.0.0.1 >nul
call :listening
if not errorlevel 1 goto open
set /a TRIES+=1
if %TRIES% lss 40 goto wait
echo The server did not start in 40 s. Last lines of webapp\server_err.log:
powershell -NoProfile -Command "Get-Content 'webapp\server_err.log' -Tail 20"
pause
exit /b 1
:open
echo Table: http://127.0.0.1:%PORT%  (the first bot loads a few seconds; press F5 if the page is early)
if "%NP_WEB_NO_OPEN%"=="1" exit /b 0
start "" "http://127.0.0.1:%PORT%"
exit /b 0
:listening
netstat -ano -p tcp | findstr /r /c:"127\.0\.0\.1:%PORT% .*LISTENING" >nul
exit /b %errorlevel%
