@echo off
setlocal

set SCRIPT_DIR=%~dp0
set VENV_DIR=%SCRIPT_DIR%\.venv
set PYTHON_EXE=%VENV_DIR%\Scripts\python.exe

if not exist "%VENV_DIR%" (
  python -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo Failed to create local venv at:
    echo   %VENV_DIR%
    exit /b 1
  )
)

"%PYTHON_EXE%" -m pip install --upgrade pip
if errorlevel 1 exit /b 1

"%PYTHON_EXE%" -m pip install -r "%SCRIPT_DIR%\requirements.txt"
if errorlevel 1 exit /b 1

echo Recorder dependencies installed in:
echo   %VENV_DIR%
endlocal
