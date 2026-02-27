@echo off
setlocal

set HOST=0.0.0.0
set PORT=8010

set SCRIPT_DIR=%~dp0
set LOCAL_VENV_PY=%SCRIPT_DIR%\.venv\Scripts\python.exe
for %%I in ("%SCRIPT_DIR%..\..") do set REPO_ROOT=%%~fI
if not exist "%LOCAL_VENV_PY%" (
  echo Missing local venv for recorder backend:
  echo   %LOCAL_VENV_PY%
  echo.
  echo Install local dependencies first:
  echo   "%SCRIPT_DIR%\install_recorder_deps.bat"
  exit /b 1
)

set PYTHON_EXEC=%LOCAL_VENV_PY%
"%PYTHON_EXEC%" -c "import uvicorn, fastapi, multipart" 1>nul 2>nul
if errorlevel 1 (
  echo Missing recorder backend dependencies in local venv.
  echo Install them with:
  echo   "%SCRIPT_DIR%\install_recorder_deps.bat"
  exit /b 1
)

set PYTHONPATH=%REPO_ROOT%
"%PYTHON_EXEC%" -m uvicorn recorder_backend:app --host %HOST% --port %PORT% --app-dir "%SCRIPT_DIR%"
endlocal
