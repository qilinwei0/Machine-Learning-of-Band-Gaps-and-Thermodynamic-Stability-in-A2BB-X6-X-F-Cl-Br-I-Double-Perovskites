@echo off
setlocal
if "%~2"=="" (
  echo Usage: run_v9_ml.bat INPUT_DIR OUTPUT_ROOT
  exit /b 2
)
set "SCRIPT_DIR=%~dp0"
python "%SCRIPT_DIR%generate_v9_ml_data.py" --input-dir "%~1" --output-root "%~2"
if errorlevel 1 exit /b %errorlevel%
python "%SCRIPT_DIR%validate_v9_data.py" --package-root "%~2"
exit /b %errorlevel%
