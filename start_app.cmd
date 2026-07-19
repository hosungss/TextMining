@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "APP_PYTHON=.venv\Scripts\python.exe"
if not exist "%APP_PYTHON%" (
  echo [1/3] Python 가상환경을 만듭니다.
  python -m venv .venv
  if errorlevel 1 (
    echo Python 가상환경을 만들지 못했습니다. Python 3.10 이상 설치를 확인하세요.
    pause
    exit /b 1
  )
)

echo [2/3] 필요한 라이브러리를 확인합니다.
"%APP_PYTHON%" -c "import streamlit, plotly, scipy, statsmodels, sklearn, networkx" 2>nul
if errorlevel 1 (
  "%APP_PYTHON%" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo 필요한 라이브러리를 설치하지 못했습니다. 인터넷 연결을 확인하세요.
    pause
    exit /b 1
  )
)

echo [3/3] YouTube Research Studio를 시작합니다.
"%APP_PYTHON%" run_app.py
if errorlevel 1 pause
