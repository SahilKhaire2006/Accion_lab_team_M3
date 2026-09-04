@echo off
echo ========================================
echo Installing dependencies with fix for av
echo ========================================

REM Activate virtual environment
call venv\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

REM Install dependencies that don't require compilation first
pip install numpy==1.26.3
pip install fastapi==0.109.0 uvicorn==0.27.0 python-multipart==0.0.6
pip install soundfile==0.12.1
pip install pydub==0.25.1
pip install librosa==0.10.1
pip install python-dotenv==1.0.0

REM Try to install av with precompiled wheel
echo.
echo Installing PyAV (av) - this may take a moment...
pip install av

REM Install faster-whisper
echo.
echo Installing faster-whisper...
pip install faster-whisper==1.0.0

echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.
echo To start the server, run:
echo   venv\Scripts\activate
echo   python -m uvicorn app.main:app --reload
echo.
pause
