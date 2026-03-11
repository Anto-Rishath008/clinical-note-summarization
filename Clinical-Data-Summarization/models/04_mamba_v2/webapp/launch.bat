@echo off
cd /d "%~dp0.."
echo.
echo ============================================
echo   MambaClinic - Clinical Note Summarization
echo ============================================
echo.
echo Installing dependencies...
"c:\Users\antor\OneDrive\Desktop\3rd year\SEMESTER-6\NLP\Project\Codes\.conda\python.exe" -m pip install flask --quiet 2>nul
echo.
echo Starting server...
echo.
"c:\Users\antor\OneDrive\Desktop\3rd year\SEMESTER-6\NLP\Project\Codes\.conda\python.exe" webapp/server.py --port 5000
pause
