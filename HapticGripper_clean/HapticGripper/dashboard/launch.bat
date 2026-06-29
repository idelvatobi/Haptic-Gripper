@echo off
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Starting Haptic Gripper Dashboard...
streamlit run app.py
pause
