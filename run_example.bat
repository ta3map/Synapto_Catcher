@echo off
REM Activate the virtual environment
call ПУТЬ К ОКРУЖЕНИЮ venv\Scripts\activate.bat
REM Run the Python script
pythonw ПУТЬ К СКРИПТУ syn_catch\syn_catch_GUI.py
REM Deactivate the virtual environment (optional)
deactivate
