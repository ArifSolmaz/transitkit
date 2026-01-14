@echo off
echo Building TransitKit executable for Windows...

REM Clean previous builds
rmdir /s /q build 2>nul
rmdir /s /q dist 2>nul

REM Install PyInstaller
pip install pyinstaller

REM Build executable
pyinstaller ^
    --name transitkit ^
    --onefile ^
    --hidden-import numpy ^
    --hidden-import scipy ^
    --hidden-import matplotlib ^
    --add-data "%python%\Lib\site-packages\matplotlib\mpl-data;matplotlib-data" ^
    src/transitkit/cli.py

echo Build complete! Executable is in dist\transitkit.exe