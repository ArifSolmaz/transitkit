#!/bin/bash
# Build executables for different platforms

echo "Building TransitKit executables..."

# Clean previous builds
rm -rf build/ dist/

# Install PyInstaller if not installed
pip install pyinstaller

# Build for current platform
echo "Building for $(uname)..."
pyinstaller \
    --name transitkit \
    --onefile \
    --hidden-import numpy \
    --hidden-import scipy \
    --hidden-import matplotlib \
    --add-data "$(python -c 'import matplotlib; print(matplotlib.get_data_path())'):matplotlib-data" \
    src/transitkit/cli.py

echo "Build complete! Executable is in dist/"