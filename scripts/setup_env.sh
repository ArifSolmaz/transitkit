#!/bin/bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev,full]"
pre-commit install
