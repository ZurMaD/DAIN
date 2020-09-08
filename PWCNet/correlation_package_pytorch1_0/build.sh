#!/usr/bin/env bash

shopt -s expand_aliases
alias pip="/root/.local/bin/pip3.6"
alias python="python3.6"

echo "Need pytorch>=1.0.0"
source activate pytorch1.0.0


rm -rf build *.egg-info dist
python setup.py install
