#!/usr/bin/env bash

shopt -s expand_aliases
alias pip="/root/.local/bin/pip3.6"
alias python="python3.6"

echo "Need pytorch>=1.0.0"
source activate pytorch1.0.0

cd MinDepthFlowProjection
rm -rf build *.egg-info dist
python setup.py install
cd ..

cd FlowProjection
rm -rf build *.egg-info dist
python setup.py install
cd ..

cd SeparableConv
rm -rf build *.egg-info dist
python setup.py install
cd ..

cd InterpolationCh
rm -rf build *.egg-info dist
python setup.py install
cd ..

cd DepthFlowProjection
rm -rf build *.egg-info dist
python setup.py install
cd ..

cd Interpolation
rm -rf build *.egg-info dist
python setup.py install
cd ..

cd SeparableConvFlow
rm -rf build *.egg-info dist
python setup.py install
cd ..

cd FilterInterpolation
rm -rf build *.egg-info dist
python setup.py install
cd ..

