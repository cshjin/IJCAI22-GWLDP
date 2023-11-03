#!/bin/bash

CUDA="${1:=0}"

# create virtual env
conda create -n gwldp python=3.8 -y
# DEBUG: activate problem
conda activate gwldp

# install packages for GPU/CPU
if [ $CUDA -eq 1 ];
then
  conda install -n gwldp \
        pytorch torchvision torchaudio cudatoolkit=11.3 \
        pyg \
        pot \
        -c pytorch \
        -c pyg \
        -c conda-forge \
        -y
else
  conda install -n gwldp \
        pytorch torchvision torchaudio cpuonly \
        pyg \
        pot \
        -c pytorch \
        -c pyg \
        -c conda-forge \
        -y
fi

