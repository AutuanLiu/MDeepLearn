#!/bin/bash

# 用于 Ubuntu 环境配置, 当然可以用于配置 云服务器环境
# 参考我的另一个库 https://github.com/AutuanLiu/ML-Docker-Env
# autuanliu@163.com
# status: ubuntu 16.04 测试 pass
# 复制 pip_pkg.txt 文件到同一目录
# chmod +x env_configure.sh
# ./env_configure.sh

condaDir=~/softwares/conda

# MiniConda install
#curl -L https://repo.continuum.io/miniconda/Miniconda3-4.3.27-Linux-x86_64.sh -o ~/anaconda.sh
bash ~/anaconda.sh -b -p $condaDir
#rm ~/anaconda.sh

# add to path
export PATH=$condaDir/bin:$PATH

# channel set
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ &&
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ &&
conda config --set show_channel_urls yes

# opencv
sudo apt -qq install -y libsm6 libxext6 && pip install -q -U opencv-python &&

# graphviz
sudo apt -qq install -y graphviz && pip install -q pydot

# conda pkgs install
conda install -y pillow scipy

# xgboost
conda install -y -c conda-forge xgboost

# pytorch
conda install pytorch torchvision -c pytorch

# R
conda config --system --append channels r &&
conda install -y rpy2=2.8* r-base=3.3.2 r-irkernel=0.7* r-plyr=1.8* r-devtools=1.12* r-tidyverse=1.0* &&
conda install -y r-shiny=0.14* r-rmarkdown=1.2* r-forecast=7.3* r-rsqlite=1.1* r-reshape2=1.4* &&
conda install -y r-nycflights13=0.2* r-caret=6.0* r-rcurl=1.95* r-crayon=1.3* r-randomforest=4.6* &&
conda clean -tipsy

# pip pkgs install
pip install -r ./requirements/pip_pkgs.txt

# autoenv activation
echo "source `which activate.sh`" >> ~/.bashrc

# jupyter notebook
python -m pip install jupyter

# else
echo "PATH=$condaDir/bin:$PATH" >> ~/.bashrc && bash
