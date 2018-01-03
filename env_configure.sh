#!/usr/bin/env bash

# 用于 Ubuntu 环境配置
# autuanliu@163.com
# status: 暂未测试

# 权限
sudo -i

# MiniConda install
'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh &&
wget -nv https://repo.continuum.io/miniconda/Miniconda3-4.3.27-Linux-x86_64.sh -O ~/anaconda.sh &&
/bin/bash ~/anaconda.sh -b -p /opt/conda && rm ~/anaconda.sh

# channel set
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ &&
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ &&
conda config --set show_channel_urls yes

# opencv
apt-get -qq install -y libsm6 libxext6 && pip3 install -q -U opencv-python

# graphviz
apt-get -qq install -y graphviz && pip3 install -q pydot

# conda pkgs install
conda install pillow, scipy

# pip pkgs install
pip3 install -r pip_pkgs.log
