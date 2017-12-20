#!/bin/sh

curl -fsSL get.docker.com -o get-docker.sh
sudo sh get-docker.sh --mirror Aliyun
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER

docker build -t ml-example:latest -f Dockerfile .
