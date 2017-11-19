#!/bin/bash

# 1. install java
sudo add-apt-repository ppa:openjdk-r/ppa
sudo apt-get update
sudo apt-get -y install openjdk-8-jdk

# 2. install docker

# 2-1. Set up the docker repository
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

# 2-2. Install Docker CE
sudo apt-get update
sudo apt-get install -y docker-ce

# 2-3. Install docker-compose
sudo curl -L https://github.com/docker/compose/releases/download/1.17.0/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose