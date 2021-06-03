#!/bin/bash -e

SRC_DIR=$(readlink -f "`dirname $0`/../../../")
source ${SRC_DIR}/scripts/whl/utils/utils.sh
config_docker_file

cd $(dirname $0)
echo "docker_file is ${docker_file}"
docker build -t env_manylinux2014:latest -f ${docker_file} .
