#!/bin/bash -e

SRC_DIR=$(readlink -f "`dirname $0`/../../../")
source ${SRC_DIR}/scripts/whl/utils/utils.sh
config_docker_file

cd $(dirname $0)
echo "docker_file is ${docker_file}"
docker build -t env_manylinux2014:latest -f ${docker_file} .

OS=$(uname -m)
if [ $OS = "aarch64"  ] && ["${cuda_version}" != "cann_8_0_RC1_alpha003"]; then
    # as nivida trt-8.5.3.1 build from ubuntu20.04, which depends on new glibc, env_manylinux2014 do not satisfy, so need build aarch64-ubuntu20.04 docker
    echo "as in aarch64, need build aarch64-ubuntu20.04 docker"
    docker build -t ubuntu2004:latest -f Dockerfile_aarch64_ubuntu2004 .
fi
