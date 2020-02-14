#!/bin/bash -e

CWD=$(dirname $0)
BASEDIR=$(readlink -f ${CWD}/../../..)
OUTPUTDIR=$(readlink -f ${CWD}/output)
USERID=$(id -u)
TMPFS_ARGS="--tmpfs /tmp:exec"

pushd ${BASEDIR}/third_party >/dev/null
    ./prepare.sh
popd >/dev/null

cd ${CWD}
mkdir -p ${OUTPUTDIR}

if [[ -z ${CUDA_ROOT_DIR} ]]; then
echo "Environment variable CUDA_ROOT_DIR not set."
exit -1
fi
if [[ -z ${CUDNN_ROOT_DIR} ]]; then
echo "Environment variable CUDNN_ROOT_DIR not set."
exit -1
fi
if [[ -z ${TENSORRT_ROOT_DIR} ]]; then
echo "Environment variable TENSORRT_ROOT_DIR not set."
exit -1
fi

docker run -it --rm $TMPFS_ARGS -e UID=${USERID} -e LOCAL_VERSION=${LOCAL_VERSION} -e ALL_PYTHON=${ALL_PYTHON} -v ${CUDA_ROOT_DIR}:/usr/local/cuda -v ${CUDNN_ROOT_DIR}:/opt/cudnn -v ${TENSORRT_ROOT_DIR}:/opt/tensorrt -v ${BASEDIR}:/home/code -v ${OUTPUTDIR}:/home/output:rw env_manylinux2010:latest /home/code/ci/docker_env/manylinux2010/do_build.sh


