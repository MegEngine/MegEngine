#!/bin/bash -e

cd $(dirname $0)

source ../ci/utils.sh

requiredGitVersion="1.8.4"
currentGitVersion="$(git --version | awk '{print $3}')"
if [ "$(printf '%s\n' "$requiredGitVersion" "$currentGitVersion" | sort -V | head -n1)" = "$currentGitVersion" ]; then
    echo "Please update your Git version. (foud version $currentGitVersion, required version >= $requiredGitVersion)"
    exit -1
fi

log "Start downloading git submodules"
git submodule sync
git submodule update -f --init midout
git submodule update -f --init intel-mkl-dnn
git submodule update -f --init HalidePrebuilt
git submodule update -f --init Halide
git submodule update -f --init protobuf
git submodule update -f --init gtest
git submodule update -f --init flatbuffers
git submodule update -f --init cutlass
git submodule update -f --init Json

git submodule update -f --init cpuinfo

name=`git config --get user.name`
if [ -z "$name" ]; then
    name="default"
fi
email=`git config --get user.email`
if [ -z "$email" ]; then
    email="default"
fi

source ./apply-patches.sh
apply_cpuinfo_patches ${name} ${email}

git submodule update -f --init OpenBLAS
if [[ ! -d mkl/$(uname -m) ]]; then
    git submodule update -f --init mkl
fi

git submodule update -f --init libzmq
git submodule update -f --init cppzmq

git submodule update -f --init MegRay
pushd MegRay/third_party >/dev/null
    git submodule sync
    git submodule update -f --init nccl
    git submodule update -f --init gdrcopy
    git submodule update -f --init ucx
    git submodule update -f --init rccl
popd >/dev/null

git submodule update -f --init pybind11
git submodule update -f --init llvm-project
git submodule update -f --init mc40
git submodule update -f --init range-v3

log "Finished downloading git submodules"
