#!/bin/bash -e

git_version=$(git --version)
if [ >= "1.8.4" ]; then
    echo "Since git 1.8.4 (August 2013), and commit 091a6eb, you don't have to be at top-level to run git submodule update."
else
    echo "You have to update your git version to 1.8.4 or later."
    exit -1
fi

cd $(dirname $0)

git submodule sync
git submodule update --init intel-mkl-dnn
git submodule update --init Halide
git submodule update --init protobuf
git submodule update --init flatbuffers
git submodule update --init gtest

git submodule update --init OpenBLAS
git submodule update --init libzmq
git submodule update --init cppzmq

git submodule update --init MegRay
pushd MegRay/third_party >/dev/null
    git submodule sync
    git submodule update --init nccl
    git submodule update --init gdrcopy
    git submodule update --init ucx
popd >/dev/null
