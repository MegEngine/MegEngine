#!/bin/bash -e

cd $(dirname $0)

requiredGitVersion="1.8.4"
currentGitVersion="$(git --version | awk '{print $3}')"
if [ "$(printf '%s\n' "$requiredGitVersion" "$currentGitVersion" | sort -V | head -n1)" = "$currentGitVersion" ]; then
    echo "Found Git: $(which git). (foud version $currentGitVersion, required version >= $requiredGitVersion)"
else
    echo "Please update your Git version. (foud version $currentGitVersion, required version >= $requiredGitVersion)"
    exit 1
fi

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
