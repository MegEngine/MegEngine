#!/bin/bash -e

cd $(dirname $0)

# force use /usr/bin/sort on windows, /c/Windows/system32/sort do not support -V
OS=$(uname -s)
SORT=sort
if [[ $OS =~ "NT" ]]; then
    SORT=/usr/bin/sort
fi

requiredGitVersion="1.8.4"
currentGitVersion="$(git --version | awk '{print $3}')"
if [ "$(printf '%s\n' "$requiredGitVersion" "$currentGitVersion" | ${SORT} -V | head -n1)" = "$currentGitVersion" ]; then
    echo "Please update your Git version. (foud version $currentGitVersion, required version >= $requiredGitVersion)"
    exit -1
fi

git submodule sync

git submodule foreach --recursive git reset --hard
git submodule foreach --recursive git clean -fd


git submodule update --init midout
git submodule update --init intel-mkl-dnn
git submodule update --init Halide
git submodule update --init protobuf
git submodule update --init flatbuffers
git submodule update --init gtest
git submodule update --init cutlass

git submodule update --init cpuinfo
source ./apply-patches.sh
apply_cpuinfo_patches

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

git submodule update --init pybind11
git submodule update --init llvm-project
git submodule update --init range-v3
