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

function update_cpuinfo() {
    git submodule update -f --init cpuinfo

    name=`git config --get user.name || true`
    if [ -z "$name" ]; then
        name="default"
    fi
    email=`git config --get user.email || true`
    if [ -z "$email" ]; then
        email="default"
    fi

    source ./apply-patches.sh
    apply_cpuinfo_patches ${name} ${email}
}

function git_submodule_update() {
    git submodule sync
    git submodule update -f --init midout
    git submodule update -f --init intel-mkl-dnn
    git submodule update -f --init Halide
    git submodule update -f --init protobuf
    git submodule update -f --init gtest
    git submodule update -f --init flatbuffers
    git submodule update -f --init cutlass
    git submodule update -f --init Json
    git submodule update -f --init pybind11
    git submodule update -f --init llvm-project
    git submodule update -f --init range-v3
    git submodule update -f --init libzmq
    git submodule update -f --init cppzmq
    git submodule update -f --init OpenBLAS
    update_cpuinfo

    git submodule update -f --init MegRay
    pushd MegRay/third_party >/dev/null
        git submodule sync
        git submodule update -f --init nccl
        git submodule update -f --init gdrcopy
        git submodule update -f --init ucx
    popd >/dev/null
}

if [[ -z "${ALREADY_UPDATE_SUBMODULES}" ]]; then
    git_submodule_update
fi
