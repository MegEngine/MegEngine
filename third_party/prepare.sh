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

SYNC_LLVM_PROJECT=True
SYNC_GTEST_PROJECT=True
SYNC_DNNL_PROJECT=True
SYNC_HALIDE_PROJECT=True
SYNC_PROTOBUF_PROJECT=True
SYNC_CUTLASS_PROJECT=True
SYNC_IMPERATIVE_RT_PROJECT=True
SYNC_DISTRIBUTED_PROJECT=True
SYNC_OPENBLAS_PROJECT=True
function usage() {
    echo "$0 args1 args2 .."
    echo "available args detail:"
    echo "-a : do not sync llvm-project"
    echo "-b : do not sync gtest"
    echo "-c : do not sync intel-mkl-dnn"
    echo "-d : do not sync Halide"
    echo "-e : do not sync protobuf"
    echo "-f : do not sync cutlass"
    echo "-g : do not sync IMPERATIVE_RT project"
    echo "-i : do not sync DISTRIBUTED project"
    echo "-j : do not sync OpenBLAS project"
    echo "-h : show usage"
    exit -1
}

while getopts "abcdefghij" arg
do
    case $arg in
        a)
            echo "do not sync llvm-project"
            SYNC_LLVM_PROJECT=False
            ;;
        b)
            echo "do not sync gtest"
            SYNC_GTEST_PROJECT=False
            ;;
        c)
            echo "do not sync intel-mkl-dnn"
            SYNC_DNNL_PROJECT=False
            ;;
        d)
            echo "do not sync Halide"
            SYNC_HALIDE_PROJECT=False
            ;;
        e)
            echo "do not sync protobuf"
            SYNC_PROTOBUF_PROJECT=False
            ;;
        f)
            echo "do not sync cutlass"
            SYNC_CUTLASS_PROJECT=False
            ;;
        g)
            echo "do not sync IMPERATIVE_RT project"
            SYNC_IMPERATIVE_RT_PROJECT=False
            ;;
        i)
            echo "do not sync DISTRIBUTED project"
            SYNC_DISTRIBUTED_PROJECT=False
            ;;
        j)
            echo "do not sync OpenBLAS project"
            SYNC_OPENBLAS_PROJECT=False
            ;;
        h)
            echo "show usage"
            usage
            ;;
        ?)
            echo "unkonw argument"
            usage
            ;;
    esac
done
function git_submodule_update() {
    git submodule sync
    git submodule update -f --init midout
    git submodule update -f --init flatbuffers
    git submodule update -f --init Json
    git submodule update -f --init gflags
    git submodule update -f --init cpuinfo
    git submodule update -f --init cpp_redis
    git submodule update -f --init tacopie
    git submodule update -f --init cudnn-frontend

    if [ ${SYNC_DNNL_PROJECT} = "True" ];then
        git submodule update -f --init intel-mkl-dnn
    fi

    if [ ${SYNC_HALIDE_PROJECT} = "True" ];then
        git submodule update -f --init Halide
    fi

    if [ ${SYNC_PROTOBUF_PROJECT} = "True" ];then
        git submodule update -f --init protobuf
    fi

    if [ ${SYNC_GTEST_PROJECT} = "True" ];then
        git submodule update -f --init gtest
    fi

    if [ ${SYNC_CUTLASS_PROJECT} = "True" ];then
        git submodule update -f --init cutlass
    fi

    if [ ${SYNC_LLVM_PROJECT} = "True" ];then
        git submodule update -f --init llvm-project
    fi

    if [ ${SYNC_IMPERATIVE_RT_PROJECT} = "True" ];then
        git submodule update -f --init pybind11
        git submodule update -f --init range-v3
    fi

    if [ ${SYNC_DISTRIBUTED_PROJECT} = "True" ];then
        git submodule update -f --init libzmq
        git submodule update -f --init cppzmq
        git submodule update -f --init MegRay
        pushd MegRay/third_party >/dev/null
        git submodule sync
        git submodule update -f --init nccl
        git submodule update -f --init gdrcopy
        git submodule update -f --init ucx
        popd >/dev/null
    fi

    if [ ${SYNC_OPENBLAS_PROJECT} = "True" ];then
        git submodule update -f --init OpenBLAS
    fi
}

if [[ -z "${ALREADY_UPDATE_SUBMODULES}" ]]; then
    git_submodule_update
fi
