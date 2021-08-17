#!/bin/bash
set -e

function apply_cpuinfo_patches() {
    if [ -d "./cpuinfo" ]; then
        cp ./patches/cpuinfo-multi-commits.patch cpuinfo/
        pushd cpuinfo >/dev/null
        git config user.name "$1"
        git config user.email "$2"
        git am cpuinfo-multi-commits.patch
        rm cpuinfo-multi-commits.patch
        popd >/dev/null
    else
        echo "cpuinfo not exist, can not apply patches."
        echo "pls run: 'git submodule update --init cpuinfo' first."
        exit -1
    fi
}
