#!/bin/bash
set -e

function apply_cpuinfo_patches() {
    if [ -d "./cpuinfo" ]; then
        cp ./patches/0001-fix-arm-linux-fix-uclibc-build-error.patch cpuinfo/
        pushd cpuinfo >/dev/null
        git am --abort || true
        git rebase --abort || true
        git reset --hard
        # Use git apply instead of git am to prevent git require name in .gitconfig
        GIT_AUTHOR_NAME='patcher' GIT_AUTHOR_EMAIL='patcher@nobody.com' git apply 0001-fix-arm-linux-fix-uclibc-build-error.patch
        rm *.patch
        popd >/dev/null
    else
        echo "cpuinfo not exist, can not apply patches."
        echo "Please run: 'git submodule update --init cpuinfo' first."
        exit -1
    fi
}
