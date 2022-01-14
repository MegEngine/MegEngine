#!/bin/bash

set -e

BASEDIR=$(readlink -f "$(dirname "$0")"/..)

source "${BASEDIR}/ci/utils.sh"

if [[ "$1" == cpu ]]; then
    DMGE_WITH_DISTRIBUTED=OFF
    DMGE_WITH_CUDA=OFF
elif [[ "$1" == cuda ]]; then
    DMGE_WITH_DISTRIBUTED=ON
    DMGE_WITH_CUDA=ON
else
    log "Argument must cpu or cuda"
    exit 1
fi


function build() {
    log "Start to build"
    local build_dir="/tmp/build/${1}"
    mkdir -p "$build_dir"
    pushd ${build_dir} >/dev/null
        cmake -S "${BASEDIR}" -B "${build_dir}" \
            -DMGE_WITH_DISTRIBUTED=${DMGE_WITH_DISTRIBUTED} \
            -DMGE_WITH_CUDA=${DMGE_WITH_CUDA} \
            -DMGE_WITH_TEST=ON \
            -DCMAKE_BUILD_TYPE=RelWithDebInfo \
	    -DMGE_WITH_CUSTOM_OP=ON
        make -j$(($(nproc) * 2)) -I ${build_dir}
        make develop
    popd >/dev/null
    log "End build: $(ls ${build_dir})"
}

build "$@"
