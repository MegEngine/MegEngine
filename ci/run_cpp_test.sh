#!/bin/bash

set -e

BASEDIR=$(readlink -f "$(dirname "$0")"/..)

export MGB_TEST_NO_LOG=1
export MGB_STABLE_RNG=1

function cpp_test {
    pushd /tmp/build/"${1}" >/dev/null
        ./dnn/test/megdnn_test
        ./test/megbrain_test
    popd >/dev/null
}


if [[ "$1" == "cpu" || "$1" == "cuda" ]] ; then
    cpp_test "$@"
else
    echo "Argument must cpu or cuda"
    exit 1
fi
