#!/bin/bash

set -e


BASEDIR=$(readlink -f "$(dirname "$0")"/..)

function python_test() {
    pushd "${BASEDIR}"/python_module >/dev/null
        pip3 install -e '.[ci]'
        export PYTHONPATH=.
        ./test/run.sh
    popd >/dev/null
}

python_test
