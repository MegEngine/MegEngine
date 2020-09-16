#!/bin/bash

set -e


BASEDIR=$(readlink -f "$(dirname "$0")"/..)

function python_test() {
    pip3 install --upgrade pip
    pushd "${BASEDIR}"/imperative/python >/dev/null
        pip3 install -e '.[ci]'
        export PYTHONPATH=.
        ./test/run.sh $1
    popd >/dev/null
}

python_test $1
