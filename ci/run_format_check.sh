#!/bin/bash

set -e

pushd imperative/python >/dev/null
    pip install -q -r requires.txt
    pip install -q -r requires-style.txt
    pip install -q -r requires-test.txt
    ./scripts/format.sh -d
popd >/dev/null
