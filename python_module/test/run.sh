#!/bin/bash -e

pushd $(dirname "${BASH_SOURCE[0]}")/.. >/dev/null
    pytest -xv -m 'not internet' \
        --json-report --json-report-file=time_python_test.json \
        --ignore test/unit/module/test_pytorch.py \
        --ignore test/pytorch_comparison \
        --ignore test/unit/hub/test_hub.py \
        --ignore test/unit/data \
        --ignore test/integration/manual \
        --ignore megengine/module/pytorch \
        megengine test
popd >/dev/null
