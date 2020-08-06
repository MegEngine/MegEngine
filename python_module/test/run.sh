#!/bin/bash -e

ignore_list="--ignore test/unit/module/test_pytorch.py \
            --ignore test/pytorch_comparison \
            --ignore test/unit/hub/test_hub.py \
            --ignore test/unit/data \
            --ignore test/integration/manual \
            --ignore megengine/module/pytorch \
            --ignore test/unit/module/test_external.py"
test_dirs="megengine test"

pushd $(dirname "${BASH_SOURCE[0]}")/.. >/dev/null
    python3 -m pytest -xv -m 'isolated_distributed' \
        --json-report --json-report-file=time_python_test.json \
        $ignore_list $test_dirs
    python3 -m pytest -xv -m 'not internet and not isolated_distributed' \
        --json-report --json-report-file=time_python_test.json \
        $ignore_list $test_dirs
popd >/dev/null
